"""Entrenamiento jerárquico de modelos ColSign.

Carga `dataset_colsign_45_154.h5` (mismas features normalizadas que usa
`train_lstm.py`) y entrena 5 modelos LSTM en cascada:

  1) Modelo raíz: clasifica entre los 4 grupos morfológicos del CSV
     `etiquetas_modelo_raiz.csv` (estático / unimanual / bimanual
     simétrico / bimanual asimétrico).

  2) 4 sub-modelos, uno por cada grupo raíz, donde cada sub-modelo
     entrena SOLO con las etiquetas pertenecientes a ese grupo:
        - estático:           ~27 clases
        - unimanual:          ~58 clases
        - bimanual simétrico: ~30 clases
        - bimanual asimétrico:~39 clases

Idea: en inferencia, el modelo raíz decide el grupo y luego el
sub-modelo correspondiente identifica la seña específica. Cada
sub-modelo enfrenta una tarea más pequeña que clasificar las 154
clases de golpe, lo cual suele mejorar la precisión.

Genera, para cada uno de los 5 modelos, los mismos artefactos que
`train_lstm.py`:
  - models/{MODEL_NAME}.keras
  - models/{MODEL_NAME}_best.keras
  - info_models/{MODEL_NAME}_train_log.txt
  - info_models/{MODEL_NAME}_labels.json
  - graphics/{MODEL_NAME}_loss.jpeg
  - graphics/{MODEL_NAME}_accuracy.jpeg

Más un resumen jerárquico en
`info_models/colsign_lstm_norm_45_154_jerarquia_resumen.json`.

Ejecutar:
    .\.venv\Scripts\python.exe -u train_lstm_cluster_labels.py
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from src.utils import load_hdf5_dataset

# stdout/stderr en UTF-8 (igual que train_lstm.py): evita UnicodeEncodeError
# cuando se redirige a archivo en Windows.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


# =====================================================================
# Configuración general
# =====================================================================

DATASET_HDF5 = 'dataset_colsign_45_154.h5'
CSV_PATH     = 'etiquetas_modelo_raiz_v2.csv'

# Nombre del modelo raíz (clasifica el grupo morfológico).
MODEL_NAME_RAIZ = 'colsign_lstm_norm_raiz_45_154_v2'

# Mapeo de grupo raíz -> nombre del sub-modelo correspondiente.
# Los nombres y la grafía deben coincidir EXACTAMENTE con los valores de la
# columna `etiqueta_raiz` del CSV (sensible a tildes y mayúsculas).
ROOT_TO_MODEL_NAME = {
    'Grupo Estático':                     'colsign_lstm_norm_estatic_45_154_v2',
    'Grupo Dinámico Unimanual':           'colsign_lstm_norm_unimanual_45_154_v2',
    #'Grupo Dinámico Bimanual Simétrico':  'colsign_lstm_norm_bi_simetrico_45_154',
    #'Grupo Dinámico Bimanual Asimétrico': 'colsign_lstm_norm_bi_asimetrico_45_154',
}

# Hiperparámetros: los mismos que dieron 85% en train_lstm.py.
SEED            = 42
TEST_SIZE       = 0.2
BATCH_SIZE      = 64
EPOCHS          = 300
LEARNING_RATE   = 1e-3
PATIENCE_EARLY  = 30
PATIENCE_LR     = 10
DROPOUT_RATE    = 0.20

# Preprocesamiento de features (idéntico a train_lstm.py)
NORMALIZE_KEYPOINTS  = True
DROP_POSE_VISIBILITY = True

# Carpetas
MODELS_DIR   = 'models'
INFO_DIR     = 'info_models'
GRAPHICS_DIR = 'graphics'
for d in (MODELS_DIR, INFO_DIR, GRAPHICS_DIR):
    os.makedirs(d, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================================
# Construcción del modelo (idéntica arquitectura a train_lstm.py)
# =====================================================================

def build_model(seq_len, num_features, num_classes):
    """Misma arquitectura LSTM que dio 85% en el modelo de 154 clases."""
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        LSTM(128, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(64, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(128, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# =====================================================================
# Función de entrenamiento reutilizable
# =====================================================================

def train_one_model(
    X,
    y_int,
    label_names,
    model_name,
    extra_log_info=None,
):
    """Entrena un LSTM y guarda todos los artefactos bajo el prefijo
    `model_name` (mismo formato que `train_lstm.py`).

    Args:
        X: ndarray (N, T, F) float32, features normalizadas.
        y_int: ndarray (N,) int64, etiquetas como índices en [0, K).
        label_names: list[str] con los K nombres de clase. label_names[i]
            es el nombre de la clase con índice i.
        model_name: prefijo de los archivos de salida (.keras, .json, .jpeg,
            .txt).
        extra_log_info: dict opcional con info adicional para el log y el
            JSON de labels (por ejemplo, indicar que es un sub-modelo del
            grupo X).

    Returns:
        dict con un resumen de las métricas finales.
    """
    print(f"\n{'=' * 70}")
    print(f"=== Entrenamiento: {model_name}")
    print('=' * 70)

    seq_len       = X.shape[1]
    num_features  = X.shape[2]
    num_classes   = len(label_names)
    total_samples = X.shape[0]
    print(f"  Samples: {total_samples}  Clases: {num_classes}  "
          f"Shape: {X.shape}")

    # ---------------- Labels JSON ----------------
    labels_json_path = os.path.join(INFO_DIR, f"{model_name}_labels.json")
    labels_payload = {
        'model_name'         : model_name,
        'num_classes'        : num_classes,
        'sequence_length'    : int(seq_len),
        'num_features'       : int(num_features),
        'normalize_keypoints': NORMALIZE_KEYPOINTS,
        'drop_pose_visibility': DROP_POSE_VISIBILITY,
        'created_at'         : datetime.now().isoformat(timespec='seconds'),
        'id_to_name'         : {str(i): n for i, n in enumerate(label_names)},
        'name_to_id'         : {n: i      for i, n in enumerate(label_names)},
    }
    if extra_log_info:
        labels_payload['extra'] = extra_log_info
    with open(labels_json_path, 'w', encoding='utf-8') as fp:
        json.dump(labels_payload, fp, ensure_ascii=False, indent=2)
    print(f"  Labels JSON:  {labels_json_path}")

    # ---------------- Split estratificado ----------------
    try:
        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X, y_int, test_size=TEST_SIZE, random_state=SEED, stratify=y_int,
        )
        split_strategy = 'estratificado por clase'
    except ValueError as e:
        print(f"  No se pudo estratificar ({e}); split aleatorio simple")
        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X, y_int, test_size=TEST_SIZE, random_state=SEED,
        )
        split_strategy = 'aleatorio (no estratificado)'

    y_train = to_categorical(y_train_int, num_classes=num_classes)
    y_test  = to_categorical(y_test_int,  num_classes=num_classes)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}  ({split_strategy})")

    # ---------------- Modelo ----------------
    model = build_model(seq_len, num_features, num_classes)
    model.summary()

    # ---------------- Callbacks ----------------
    best_ckpt_path = os.path.join(MODELS_DIR, f"{model_name}_best.keras")
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=PATIENCE_EARLY,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=PATIENCE_LR, min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_ckpt_path, monitor='val_loss',
            save_best_only=True, verbose=0,
        ),
    ]

    # ---------------- Entrenamiento ----------------
    print(f"  Entrenando hasta {EPOCHS} epochs (patience={PATIENCE_EARLY})...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    train_duration = time.time() - t0
    print(f"  Entrenamiento terminado en {train_duration:.1f}s "
          f"({train_duration/60:.1f} min)")

    # ---------------- Evaluación ----------------
    loss_train, acc_train = model.evaluate(X_train, y_train, verbose=0)
    loss_test,  acc_test  = model.evaluate(X_test,  y_test,  verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred_int   = np.argmax(y_pred_proba, axis=1)
    report_text = classification_report(
        y_test_int, y_pred_int,
        labels=list(range(num_classes)),
        target_names=label_names,
        digits=4, zero_division=0,
    )

    # ---------------- Gráficas ----------------
    hist_df = pd.DataFrame(history.history)

    plt.figure(figsize=(10, 6))
    hist_df[['loss', 'val_loss']].plot(ax=plt.gca(), grid=True)
    plt.title(f'Loss - {model_name}')
    plt.xlabel('Epoca'); plt.ylabel('Loss'); plt.tight_layout()
    loss_plot_path = os.path.join(GRAPHICS_DIR, f"{model_name}_loss.jpeg")
    plt.savefig(loss_plot_path, format='jpeg', dpi=120); plt.close()

    plt.figure(figsize=(10, 6))
    hist_df[['accuracy', 'val_accuracy']].plot(ax=plt.gca(), grid=True)
    plt.title(f'Accuracy - {model_name}')
    plt.xlabel('Epoca'); plt.ylabel('Accuracy'); plt.tight_layout()
    acc_plot_path = os.path.join(GRAPHICS_DIR, f"{model_name}_accuracy.jpeg")
    plt.savefig(acc_plot_path, format='jpeg', dpi=120); plt.close()

    # ---------------- Guardar modelo final ----------------
    final_model_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
    model.save(final_model_path)

    # ---------------- Log de texto ----------------
    best_idx   = int(hist_df['val_loss'].idxmin())
    best_epoch = best_idx + 1
    log_path   = os.path.join(INFO_DIR, f"{model_name}_train_log.txt")

    with open(log_path, 'w', encoding='utf-8') as fp:
        fp.write(f"# Entrenamiento ColSign LSTM\n")
        fp.write(f"# Modelo: {model_name}\n")
        fp.write(f"# Fecha:  {datetime.now().isoformat(timespec='seconds')}\n\n")

        fp.write("## Configuracion\n")
        fp.write(f"Dataset HDF5:        {DATASET_HDF5}\n")
        fp.write(f"Sequence length:     {seq_len}\n")
        fp.write(f"Num features:        {num_features}\n")
        fp.write(f"Normalize keypoints: {NORMALIZE_KEYPOINTS}\n")
        fp.write(f"Drop pose visibility:{DROP_POSE_VISIBILITY}\n")
        fp.write(f"Num classes:         {num_classes}\n")
        fp.write(f"Total samples:       {total_samples}\n")
        fp.write(f"Train samples:       {X_train.shape[0]}\n")
        fp.write(f"Test  samples:       {X_test.shape[0]}\n")
        fp.write(f"Test size fraction:  {TEST_SIZE}\n")
        fp.write(f"Split strategy:      {split_strategy}\n")
        fp.write(f"Random seed:         {SEED}\n")
        fp.write(f"Batch size:          {BATCH_SIZE}\n")
        fp.write(f"Epochs (config):     {EPOCHS}\n")
        fp.write(f"Epochs (entrenadas): {len(hist_df)}\n")
        fp.write(f"Learning rate inic:  {LEARNING_RATE}\n")
        fp.write(f"Dropout rate:        {DROPOUT_RATE}\n")
        fp.write(f"Patience EarlyStop:  {PATIENCE_EARLY}\n")
        fp.write(f"Patience ReduceLR:   {PATIENCE_LR}\n")
        fp.write(f"Tiempo entrenamiento:{train_duration:.1f}s "
                 f"({train_duration/60:.2f} min)\n\n")

        if extra_log_info:
            fp.write("## Info adicional\n")
            for k, v in extra_log_info.items():
                fp.write(f"{k}: {v}\n")
            fp.write("\n")

        fp.write("## Arquitectura del modelo\n")
        model.summary(print_fn=lambda s: fp.write(s + '\n'))
        fp.write("\n")

        fp.write("## Metricas finales (modelo restaurado al mejor epoch)\n")
        fp.write(f"Train loss:     {loss_train:.4f}\n")
        fp.write(f"Train accuracy: {acc_train:.4f}\n")
        fp.write(f"Test  loss:     {loss_test:.4f}\n")
        fp.write(f"Test  accuracy: {acc_test:.4f}\n\n")

        fp.write("## Mejor epoch (segun val_loss)\n")
        fp.write(f"Epoch:           {best_epoch}\n")
        fp.write(f"Train loss:      {hist_df.loc[best_idx, 'loss']:.4f}\n")
        fp.write(f"Train accuracy:  {hist_df.loc[best_idx, 'accuracy']:.4f}\n")
        fp.write(f"Val   loss:      {hist_df.loc[best_idx, 'val_loss']:.4f}\n")
        fp.write(f"Val   accuracy:  {hist_df.loc[best_idx, 'val_accuracy']:.4f}\n\n")

        fp.write("## Classification report por clase (sobre conjunto de test)\n")
        fp.write(report_text)
        fp.write("\n")

        fp.write("## Archivos generados\n")
        fp.write(f"Modelo final:     {final_model_path}\n")
        fp.write(f"Mejor checkpoint: {best_ckpt_path}\n")
        fp.write(f"Labels JSON:      {labels_json_path}\n")
        fp.write(f"Loss plot:        {loss_plot_path}\n")
        fp.write(f"Accuracy plot:    {acc_plot_path}\n")

    print(f"  Train acc: {acc_train:.4f} | Test acc: {acc_test:.4f}")
    print(f"  Mejor epoch: {best_epoch}")
    print(f"  Log:       {log_path}")

    # Liberar memoria de TF entre modelos: importante porque vamos a entrenar
    # 5 modelos en el mismo proceso y el grafo de Keras se acumula.
    tf.keras.backend.clear_session()

    return {
        'model_name'  : model_name,
        'num_classes' : int(num_classes),
        'num_samples' : int(total_samples),
        'epochs'      : int(len(hist_df)),
        'best_epoch'  : int(best_epoch),
        'train_loss'  : float(loss_train),
        'train_acc'   : float(acc_train),
        'test_loss'   : float(loss_test),
        'test_acc'    : float(acc_test),
        'duration_s'  : float(train_duration),
        'log_path'    : log_path,
        'model_path'  : final_model_path,
    }


# =====================================================================
# Main: orquesta los 5 entrenamientos
# =====================================================================

def main():
    # ---------- 1. Cargar CSV ----------
    print(f"Cargando CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')
    # normalizar espacios en bordes (por si acaso)
    df['nombre']        = df['nombre'].astype(str).str.strip()
    df['etiqueta_raiz'] = df['etiqueta_raiz'].astype(str).str.strip()
    print(f"  {len(df)} etiquetas en CSV")
    name_to_root = dict(zip(df['nombre'], df['etiqueta_raiz']))

    # ---------- 2. Cargar HDF5 normalizado ----------
    print(f"\nCargando dataset normalizado: {DATASET_HDF5}")
    t0 = time.time()
    X, y_int, label_names = load_hdf5_dataset(
        DATASET_HDF5,
        normalize=NORMALIZE_KEYPOINTS,
        drop_pose_visibility=DROP_POSE_VISIBILITY,
    )
    print(f"  X.shape={X.shape}  y.shape={y_int.shape}  "
          f"clases={len(label_names)}  ({time.time()-t0:.1f}s)")

    # ---------- 3. Validar coherencia CSV vs HDF5 ----------
    missing = [n for n in label_names if n not in name_to_root]
    if missing:
        raise ValueError(
            f"Faltan {len(missing)} etiquetas del HDF5 en el CSV "
            f"(p.ej. {missing[:5]})"
        )
    extras = [n for n in name_to_root if n not in label_names]
    if extras:
        print(f"  Advertencia: el CSV tiene {len(extras)} etiquetas no "
              f"presentes en el HDF5:")
        print(f"    {extras[:10]}")

    # ---------- 4. Mapear label -> grupo raíz ----------
    root_groups = sorted(set(name_to_root[n] for n in label_names))
    root_to_idx = {r: i for i, r in enumerate(root_groups)}
    print(f"\n  Grupos raíz detectados: {len(root_groups)}")
    for r in root_groups:
        cnt_clases = sum(1 for n in label_names if name_to_root[n] == r)
        cnt_samples = sum(
            1 for lid in y_int if name_to_root[label_names[int(lid)]] == r
        )
        print(f"    [{root_to_idx[r]}] {r}: "
              f"{cnt_clases} clases, {cnt_samples} samples")

    y_root = np.array(
        [root_to_idx[name_to_root[label_names[int(lid)]]] for lid in y_int],
        dtype=np.int64,
    )

    summary = {}

    # ---------- 5. Modelo raíz (4 clases) ----------
    summary['raiz'] = train_one_model(
        X=X,
        y_int=y_root,
        label_names=root_groups,
        model_name=MODEL_NAME_RAIZ,
        extra_log_info={
            'tipo'            : 'modelo raíz (clasifica grupo morfológico)',
            'grupos_incluidos': ', '.join(root_groups),
        },
    )

    # ---------- 6. Sub-modelos (uno por grupo raíz) ----------
    for root_name in root_groups:
        sub_label_names = sorted(
            [n for n in label_names if name_to_root[n] == root_name]
        )
        if len(sub_label_names) < 2:
            print(f"\n  Grupo '{root_name}' tiene solo "
                  f"{len(sub_label_names)} clase(s); se omite.")
            continue

        sub_name_to_idx = {n: i for i, n in enumerate(sub_label_names)}
        sub_name_set = set(sub_label_names)

        # máscara booleana de samples cuya clase original pertenece al grupo
        mask = np.array(
            [label_names[int(lid)] in sub_name_set for lid in y_int],
            dtype=bool,
        )
        X_sub = X[mask]
        # re-indexamos: globales [0, 154) -> locales [0, K) dentro del grupo
        y_sub = np.array(
            [sub_name_to_idx[label_names[int(lid)]] for lid in y_int[mask]],
            dtype=np.int64,
        )

        model_name_sub = ROOT_TO_MODEL_NAME.get(root_name)
        if not model_name_sub:
            print(f"\n  Grupo '{root_name}' no tiene nombre de modelo "
                  f"asignado en ROOT_TO_MODEL_NAME; se omite.")
            continue

        summary[root_name] = train_one_model(
            X=X_sub,
            y_int=y_sub,
            label_names=sub_label_names,
            model_name=model_name_sub,
            extra_log_info={
                'tipo'             : 'sub-modelo (clasifica seña específica)',
                'grupo_raiz'       : root_name,
                'modelo_raiz_padre': MODEL_NAME_RAIZ,
                'clases_del_grupo' : ', '.join(sub_label_names),
            },
        )

    # ---------- 7. Resumen jerárquico ----------
    print(f"\n{'=' * 70}")
    print('=== Resumen de todos los entrenamientos ===')
    print('=' * 70)
    header = f"{'modelo':<48} {'clases':>7} {'samples':>8} {'epochs':>7} {'test_acc':>10}"
    print(header)
    for _, m in summary.items():
        print(
            f"{m['model_name']:<48} {m['num_classes']:>7} "
            f"{m['num_samples']:>8} {m['epochs']:>7} {m['test_acc']:>10.4f}"
        )

    resumen_path = os.path.join(
        INFO_DIR, 'colsign_lstm_norm_45_154_jerarquia_resumen.json'
    )
    with open(resumen_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"\nResumen JSON guardado en: {resumen_path}")


if __name__ == '__main__':
    main()
