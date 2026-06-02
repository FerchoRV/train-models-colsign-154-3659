"""Modelo MLP para señas estáticas del abecedario (Grupo Estático).

Diagnóstico que motiva este script
----------------------------------
El sub-modelo LSTM `colsign_lstm_norm_estatic_45_154` se estancó en 60-64%
de val accuracy con overfitting severo (train 86% / val 64%). Las señas
estáticas no tienen dinámica temporal (una letra es una pose congelada),
por lo que una LSTM con 45 timesteps y 225 features (~727k parámetros)
tiene demasiada capacidad para 600 muestras y memoriza el ruido.

Cambios estructurales aquí
--------------------------
1. **Arquitectura MLP** sobre un único vector por muestra (no LSTM).
2. **Solo features de la mano dominante** (~63 features) -- la pose corporal
   y la otra mano son ruido para clasificar letras.
3. **Espejado horizontal** de muestras donde la mano izquierda es la
   dominante, para que el modelo vea siempre una orientación derecha
   (duplica datos efectivos y elimina la variable mano izda/dcha).
4. **Normalización por muñeca**: cada frame se centra en la muñeca y se
   escala por la distancia muñeca → middle MCP (invariante a posición y
   tamaño en pantalla).
5. **Agregación temporal por mediana** sobre los frames con detección
   válida. La mediana es robusta a frames espurios al inicio/fin del
   video.
6. **Data augmentation** sobre los keypoints (rotación, escala,
   traslación, jitter) para suplir el bajo conteo de muestras.
7. **Regularización fuerte**: dropout 0.5 + L2 + BatchNorm.

Output (igual formato que train_lstm.py / train_lstm_cluster_labels.py):
  - models/colsign_static_mlp_45_154.keras
  - models/colsign_static_mlp_45_154_best.keras
  - info_models/colsign_static_mlp_45_154_train_log.txt
  - info_models/colsign_static_mlp_45_154_labels.json
  - graphics/colsign_static_mlp_45_154_loss.jpeg
  - graphics/colsign_static_mlp_45_154_accuracy.jpeg

Ejecutar:
    .\.venv\Scripts\python.exe -u train_static_mlp.py
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
from tensorflow.keras.layers import (
    Dense, Dropout, Input, BatchNormalization,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.regularizers import l2

from src.utils import load_hdf5_dataset

# UTF-8 en stdout/stderr (consistente con los otros scripts en Windows)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


# =====================================================================
# Configuración
# =====================================================================

DATASET_HDF5 = 'dataset_colsign_45_154.h5'
CSV_PATH     = 'etiquetas_modelo_raiz.csv'

# Identificador de grupo en el CSV (atención a la tilde):
ROOT_GROUP_NAME = 'Grupo Estático'

MODEL_NAME = 'colsign_static_mlp_45_154'

# Hiperparámetros
SEED              = 42
TEST_SIZE         = 0.2
BATCH_SIZE        = 32
EPOCHS            = 500
LEARNING_RATE     = 1e-3
PATIENCE_EARLY    = 50
PATIENCE_LR       = 15
DROPOUT_RATE      = 0.5
L2_REG            = 1e-4
N_AUG_PER_SAMPLE  = 5  # Por cada muestra de train se generan 5 versiones aumentadas

# Carpetas de salida
MODELS_DIR   = 'models'
INFO_DIR     = 'info_models'
GRAPHICS_DIR = 'graphics'
for _d in (MODELS_DIR, INFO_DIR, GRAPHICS_DIR):
    os.makedirs(_d, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================================
# Layout del HDF5 raw (258 features, sin normalización pose-hombros)
#
#   [0:132]    pose (33 puntos × 4 coords: x, y, z, visibility)
#   [132:195]  mano izquierda (21 puntos × 3 coords: x, y, z)
#   [195:258]  mano derecha   (21 puntos × 3 coords: x, y, z)
#
# Para clasificar letras estáticas solo nos importan los bloques de manos.
# =====================================================================

N_HAND_POINTS   = 21
N_HAND_FEATURES = N_HAND_POINTS * 3  # 63

LEFT_HAND_START  = 132
LEFT_HAND_END    = 195
RIGHT_HAND_START = 195
RIGHT_HAND_END   = 258

# Índices de keypoints de mano (convención MediaPipe Hands)
WRIST_IDX      = 0  # punto 0 = muñeca
MIDDLE_MCP_IDX = 9  # punto 9 = articulación base del dedo medio


# =====================================================================
# Funciones de preprocesamiento de muestras
# =====================================================================

def _is_hand_detected(hand_vec, eps=1e-6):
    """Una mano se considera detectada si el vector no es todo cero."""
    return bool(np.any(np.abs(hand_vec) > eps))


def select_dominant_hand(sample_frames):
    """Elige la mano que aparece detectada en más frames como "dominante".

    Si la dominante era la izquierda, la espeja horizontalmente para
    expresarla en orientación de mano derecha. De este modo el modelo
    siempre ve una sola orientación canónica.

    Args:
        sample_frames: ndarray (T, 258) raw del HDF5.

    Returns:
        hand_seq: (T, 63) con la mano dominante en orientación derecha.
        dominant: 'left' o 'right' (cuál fue originalmente).
        det_mask: (T,) booleano, frames donde la mano dominante fue
            detectada con keypoints válidos.
    """
    T = sample_frames.shape[0]
    left  = sample_frames[:, LEFT_HAND_START:LEFT_HAND_END].copy()
    right = sample_frames[:, RIGHT_HAND_START:RIGHT_HAND_END].copy()

    left_det  = np.array([_is_hand_detected(left[t])  for t in range(T)], dtype=bool)
    right_det = np.array([_is_hand_detected(right[t]) for t in range(T)], dtype=bool)

    if right_det.sum() >= left_det.sum():
        return right, 'right', right_det

    # Mano izquierda es la dominante -> espejar horizontalmente.
    # Las coordenadas X de MediaPipe están en [0, 1] (espacio de imagen).
    # Espejado: x' = 1 - x. Z y Y no cambian.
    hand_pts = left.reshape(T, N_HAND_POINTS, 3)
    hand_pts[:, :, 0] = 1.0 - hand_pts[:, :, 0]

    # Importante: en frames NO detectados los keypoints originales eran 0,
    # y `1 - 0 = 1` introduciría datos espurios. Re-anulamos esos frames.
    for t in range(T):
        if not left_det[t]:
            hand_pts[t] = 0.0
    return hand_pts.reshape(T, N_HAND_FEATURES), 'left', left_det


def normalize_hand_per_frame(hand_seq, det_mask):
    """Centra cada frame en su muñeca y escala por |muñeca→middle MCP|.

    Esto vuelve los keypoints invariantes a la posición (X, Y, Z) y al
    tamaño aparente de la mano en la imagen.

    Frames no detectados se dejan en 0.
    """
    T = hand_seq.shape[0]
    out = np.zeros_like(hand_seq)
    for t in range(T):
        if not det_mask[t]:
            continue
        pts = hand_seq[t].reshape(N_HAND_POINTS, 3)
        wrist = pts[WRIST_IDX].copy()
        pts_c = pts - wrist[None, :]
        scale = float(np.linalg.norm(pts_c[MIDDLE_MCP_IDX]))
        if scale > 1e-6:
            pts_c = pts_c / scale
        out[t] = pts_c.flatten()
    return out


def aggregate_temporal(hand_seq, det_mask):
    """Agrega los frames detectados a un único vector usando la mediana
    coordenada-a-coordenada. La mediana es robusta a outliers (mano que
    cambia de pose al inicio/fin del video).
    """
    if not det_mask.any():
        return np.zeros(N_HAND_FEATURES, dtype=np.float32)
    return np.median(hand_seq[det_mask], axis=0).astype(np.float32)


def preprocess_sample(sample_frames):
    """Pipeline completo de una muestra: (T, 258) -> (63,) vector estático."""
    hand_seq, dom, det_mask = select_dominant_hand(sample_frames)
    hand_seq = normalize_hand_per_frame(hand_seq, det_mask)
    return aggregate_temporal(hand_seq, det_mask), dom, int(det_mask.sum())


# =====================================================================
# Carga del dataset filtrado a clases estáticas
# =====================================================================

def load_static_dataset():
    """Carga el HDF5 raw, filtra muestras de clases estáticas y aplica
    todo el preprocesamiento.
    """
    print(f"Cargando CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')
    df['nombre']        = df['nombre'].astype(str).str.strip()
    df['etiqueta_raiz'] = df['etiqueta_raiz'].astype(str).str.strip()

    static_labels = sorted(
        df[df['etiqueta_raiz'] == ROOT_GROUP_NAME]['nombre'].tolist()
    )
    if not static_labels:
        raise ValueError(
            f"No se encontraron etiquetas para '{ROOT_GROUP_NAME}' en el CSV. "
            f"Valores únicos: {sorted(df['etiqueta_raiz'].unique())}"
        )
    print(f"  Clases del grupo '{ROOT_GROUP_NAME}': {len(static_labels)}")
    print(f"  {static_labels}")
    label_to_idx = {n: i for i, n in enumerate(static_labels)}

    print(f"\nCargando HDF5 raw (sin normalización pose-hombros): {DATASET_HDF5}")
    X_raw, y_raw_global, all_label_names = load_hdf5_dataset(
        DATASET_HDF5,
        normalize=False,       # queremos los keypoints crudos
        drop_pose_visibility=True,  # ignorado cuando normalize=False
    )
    print(f"  X_raw.shape={X_raw.shape}  ({X_raw.shape[2]} features esperadas: 258)")
    if X_raw.shape[2] != 258:
        raise ValueError(
            f"Se esperaba (N, T, 258) pero llegó {X_raw.shape}. ¿El HDF5 está "
            f"normalizado o sin visibility ya? Este script asume features raw."
        )

    static_set = set(static_labels)
    mask = np.array(
        [all_label_names[int(g)] in static_set for g in y_raw_global],
        dtype=bool,
    )
    X_static_raw   = X_raw[mask]
    y_static_global = y_raw_global[mask]
    print(f"  Muestras de clases estáticas: {X_static_raw.shape[0]}")

    y = np.array(
        [label_to_idx[all_label_names[int(g)]] for g in y_static_global],
        dtype=np.int64,
    )

    print("\nPreprocesando muestras "
          "(mano dominante + espejado + normalización por muñeca + agregación temporal)...")
    N = X_static_raw.shape[0]
    X = np.zeros((N, N_HAND_FEATURES), dtype=np.float32)
    dom_count = {'left': 0, 'right': 0}
    det_counts = []

    for i in range(N):
        feat, dom, n_det = preprocess_sample(X_static_raw[i])
        X[i] = feat
        dom_count[dom] += 1
        det_counts.append(n_det)

    det_counts = np.array(det_counts)
    print(f"  Mano dominante:  derecha={dom_count['right']}  "
          f"izquierda={dom_count['left']} (espejadas)")
    print(f"  Frames detectados/muestra: "
          f"min={det_counts.min()}  max={det_counts.max()}  "
          f"mean={det_counts.mean():.1f}")

    # Detección de muestras "vacías" (cero frames detectados): las descartamos
    # porque su vector es todo cero y solo introducirían ruido.
    empty_mask = det_counts == 0
    n_empty = int(empty_mask.sum())
    if n_empty:
        print(f"  Descartando {n_empty} muestras sin ningún frame detectado.")
        keep = ~empty_mask
        X = X[keep]
        y = y[keep]

    meta = {
        'n_samples_inicial' : int(N),
        'n_samples_final'   : int(X.shape[0]),
        'n_descartadas'     : n_empty,
        'n_classes'         : int(len(static_labels)),
        'dominant_right'    : int(dom_count['right']),
        'dominant_left'     : int(dom_count['left']),
        'detection_min'     : int(det_counts.min()),
        'detection_max'     : int(det_counts.max()),
        'detection_mean'    : float(det_counts.mean()),
    }
    return X, y, static_labels, meta


# =====================================================================
# Data augmentation sobre keypoints normalizados
# =====================================================================

def augment_hand_vector(v, rng):
    """Aplica una transformación aleatoria sobre el vector (63,):

    - Rotación 2D en plano XY (±15°).
    - Escala uniforme (±10%).
    - Traslación pequeña en XY (±0.05).
    - Jitter aditivo gaussiano por punto (σ=0.005).

    Como los puntos ya están centrados en muñeca y escalados al rango
    aprox [-1, 1], estas perturbaciones simulan variaciones realistas:
    ángulos ligeramente distintos de la mano, manos un poco más grandes
    o pequeñas, y ruido de detección.
    """
    pts = v.reshape(N_HAND_POINTS, 3).copy()

    angle = rng.uniform(-15.0, 15.0) * np.pi / 180.0
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)
    pts[:, :2] = pts[:, :2] @ R.T

    s = float(rng.uniform(0.9, 1.1))
    pts *= s

    t = rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
    pts[:, :2] += t

    pts += rng.normal(0.0, 0.005, size=pts.shape).astype(np.float32)
    return pts.flatten().astype(np.float32)


def augment_dataset(X, y, n_aug_per_sample, rng):
    """Devuelve X, y aumentados (1 + n_aug_per_sample) veces."""
    N, F = X.shape
    total = N * (1 + n_aug_per_sample)
    X_out = np.zeros((total, F), dtype=np.float32)
    y_out = np.zeros(total, dtype=y.dtype)

    X_out[:N] = X
    y_out[:N] = y
    for k in range(n_aug_per_sample):
        offset = N * (1 + k)
        for i in range(N):
            X_out[offset + i] = augment_hand_vector(X[i], rng)
            y_out[offset + i] = y[i]
    return X_out, y_out


# =====================================================================
# Modelo
# =====================================================================

def build_model(n_features, n_classes):
    """MLP pequeño y bien regularizado. ~11k parámetros vs los 727k del
    LSTM original (ratio mucho más sano para 600 muestras de train)."""
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(128, activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(64,  activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        Dropout(DROPOUT_RATE),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# =====================================================================
# Main
# =====================================================================

def main():
    print('=' * 70)
    print(f'=== Entrenamiento {MODEL_NAME} (MLP para señas estáticas) ===')
    print('=' * 70)

    X, y, label_names, meta = load_static_dataset()
    n_classes = len(label_names)
    print(f"\nDataset final: X.shape={X.shape}  y.shape={y.shape}  K={n_classes}")

    # Split estratificado
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    print(f"\nTrain pre-aug: {X_train.shape}   Test: {X_test.shape}")

    # Augmentation SOLO en train
    rng = np.random.default_rng(SEED)
    X_train_aug, y_train_aug_int = augment_dataset(
        X_train, y_train_int, N_AUG_PER_SAMPLE, rng,
    )
    print(f"Train post-aug: {X_train_aug.shape} (x{1 + N_AUG_PER_SAMPLE})")

    # Labels JSON
    labels_path = os.path.join(INFO_DIR, f"{MODEL_NAME}_labels.json")
    labels_payload = {
        'model_name'        : MODEL_NAME,
        'architecture'      : 'MLP',
        'num_classes'       : n_classes,
        'num_features'      : int(X.shape[1]),
        'input_shape'       : [int(X.shape[1])],
        'feature_description': (
            "Vector de 63 valores (21 puntos × 3 coords) de la mano "
            "dominante, espejada a orientación derecha si era izquierda, "
            "centrada en muñeca y escalada por |muñeca→middle MCP|, "
            "agregada temporalmente por mediana sobre frames detectados."
        ),
        'preprocessing': {
            'source_features_raw_hdf5'   : 258,
            'select_dominant_hand'       : True,
            'mirror_left_to_right'       : True,
            'normalize_by_wrist_center'  : True,
            'scale_by_wrist_to_middle_mcp': True,
            'temporal_aggregation'       : 'median',
            'augmentation_per_train_sample': N_AUG_PER_SAMPLE,
        },
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'id_to_name': {str(i): n for i, n in enumerate(label_names)},
        'name_to_id': {n: i for i, n in enumerate(label_names)},
        'preprocessing_meta': meta,
    }
    with open(labels_path, 'w', encoding='utf-8') as fp:
        json.dump(labels_payload, fp, ensure_ascii=False, indent=2)
    print(f"Labels JSON: {labels_path}")

    y_train = to_categorical(y_train_aug_int, num_classes=n_classes)
    y_test  = to_categorical(y_test_int,  num_classes=n_classes)

    # Modelo
    model = build_model(X.shape[1], n_classes)
    model.summary()

    # Callbacks
    best_ckpt = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.keras")
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
            filepath=best_ckpt, monitor='val_loss',
            save_best_only=True, verbose=0,
        ),
    ]

    print(f"\nEntrenando hasta {EPOCHS} epochs (patience={PATIENCE_EARLY})...")
    t0 = time.time()
    history = model.fit(
        X_train_aug, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )
    duration = time.time() - t0
    print(f"Entrenamiento terminado en {duration:.1f}s ({duration/60:.2f} min)")

    # Evaluación
    loss_train, acc_train = model.evaluate(X_train_aug, y_train, verbose=0)
    loss_test,  acc_test  = model.evaluate(X_test,  y_test,  verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(
        y_test_int, y_pred,
        labels=list(range(n_classes)),
        target_names=label_names,
        digits=4, zero_division=0,
    )

    # Gráficas
    hist_df = pd.DataFrame(history.history)

    plt.figure(figsize=(10, 6))
    hist_df[['loss', 'val_loss']].plot(ax=plt.gca(), grid=True)
    plt.title(f'Loss - {MODEL_NAME}')
    plt.xlabel('Epoca'); plt.ylabel('Loss'); plt.tight_layout()
    loss_plot = os.path.join(GRAPHICS_DIR, f"{MODEL_NAME}_loss.jpeg")
    plt.savefig(loss_plot, format='jpeg', dpi=120); plt.close()

    plt.figure(figsize=(10, 6))
    hist_df[['accuracy', 'val_accuracy']].plot(ax=plt.gca(), grid=True)
    plt.title(f'Accuracy - {MODEL_NAME}')
    plt.xlabel('Epoca'); plt.ylabel('Accuracy'); plt.tight_layout()
    acc_plot = os.path.join(GRAPHICS_DIR, f"{MODEL_NAME}_accuracy.jpeg")
    plt.savefig(acc_plot, format='jpeg', dpi=120); plt.close()

    final_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras")
    model.save(final_path)

    # Log
    best_idx   = int(hist_df['val_loss'].idxmin())
    best_epoch = best_idx + 1
    log_path   = os.path.join(INFO_DIR, f"{MODEL_NAME}_train_log.txt")
    with open(log_path, 'w', encoding='utf-8') as fp:
        fp.write("# Entrenamiento ColSign MLP (señas estáticas)\n")
        fp.write(f"# Modelo: {MODEL_NAME}\n")
        fp.write(f"# Fecha:  {datetime.now().isoformat(timespec='seconds')}\n\n")

        fp.write("## Configuracion\n")
        fp.write(f"Dataset HDF5:           {DATASET_HDF5}\n")
        fp.write(f"Arquitectura:           MLP (NO recurrente)\n")
        fp.write(f"Num features input:     {X.shape[1]} (mano dominante, 21 puntos x 3)\n")
        fp.write(f"Num classes:            {n_classes}\n")
        fp.write(f"Total samples inicial:  {meta['n_samples_inicial']}\n")
        fp.write(f"Total samples final:    {meta['n_samples_final']}  "
                 f"(descartadas {meta['n_descartadas']} sin detección)\n")
        fp.write(f"  mano derecha:         {meta['dominant_right']}\n")
        fp.write(f"  mano izquierda:       {meta['dominant_left']} (espejadas)\n")
        fp.write(f"Detección por muestra:  "
                 f"min={meta['detection_min']}  "
                 f"max={meta['detection_max']}  "
                 f"mean={meta['detection_mean']:.1f}\n")
        fp.write(f"Train samples (pre):    {X_train.shape[0]}\n")
        fp.write(f"Train samples (+aug):   {X_train_aug.shape[0]} "
                 f"(x{1 + N_AUG_PER_SAMPLE})\n")
        fp.write(f"Test  samples:          {X_test.shape[0]}\n")
        fp.write(f"Test size fraction:     {TEST_SIZE}\n")
        fp.write(f"Random seed:            {SEED}\n")
        fp.write(f"Batch size:             {BATCH_SIZE}\n")
        fp.write(f"Epochs (config):        {EPOCHS}\n")
        fp.write(f"Epochs (entrenadas):    {len(hist_df)}\n")
        fp.write(f"Learning rate inicial:  {LEARNING_RATE}\n")
        fp.write(f"Dropout rate:           {DROPOUT_RATE}\n")
        fp.write(f"L2 regularization:      {L2_REG}\n")
        fp.write(f"Patience EarlyStop:     {PATIENCE_EARLY}\n")
        fp.write(f"Patience ReduceLR:      {PATIENCE_LR}\n")
        fp.write(f"Tiempo entrenamiento:   {duration:.1f}s "
                 f"({duration/60:.2f} min)\n\n")

        fp.write("## Preprocesamiento\n")
        fp.write("- Selección de mano dominante por número de frames con detección.\n")
        fp.write("- Espejado horizontal (x' = 1 - x) para muestras de mano izquierda.\n")
        fp.write("- Centrado en muñeca (punto 0); escala por |muñeca→middle MCP (punto 9)|.\n")
        fp.write("- Agregación temporal: mediana sobre frames con detección válida.\n")
        fp.write(f"- Augmentation por muestra train: x{N_AUG_PER_SAMPLE} con "
                 f"rotación ±15°, escala ±10%, traslación ±0.05, jitter σ=0.005.\n\n")

        fp.write("## Arquitectura del modelo\n")
        model.summary(print_fn=lambda s: fp.write(s + '\n'))
        fp.write("\n")

        fp.write("## Metricas finales (modelo restaurado al mejor epoch)\n")
        fp.write(f"Train loss (aug):  {loss_train:.4f}\n")
        fp.write(f"Train accuracy:    {acc_train:.4f}\n")
        fp.write(f"Test  loss:        {loss_test:.4f}\n")
        fp.write(f"Test  accuracy:    {acc_test:.4f}\n\n")

        fp.write(f"## Mejor epoch (segun val_loss): {best_epoch}\n")
        fp.write(f"Train loss:     {hist_df.loc[best_idx, 'loss']:.4f}\n")
        fp.write(f"Train accuracy: {hist_df.loc[best_idx, 'accuracy']:.4f}\n")
        fp.write(f"Val   loss:     {hist_df.loc[best_idx, 'val_loss']:.4f}\n")
        fp.write(f"Val   accuracy: {hist_df.loc[best_idx, 'val_accuracy']:.4f}\n\n")

        fp.write("## Classification report por clase (sobre conjunto de test)\n")
        fp.write(report)
        fp.write("\n")

        fp.write("## Archivos generados\n")
        fp.write(f"Modelo final:     {final_path}\n")
        fp.write(f"Mejor checkpoint: {best_ckpt}\n")
        fp.write(f"Labels JSON:      {labels_path}\n")
        fp.write(f"Loss plot:        {loss_plot}\n")
        fp.write(f"Accuracy plot:    {acc_plot}\n")

    print(f"\nTrain acc: {acc_train:.4f}  |  Test acc: {acc_test:.4f}  "
          f"|  Mejor epoch: {best_epoch}")
    print(f"Log: {log_path}")


if __name__ == '__main__':
    main()
