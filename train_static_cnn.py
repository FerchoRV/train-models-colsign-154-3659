"""Entrenamiento CNN para señas estáticas del abecedario.

Por qué CNN en lugar de LSTM/MLP sobre keypoints
-------------------------------------------------
Diagnóstico con k-NN y SVM sobre las representaciones de keypoints
(coordenadas XYZ + features geométricas) mostró un techo real ~48%.
Las letras del abecedario LSC se diferencian por detalles finos
(sombras, ángulo de dedos, separaciones milimétricas) que la imagen
RGB captura completamente pero los 21 keypoints no.

Este script entrena una CNN (MobileNetV2 con transfer learning desde
ImageNet) sobre los crops de mano generados por
`extract_static_hand_crops.py`.

Input
-----
`dataset_static_crops/{label}/{video_id}.jpg`
  Estructura producida por extract_static_hand_crops.py.
  Una imagen 128×128 RGB por video, mostrando la mano dominante
  centrada y con 25% de padding.

Output (mismo formato que train_lstm.py / train_lstm_cluster_labels.py)
-----------------------------------------------------------------------
  - models/colsign_static_cnn_45_154.keras
  - models/colsign_static_cnn_45_154_best.keras
  - info_models/colsign_static_cnn_45_154_train_log.txt
  - info_models/colsign_static_cnn_45_154_labels.json
  - graphics/colsign_static_cnn_45_154_loss.jpeg
  - graphics/colsign_static_cnn_45_154_accuracy.jpeg

Estrategia
----------
1. **Carga**: `tf.keras.utils.image_dataset_from_directory` con split
   train/val (80/20) y orden determinista.
2. **Augmentation**: random_flip_horizontal (canonicaliza orientación
   left/right sin perder datos), random_rotation ±10%, random_zoom ±10%,
   random_contrast/brightness moderado.
3. **Modelo base**: MobileNetV2 preentrenado en ImageNet
   (~2.3M parámetros, ligero y rápido).
4. **Cabeza**: GlobalAveragePooling2D + Dense(128, relu) + Dropout(0.4)
   + Dense(num_classes, softmax).
5. **Fase 1**: entrenar solo la cabeza (base congelada) ~30 epochs.
6. **Fase 2**: fine-tuning de las últimas capas del backbone con LR
   bajo (~1e-5) ~30 epochs.

Ejecutar:
    .\.venv\Scripts\python.exe -u train_static_cnn.py
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

from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)

# UTF-8 en Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


# =====================================================================
# Configuración
# =====================================================================

CROPS_DIR  = 'dataset_static_crops_expanded'  # golden + matches por similitud de pose
MODEL_NAME = 'colsign_static_cnn_45_154'

# Algunas clases tienen nombres con caracteres no-ASCII que rompen el listado
# de directorios en TensorFlow/Keras en Windows (UnicodeDecodeError 0xf1 para ñ).
# Workaround: la carpeta en disco usa un alias ASCII (ej. "enie") y aquí
# mapeamos de vuelta al nombre real al construir los class_names "display".
# Si en el futuro hay más colisiones, se agregan aquí.
FOLDER_TO_DISPLAY = {
    'enie': 'ñ',
}
DISPLAY_TO_FOLDER = {v: k for k, v in FOLDER_TO_DISPLAY.items()}


def folder_to_display(folder_name):
    return FOLDER_TO_DISPLAY.get(folder_name, folder_name)

IMG_SIZE   = 128                # mismo tamaño que produjo extract_static_hand_crops
BATCH_SIZE = 32
SEED       = 42
VAL_SPLIT  = 0.2

# Hiperparámetros - Fase 1 (cabeza)
EPOCHS_HEAD          = 40
LR_HEAD              = 1e-3
DROPOUT_HEAD         = 0.4
PATIENCE_EARLY_HEAD  = 15
PATIENCE_LR_HEAD     = 6

# Hiperparámetros - Fase 2 (fine-tuning)
EPOCHS_FT            = 40
LR_FT                = 1e-5
FINE_TUNE_LAYERS     = 30   # cuántas capas finales del backbone descongelar
PATIENCE_EARLY_FT    = 20
PATIENCE_LR_FT       = 8

MODELS_DIR   = 'models'
INFO_DIR     = 'info_models'
GRAPHICS_DIR = 'graphics'
for _d in (MODELS_DIR, INFO_DIR, GRAPHICS_DIR):
    os.makedirs(_d, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================================
# Carga de imágenes con tf.data
# =====================================================================

def load_image_datasets():
    """Carga `dataset_static_crops/` como train_ds y val_ds usando un
    split determinista del 80/20 estratificado.

    Returns:
        train_ds: tf.data.Dataset (X, y) de train.
        val_ds:   tf.data.Dataset (X, y) de val.
        class_names: list[str] con los nombres de clase en orden.
        meta: dict con conteos por clase.
    """
    print(f"Cargando imágenes desde {CROPS_DIR}/ ...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        CROPS_DIR,
        validation_split=VAL_SPLIT,
        subset='training',
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        CROPS_DIR,
        validation_split=VAL_SPLIT,
        subset='validation',
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False,
    )
    # `class_names` viene de los nombres de carpeta en disco (ASCII safe).
    # Construimos también la versión "display" con los nombres reales
    # (mapeando alias como 'enie' → 'ñ').
    folder_names = train_ds.class_names
    class_names  = [folder_to_display(cn) for cn in folder_names]
    print(f"  Clases ({len(class_names)}): {class_names}")
    if folder_names != class_names:
        print(f"  Folder aliases activos: "
              f"{[(f, d) for f, d in zip(folder_names, class_names) if f != d]}")

    # Conteo por clase (usando el nombre real para el JSON)
    counts = {}
    for fn, cn in zip(folder_names, class_names):
        d = os.path.join(CROPS_DIR, fn)
        counts[cn] = len([f for f in os.listdir(d) if f.lower().endswith('.jpg')])
    total = sum(counts.values())
    print(f"  Total imágenes: {total}")

    # Optimización de I/O: cachear + prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1024, seed=SEED).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names, counts


# =====================================================================
# Modelo
# =====================================================================

def build_augmentation():
    """Pipeline de augmentation: solo en training, dentro del modelo."""
    return models.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.08),       # ±10° aprox
        layers.RandomZoom(0.10),           # ±10%
        layers.RandomContrast(0.10),
        layers.RandomBrightness(0.10),
    ], name='augmentation')


def build_model(num_classes):
    """MobileNetV2 + cabeza personalizada. Compatibles con fine-tuning."""
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Augmentation (solo activa durante training)
    x = build_augmentation()(inputs)

    # Preprocesamiento estándar de MobileNetV2: scale a [-1, 1]
    x = applications.mobilenet_v2.preprocess_input(x)

    base = applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False  # fase 1: backbone congelado

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_HEAD)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(DROPOUT_HEAD)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name=MODEL_NAME)
    return model, base


# =====================================================================
# Entrenamiento
# =====================================================================

def compile_phase(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


def fit_phase(model, train_ds, val_ds, epochs, patience_early, patience_lr,
              best_ckpt_path, phase_label):
    callbacks = [
        EarlyStopping(
            monitor='val_loss', patience=patience_early,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=patience_lr, min_lr=1e-7, verbose=1,
        ),
        ModelCheckpoint(
            filepath=best_ckpt_path, monitor='val_loss',
            save_best_only=True, verbose=0,
        ),
    ]
    print(f"\n--- {phase_label}: hasta {epochs} epochs ---")
    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks, verbose=2,
    )
    duration = time.time() - t0
    print(f"--- {phase_label} terminada en {duration:.1f}s "
          f"({duration/60:.2f} min) ---")
    return history, duration


# =====================================================================
# Main
# =====================================================================

def main():
    print('=' * 70)
    print(f'=== {MODEL_NAME} (CNN sobre crops RGB) ===')
    print('=' * 70)

    if not os.path.isdir(CROPS_DIR):
        raise SystemExit(
            f"No existe {CROPS_DIR}/. Ejecuta primero "
            f"extract_static_hand_crops.py"
        )

    train_ds, val_ds, class_names, counts = load_image_datasets()
    num_classes = len(class_names)

    # Labels JSON
    labels_path = os.path.join(INFO_DIR, f"{MODEL_NAME}_labels.json")
    labels_payload = {
        'model_name'         : MODEL_NAME,
        'architecture'       : 'MobileNetV2 + head (transfer learning)',
        'num_classes'        : num_classes,
        'image_size'         : IMG_SIZE,
        'input_shape'        : [IMG_SIZE, IMG_SIZE, 3],
        'feature_description': (
            f"Crop cuadrado {IMG_SIZE}×{IMG_SIZE} RGB de la mano dominante, "
            f"con 25% de padding alrededor del bbox de keypoints, "
            f"extraído del frame medoide (más estable temporalmente) del "
            f"video. No se aplica espejado manual; la red aprende "
            f"invariancia con random_flip_horizontal."
        ),
        'preprocessing': {
            'source': 'dataset_static_crops/ generado por extract_static_hand_crops.py',
            'crop_size': IMG_SIZE,
            'padding_ratio': 0.25,
            'preprocessing_mode': 'mobilenet_v2_preprocess_input (-1, 1)',
        },
        'augmentation': {
            'random_flip_horizontal': True,
            'random_rotation': 0.08,
            'random_zoom': 0.10,
            'random_contrast': 0.10,
            'random_brightness': 0.10,
        },
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'id_to_name': {str(i): n for i, n in enumerate(class_names)},
        'name_to_id': {n: i for i, n in enumerate(class_names)},
        'samples_per_class': counts,
    }
    with open(labels_path, 'w', encoding='utf-8') as fp:
        json.dump(labels_payload, fp, ensure_ascii=False, indent=2)
    print(f"\nLabels JSON: {labels_path}")

    # Modelo
    model, base = build_model(num_classes)
    model.summary(line_length=100)

    best_ckpt_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.keras")

    # ---------------- Fase 1: cabeza ----------------
    compile_phase(model, LR_HEAD)
    hist_head, dur_head = fit_phase(
        model, train_ds, val_ds,
        EPOCHS_HEAD, PATIENCE_EARLY_HEAD, PATIENCE_LR_HEAD,
        best_ckpt_path, 'Fase 1 (cabeza)',
    )

    # Métricas tras fase 1
    loss_h, acc_h = model.evaluate(val_ds, verbose=0)
    print(f"\nFase 1 - val_loss: {loss_h:.4f}  val_acc: {acc_h:.4f}")

    # ---------------- Fase 2: fine-tuning ----------------
    base.trainable = True
    # congelar todas menos las últimas N
    for layer in base.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    print(f"\nFase 2: descongelando últimas {FINE_TUNE_LAYERS} capas del backbone "
          f"(LR={LR_FT})")
    compile_phase(model, LR_FT)

    hist_ft, dur_ft = fit_phase(
        model, train_ds, val_ds,
        EPOCHS_FT, PATIENCE_EARLY_FT, PATIENCE_LR_FT,
        best_ckpt_path, 'Fase 2 (fine-tuning)',
    )

    # Métricas finales
    loss_train, acc_train = model.evaluate(train_ds, verbose=0)
    loss_val,   acc_val   = model.evaluate(val_ds,   verbose=0)
    print(f"\nFinal - train_acc: {acc_train:.4f}  val_acc: {acc_val:.4f}")

    # Classification report
    y_true = []
    y_pred = []
    for X_batch, y_batch in val_ds:
        proba = model.predict(X_batch, verbose=0)
        y_pred.extend(np.argmax(proba, axis=1).tolist())
        y_true.extend(y_batch.numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=class_names, digits=4, zero_division=0,
    )

    # Historia combinada (concatenar fase 1 + fase 2)
    hist_combined = {
        k: list(hist_head.history.get(k, [])) + list(hist_ft.history.get(k, []))
        for k in set(hist_head.history) | set(hist_ft.history)
    }
    hist_df = pd.DataFrame(hist_combined)
    n_epochs_total = len(hist_df)

    # Gráficas
    plt.figure(figsize=(11, 6))
    hist_df[['loss', 'val_loss']].plot(ax=plt.gca(), grid=True)
    plt.axvline(x=len(hist_head.history['loss']) - 0.5, color='gray',
                linestyle='--', label='Inicio fase 2')
    plt.legend()
    plt.title(f'Loss - {MODEL_NAME}')
    plt.xlabel('Epoca'); plt.ylabel('Loss'); plt.tight_layout()
    loss_plot = os.path.join(GRAPHICS_DIR, f"{MODEL_NAME}_loss.jpeg")
    plt.savefig(loss_plot, format='jpeg', dpi=120); plt.close()

    plt.figure(figsize=(11, 6))
    hist_df[['accuracy', 'val_accuracy']].plot(ax=plt.gca(), grid=True)
    plt.axvline(x=len(hist_head.history['accuracy']) - 0.5, color='gray',
                linestyle='--', label='Inicio fase 2')
    plt.legend()
    plt.title(f'Accuracy - {MODEL_NAME}')
    plt.xlabel('Epoca'); plt.ylabel('Accuracy'); plt.tight_layout()
    acc_plot = os.path.join(GRAPHICS_DIR, f"{MODEL_NAME}_accuracy.jpeg")
    plt.savefig(acc_plot, format='jpeg', dpi=120); plt.close()

    final_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras")
    model.save(final_path)

    # Log de texto
    best_idx   = int(hist_df['val_loss'].idxmin())
    best_epoch = best_idx + 1
    log_path   = os.path.join(INFO_DIR, f"{MODEL_NAME}_train_log.txt")

    with open(log_path, 'w', encoding='utf-8') as fp:
        fp.write(f"# Entrenamiento ColSign CNN (señas estáticas, RGB)\n")
        fp.write(f"# Modelo: {MODEL_NAME}\n")
        fp.write(f"# Fecha:  {datetime.now().isoformat(timespec='seconds')}\n\n")

        fp.write("## Configuracion\n")
        fp.write(f"Crops directorio:    {CROPS_DIR}\n")
        fp.write(f"Arquitectura:        MobileNetV2 + cabeza Dense + Dropout\n")
        fp.write(f"Image size:          {IMG_SIZE}x{IMG_SIZE} RGB\n")
        fp.write(f"Num classes:         {num_classes}\n")
        total_imgs = sum(counts.values())
        fp.write(f"Total imágenes:      {total_imgs}\n")
        fp.write(f"Val split:           {VAL_SPLIT}\n")
        fp.write(f"Batch size:          {BATCH_SIZE}\n")
        fp.write(f"Random seed:         {SEED}\n\n")

        fp.write("## Fase 1 (entrenar solo cabeza, backbone congelado)\n")
        fp.write(f"Epochs config:       {EPOCHS_HEAD}\n")
        fp.write(f"Epochs entrenados:   {len(hist_head.history['loss'])}\n")
        fp.write(f"Learning rate:       {LR_HEAD}\n")
        fp.write(f"Dropout (cabeza):    {DROPOUT_HEAD}\n")
        fp.write(f"Patience early/lr:   {PATIENCE_EARLY_HEAD} / {PATIENCE_LR_HEAD}\n")
        fp.write(f"Duración:            {dur_head:.1f}s\n\n")

        fp.write("## Fase 2 (fine-tuning últimas capas del backbone)\n")
        fp.write(f"Epochs config:       {EPOCHS_FT}\n")
        fp.write(f"Epochs entrenados:   {len(hist_ft.history['loss'])}\n")
        fp.write(f"Learning rate:       {LR_FT}\n")
        fp.write(f"Capas descongeladas: últimas {FINE_TUNE_LAYERS}\n")
        fp.write(f"Patience early/lr:   {PATIENCE_EARLY_FT} / {PATIENCE_LR_FT}\n")
        fp.write(f"Duración:            {dur_ft:.1f}s\n\n")
        fp.write(f"Duración total:      {dur_head + dur_ft:.1f}s "
                 f"({(dur_head+dur_ft)/60:.2f} min)\n\n")

        fp.write("## Augmentation aplicada\n")
        fp.write("- RandomFlip horizontal (canonicaliza orientación left/right)\n")
        fp.write("- RandomRotation ±~10°\n")
        fp.write("- RandomZoom ±10%\n")
        fp.write("- RandomContrast ±10%\n")
        fp.write("- RandomBrightness ±10%\n\n")

        fp.write("## Arquitectura\n")
        model.summary(print_fn=lambda s: fp.write(s + '\n'), line_length=100)
        fp.write("\n")

        fp.write("## Metricas finales (mejores pesos restaurados por EarlyStopping)\n")
        fp.write(f"Train loss:     {loss_train:.4f}\n")
        fp.write(f"Train accuracy: {acc_train:.4f}\n")
        fp.write(f"Val   loss:     {loss_val:.4f}\n")
        fp.write(f"Val   accuracy: {acc_val:.4f}\n\n")

        fp.write(f"## Mejor epoch global (según val_loss): {best_epoch}/{n_epochs_total}\n")
        fp.write(f"Train loss:     {hist_df.loc[best_idx, 'loss']:.4f}\n")
        fp.write(f"Train accuracy: {hist_df.loc[best_idx, 'accuracy']:.4f}\n")
        fp.write(f"Val   loss:     {hist_df.loc[best_idx, 'val_loss']:.4f}\n")
        fp.write(f"Val   accuracy: {hist_df.loc[best_idx, 'val_accuracy']:.4f}\n\n")

        fp.write("## Classification report (validación)\n")
        fp.write(report)
        fp.write("\n")

        fp.write("## Conteo por clase\n")
        for cn in class_names:
            fp.write(f"  {cn:<6} {counts[cn]:>4}\n")
        fp.write("\n")

        fp.write("## Archivos generados\n")
        fp.write(f"Modelo final:     {final_path}\n")
        fp.write(f"Mejor checkpoint: {best_ckpt_path}\n")
        fp.write(f"Labels JSON:      {labels_path}\n")
        fp.write(f"Loss plot:        {loss_plot}\n")
        fp.write(f"Accuracy plot:    {acc_plot}\n")

    print(f"\nVal acc: {acc_val:.4f}  |  Train acc: {acc_train:.4f}  "
          f"|  Mejor epoch: {best_epoch}/{n_epochs_total}")
    print(f"Log: {log_path}")


if __name__ == '__main__':
    main()
