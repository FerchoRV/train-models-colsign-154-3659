"""Entrenamiento de una LSTM para reconocimiento de señas (ColSign).

Carga el dataset desde `dataset_colsign.h5` (pose + manos, 120 frames) y
normaliza los keypoints antes de entrenar una LSTM con 154 clases (etiquetas tomadas
automáticamente de los grupos del HDF5).

Genera, todo bajo el prefijo MODEL_NAME:
  - models/{MODEL_NAME}.keras         -> modelo final (mejor por val_loss)
  - models/{MODEL_NAME}_best.keras    -> mejor checkpoint durante training
  - info_models/{MODEL_NAME}_train_log.txt -> log de configuración, métricas
        finales, métricas del mejor epoch y classification_report por clase.
  - info_models/{MODEL_NAME}_labels.json   -> mapping id <-> nombre de clase.
        Keras solo trabaja con índices enteros (0..N-1); este JSON es lo
        que se necesita en inferencia para traducir esos índices a la
        etiqueta legible.
  - graphics/{MODEL_NAME}_loss.jpeg
  - graphics/{MODEL_NAME}_accuracy.jpeg

Ejecutar::

    .\.venv\Scripts\python.exe -u train_lstm.py *> log_train.txt
"""

import os
import sys
import json
import time
from datetime import datetime

# En Windows, stdout/stderr usan cp1252 por defecto cuando se redirige a
# archivo (`*> log.txt`). Keras imprime su barra de progreso con caracteres
# Unicode (━━━━), y al escribirlos en cp1252 se lanza UnicodeEncodeError.
# Forzamos UTF-8 desde el script para que el redireccionamiento funcione
# sin importar las variables de entorno.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend no interactivo: evita que se abran ventanas
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


# =====================================================================
# Configuración
# =====================================================================

DATASET_HDF5 = 'dataset_colsign_45_154.h5'
MODEL_NAME   = 'colsign_lstm_norm_45_154'

# Preprocesamiento de features.
# El HDF5 conserva los keypoints crudos (258 features). Para entrenar, usamos:
#   - Normalización por frame: centro de hombros como origen y distancia entre
#     hombros como escala.
#   - Sin visibility de pose: 258 -> 225 features.
NORMALIZE_KEYPOINTS  = True
DROP_POSE_VISIBILITY = True

# Reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Split
TEST_SIZE = 0.2

# Entrenamiento
BATCH_SIZE     = 64
EPOCHS         = 300
LEARNING_RATE  = 1e-3
PATIENCE_EARLY = 30   # epochs sin mejora antes de cortar
PATIENCE_LR    = 10   # epochs sin mejora antes de bajar LR
DROPOUT_RATE   = 0.20

# Carpetas de salida
MODELS_DIR   = 'models'
INFO_DIR     = 'info_models'
GRAPHICS_DIR = 'graphics'
for d in (MODELS_DIR, INFO_DIR, GRAPHICS_DIR):
    os.makedirs(d, exist_ok=True)


# =====================================================================
# 1. Cargar dataset HDF5
# =====================================================================

print(f"Cargando dataset: {DATASET_HDF5}")
t0 = time.time()
X, y_int, label_names = load_hdf5_dataset(
    DATASET_HDF5,
    normalize=NORMALIZE_KEYPOINTS,
    drop_pose_visibility=DROP_POSE_VISIBILITY,
)
t_load = time.time() - t0

NUM_CLASSES     = len(label_names)
SEQUENCE_LENGTH = X.shape[1]
NUM_FEATURES    = X.shape[2]
print(
    f"  X shape: {X.shape}  dtype={X.dtype}\n"
    f"  y shape: {y_int.shape}  dtype={y_int.dtype}\n"
    f"  Clases : {NUM_CLASSES}\n"
    f"  Tiempo de carga: {t_load:.1f}s"
)


# =====================================================================
# 2. Guardar mapping id <-> nombre de clase
# =====================================================================
#
# Keras NO guarda los nombres de etiqueta dentro del .keras; solo aprende
# a predecir un índice 0..N-1. Este JSON es el "diccionario" que conecta
# esos índices con los nombres reales del dataset, en exactamente el
# mismo orden en el que el modelo está entrenado.

labels_json_path = os.path.join(INFO_DIR, f"{MODEL_NAME}_labels.json")
labels_payload = {
    'model_name'     : MODEL_NAME,
    'num_classes'    : NUM_CLASSES,
    'sequence_length': SEQUENCE_LENGTH,
    'num_features'   : NUM_FEATURES,
    'normalize_keypoints' : NORMALIZE_KEYPOINTS,
    'drop_pose_visibility': DROP_POSE_VISIBILITY,
    'created_at'     : datetime.now().isoformat(timespec='seconds'),
    'id_to_name'     : {str(i): n for i, n in enumerate(label_names)},
    'name_to_id'     : {n: i      for i, n in enumerate(label_names)},
}
with open(labels_json_path, 'w', encoding='utf-8') as fp:
    json.dump(labels_payload, fp, ensure_ascii=False, indent=2)
print(f"  Mapping etiquetas: {labels_json_path}")


# =====================================================================
# 3. Split estratificado train/test
# =====================================================================
#
# Stratify es importante con 154 clases y ~17-30 muestras por clase:
# garantiza que cada clase tenga representación tanto en train como en
# test (sin stratify, alguna clase podría quedar sin samples de test).

try:
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y_int,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_int,
    )
    split_strategy = 'estratificado por clase'
except ValueError as e:
    print(f"  No se pudo estratificar ({e}); split aleatorio simple")
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y_int,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    split_strategy = 'aleatorio (no estratificado)'

y_train = to_categorical(y_train_int, num_classes=NUM_CLASSES)
y_test  = to_categorical(y_test_int,  num_classes=NUM_CLASSES)
print(f"  Train: {X_train.shape}  Test: {X_test.shape}  ({split_strategy})")


# =====================================================================
# 4. Definir modelo
# =====================================================================
#
# Cambios respecto a la versión anterior (abecedario):
#   - input_shape dinámico desde el dataset cargado, normalmente (120, 225)
#     porque entrenamos con keypoints normalizados y sin visibility.
#   - 154 clases en vez de 27.
#   - SIN activation='relu' en LSTM: por defecto Keras usa tanh, que es
#     lo que las celdas LSTM esperan internamente. Forzar relu en LSTM
#     suele causar gradientes que explotan.
#   - Dropout moderado entre capas para evitar overfitting sin impedir
#     que el modelo aprenda el baseline normalizado.

model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    LSTM(128, return_sequences=True),
    Dropout(DROPOUT_RATE),
    LSTM(64,  return_sequences=False),
    Dropout(DROPOUT_RATE),
    Dense(128, activation='relu'),
    Dropout(DROPOUT_RATE),
    Dense(NUM_CLASSES, activation='softmax'),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()


# =====================================================================
# 5. Callbacks
# =====================================================================

best_ckpt_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_best.keras")
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE_EARLY,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=PATIENCE_LR,
        min_lr=1e-6,
        verbose=1,
    ),
    ModelCheckpoint(
        filepath=best_ckpt_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0,
    ),
]


# =====================================================================
# 6. Entrenar
# =====================================================================

print(f"\nEntrenando hasta {EPOCHS} epochs (early stopping patience={PATIENCE_EARLY})...")
t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    # verbose=2: una línea por epoch sin barra animada con caracteres
    # Unicode. Mucho más limpio cuando se redirige stdout a un log.
    verbose=2,
)
train_duration = time.time() - t0
print(f"\nEntrenamiento finalizado en {train_duration:.1f}s ({train_duration/60:.1f} min)")


# =====================================================================
# 7. Evaluación final
# =====================================================================

loss_train, acc_train = model.evaluate(X_train, y_train, verbose=0)
loss_test,  acc_test  = model.evaluate(X_test,  y_test,  verbose=0)

y_pred_proba = model.predict(X_test, verbose=0)
y_pred_int   = np.argmax(y_pred_proba, axis=1)

report_text = classification_report(
    y_test_int, y_pred_int,
    labels=list(range(NUM_CLASSES)),
    target_names=label_names,
    digits=4,
    zero_division=0,
)


# =====================================================================
# 8. Gráficas
# =====================================================================

hist_df = pd.DataFrame(history.history)

plt.figure(figsize=(10, 6))
hist_df[['loss', 'val_loss']].plot(ax=plt.gca(), grid=True)
plt.title(f'Loss - {MODEL_NAME}')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.tight_layout()
loss_plot_path = os.path.join(GRAPHICS_DIR, f"{MODEL_NAME}_loss.jpeg")
plt.savefig(loss_plot_path, format='jpeg', dpi=120)
plt.close()

plt.figure(figsize=(10, 6))
hist_df[['accuracy', 'val_accuracy']].plot(ax=plt.gca(), grid=True)
plt.title(f'Accuracy - {MODEL_NAME}')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.tight_layout()
acc_plot_path = os.path.join(GRAPHICS_DIR, f"{MODEL_NAME}_accuracy.jpeg")
plt.savefig(acc_plot_path, format='jpeg', dpi=120)
plt.close()


# =====================================================================
# 9. Guardar modelo final
# =====================================================================
#
# EarlyStopping con restore_best_weights=True ya restauró los pesos del
# mejor epoch; este save guarda esa versión.

final_model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.keras")
model.save(final_model_path)


# =====================================================================
# 10. Log de texto en info_models/
# =====================================================================

best_idx       = int(hist_df['val_loss'].idxmin())
best_epoch     = best_idx + 1
log_path       = os.path.join(INFO_DIR, f"{MODEL_NAME}_train_log.txt")

with open(log_path, 'w', encoding='utf-8') as fp:
    fp.write(f"# Entrenamiento ColSign LSTM\n")
    fp.write(f"# Modelo: {MODEL_NAME}\n")
    fp.write(f"# Fecha:  {datetime.now().isoformat(timespec='seconds')}\n\n")

    fp.write("## Configuracion\n")
    fp.write(f"Dataset HDF5:        {DATASET_HDF5}\n")
    fp.write(f"Sequence length:     {SEQUENCE_LENGTH}\n")
    fp.write(f"Num features:        {NUM_FEATURES}\n")
    fp.write(f"Normalize keypoints: {NORMALIZE_KEYPOINTS}\n")
    fp.write(f"Drop pose visibility:{DROP_POSE_VISIBILITY}\n")
    fp.write(f"Num classes:         {NUM_CLASSES}\n")
    fp.write(f"Total samples:       {X.shape[0]}\n")
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
    fp.write(f"Tiempo entrenamiento:{train_duration:.1f}s ({train_duration/60:.2f} min)\n\n")

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


# =====================================================================
# 11. Resumen en consola
# =====================================================================

print(f"\n=== Resumen ===")
print(f"  Modelo final:     {final_model_path}")
print(f"  Mejor checkpoint: {best_ckpt_path}")
print(f"  Labels JSON:      {labels_json_path}")
print(f"  Log de texto:     {log_path}")
print(f"  Loss plot:        {loss_plot_path}")
print(f"  Accuracy plot:    {acc_plot_path}")
print(f"\n  Train acc: {acc_train:.4f}  loss: {loss_train:.4f}")
print(f"  Test  acc: {acc_test:.4f}   loss: {loss_test:.4f}")
print(f"  Mejor epoch: {best_epoch} (val_acc={hist_df.loc[best_idx, 'val_accuracy']:.4f})")
