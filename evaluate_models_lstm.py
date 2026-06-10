"""Evaluación de un modelo LSTM sobre la carpeta `dataset_videos_evaluate`.

A diferencia de `piplane_lstm_all.py` (que lee las rutas de un CSV), aquí
se recorre directamente el árbol de carpetas:

    dataset_videos_evaluate/
      ├── a/            <- la ETIQUETA REAL es el nombre de la carpeta
      │     ├── a_xxx.mp4
      │     └── ...
      ├── A veces/
      └── ...

Flujo:
  1. Listar cada subcarpeta (etiqueta) y sus videos.
  2. Predecir cada video con el modelo (un único Holistic reutilizado).
  3. La etiqueta real es el nombre de la carpeta contenedora.
  4. Calcular accuracy global y por clase, y guardar CSV + TXT.

Ejecutar:
    .\.venv\Scripts\python.exe -u evaluate_models_lstm.py
"""

import os
import sys
import csv
import time
from collections import defaultdict
from datetime import datetime

# UTF-8 en consola (consistente con el resto del proyecto en Windows)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

from tqdm import tqdm

from src.utils_pipeplanes import (
    load_model,
    make_holistic,
    predict,
)


# =====================================================================
# Configuración
# =====================================================================
PATH_DATASET_EVALUATE = 'dataset_videos_evaluate'
MODEL_NAME            = 'colsign_lstm_norm_45_154_aug'  # se prefiere el `_best`
OUTPUT_TXT            = 'results_evaluate/accuracy_colsign_norm_45_154_aug.txt'
OUTPUT_CSV            = 'results_evaluate/resuls_colsign_norm_45_154_aug.csv'

VIDEO_EXTS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm')


# =====================================================================
# Utilidades
# =====================================================================

def construir_items(carpeta):
    """Recorre `carpeta` y devuelve una lista de `(label, video_path)`.

    `label` es el nombre de la subcarpeta a la que pertenece el video.
    """
    if not os.path.isdir(carpeta):
        raise SystemExit(
            f"No existe la carpeta {carpeta!r}. Crea la carpeta con "
            f"subcarpetas por etiqueta y coloca los videos dentro."
        )
    items = []
    for label in sorted(os.listdir(carpeta)):
        label_dir = os.path.join(carpeta, label)
        if not os.path.isdir(label_dir):
            continue
        for nombre in sorted(os.listdir(label_dir)):
            if nombre.lower().endswith(VIDEO_EXTS):
                items.append((label, os.path.join(label_dir, nombre)))
    return items


# =====================================================================
# Main
# =====================================================================

def main():
    t_start = time.time()

    print(f"=== evaluate_models_lstm ===")
    print(f"Fecha:       {datetime.now().isoformat(timespec='seconds')}")
    print(f"Carpeta:     {PATH_DATASET_EVALUATE}")
    print(f"Modelo:      {MODEL_NAME}  (se prefiere checkpoint _best)")
    print(f"Output CSV:  {OUTPUT_CSV}")
    print(f"Output TXT:  {OUTPUT_TXT}")
    print()

    # 1) Listar videos (etiqueta = carpeta contenedora)
    items = construir_items(PATH_DATASET_EVALUATE)
    labels_presentes = sorted({lab for lab, _ in items})
    print(f"Etiquetas encontradas: {len(labels_presentes)}")
    print(f"Videos a evaluar:      {len(items)}")
    if not items:
        raise SystemExit(f"No hay videos ({VIDEO_EXTS}) en {PATH_DATASET_EVALUATE!r}.")

    # 2) Cargar el modelo (carga perezosa de TensorFlow)
    print(f"Cargando modelo {MODEL_NAME}...")
    t0 = time.time()
    model, info = load_model(MODEL_NAME)
    print(f"  Modelo cargado en {time.time() - t0:.1f}s")
    print(f"  Arquitectura: {info.architecture}")
    print(f"  Input shape:  {info.input_shape}")
    print(f"  Num classes:  {info.num_classes}")
    print(f"  Keras path:   {info.keras_path}")

    # 3) Avisar si alguna etiqueta real no está en el vocabulario del modelo
    unknown_labels = sorted(set(labels_presentes) - set(info.name_to_id))
    if unknown_labels:
        print(f"\nADVERTENCIA: {len(unknown_labels)} etiquetas de la carpeta no "
              f"están en el vocabulario del modelo. Esos videos contarán "
              f"siempre como incorrectos. Ej: {unknown_labels[:5]}")

    # 4) Procesar cada video reutilizando un único Holistic
    results = []
    correct = 0
    failed  = 0

    print("\nProcesando videos...")
    with make_holistic() as holistic:
        for label_real, video_path in tqdm(items, desc='Predict', unit='video'):
            try:
                pred = predict(model, info, video_path, holistic=holistic)
                label_pred = pred['label']
                prob = float(pred['prob'])
            except Exception as e:  # noqa: BLE001 - resiliente
                label_pred = f'<error:{type(e).__name__}>'
                prob = 0.0
                failed += 1

            if label_pred == label_real:
                correct += 1

            results.append({
                'label':          label_real,
                'path':           video_path,
                'label_predicha': label_pred,
                'prob':           f'{prob:.4f}',
            })

    # 5) Guardar CSV de resultados
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(
            fp, fieldnames=['label', 'path', 'label_predicha', 'prob'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV de resultados: {OUTPUT_CSV}  ({len(results)} filas)")

    # 6) Métricas
    total = len(results)
    accuracy = correct / total if total else 0.0

    per_class_ok    = defaultdict(int)
    per_class_total = defaultdict(int)
    for r in results:
        per_class_total[r['label']] += 1
        if r['label'] == r['label_predicha']:
            per_class_ok[r['label']] += 1

    duration = time.time() - t_start

    # 7) Guardar reporte de accuracy
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as fp:
        fp.write(f"# Evaluación de modelo {MODEL_NAME}\n")
        fp.write(f"# Fecha:    {datetime.now().isoformat(timespec='seconds')}\n")
        fp.write(f"# Carpeta:  {PATH_DATASET_EVALUATE}\n")
        fp.write(f"# Modelo:   {info.keras_path}\n")
        fp.write(f"# Duración: {duration:.1f}s ({duration/60:.2f} min)\n\n")

        fp.write("## Resumen global\n")
        fp.write(f"Etiquetas:         {len(labels_presentes)}\n")
        fp.write(f"Total videos:      {total}\n")
        fp.write(f"Aciertos:          {correct}\n")
        fp.write(f"Errores:           {total - correct}\n")
        fp.write(f"Fallos de proceso: {failed}\n")
        fp.write(f"Accuracy:          {accuracy:.4f}  ({accuracy*100:.2f}%)\n")
        if unknown_labels:
            fp.write(f"\nAdvertencia: {len(unknown_labels)} etiquetas de la carpeta "
                     f"no existían en el vocabulario del modelo:\n")
            for u in unknown_labels:
                fp.write(f"  - {u}\n")
        fp.write("\n")

        fp.write("## Accuracy por clase\n")
        fp.write(f"{'label':<35} {'ok':>5} {'total':>5} {'acc':>7}\n")
        fp.write('-' * 56 + '\n')
        for label in sorted(per_class_total.keys()):
            n = per_class_total[label]
            ok = per_class_ok[label]
            acc = ok / n if n else 0.0
            fp.write(f"{label:<35} {ok:>5} {n:>5} {acc*100:>6.2f}%\n")
        fp.write('-' * 56 + '\n')
        fp.write(f"{'TOTAL':<35} {correct:>5} {total:>5} {accuracy*100:>6.2f}%\n")

    print(f"Reporte de accuracy: {OUTPUT_TXT}")
    print(f"\nAccuracy global: {accuracy*100:.2f}%  ({correct}/{total})")
    print(f"Duración total:  {duration:.1f}s ({duration/60:.2f} min)")


if __name__ == '__main__':
    main()
