"""Pipeline de evaluación del modelo plano (154 clases).

Lee `video_path_test_piplane.csv`, predice cada video con el modelo
`colsign_lstm_norm_45_154_best.keras` y genera:

  - `resuls_colsign_norm_45_154.csv`   : columnas label, path, label_predicha
  - `accuracy_colsign_norm_45_154.txt` : accuracy global + resumen + accuracy
    por clase + matches/total.

Ejecutar:
    .\.venv\Scripts\python.exe -u piplane_lstm_all.py
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

INPUT_CSV   = 'video_path_test_piplane.csv'
MODEL_NAME  = 'colsign_lstm_norm_45_154'   # busca automáticamente el `_best`
OUTPUT_CSV  = 'results_piplanes/resuls_colsign_norm_45_154.csv'
OUTPUT_TXT  = 'results_piplanes/accuracy_colsign_norm_45_154.txt'


# =====================================================================
# Main
# =====================================================================

def main():
    t_start = time.time()

    print(f"=== piplane_lstm_all ===")
    print(f"Fecha:       {datetime.now().isoformat(timespec='seconds')}")
    print(f"Input CSV:   {INPUT_CSV}")
    print(f"Modelo:      {MODEL_NAME}  (se prefiere checkpoint _best)")
    print(f"Output CSV:  {OUTPUT_CSV}")
    print(f"Output TXT:  {OUTPUT_TXT}")
    print()

    # 1) Cargar el set de evaluación
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"No se encontró {INPUT_CSV}. "
                         f"Ejecuta primero selecction_video_test_piplanes.py")
    with open(INPUT_CSV, 'r', encoding='utf-8', newline='') as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    print(f"Videos a evaluar: {len(rows)}")

    # 2) Cargar el modelo (carga perezosa de TensorFlow)
    print(f"Cargando modelo {MODEL_NAME}...")
    t0 = time.time()
    model, info = load_model(MODEL_NAME)
    print(f"  Modelo cargado en {time.time() - t0:.1f}s")
    print(f"  Arquitectura: {info.architecture}")
    print(f"  Input shape:  {info.input_shape}")
    print(f"  Num classes:  {info.num_classes}")
    print(f"  Keras path:   {info.keras_path}")

    # 3) Verificar que TODAS las etiquetas reales del CSV existan en el
    # vocabulario del modelo. Si alguna no existe, advertimos antes de
    # gastar 40 min procesando para que el usuario decida qué hacer.
    unknown_labels = sorted({r['label'] for r in rows} - set(info.name_to_id))
    if unknown_labels:
        print(f"\nADVERTENCIA: {len(unknown_labels)} etiquetas del CSV no están "
              f"en el vocabulario del modelo ({MODEL_NAME}). Esas filas "
              f"contarán siempre como incorrectas. Ej: {unknown_labels[:5]}")

    # 4) Procesar cada video reutilizando un único Holistic
    results = []
    correct = 0
    failed  = 0

    print("\nProcesando videos...")
    with make_holistic() as holistic:
        for row in tqdm(rows, desc='Predict', unit='video'):
            label_real = row['label']
            video_path = row['path']
            try:
                pred = predict(model, info, video_path, holistic=holistic)
                label_pred = pred['label']
            except Exception as e:  # noqa: BLE001 - resiliente al procesar 1000+ videos
                label_pred = f'<error:{type(e).__name__}>'
                failed += 1

            is_correct = (label_pred == label_real)
            if is_correct:
                correct += 1

            results.append({
                'label':          label_real,
                'path':           video_path,
                'label_predicha': label_pred,
            })

    # 5) Guardar CSV de resultados
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['label', 'path', 'label_predicha'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV de resultados: {OUTPUT_CSV}  ({len(results)} filas)")

    # 6) Métricas
    total = len(results)
    accuracy = correct / total if total else 0.0

    # accuracy por clase (también útil para diagnóstico)
    per_class_ok    = defaultdict(int)
    per_class_total = defaultdict(int)
    for r in results:
        per_class_total[r['label']] += 1
        if r['label'] == r['label_predicha']:
            per_class_ok[r['label']] += 1

    duration = time.time() - t_start

    # 7) Guardar reporte de accuracy
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as fp:
        fp.write(f"# Evaluación de modelo plano {MODEL_NAME}\n")
        fp.write(f"# Fecha:    {datetime.now().isoformat(timespec='seconds')}\n")
        fp.write(f"# Input:    {INPUT_CSV}\n")
        fp.write(f"# Modelo:   {info.keras_path}\n")
        fp.write(f"# Duración: {duration:.1f}s ({duration/60:.2f} min)\n\n")

        fp.write("## Resumen global\n")
        fp.write(f"Total videos:      {total}\n")
        fp.write(f"Aciertos:          {correct}\n")
        fp.write(f"Errores:           {total - correct}\n")
        fp.write(f"Fallos de proceso: {failed}\n")
        fp.write(f"Accuracy:          {accuracy:.4f}  ({accuracy*100:.2f}%)\n")
        if unknown_labels:
            fp.write(f"\nAdvertencia: {len(unknown_labels)} etiquetas del CSV no "
                     f"existían en el vocabulario del modelo:\n")
            for u in unknown_labels:
                fp.write(f"  - {u}\n")
        fp.write("\n")

        fp.write("## Accuracy por clase (sobre el conjunto de test del CSV)\n")
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
