"""Genera `video_path_test_piplane.csv` con un 30% de los videos por
etiqueta seleccionados aleatoriamente.

Estructura esperada de entrada
------------------------------
    dataset_videos/
        <label_1>/
            video_a.mp4
            video_b.mp4
            ...
        <label_2>/
            ...

Salida
------
    video_path_test_piplane.csv  (separador ',', encoding utf-8)
        columnas: label, path

Notas
-----
- La selección por etiqueta usa `math.ceil(n * 0.3)`, así una clase con
  3 videos aporta 1, una con 5 aporta 2, etc. Si una clase tiene 0
  videos, se omite.
- Se usa un seed fijo para que la selección sea reproducible.
- Las rutas se guardan como ruta relativa al proyecto con separador
  "/" para que sea portable entre Windows y Linux.
"""

import os
import sys
import csv
import math
import random
from datetime import datetime

# UTF-8 en consola (consistente con los demás scripts del proyecto en Windows)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


DATASET_VIDEOS_DIR = 'dataset_videos'
OUTPUT_CSV         = 'video_path_test_piplane.csv'
TEST_FRACTION      = 0.30
SEED               = 42
VIDEO_EXTS         = ('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm')


def to_posix(path):
    """Normaliza separadores a '/' para que el CSV sea portable."""
    return path.replace('\\', '/')


def main():
    random.seed(SEED)

    if not os.path.isdir(DATASET_VIDEOS_DIR):
        raise SystemExit(
            f"No se encontró la carpeta '{DATASET_VIDEOS_DIR}'. "
            f"Ejecuta este script desde la raíz del proyecto."
        )

    print(f"=== selecction_video_test_piplanes ===")
    print(f"Fecha:           {datetime.now().isoformat(timespec='seconds')}")
    print(f"Directorio:      {DATASET_VIDEOS_DIR}")
    print(f"Fracción test:   {TEST_FRACTION:.0%}")
    print(f"Random seed:     {SEED}")
    print()

    labels = sorted(
        d for d in os.listdir(DATASET_VIDEOS_DIR)
        if os.path.isdir(os.path.join(DATASET_VIDEOS_DIR, d))
    )
    print(f"Etiquetas encontradas: {len(labels)}")

    rows = []
    total_videos = 0
    total_selected = 0
    per_label_stats = []

    for label in labels:
        label_dir = os.path.join(DATASET_VIDEOS_DIR, label)
        videos = sorted(
            f for f in os.listdir(label_dir)
            if f.lower().endswith(VIDEO_EXTS)
        )
        n = len(videos)
        total_videos += n
        if n == 0:
            per_label_stats.append((label, 0, 0))
            continue

        n_test = max(1, math.ceil(n * TEST_FRACTION))
        chosen = random.sample(videos, n_test)
        for vname in chosen:
            full_path = to_posix(os.path.join(DATASET_VIDEOS_DIR, label, vname))
            rows.append({'label': label, 'path': full_path})

        total_selected += n_test
        per_label_stats.append((label, n, n_test))

    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['label', 'path'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'label':<35} {'total':>6} {'test':>6}")
    print('-' * 55)
    for label, n, k in per_label_stats:
        print(f"{label:<35} {n:>6} {k:>6}")
    print('-' * 55)
    print(f"{'TOTAL':<35} {total_videos:>6} {total_selected:>6}")
    print(f"\nCSV generado: {OUTPUT_CSV}")
    print(f"Filas escritas: {len(rows)}")


if __name__ == '__main__':
    main()
