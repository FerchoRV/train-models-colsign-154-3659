"""Gestión de experimentos: rutas de video y registro en CSV.

Cada experimento (una sesión de traducción) guarda:
  - un video .mp4 en `dataset_time_real/`
  - una fila en `dataset_time_real/experiments.csv` con la ruta del video y
    la lista de señas detectadas en ese experimento.
"""

import csv
import os
from datetime import datetime

from . import config

CSV_FIELDS = ['experiment_id', 'timestamp', 'modelo', 'video_path',
              'num_senas', 'senas']


def ensure_dirs():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def new_experiment_paths():
    """Devuelve `(experiment_id, video_path)` con timestamp único."""
    ensure_dirs()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_id = f'exp_{ts}'
    video_path = os.path.join(config.OUTPUT_DIR, f'{exp_id}.mp4')
    return exp_id, video_path


def append_experiment_row(experiment_id, video_path, signs, modelo=''):
    """Agrega (o crea) una fila al CSV de experimentos.

    `signs` es la lista ordenada de etiquetas detectadas en el experimento.
    `modelo` es la estrategia usada ('colsign 154' | 'colsign jerarquico').
    """
    ensure_dirs()
    is_new = not os.path.exists(config.CSV_PATH)
    rel_video = os.path.relpath(video_path, config.PROJECT_ROOT).replace('\\', '/')
    with open(config.CSV_PATH, 'a', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        if is_new:
            writer.writerow(CSV_FIELDS)
        writer.writerow([
            experiment_id,
            datetime.now().isoformat(timespec='seconds'),
            modelo,
            rel_video,
            len(signs),
            ' | '.join(signs),
        ])
    return config.CSV_PATH
