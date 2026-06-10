"""Extracción paralela de keypoints (pose + manos) con MediaPipe Holistic
y escritura a un único archivo HDF5.

Arquitectura
------------
    [Worker 1] ─┐
    [Worker 2] ─┤
    [Worker 3] ─┼─→  (label, idx, ndarray (T, F))  ──→  [Main writer]  ──→  dataset_colsign.h5
       ...      ─┤
    [Worker 8] ─┘

  - Workers (multiprocessing.Pool): CPU-bound. Cada uno tiene su propia
    instancia de MediaPipe Holistic (creada una sola vez en `_init_worker`)
    y devuelve el array de keypoints por la pipe del Pool.
  - Proceso principal: I/O-bound. Recibe los arrays con `imap_unordered`
    y los escribe en HDF5 secuencialmente. HDF5 NO es thread-safe para
    escritura concurrente; un único writer en el proceso principal es la
    forma correcta de usarlo con multiprocessing.

Reanudable
----------
El HDF5 se abre en modo 'a' (append). Antes de mandar los jobs al Pool
se filtran los que ya existen en el archivo (mismo `label/idx`), así que
si interrumpes y vuelves a correr, salta lo ya procesado.

Estructura del HDF5 generado
----------------------------
    dataset_colsign.h5
      ├── "A veces"/
      │     ├── "0"   shape (120, 258), float32, gzip
      │     ├── "1"
      │     └── ...
      ├── "Abandonar"/
      │     └── ...
      └── attrs: sequence_length, num_features, type_extract, ...

Para cargar al entrenar, ver `src.utils.load_hdf5_dataset`.

Ejecución::

    .\.venv\Scripts\python.exe mediapipe_point_extraction.py
"""

import os
import time
import multiprocessing as mp_proc

import h5py
import numpy as np
from tqdm import tqdm

from src.utils import (
    extract_video_keypoints_with_holistic,
    KP_SIZE_HANDS,
    KP_SIZE_POSE_HANDS,
)


# ---------------- configuración ----------------

VIDEOS_PATH = 'dataset_videos'
#OUTPUT_HDF5 = 'dataset_colsign_45_154.h5'
OUTPUT_HDF5 = 'dataset_colsign_15_154.h5'

# 45 frames distribuidos uniformemente con np.linspace.
# A 45 frames sobre la parte útil del video (~3 s) → ~15 fps efectivos,
# que es lo estándar en SLR con MediaPipe.
SEQUENCE_LENGTH = 15

# Recorte temporal de la cola del video:
# si dura más de TRIM_THRESHOLD_S, se ignoran los últimos TRIM_TAIL_S
# (típicamente pantalla negra o mano abajo al final del clip).
TRIM_THRESHOLD_S = 3.0
TRIM_TAIL_S      = 1.0

TYPE_EXTRACT = 'pose_hands'  # 'hands' o 'pose_hands'
NUM_FEATURES = KP_SIZE_POSE_HANDS if TYPE_EXTRACT == 'pose_hands' else KP_SIZE_HANDS

VIDEO_EXTENSIONS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm')

# Ryzen 7 5700G: 8 cores físicos / 16 hilos SMT.
NUM_WORKERS = max(1, (os.cpu_count() or 4) // 2)

# 0=lite (más rápido), 1=full (default), 2=heavy
MODEL_COMPLEXITY = 1

# Compresión por dataset. 'gzip' es buena relación tamaño/velocidad.
HDF5_COMPRESSION = 'gzip'
HDF5_COMPRESSION_OPTS = 4  # 0-9


# ---------------- worker (uno por proceso) ----------------

_HOLISTIC = None


def _init_worker():
    """Inicializa una única instancia Holistic por proceso del Pool.

    `static_image_mode=True` porque vamos a procesar solo los frames
    muestreados (no consecutivos): en modo tracking, MediaPipe se confunde
    al saltar frames.
    """
    import mediapipe as mp
    global _HOLISTIC
    _HOLISTIC = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=False,
        # 0.3 en vez de 0.5: el dataset real tiene videos con iluminación,
        # encuadre y resolución variable. 0.5 es muy estricto para algunos
        # videos VP8 a baja resolución y descarta demasiadas detecciones.
        min_detection_confidence=0.3,
    )


def _procesar_job(job):
    """Procesa un video y devuelve `(action, sequence, array, error)`.

    `array` es ndarray (SEQUENCE_LENGTH, NUM_FEATURES) float32, o None si
    hubo error irrecuperable.
    """
    action, sequence, video_path = job
    try:
        arr = extract_video_keypoints_with_holistic(
            holistic=_HOLISTIC,
            url_video=video_path,
            sequence_length=SEQUENCE_LENGTH,
            type_extract=TYPE_EXTRACT,
            trim_threshold_s=TRIM_THRESHOLD_S,
            trim_tail_s=TRIM_TAIL_S,
        )
        return (action, sequence, video_path, arr, None)
    except Exception as e:
        return (action, sequence, video_path, None, repr(e))


# ---------------- construcción de jobs ----------------

def construir_jobs():
    """Lista plana de todos los videos a procesar."""
    if not os.path.isdir(VIDEOS_PATH):
        raise FileNotFoundError(f"No existe la carpeta de videos: {VIDEOS_PATH}")

    actions = sorted([
        d for d in os.listdir(VIDEOS_PATH)
        if os.path.isdir(os.path.join(VIDEOS_PATH, d))
    ])

    jobs = []
    for action in actions:
        action_dir = os.path.join(VIDEOS_PATH, action)
        videos = sorted([
            v for v in os.listdir(action_dir)
            if v.lower().endswith(VIDEO_EXTENSIONS)
        ])
        for seq_idx, video_name in enumerate(videos):
            video_path = os.path.join(action_dir, video_name)
            jobs.append((action, seq_idx, video_path))

    return actions, jobs


def filtrar_jobs_pendientes(jobs, hdf5_path):
    """Quita los jobs cuyo dataset ya existe en el HDF5 (reanudación)."""
    if not os.path.exists(hdf5_path):
        return jobs
    pendientes = []
    with h5py.File(hdf5_path, 'r') as f:
        for job in jobs:
            action, seq_idx, _ = job
            if action in f and str(seq_idx) in f[action]:
                continue
            pendientes.append(job)
    return pendientes


# ---------------- main ----------------

def _validar_hdf5_existente(hdf5_path):
    """Si el HDF5 ya existe, verifica que sus parámetros coincidan con los
    actuales. Esto evita mezclar versiones (ej. datasets viejos de 120
    frames + nuevos de 45 frames en el mismo archivo).
    """
    if not os.path.exists(hdf5_path):
        return
    with h5py.File(hdf5_path, 'r') as f:
        existing_seq_len = int(f.attrs.get('sequence_length', -1))
        existing_features = int(f.attrs.get('num_features', -1))
        existing_type = str(f.attrs.get('type_extract', ''))
    incompatibilidades = []
    if existing_seq_len not in (-1, SEQUENCE_LENGTH):
        incompatibilidades.append(
            f"sequence_length: archivo={existing_seq_len}, config={SEQUENCE_LENGTH}"
        )
    if existing_features not in (-1, NUM_FEATURES):
        incompatibilidades.append(
            f"num_features: archivo={existing_features}, config={NUM_FEATURES}"
        )
    if existing_type and existing_type != TYPE_EXTRACT:
        incompatibilidades.append(
            f"type_extract: archivo={existing_type!r}, config={TYPE_EXTRACT!r}"
        )
    if incompatibilidades:
        raise RuntimeError(
            f"El HDF5 existente {hdf5_path!r} no es compatible con la "
            f"configuración actual:\n  - "
            + "\n  - ".join(incompatibilidades)
            + "\n\nOpciones:\n"
              "  1) Cambia OUTPUT_HDF5 a otro nombre (p.ej. dataset_colsign_v3.h5).\n"
              "  2) Borra o renombra el archivo existente y vuelve a correr.\n"
        )


def main():
    _validar_hdf5_existente(OUTPUT_HDF5)

    actions, jobs_total = construir_jobs()
    jobs = filtrar_jobs_pendientes(jobs_total, OUTPUT_HDF5)
    ya_hechos = len(jobs_total) - len(jobs)

    print(
        f"Etiquetas: {len(actions)} | "
        f"Videos totales: {len(jobs_total)} | "
        f"Pendientes: {len(jobs)} (ya hechos: {ya_hechos}) | "
        f"Workers: {NUM_WORKERS} | "
        f"model_complexity: {MODEL_COMPLEXITY} | "
        f"seq_len: {SEQUENCE_LENGTH} | "
        f"trim: >{TRIM_THRESHOLD_S}s -{TRIM_TAIL_S}s | "
        f"output: {OUTPUT_HDF5}"
    )

    if not jobs:
        print("Nada que procesar. El HDF5 ya contiene todos los videos.")
        return

    inicio = time.time()
    ok = fail = errores = 0

    with h5py.File(OUTPUT_HDF5, 'a') as h5f, \
            mp_proc.Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:

        # Metadatos (se sobrescriben en cada ejecución, lo cual está bien).
        h5f.attrs['sequence_length']  = SEQUENCE_LENGTH
        h5f.attrs['num_features']     = NUM_FEATURES
        h5f.attrs['type_extract']     = TYPE_EXTRACT
        h5f.attrs['model_complexity'] = MODEL_COMPLEXITY
        h5f.attrs['trim_threshold_s'] = float(TRIM_THRESHOLD_S)
        h5f.attrs['trim_tail_s']      = float(TRIM_TAIL_S)

        # Pre-creamos los grupos de etiqueta (no cuesta nada).
        for action in actions:
            if action not in h5f:
                h5f.create_group(action)

        for action, sequence, video_path, arr, err in tqdm(
            pool.imap_unordered(_procesar_job, jobs, chunksize=2),
            total=len(jobs),
            desc='Procesando videos',
            unit='video',
        ):
            if err is not None:
                errores += 1
                tqdm.write(
                    f"  ERROR [{action}/{sequence}] "
                    f"{os.path.basename(video_path)}: {err}"
                )
                continue

            if arr is None or not np.any(arr):
                fail += 1
                tqdm.write(
                    f"  FALLO [{action}/{sequence}] "
                    f"{os.path.basename(video_path)} (todo ceros)"
                )
                # Lo guardamos igual para no reintentarlo eternamente.

            grupo = h5f[action]
            ds_name = str(sequence)
            if ds_name in grupo:
                # raro: alguien lo escribió entre filtrar y procesar. Lo borramos
                # y reescribimos para consistencia.
                del grupo[ds_name]
            grupo.create_dataset(
                ds_name,
                data=arr,
                dtype='float32',
                compression=HDF5_COMPRESSION,
                compression_opts=HDF5_COMPRESSION_OPTS,
            )
            grupo[ds_name].attrs['video_filename'] = os.path.basename(video_path)
            ok += 1

        h5f.flush()

    elapsed = time.time() - inicio
    minutos = elapsed / 60
    procesados = ok + fail + errores
    vps = procesados / elapsed if elapsed > 0 else 0
    tam_mb = os.path.getsize(OUTPUT_HDF5) / (1024 * 1024)
    print(
        f"\nFinalizado. OK={ok} | FALLO={fail} | ERROR={errores}"
        f" | Tiempo: {elapsed:.1f}s ({minutos:.1f} min)"
        f" | Throughput: {vps:.2f} videos/s"
        f" | Tamaño HDF5: {tam_mb:.1f} MB"
    )


if __name__ == '__main__':
    # Windows usa 'spawn': el guard `if __name__ == '__main__'` es obligatorio.
    mp_proc.freeze_support()
    main()
