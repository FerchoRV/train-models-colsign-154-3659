"""Extrae crops cuadrados de la mano dominante (RGB) para cada video del
grupo "Grupo Estático" del CSV `etiquetas_modelo_raiz.csv`.

Para cada video:
  1. Cuenta frames reales (pasada 1, cap.grab()) -- evita el bug de
     metadata VP8 que ya nos pegó antes.
  2. Calcula `linspace` de 45 índices a procesar.
  3. Procesa esos frames con MediaPipe Holistic y guarda en memoria
     tanto la imagen original como los keypoints 2D de ambas manos.
  4. Determina la mano dominante (la detectada en más frames).
  5. Sobre los frames donde la dominante fue detectada, normaliza los
     keypoints (centro en muñeca + escala por mean dist) y encuentra el
     frame MEDOIDE (el más cercano a la pose promedio del video). Ese es
     el frame "más estable" temporalmente.
  6. Calcula la bbox 2D de la mano dominante en píxeles, le agrega 25%
     de padding, la cuadra y recorta. Resize a 128×128. Guarda como JPG.

NO se espeja la mano izquierda: el modelo CNN aprenderá invariancia
con `random_flip_horizontal` durante augmentation.

Estructura de salida (compatible con `image_dataset_from_directory`):

  dataset_static_crops/
    a/      <video_id>.jpg
    b/      <video_id>.jpg
    ...
    ñ/      <video_id>.jpg

Multiprocessing con un Holistic por worker (mismo patrón que
`mediapipe_point_extraction.py`).

Ejecutar:
    .\.venv\Scripts\python.exe -u extract_static_hand_crops.py
"""

import os
import sys
import time
import argparse
import multiprocessing as mpr
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# UTF-8 en Windows
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


# =====================================================================
# Configuración
# =====================================================================

DATASET_VIDEOS_DIR = 'dataset_videos'
OUTPUT_DIR         = 'dataset_static_crops'
CSV_PATH           = 'etiquetas_modelo_raiz.csv'
ROOT_GROUP_NAME    = 'Grupo Estático'

SEQUENCE_LENGTH          = 45
CROP_SIZE                = 224       # resolución nativa de MobileNetV2 ImageNet
PADDING_RATIO            = 0.25      # 25% de padding alrededor del bbox
MIN_DETECTION_CONFIDENCE = 0.3
JPG_QUALITY              = 92
VIDEO_EXTS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv')

# Algunas etiquetas tienen caracteres no-ASCII que rompen TensorFlow al cargar
# directorios en Windows (UnicodeDecodeError con bytes como 0xf1 para ñ).
# Usamos un alias en disco para esos casos; train_static_cnn.py mapea de
# vuelta al nombre real al guardar el JSON de labels y los reports.
LABEL_TO_FOLDER_ALIAS = {
    'ñ': 'enie',
}


def label_to_folder(label):
    return LABEL_TO_FOLDER_ALIAS.get(label, label)


def sanitize_filename(name):
    """Reemplaza caracteres no-ASCII que rompen TensorFlow en Windows.
    Conserva guiones, guiones-bajos, puntos y caracteres alfanuméricos."""
    return name.replace('ñ', 'nh').replace('Ñ', 'NH')


# =====================================================================
# Worker MediaPipe (uno por proceso)
# =====================================================================

_holistic = None  # global por proceso

def _init_worker():
    """Crea un Holistic exclusivo para este worker."""
    global _holistic
    _holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
    )


def _close_worker():
    global _holistic
    if _holistic is not None:
        _holistic.close()
        _holistic = None


# =====================================================================
# Pipeline por video
# =====================================================================

def _count_real_frames(cap):
    """Cuenta los frames realmente decodificables. Workaround del bug VP8
    donde CAP_PROP_FRAME_COUNT miente."""
    n = 0
    while cap.grab():
        n += 1
    return n


def _extract_one_crop(video_path, out_path):
    """Procesa un video y guarda el crop de la mano dominante (frame medoide).

    Returns:
        (ok: bool, reason: str)
    """
    global _holistic
    if _holistic is None:
        _init_worker()

    if os.path.exists(out_path):
        return True, 'skip_existing'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, 'no_open'

    real_frames = _count_real_frames(cap)
    cap.release()
    if real_frames < 5:
        return False, 'too_few_frames'

    # frames a muestrear (linspace)
    indices = np.linspace(0, real_frames - 1, SEQUENCE_LENGTH).astype(int)
    indices_set = set(int(i) for i in indices)

    # pasada 2: leer y procesar solo los frames muestreados
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, 'no_reopen'

    frames_imgs    = {}  # frame_idx -> BGR ndarray
    frames_kp_left = {}  # frame_idx -> (21, 2) o None
    frames_kp_right= {}

    fi = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if fi in indices_set:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = _holistic.process(rgb)
            lh = (
                np.array([[lm.x, lm.y] for lm in res.left_hand_landmarks.landmark],
                         dtype=np.float32)
                if res.left_hand_landmarks else None
            )
            rh = (
                np.array([[lm.x, lm.y] for lm in res.right_hand_landmarks.landmark],
                         dtype=np.float32)
                if res.right_hand_landmarks else None
            )
            frames_imgs[fi]     = frame_bgr
            frames_kp_left[fi]  = lh
            frames_kp_right[fi] = rh
        fi += 1
    cap.release()

    if not frames_imgs:
        return False, 'no_sampled_frames'

    n_left  = sum(1 for v in frames_kp_left.values()  if v is not None)
    n_right = sum(1 for v in frames_kp_right.values() if v is not None)
    if n_left == 0 and n_right == 0:
        return False, 'no_hand_detected'

    use_left = n_left > n_right

    # frames con la mano dominante detectada, en orden temporal
    valid = []
    kp_source = frames_kp_left if use_left else frames_kp_right
    for fi in sorted(frames_imgs.keys()):
        kp = kp_source[fi]
        if kp is not None:
            valid.append((fi, kp))

    if not valid:
        return False, 'no_dominant_frames'

    # Normalizar cada keypoint set y encontrar el medoide
    normed = []
    for _, kp in valid:
        wrist = kp[0]
        p = kp - wrist
        # escala = distancia promedio de puntos 1..20 a la muñeca
        scale = float(np.linalg.norm(p[1:], axis=1).mean())
        if scale > 1e-6:
            p = p / scale
        normed.append(p)
    normed = np.array(normed)  # (T_valid, 21, 2)
    centroid = normed.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(normed - centroid, axis=(1, 2))
    medoid_idx = int(np.argmin(dists))

    chosen_fi, chosen_kp = valid[medoid_idx]
    img_bgr = frames_imgs[chosen_fi]
    H, W = img_bgr.shape[:2]

    # bbox en píxeles
    xs = chosen_kp[:, 0] * W
    ys = chosen_kp[:, 1] * H
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    half = max(x_max - x_min, y_max - y_min) * 0.5
    half *= (1.0 + PADDING_RATIO)

    if half < 10:
        return False, 'tiny_bbox'

    x0 = int(max(0,     round(cx - half)))
    y0 = int(max(0,     round(cy - half)))
    x1 = int(min(W,     round(cx + half)))
    y1 = int(min(H,     round(cy + half)))
    if (x1 - x0) < 10 or (y1 - y0) < 10:
        return False, 'tiny_crop'

    crop = img_bgr[y0:y1, x0:x1]
    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                              interpolation=cv2.INTER_AREA)

    # Canonicalización de orientación: si la mano dominante era la izquierda
    # (típicamente, persona diestra grabada en modo selfie/espejo), espejamos
    # horizontalmente la imagen para que TODAS las muestras queden en la
    # orientación canónica de mano derecha. Las letras del abecedario LSC son
    # sensibles a la orientación (b↔d, p↔q, etc.), por lo que NO usamos
    # RandomFlip en augmentation; canonicalizamos en disco.
    if use_left:
        crop_resized = cv2.flip(crop_resized, 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # cv2.imwrite tiene un bug en Windows con paths que contienen caracteres
    # no-ASCII (ej. la letra "ñ"): falla silenciosamente sin levantar excepción.
    # Workaround: codificar en memoria con cv2.imencode y escribir bytes con
    # open() (Python sí maneja UTF-8 en rutas).
    ok_enc, encoded = cv2.imencode(
        '.jpg', crop_resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY],
    )
    if not ok_enc:
        return False, 'encode_failed'
    try:
        with open(out_path, 'wb') as fp:
            fp.write(encoded.tobytes())
    except OSError as e:
        return False, f'write_failed:{e}'

    return True, f"ok ({'left' if use_left else 'right'} frame={chosen_fi})"


def _job(args):
    video_path, out_path = args
    try:
        return _extract_one_crop(video_path, out_path)
    except Exception as e:
        return False, f'error:{type(e).__name__}:{e}'


# =====================================================================
# Main
# =====================================================================

def collect_jobs(static_labels):
    jobs = []
    n_per_class = {}
    for label in static_labels:
        src_dir = os.path.join(DATASET_VIDEOS_DIR, label)
        if not os.path.isdir(src_dir):
            print(f"  WARN: no existe {src_dir}")
            n_per_class[label] = 0
            continue
        out_class_dir = os.path.join(OUTPUT_DIR, label_to_folder(label))
        videos = [f for f in os.listdir(src_dir) if f.lower().endswith(VIDEO_EXTS)]
        n_per_class[label] = len(videos)
        for vname in videos:
            vid = sanitize_filename(os.path.splitext(vname)[0])
            out_path = os.path.join(out_class_dir, f"{vid}.jpg")
            jobs.append((os.path.join(src_dir, vname), out_path))
    return jobs, n_per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mpr.cpu_count() - 1))
    parser.add_argument('--limit',   type=int, default=None,
                        help='Procesar solo los primeros N videos (para pruebas)')
    args = parser.parse_args()

    print(f"=== extract_static_hand_crops ===")
    print(f"Inicio:  {datetime.now().isoformat(timespec='seconds')}")
    print(f"CSV:     {CSV_PATH}")
    print(f"Videos:  {DATASET_VIDEOS_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Workers: {args.workers}")
    print(f"Crop:    {CROP_SIZE}x{CROP_SIZE}  padding={PADDING_RATIO*100:.0f}%")
    print()

    df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')
    df['nombre']        = df['nombre'].astype(str).str.strip()
    df['etiqueta_raiz'] = df['etiqueta_raiz'].astype(str).str.strip()
    static_labels = sorted(
        df[df['etiqueta_raiz'] == ROOT_GROUP_NAME]['nombre'].tolist()
    )
    if not static_labels:
        raise SystemExit(f"No hay etiquetas con etiqueta_raiz='{ROOT_GROUP_NAME}'")
    print(f"Clases del grupo: {len(static_labels)}")
    print(f"  {static_labels}\n")

    jobs, n_per_class = collect_jobs(static_labels)
    total = len(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"LIMIT: solo {len(jobs)} de {total} videos")
    else:
        print(f"Total de videos a procesar: {total}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    stats = {'ok': 0, 'skip': 0, 'fail': 0, 'reasons': {}}

    if args.workers > 1:
        with mpr.Pool(processes=args.workers, initializer=_init_worker) as pool:
            for ok, reason in tqdm(
                pool.imap_unordered(_job, jobs, chunksize=4),
                total=len(jobs), desc='Procesando',
            ):
                if ok and 'skip' in reason:
                    stats['skip'] += 1
                elif ok:
                    stats['ok'] += 1
                else:
                    stats['fail'] += 1
                    stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
    else:
        _init_worker()
        for j in tqdm(jobs, desc='Procesando'):
            ok, reason = _job(j)
            if ok and 'skip' in reason:
                stats['skip'] += 1
            elif ok:
                stats['ok'] += 1
            else:
                stats['fail'] += 1
                stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
        _close_worker()

    duration = time.time() - t0
    print(f"\n--- Resumen ---")
    print(f"Total:      {len(jobs)}")
    print(f"Generados:  {stats['ok']}")
    print(f"Existentes: {stats['skip']}")
    print(f"Fallos:     {stats['fail']}")
    if stats['reasons']:
        print("Razones de fallo:")
        for r, c in sorted(stats['reasons'].items(), key=lambda x: -x[1]):
            print(f"  {r}: {c}")
    print(f"Duración:   {duration:.1f}s  ({duration/60:.1f} min)")
    print(f"Output:     {OUTPUT_DIR}")

    # contar imágenes finales por carpeta
    print(f"\nImágenes generadas por clase:")
    for label in static_labels:
        d = os.path.join(OUTPUT_DIR, label)
        n = len([f for f in os.listdir(d) if f.lower().endswith('.jpg')]) \
            if os.path.isdir(d) else 0
        print(f"  {label:<6} {n:>4} / {n_per_class.get(label, 0):>4}")


if __name__ == '__main__':
    main()
