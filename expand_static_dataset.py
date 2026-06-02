"""Expande el dataset de señas estáticas usando las imágenes limpias del
usuario como TEMPLATES GOLDEN.

Idea
----
El usuario revisó manualmente `dataset_static_crops/` y dejó solo 149
imágenes con la pose canónica de cada letra (el resto eran frames de
transición / ajuste). En lugar de regenerar todo automáticamente
(perdiendo esa curación), usamos esas 149 imágenes como referencia:

  1) Para cada imagen golden, extraemos sus 21 keypoints de mano
     normalizados (MediaPipe Hands sobre el crop ya recortado).
  2) Calibramos un umbral automático basado en las distancias
     intra-clase entre goldens (percentil 90).
  3) Procesamos cada video del grupo estático:
       - Sampleo de 45 frames con MediaPipe Holistic.
       - Para cada frame: keypoints normalizados de mano dominante.
       - Comparación con los goldens de su clase, considerando AMBAS
         orientaciones (directa y espejada horizontalmente).
       - Si min(distancia) < umbral, guardamos el crop (espejándolo si
         el match fue con la versión espejada del frame).
       - Limitamos a top-K matches por video para no saturar con frames
         casi idénticos.

Salida
------
`dataset_static_crops_expanded/{label}/<video_id>_f<frame>.jpg`

Junto con un resumen `expand_summary.json` que reporta:
  - umbral calculado
  - matches por clase
  - distancia min/max/mean de los matches

Ejecutar:
    .\.venv\Scripts\python.exe -u expand_static_dataset.py
"""

import os
import sys
import json
import time
import argparse
import multiprocessing as mpr
from datetime import datetime
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


# =====================================================================
# Configuración
# =====================================================================

DATASET_VIDEOS_DIR = 'dataset_videos'
GOLDEN_DIR         = 'dataset_static_crops'           # las 149 limpias del usuario
OUTPUT_DIR         = 'dataset_static_crops_expanded'  # dataset expandido (incluye golden + matches)
CSV_PATH           = 'etiquetas_modelo_raiz.csv'
ROOT_GROUP_NAME    = 'Grupo Estático'

SEQUENCE_LENGTH          = 45
CROP_SIZE                = 128       # mismo que las imágenes golden (sin re-escalar)
PADDING_RATIO            = 0.25
MIN_DETECTION_CONFIDENCE = 0.3
JPG_QUALITY              = 92
MAX_MATCHES_PER_VIDEO    = 6         # top-K frames matchados a guardar por video
MIN_MATCHES_PER_CLASS    = 60        # piso mínimo de imágenes por clase
INTRA_PERCENTILE         = 90        # umbral = percentil de distancias intra-clase
RELAX_FACTOR             = 1.5       # si una clase tiene < MIN matches, relajar umbral por este factor

VIDEO_EXTS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv')

# Alias para directorios con caracteres no-ASCII (carpetas en disco)
LABEL_TO_FOLDER_ALIAS = {'ñ': 'enie'}
def label_to_folder(label):
    return LABEL_TO_FOLDER_ALIAS.get(label, label)
def folder_to_label(folder):
    inv = {v: k for k, v in LABEL_TO_FOLDER_ALIAS.items()}
    return inv.get(folder, folder)

def sanitize_filename(name):
    return name.replace('ñ', 'nh').replace('Ñ', 'NH')


# =====================================================================
# Etapa 1: calcular keypoints golden a partir de las imágenes limpias
# =====================================================================

def normalize_hand_kp(kp_xyz):
    """kp_xyz: (21, 3). Centra en muñeca y escala por mean(|wrist-points|)."""
    p = kp_xyz - kp_xyz[0]
    scale = float(np.linalg.norm(p[1:], axis=1).mean())
    if scale > 1e-6:
        p = p / scale
    return p

def mirror_kp(kp_normalized):
    """Espejado horizontal de keypoints centrados en muñeca."""
    m = kp_normalized.copy()
    m[:, 0] = -m[:, 0]
    return m


def compute_golden_keypoints(golden_dir):
    """Procesa las imágenes golden con MediaPipe Hands y devuelve
    {label_display: list[(21, 3) ndarray]}.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.3,
    )

    golden = defaultdict(list)
    skipped = defaultdict(int)

    folders = sorted(
        [d for d in os.listdir(golden_dir)
         if os.path.isdir(os.path.join(golden_dir, d))]
    )
    print(f"Procesando golden de {len(folders)} clases...")
    for folder in folders:
        fpath = os.path.join(golden_dir, folder)
        label = folder_to_label(folder)
        for fn in sorted(os.listdir(fpath)):
            if not fn.lower().endswith('.jpg'):
                continue
            full = os.path.join(fpath, fn)
            img = cv2.imread(full)
            if img is None:
                skipped[label] += 1
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if not res.multi_hand_landmarks:
                skipped[label] += 1
                continue
            lms = res.multi_hand_landmarks[0].landmark
            kp = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            golden[label].append(normalize_hand_kp(kp))
    hands.close()

    print(f"\nGolden por clase (detectados / saltados):")
    for folder in folders:
        label = folder_to_label(folder)
        n_det = len(golden[label])
        n_skip = skipped[label]
        print(f"  {label!s:<6} {n_det:>4}  (skipped {n_skip})")

    return dict(golden)


def calibrate_threshold(golden):
    """Devuelve un dict {label: threshold_dist} basado en las distancias
    intra-clase entre los keypoints golden de cada clase. Si una clase
    tiene <2 goldens, usa el threshold global como fallback."""
    all_intra = []
    per_class = {}
    for label, kps in golden.items():
        if len(kps) < 2:
            continue
        flat = np.array([k.flatten() for k in kps])  # (n, 63)
        # Distancias pares
        dists = []
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                # Tomar min entre directa y mirrored (considerar zurdos)
                d_dir = float(np.linalg.norm(flat[i] - flat[j]))
                d_mir = float(np.linalg.norm(flat[i] - mirror_kp(kps[j]).flatten()))
                dists.append(min(d_dir, d_mir))
        per_class[label] = float(np.percentile(dists, INTRA_PERCENTILE))
        all_intra.extend(dists)

    global_threshold = float(np.percentile(all_intra, INTRA_PERCENTILE)) if all_intra else 1.0
    # Fallback para clases con 1 solo golden
    full = {label: per_class.get(label, global_threshold) for label in golden.keys()}

    print(f"\nUmbral global (p{INTRA_PERCENTILE} intra-clase): {global_threshold:.4f}")
    print(f"Umbral por clase:")
    for label in sorted(full.keys()):
        marker = ' (fallback global)' if label not in per_class else ''
        print(f"  {label!s:<6} {full[label]:.4f}{marker}")
    return full, global_threshold


# =====================================================================
# Etapa 2: worker MediaPipe Holistic por proceso
# =====================================================================

_holistic = None

def _init_worker():
    global _holistic
    _holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
    )


def _count_real_frames(cap):
    n = 0
    while cap.grab():
        n += 1
    return n


LEFT_HAND_BLOCK, RIGHT_HAND_BLOCK = slice(132, 195), slice(195, 258)


def _process_video_for_matches(args):
    """Procesa un video y devuelve hasta MAX_MATCHES_PER_VIDEO crops
    cuya keypoints estén bajo el umbral de su clase.

    Args:
        args: (video_path, label_display, threshold, goldens_flat, out_dir)
              goldens_flat: (n_goldens, 63) ndarray (incluye solo direct,
              la versión espejada se compara también acá).
    """
    global _holistic
    if _holistic is None:
        _init_worker()

    video_path, label, threshold, goldens_flat, out_dir, video_id = args

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        real_frames = _count_real_frames(cap)
        cap.release()
        if real_frames < 5:
            return []

        indices = np.linspace(0, real_frames - 1, SEQUENCE_LENGTH).astype(int)
        indices_set = set(int(i) for i in indices)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        candidates = []  # list of (dist, frame_idx, crop_bgr, use_mirror)

        fi = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if fi in indices_set:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                res = _holistic.process(rgb)
                lh = res.left_hand_landmarks
                rh = res.right_hand_landmarks
                # Probar ambas manos detectadas
                for hand_lms, hand_side in (
                    (lh, 'left'), (rh, 'right'),
                ):
                    if hand_lms is None:
                        continue
                    kp = np.array(
                        [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                        dtype=np.float32,
                    )
                    kp_norm = normalize_hand_kp(kp)
                    kp_flat = kp_norm.flatten()
                    kp_mir_flat = mirror_kp(kp_norm).flatten()

                    # Comparar contra todos los goldens
                    d_direct = float(np.linalg.norm(
                        goldens_flat - kp_flat[None, :], axis=1,
                    ).min())
                    d_mirror = float(np.linalg.norm(
                        goldens_flat - kp_mir_flat[None, :], axis=1,
                    ).min())

                    if d_direct <= d_mirror:
                        best_dist = d_direct
                        use_mirror = False
                    else:
                        best_dist = d_mirror
                        use_mirror = True

                    if best_dist > threshold:
                        continue

                    # Hacer crop con padding y resize a CROP_SIZE
                    H, W = frame_bgr.shape[:2]
                    xs = kp[:, 0] * W
                    ys = kp[:, 1] * H
                    x_min, x_max = float(xs.min()), float(xs.max())
                    y_min, y_max = float(ys.min()), float(ys.max())
                    cx = (x_min + x_max) * 0.5
                    cy = (y_min + y_max) * 0.5
                    half = max(x_max - x_min, y_max - y_min) * 0.5 * (1 + PADDING_RATIO)
                    if half < 10:
                        continue
                    x0 = int(max(0, round(cx - half)))
                    y0 = int(max(0, round(cy - half)))
                    x1 = int(min(W, round(cx + half)))
                    y1 = int(min(H, round(cy + half)))
                    if (x1 - x0) < 10 or (y1 - y0) < 10:
                        continue
                    crop = frame_bgr[y0:y1, x0:x1]
                    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                      interpolation=cv2.INTER_AREA)
                    if use_mirror:
                        crop = cv2.flip(crop, 1)
                    candidates.append((best_dist, fi, crop, use_mirror, hand_side))
            fi += 1
        cap.release()

        # ordenar por menor distancia y quedarse con top-K
        candidates.sort(key=lambda c: c[0])
        top = candidates[:MAX_MATCHES_PER_VIDEO]

        saved = []
        os.makedirs(out_dir, exist_ok=True)
        for dist, frame_idx, crop, use_mirror, hand_side in top:
            out_name = f"{video_id}_f{frame_idx:03d}.jpg"
            out_path = os.path.join(out_dir, out_name)
            ok_enc, encoded = cv2.imencode(
                '.jpg', crop, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY],
            )
            if not ok_enc:
                continue
            try:
                with open(out_path, 'wb') as fp:
                    fp.write(encoded.tobytes())
            except OSError:
                continue
            saved.append({
                'video': os.path.basename(video_path),
                'frame': int(frame_idx),
                'dist':  float(dist),
                'mirror': bool(use_mirror),
                'hand':  hand_side,
            })

        return saved
    except Exception as e:
        return [{'error': f'{type(e).__name__}: {e}', 'video': video_path}]


# =====================================================================
# Main
# =====================================================================

def collect_jobs_per_class(static_labels):
    """Devuelve dict label -> [(video_path, out_dir, video_id), ...]"""
    jobs = defaultdict(list)
    for label in static_labels:
        src_dir = os.path.join(DATASET_VIDEOS_DIR, label)
        if not os.path.isdir(src_dir):
            continue
        out_dir = os.path.join(OUTPUT_DIR, label_to_folder(label))
        for vname in os.listdir(src_dir):
            if not vname.lower().endswith(VIDEO_EXTS):
                continue
            vid = sanitize_filename(os.path.splitext(vname)[0])
            jobs[label].append(
                (os.path.join(src_dir, vname), out_dir, vid)
            )
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=max(1, mpr.cpu_count() - 1))
    parser.add_argument('--keep-golden', action='store_true', default=True,
                        help='Copiar las imágenes golden al output (default: True)')
    args = parser.parse_args()

    t_start = time.time()
    print('=' * 70)
    print('=== expand_static_dataset ===')
    print(f'Inicio: {datetime.now().isoformat(timespec="seconds")}')
    print('=' * 70)

    # CSV
    df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')
    df['nombre']        = df['nombre'].astype(str).str.strip()
    df['etiqueta_raiz'] = df['etiqueta_raiz'].astype(str).str.strip()
    static_labels = sorted(
        df[df['etiqueta_raiz'] == ROOT_GROUP_NAME]['nombre'].tolist()
    )
    print(f"\nClases estáticas: {len(static_labels)}")

    # 1) Golden keypoints
    print("\n--- 1) Procesando imágenes golden ---")
    golden = compute_golden_keypoints(GOLDEN_DIR)

    # 2) Umbrales
    print("\n--- 2) Calibrando umbrales ---")
    thresholds, global_thr = calibrate_threshold(golden)

    # Prefiltering: agrupamos goldens por clase como matriz (n,63) que se
    # le pasa al worker (incluyendo ambas orientaciones para que el worker
    # compare cualquiera de las dos).
    goldens_by_label_flat = {}
    for label, kps in golden.items():
        if not kps:
            continue
        arr_direct = np.array([k.flatten() for k in kps], dtype=np.float32)
        arr_mirror = np.array([mirror_kp(k).flatten() for k in kps], dtype=np.float32)
        goldens_by_label_flat[label] = np.concatenate([arr_direct, arr_mirror], axis=0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2.5) (Opcional) Copiar las imágenes golden al expanded para que sean
    # también parte del entrenamiento.
    if args.keep_golden:
        import shutil
        n_copied = 0
        for folder in os.listdir(GOLDEN_DIR):
            src = os.path.join(GOLDEN_DIR, folder)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(OUTPUT_DIR, folder)
            os.makedirs(dst, exist_ok=True)
            for fn in os.listdir(src):
                if fn.lower().endswith('.jpg'):
                    shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
                    n_copied += 1
        print(f"\nGolden copiadas al expanded: {n_copied} imágenes")

    # 3) Procesar videos por clase
    print("\n--- 3) Procesando videos con multiprocessing ---")
    jobs_by_label = collect_jobs_per_class(static_labels)
    all_jobs = []
    for label, jobs in jobs_by_label.items():
        if label not in goldens_by_label_flat:
            print(f"  WARN: clase '{label}' sin goldens, omitiendo {len(jobs)} videos")
            continue
        thr = thresholds[label]
        goldens_flat = goldens_by_label_flat[label]
        for video_path, out_dir, vid in jobs:
            all_jobs.append((video_path, label, thr, goldens_flat, out_dir, vid))
    print(f"Total videos a procesar: {len(all_jobs)}")
    print(f"Workers: {args.workers}")

    all_saved = []
    with mpr.Pool(processes=args.workers, initializer=_init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_video_for_matches, all_jobs, chunksize=4),
            total=len(all_jobs), desc='Videos',
        ):
            all_saved.extend(result)

    # 4) Resumen
    per_class = defaultdict(int)
    distances = []
    mirrors = 0
    errors = []
    for s in all_saved:
        if 'error' in s:
            errors.append(s)
            continue
        # Folder = primer carácter del crop name antes de '_'
        # Mejor: inferir del path (no lo tenemos). Lo guardamos por el out_dir cuando lo procesemos.
        distances.append(s['dist'])
        if s['mirror']:
            mirrors += 1

    # Recontar imágenes por carpeta
    print(f"\n--- 4) Resumen final ---")
    print(f"Total matches guardados: {len([s for s in all_saved if 'error' not in s])}")
    print(f"Errores: {len(errors)}")
    if distances:
        print(f"Distancia matches: min={min(distances):.4f}  "
              f"max={max(distances):.4f}  mean={np.mean(distances):.4f}")
    print(f"Matches con espejado: {mirrors}")

    print(f"\nImágenes finales por clase (golden + expansion):")
    total = 0
    counts_per_class = {}
    for label in static_labels:
        folder = label_to_folder(label)
        d = os.path.join(OUTPUT_DIR, folder)
        n = len([f for f in os.listdir(d) if f.lower().endswith('.jpg')]) \
            if os.path.isdir(d) else 0
        counts_per_class[label] = n
        total += n
        marker = ' ⚠' if n < MIN_MATCHES_PER_CLASS else ''
        print(f"  {label!s:<6} {n:>4}{marker}")
    print(f"  TOTAL: {total}")

    duration = time.time() - t_start
    print(f"\nDuración total: {duration:.1f}s ({duration/60:.2f} min)")

    # Guardar resumen JSON
    summary = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'golden_dir': GOLDEN_DIR,
        'output_dir': OUTPUT_DIR,
        'global_threshold_p90_intra': global_thr,
        'per_class_threshold': {str(k): v for k, v in thresholds.items()},
        'max_matches_per_video': MAX_MATCHES_PER_VIDEO,
        'counts_per_class': {str(k): v for k, v in counts_per_class.items()},
        'total_images': total,
        'matches_with_mirror': mirrors,
        'distance_stats': {
            'min': float(min(distances)) if distances else None,
            'max': float(max(distances)) if distances else None,
            'mean': float(np.mean(distances)) if distances else None,
        },
        'duration_seconds': duration,
    }
    summary_path = os.path.join(OUTPUT_DIR, 'expand_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"\nResumen JSON: {summary_path}")


if __name__ == '__main__':
    main()
