"""Búsqueda exhaustiva de DIFERENCIAS entre los videos de entrenamiento
(`dataset_videos`) y los de evaluación de usuarios nuevos
(`dataset_videos_evaluate`).

Motivación
----------
Al evaluar los modelos sobre videos de usuarios nuevos la accuracy cae
fuertemente. Diagnósticos previos (encuadre, distancia focal, fps) apuntaron
a que el cuello de botella NO es el modelo sino la **dificultad de MediaPipe
para detectar los puntos de control de las manos** en los videos de
evaluación (manos fuera de cuadro / usuario demasiado cerca). Este script
cuantifica esas diferencias de forma sistemática y entrega un reporte.

Qué mide (por video, agregando sobre frames muestreados)
--------------------------------------------------------
1. Detección de MediaPipe (tasas 0..1):
   - pose, mano izq., mano der., cualquier mano, ambas manos.
2. Encuadre / distancia a cámara (sobre frames con pose):
   - distancia inter-hombros (norm.), bbox de pose (ancho/alto/área),
     posición vertical de hombros y nariz, completitud de pose (visibilidad).
3. Calidad de manos detectadas:
   - fracción de landmarks de la mano fuera de [0,1] y pegados al borde.
4. Metadatos de video:
   - resolución (ancho/alto), fps reportado, nº de frames, duración, aspect.

Cómo decide los hallazgos PRINCIPALES
-------------------------------------
Para cada métrica calcula el tamaño de efecto (Cohen's d) entre
entrenamiento y evaluación y rankea por |d|. Las de mayor |d| se reportan
como hallazgos principales (típicamente dominan las de detección de manos).

Salida (carpeta `results_diff/`)
--------------------------------
- `differences_data_video.txt`            -> reporte de hallazgos.
- `differences_data_video_per_video.csv`  -> métricas crudas por video.

Configuración de MediaPipe: la MISMA que usa la inferencia/entrenamiento
(`make_holistic()`: static_image_mode=True, model_complexity=1,
min_detection_confidence=0.3), para que las tasas reflejen lo que el modelo
realmente recibe.

Ejecutar:
    .\.venv\Scripts\python.exe -u diffrences_data_video.py
"""

import os
import sys
import csv
import math
import time
import random
from datetime import datetime

# UTF-8 en consola (Windows)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

import numpy as np
import cv2
from tqdm import tqdm

from src.utils import mediapipe_detection, compute_target_indices
from src.utils_pipeplanes import make_holistic


# =====================================================================
# Configuración
# =====================================================================
PATH_TRAIN = 'dataset_videos'
PATH_EVAL  = 'dataset_videos_evaluate'

RESULTS_DIR = 'results_diff'
OUTPUT_TXT  = os.path.join(RESULTS_DIR, 'differences_data_video.txt')
OUTPUT_CSV  = os.path.join(RESULTS_DIR, 'differences_data_video_per_video.csv')

VIDEO_EXTS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm')

# Muestreo (entrenamiento tiene miles de videos; evaluación se procesa entero).
MAX_TRAIN_VIDEOS     = 250    # 0 = todos
MAX_FRAMES_PER_VIDEO = 45     # frames muestreados por video (uniforme, como train)
MAX_READ_FRAMES      = 900    # tope de frames leídos por video (memoria)
SEED                 = 42

# Índices de landmarks de pose (MediaPipe)
LSHOULDER, RSHOULDER, NOSE = 11, 12, 0

# Métricas: clave -> (nombre legible, grupo). El signo del efecto se reporta
# como train - eval (d>0 => entrenamiento mayor que evaluación).
METRIC_INFO = [
    ('pose_rate',         'Tasa detección POSE',                 'Detección'),
    ('lh_rate',           'Tasa detección mano IZQUIERDA',       'Detección'),
    ('rh_rate',           'Tasa detección mano DERECHA',         'Detección'),
    ('any_hand_rate',     'Tasa detección CUALQUIER mano',       'Detección'),
    ('both_hands_rate',   'Tasa detección AMBAS manos',          'Detección'),
    ('inter_shoulder',    'Distancia inter-hombros (norm.)',     'Encuadre/Distancia'),
    ('bbox_w',            'Ancho bbox de pose (norm.)',          'Encuadre/Distancia'),
    ('bbox_h',            'Alto bbox de pose (norm.)',           'Encuadre/Distancia'),
    ('bbox_area',         'Área bbox de pose (norm.)',           'Encuadre/Distancia'),
    ('shoulder_y',        'Posición vertical hombros (y norm.)', 'Encuadre/Distancia'),
    ('nose_y',            'Posición vertical nariz (y norm.)',   'Encuadre/Distancia'),
    ('pose_visible_frac', 'Completitud de pose (visib.>0.5)',    'Encuadre/Distancia'),
    ('hand_oob',          'Fracción landmarks mano FUERA cuadro','Calidad manos'),
    ('hand_border',       'Fracción landmarks mano EN borde',    'Calidad manos'),
    ('width',             'Ancho del video (px)',                'Metadatos'),
    ('height',            'Alto del video (px)',                 'Metadatos'),
    ('aspect',            'Relación de aspecto (w/h)',           'Metadatos'),
    ('fps',               'FPS reportado (corrupto ~1000)',      'Metadatos'),
    ('frame_count',       'Nº de frames (reportado)',            'Metadatos'),
    ('n_frames_measured', 'Nº de frames (medido, fiable)',       'Metadatos'),
    ('duration_s',        'Duración (s, fps fiable)',            'Metadatos'),
]
METRIC_KEYS = [k for k, _, _ in METRIC_INFO]


# =====================================================================
# Utilidades
# =====================================================================
def list_videos(folder):
    """Devuelve [(label, path)] recorriendo subcarpetas (label = carpeta)."""
    items = []
    if not os.path.isdir(folder):
        return items
    for label in sorted(os.listdir(folder)):
        label_dir = os.path.join(folder, label)
        if not os.path.isdir(label_dir):
            continue
        for nombre in sorted(os.listdir(label_dir)):
            if nombre.lower().endswith(VIDEO_EXTS):
                items.append((label, os.path.join(label_dir, nombre)))
    return items


def read_video(path):
    """Lee metadatos y frames (hasta MAX_READ_FRAMES). Devuelve (meta, frames).

    Saneo defensivo: varios videos del proyecto traen metadata corrupta
    (fps reportado ~1000 y frame_count absurdo/negativo). Esos valores se
    convierten a NaN para no contaminar las medias; el conteo fiable es el
    número de frames realmente leídos (`n_frames_measured`).
    """
    cap = cv2.VideoCapture(path)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps_raw = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    fc_raw = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    while len(frames) < MAX_READ_FRAMES:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    fc = fc_raw if (0 < fc_raw < 1e6) else float('nan')          # frame_count plausible
    fps_ok = fps_raw if (0 < fps_raw <= 120) else float('nan')   # fps plausible (1000 = corrupto)
    meta = {
        'width':             width,
        'height':            height,
        'aspect':            (width / height) if height else float('nan'),
        'fps':               fps_raw,            # se reporta crudo (suele ser corrupto)
        'frame_count':       fc,
        'n_frames_measured': float(len(frames)),
        'duration_s':        (fc / fps_ok) if (not math.isnan(fc) and not math.isnan(fps_ok))
                             else float('nan'),
    }
    return meta, frames


def _hand_oob_border(hand):
    pts = hand.landmark
    n = len(pts)
    oob = sum(1 for p in pts if p.x < 0 or p.x > 1 or p.y < 0 or p.y > 1) / n
    border = sum(1 for p in pts if p.x < 0.02 or p.x > 0.98 or p.y < 0.02 or p.y > 0.98) / n
    return oob, border


def frame_metrics(results):
    """Métricas de un frame a partir de los resultados de MediaPipe Holistic."""
    pose = results.pose_landmarks
    lh = results.left_hand_landmarks
    rh = results.right_hand_landmarks

    m = {
        'pose':       1.0 if pose else 0.0,
        'lh':         1.0 if lh else 0.0,
        'rh':         1.0 if rh else 0.0,
        'any_hand':   1.0 if (lh or rh) else 0.0,
        'both_hands': 1.0 if (lh and rh) else 0.0,
    }

    if pose:
        L = pose.landmark
        ls, rs = L[LSHOULDER], L[RSHOULDER]
        m['inter_shoulder'] = math.hypot(ls.x - rs.x, ls.y - rs.y)
        m['shoulder_y'] = (ls.y + rs.y) / 2.0
        m['nose_y'] = L[NOSE].y
        vis_pts = [p for p in L if p.visibility > 0.5]
        m['pose_visible_frac'] = len(vis_pts) / len(L)
        if vis_pts:
            xs = [p.x for p in vis_pts]
            ys = [p.y for p in vis_pts]
            m['bbox_w'] = max(xs) - min(xs)
            m['bbox_h'] = max(ys) - min(ys)
            m['bbox_area'] = m['bbox_w'] * m['bbox_h']

    oobs, borders = [], []
    for h in (lh, rh):
        if h:
            o, b = _hand_oob_border(h)
            oobs.append(o)
            borders.append(b)
    if oobs:
        m['hand_oob'] = sum(oobs) / len(oobs)
        m['hand_border'] = sum(borders) / len(borders)
    return m


def analyze_video(path, holistic):
    """Devuelve un dict de métricas por video, o None si no se pudo procesar."""
    meta, frames = read_video(path)
    if not frames:
        return None

    idxs = compute_target_indices(len(frames), MAX_FRAMES_PER_VIDEO)
    per_frame = []
    for i in idxs:
        _, results = mediapipe_detection(frames[int(i)], holistic)
        per_frame.append(frame_metrics(results))

    total = len(per_frame)
    out = {}

    # Tasas (denominador = todos los frames muestreados)
    for key in ('pose', 'lh', 'rh', 'any_hand', 'both_hands'):
        out[key + '_rate'] = sum(fm[key] for fm in per_frame) / total

    # Condicionales (promedio sobre frames donde la métrica existe)
    for key in ('inter_shoulder', 'shoulder_y', 'nose_y', 'pose_visible_frac',
                'bbox_w', 'bbox_h', 'bbox_area', 'hand_oob', 'hand_border'):
        vals = [fm[key] for fm in per_frame if key in fm]
        out[key] = (sum(vals) / len(vals)) if vals else float('nan')

    # Metadatos
    for key in ('width', 'height', 'aspect', 'fps', 'frame_count',
                'n_frames_measured', 'duration_s'):
        out[key] = meta[key]

    out['n_frames_read'] = float(len(frames))
    return out


def analyze_dataset(items, holistic, max_videos, desc):
    """Procesa una lista de (label, path) y devuelve lista de dicts de métricas."""
    if max_videos and len(items) > max_videos:
        rng = random.Random(SEED)
        items = rng.sample(items, max_videos)
    rows, failed = [], 0
    for label, path in tqdm(items, desc=desc, unit='video'):
        try:
            m = analyze_video(path, holistic)
        except Exception:  # noqa: BLE001 - resiliente a videos corruptos
            m = None
        if m is None:
            failed += 1
            continue
        m['label'] = label
        m['path'] = path
        rows.append(m)
    return rows, failed


# =====================================================================
# Estadística
# =====================================================================
def _clean(arr):
    a = np.asarray(arr, dtype=float)
    return a[~np.isnan(a)]


def summarize(values):
    a = _clean(values)
    if a.size == 0:
        return dict(n=0, mean=float('nan'), std=float('nan'),
                    median=float('nan'), p10=float('nan'), p90=float('nan'))
    return dict(
        n=int(a.size),
        mean=float(a.mean()),
        std=float(a.std(ddof=1)) if a.size > 1 else 0.0,
        median=float(np.median(a)),
        p10=float(np.percentile(a, 10)),
        p90=float(np.percentile(a, 90)),
    )


def cohens_d(train_vals, eval_vals):
    """Tamaño de efecto (train - eval). |d|>=0.8 ~ grande; >=0.5 medio."""
    a = _clean(train_vals)
    b = _clean(eval_vals)
    if a.size < 2 or b.size < 2:
        return float('nan')
    na, nb = a.size, b.size
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    sp = math.sqrt(sp2)
    if sp == 0:
        return 0.0
    return (a.mean() - b.mean()) / sp


def magnitude_label(d):
    ad = abs(d)
    if math.isnan(ad):
        return 'n/d'
    if ad >= 0.8:
        return 'GRANDE'
    if ad >= 0.5:
        return 'medio'
    if ad >= 0.2:
        return 'pequeño'
    return 'mínimo'


# =====================================================================
# Reporte
# =====================================================================
def write_report(train_rows, eval_rows, n_train_total, n_eval_total,
                 failed_train, failed_eval, duration):
    names = {k: name for k, name, _ in METRIC_INFO}
    groups = {k: grp for k, _, grp in METRIC_INFO}

    # Stats + efecto por métrica
    stats = []
    for key in METRIC_KEYS:
        tr = [r.get(key, float('nan')) for r in train_rows]
        ev = [r.get(key, float('nan')) for r in eval_rows]
        st_tr = summarize(tr)
        st_ev = summarize(ev)
        d = cohens_d(tr, ev)
        rel = float('nan')
        if not math.isnan(st_tr['mean']) and st_tr['mean'] != 0:
            rel = (st_ev['mean'] - st_tr['mean']) / abs(st_tr['mean']) * 100.0
        stats.append({
            'key': key, 'name': names[key], 'group': groups[key],
            'train': st_tr, 'eval': st_ev, 'd': d, 'rel': rel,
        })

    ranked = sorted(
        stats,
        key=lambda s: (abs(s['d']) if not math.isnan(s['d']) else -1.0),
        reverse=True,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as fp:
        fp.write("=" * 78 + "\n")
        fp.write("DIFERENCIAS ENTRE VIDEOS DE ENTRENAMIENTO Y DE EVALUACIÓN\n")
        fp.write("=" * 78 + "\n")
        fp.write(f"Fecha:            {datetime.now().isoformat(timespec='seconds')}\n")
        fp.write(f"Entrenamiento:    {PATH_TRAIN}  (analizados {len(train_rows)} "
                 f"de {n_train_total}; fallidos {failed_train})\n")
        fp.write(f"Evaluación:       {PATH_EVAL}  (analizados {len(eval_rows)} "
                 f"de {n_eval_total}; fallidos {failed_eval})\n")
        fp.write(f"Frames/video:     {MAX_FRAMES_PER_VIDEO} (muestreo uniforme)\n")
        fp.write(f"MediaPipe:        static_image_mode=True, model_complexity=1, "
                 f"min_detection_confidence=0.3\n")
        fp.write(f"Duración total:   {duration:.1f}s ({duration/60:.2f} min)\n")
        fp.write("\nNota: el signo del efecto es (train - eval). d>0 => el valor "
                 "es MAYOR en entrenamiento.\n\n")

        # ---- Hallazgos principales ----
        fp.write("-" * 78 + "\n")
        fp.write("HALLAZGOS PRINCIPALES (ordenados por tamaño de efecto |d|)\n")
        fp.write("-" * 78 + "\n")
        top = [s for s in ranked if not math.isnan(s['d'])][:6]
        for i, s in enumerate(top, 1):
            direction = ('menor' if s['eval']['mean'] < s['train']['mean'] else 'mayor')
            fp.write(
                f"{i}. {s['name']} [{s['group']}]\n"
                f"     train = {s['train']['mean']:.4f} (±{s['train']['std']:.4f})   "
                f"eval = {s['eval']['mean']:.4f} (±{s['eval']['std']:.4f})\n"
                f"     d = {s['d']:+.2f} ({magnitude_label(s['d'])}) | "
                f"eval es {direction} que train | Δrel = {s['rel']:+.1f}%\n"
            )
        fp.write("\n")

        # ---- Conclusión automática ----
        fp.write("-" * 78 + "\n")
        fp.write("CONCLUSIÓN AUTOMÁTICA\n")
        fp.write("-" * 78 + "\n")
        hand_keys = {'lh_rate', 'rh_rate', 'any_hand_rate', 'both_hands_rate'}
        hand_stats = [s for s in stats if s['key'] in hand_keys]
        worst_hand = min(hand_stats, key=lambda s: s['eval']['mean']) if hand_stats else None
        top1 = top[0] if top else None
        if top1 and top1['key'] in hand_keys and top1['eval']['mean'] < top1['train']['mean']:
            fp.write("La mayor diferencia entre datasets es una métrica de DETECCIÓN DE "
                     "MANOS de MediaPipe, más baja en evaluación. Esto confirma que el "
                     "cuello de botella es la dificultad de MediaPipe para detectar los "
                     "puntos de control de las manos en los videos de usuarios nuevos, "
                     "no el modelo de clasificación.\n")
        else:
            fp.write("La mayor diferencia no es directamente la detección de manos; "
                     "revisar la tabla completa para interpretar el patrón.\n")
        if worst_hand:
            fp.write(f"\nDetección de manos más afectada: {worst_hand['name']} -> "
                     f"train {worst_hand['train']['mean']*100:.1f}% vs "
                     f"eval {worst_hand['eval']['mean']*100:.1f}%.\n")

        fps_stat = next((s for s in stats if s['key'] == 'fps'), None)
        if fps_stat:
            fp.write(f"\nNota fps: el FPS reportado es NO fiable "
                     f"(mediana train {fps_stat['train']['median']:.0f} / "
                     f"eval {fps_stat['eval']['median']:.0f}; típicamente 1000). "
                     f"Es una corrupción de metadata COMPARTIDA por ambos datasets, "
                     f"así que no distingue train de eval; la duración basada en "
                     f"metadata no es comparable. Usar 'Nº de frames (medido)'.\n")
        fp.write("\nNota (diagnóstico previo): ajustar parámetros de MediaPipe "
                 "(mayor model_complexity, menor min_detection_confidence, upscaling, "
                 "fallback con Hands) NO mejoró significativamente la detección de manos "
                 "en estos videos, lo que refuerza que el problema es de captura/encuadre.\n\n")

        # ---- Tabla completa por grupos ----
        fp.write("-" * 78 + "\n")
        fp.write("TABLA COMPLETA (media ± sd | mediana | d)\n")
        fp.write("-" * 78 + "\n")
        last_group = None
        for s in stats:
            if s['group'] != last_group:
                fp.write(f"\n[{s['group']}]\n")
                last_group = s['group']
            fp.write(
                f"  {s['name']:<38}\n"
                f"     train: {s['train']['mean']:>10.4f} ± {s['train']['std']:<9.4f} "
                f"med {s['train']['median']:>9.4f}  (n={s['train']['n']})\n"
                f"     eval : {s['eval']['mean']:>10.4f} ± {s['eval']['std']:<9.4f} "
                f"med {s['eval']['median']:>9.4f}  (n={s['eval']['n']})\n"
                f"     d = {s['d']:+.2f} ({magnitude_label(s['d'])})   Δrel = {s['rel']:+.1f}%\n"
            )

    # CSV crudo por video
    fieldnames = ['dataset', 'label', 'path'] + METRIC_KEYS + ['n_frames_read']
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in train_rows:
            writer.writerow({'dataset': 'train', **r})
        for r in eval_rows:
            writer.writerow({'dataset': 'eval', **r})

    return ranked


# =====================================================================
# Main
# =====================================================================
def main():
    t0 = time.time()
    print("=== diffrences_data_video ===")
    print(f"Train: {PATH_TRAIN}   Eval: {PATH_EVAL}")

    train_items = list_videos(PATH_TRAIN)
    eval_items = list_videos(PATH_EVAL)
    print(f"Videos entrenamiento: {len(train_items)}")
    print(f"Videos evaluación:    {len(eval_items)}")
    if not eval_items:
        raise SystemExit(f"No hay videos en {PATH_EVAL!r}.")
    if not train_items:
        raise SystemExit(f"No hay videos en {PATH_TRAIN!r}.")

    with make_holistic() as holistic:
        print("\nAnalizando evaluación...")
        eval_rows, failed_eval = analyze_dataset(eval_items, holistic, 0, 'Eval')
        print("Analizando entrenamiento (muestra)...")
        train_rows, failed_train = analyze_dataset(
            train_items, holistic, MAX_TRAIN_VIDEOS, 'Train')

    duration = time.time() - t0
    ranked = write_report(
        train_rows, eval_rows, len(train_items), len(eval_items),
        failed_train, failed_eval, duration)

    print(f"\nReporte:  {OUTPUT_TXT}")
    print(f"CSV:      {OUTPUT_CSV}")
    print(f"Duración: {duration:.1f}s ({duration/60:.2f} min)")
    print("\nTop 3 diferencias (|d|):")
    for s in [r for r in ranked if not math.isnan(r['d'])][:3]:
        print(f"  - {s['name']}: train={s['train']['mean']:.3f} "
              f"eval={s['eval']['mean']:.3f} d={s['d']:+.2f}")


if __name__ == '__main__':
    main()
