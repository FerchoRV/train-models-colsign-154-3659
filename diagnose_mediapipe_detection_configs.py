"""Diagnostica configuraciones de MediaPipe para videos de evaluación.

El objetivo es medir si parámetros más permisivos recuperan detección de
manos en `dataset_videos_evaluate`, antes de re-extraer todo el HDF5.

Evalúa varias configuraciones:
  - baseline actual: Holistic complexity=1, conf=0.3
  - Holistic heavy: complexity=2, conf=0.3
  - Holistic heavy + conf baja: complexity=2, conf=0.1
  - Holistic heavy + conf baja + upscale 1.5x
  - fallback Hands standalone sobre frames donde Holistic no detecta manos

Genera:
  - results_evaluate/diagnose_mediapipe_detection_configs.txt
  - results_evaluate/diagnose_mediapipe_detection_configs.csv

Ejecutar:
    .\.venv\Scripts\python.exe -u diagnose_mediapipe_detection_configs.py
"""

import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


EVAL_DIR = 'dataset_videos_evaluate'
OUTPUT_TXT = 'results_evaluate/diagnose_mediapipe_detection_configs.txt'
OUTPUT_CSV = 'results_evaluate/diagnose_mediapipe_detection_configs.csv'

VIDEO_EXTS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm')
SEQUENCE_LENGTH = 45


@dataclass(frozen=True)
class MpConfig:
    name: str
    model_complexity: int
    min_detection_confidence: float
    upscale: float = 1.0
    use_hands_fallback: bool = False


CONFIGS = [
    MpConfig('baseline_c1_conf03', 1, 0.3, 1.0, False),
    MpConfig('heavy_c2_conf03', 2, 0.3, 1.0, False),
    MpConfig('heavy_c2_conf01', 2, 0.1, 1.0, False),
    MpConfig('heavy_c2_conf01_up15', 2, 0.1, 1.5, False),
    MpConfig('heavy_c2_conf01_up15_handsfb', 2, 0.1, 1.5, True),
]


def list_videos(base_dir: str) -> List[tuple]:
    rows = []
    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for name in sorted(os.listdir(label_dir)):
            if name.lower().endswith(VIDEO_EXTS):
                rows.append((label, os.path.join(label_dir, name)))
    return rows


def target_indices(n_total: int, sequence_length: int) -> np.ndarray:
    if n_total <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, n_total - 1, sequence_length, dtype=int)


def read_sampled_frames(path: str, sequence_length: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return []
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return []
    return [frames[int(i)] for i in target_indices(len(frames), sequence_length)]


def maybe_upscale(frame_bgr: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    return cv2.resize(
        frame_bgr,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def process_rgb(model, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    out = model.process(rgb)
    rgb.flags.writeable = True
    return out


def detect_with_config(label: str, path: str, cfg: MpConfig) -> Dict[str, float]:
    frames = read_sampled_frames(path, SEQUENCE_LENGTH)
    n = len(frames)
    if n == 0:
        return {
            'label': label,
            'video': os.path.basename(path),
            'config': cfg.name,
            'frames': 0,
            'pose': 0.0,
            'lh': 0.0,
            'rh': 0.0,
            'any_hand': 0.0,
            'both_hands': 0.0,
            'hands_fb_any': 0.0,
            'combined_any_hand': 0.0,
        }

    pose_hits = lh_hits = rh_hits = any_hits = both_hits = 0
    hands_fb_hits = combined_any_hits = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=cfg.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=cfg.min_detection_confidence,
    ) as holistic:
        hands_model = None
        if cfg.use_hands_fallback:
            hands_model = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=cfg.min_detection_confidence,
            )

        try:
            for frame in frames:
                frame_proc = maybe_upscale(frame, cfg.upscale)
                res = process_rgb(holistic, frame_proc)

                has_pose = res.pose_landmarks is not None
                has_lh = res.left_hand_landmarks is not None
                has_rh = res.right_hand_landmarks is not None
                has_any = has_lh or has_rh
                has_both = has_lh and has_rh

                fb_any = False
                if cfg.use_hands_fallback and not has_any and hands_model is not None:
                    hres = process_rgb(hands_model, frame_proc)
                    fb_any = hres.multi_hand_landmarks is not None

                pose_hits += int(has_pose)
                lh_hits += int(has_lh)
                rh_hits += int(has_rh)
                any_hits += int(has_any)
                both_hits += int(has_both)
                hands_fb_hits += int(fb_any)
                combined_any_hits += int(has_any or fb_any)
        finally:
            if hands_model is not None:
                hands_model.close()

    return {
        'label': label,
        'video': os.path.basename(path),
        'config': cfg.name,
        'frames': n,
        'pose': pose_hits / n,
        'lh': lh_hits / n,
        'rh': rh_hits / n,
        'any_hand': any_hits / n,
        'both_hands': both_hits / n,
        'hands_fb_any': hands_fb_hits / n,
        'combined_any_hand': combined_any_hits / n,
    }


def summarize(rows: List[dict]) -> Dict[str, dict]:
    out = {}
    metrics = ['pose', 'lh', 'rh', 'any_hand', 'both_hands', 'hands_fb_any', 'combined_any_hand']
    for cfg in CONFIGS:
        cfg_rows = [r for r in rows if r['config'] == cfg.name]
        out[cfg.name] = {}
        for m in metrics:
            vals = np.array([r[m] for r in cfg_rows], dtype=float)
            out[cfg.name][m] = float(vals.mean()) if len(vals) else 0.0
    return out


def main():
    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    videos = list_videos(EVAL_DIR)
    if not videos:
        raise SystemExit(f"No se encontraron videos en {EVAL_DIR!r}")

    print(f"Videos: {len(videos)} | configs: {len(CONFIGS)}")
    print(f"Salida TXT: {OUTPUT_TXT}")
    print(f"Salida CSV: {OUTPUT_CSV}")

    t0 = time.time()
    rows = []
    for cfg in CONFIGS:
        print(f"\n== {cfg.name} ==")
        for label, path in tqdm(videos, desc=cfg.name, unit='video'):
            rows.append(detect_with_config(label, path, cfg))

    summary = summarize(rows)
    duration = time.time() - t0

    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as fp:
        fieldnames = [
            'label', 'video', 'config', 'frames',
            'pose', 'lh', 'rh', 'any_hand', 'both_hands',
            'hands_fb_any', 'combined_any_hand',
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(OUTPUT_TXT, 'w', encoding='utf-8') as fp:
        fp.write("# Diagnóstico MediaPipe: configuraciones de detección\n")
        fp.write(f"# Fecha:    {datetime.now().isoformat(timespec='seconds')}\n")
        fp.write(f"# Dataset:  {EVAL_DIR}\n")
        fp.write(f"# Videos:   {len(videos)}\n")
        fp.write(f"# Frames/video muestreados: {SEQUENCE_LENGTH}\n")
        fp.write(f"# Duración: {duration:.1f}s ({duration/60:.2f} min)\n\n")

        fp.write("## Promedio por configuración\n")
        fp.write(
            f"{'config':<32} {'pose':>6} {'lh':>6} {'rh':>6} "
            f"{'any':>6} {'both':>6} {'fb_any':>7} {'comb_any':>9}\n"
        )
        fp.write('-' * 87 + '\n')
        for cfg in CONFIGS:
            s = summary[cfg.name]
            fp.write(
                f"{cfg.name:<32} {s['pose']:>6.3f} {s['lh']:>6.3f} {s['rh']:>6.3f} "
                f"{s['any_hand']:>6.3f} {s['both_hands']:>6.3f} "
                f"{s['hands_fb_any']:>7.3f} {s['combined_any_hand']:>9.3f}\n"
            )

        fp.write("\n## Lectura\n")
        fp.write("- `lh`/`rh`: fracción de frames con mano izquierda/derecha detectada por Holistic.\n")
        fp.write("- `any`: al menos una mano por Holistic.\n")
        fp.write("- `both`: ambas manos por Holistic.\n")
        fp.write("- `fb_any`: frames donde Holistic no vio manos, pero Hands standalone sí vio alguna.\n")
        fp.write("- `comb_any`: al menos una mano con Holistic o fallback Hands.\n")
        fp.write("\nSi `heavy/upscale/conf01` sube poco, Holistic no resolverá el dominio.\n")
        fp.write("Si `fb_any` sube bastante, conviene implementar fallback Hands en extracción + inferencia.\n")

    print("\nResumen:")
    print(f"{'config':<32} {'pose':>6} {'lh':>6} {'rh':>6} {'any':>6} {'both':>6} {'fb_any':>7} {'comb_any':>9}")
    print('-' * 87)
    for cfg in CONFIGS:
        s = summary[cfg.name]
        print(
            f"{cfg.name:<32} {s['pose']:>6.3f} {s['lh']:>6.3f} {s['rh']:>6.3f} "
            f"{s['any_hand']:>6.3f} {s['both_hands']:>6.3f} "
            f"{s['hands_fb_any']:>7.3f} {s['combined_any_hand']:>9.3f}"
        )
    print(f"\nListo en {duration:.1f}s ({duration/60:.2f} min)")


if __name__ == '__main__':
    main()
