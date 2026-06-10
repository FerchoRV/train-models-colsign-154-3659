"""Pipeline de inferencia sobre videos de frases/secuencias (grup_sign).

A diferencia de `piplane_lstm_all.py` (que evalúa videos de UNA seña con
etiqueta conocida), aquí los videos de `PATH_GRUP_SING` contienen varias
señas seguidas. La estrategia de procesamiento es:

  1. Recorrer todos los videos de la carpeta `PATH_GRUP_SING`.
  2. Cargar cada video y recorrerlo con una VENTANA DESLIZANTE de
     `WINDOW_SIZE` frames con SOLAPAMIENTO (paso = `STRIDE` frames). El
     tamaño se mide en frames crudos porque el fps de estos videos es poco
     confiable (muchos reportan 1000 fps). La ventana aproxima la duración
     de una seña; `extract_lstm_features` la submuestrea internamente a la
     `sequence_length` del modelo.
  3. Enviar cada ventana al modelo y quedarse con su softmax (top-1).
  4. FILTRO DE CONFIANZA: descartar ventanas con prob < `CONF_THRESHOLD`
     (las transiciones entre señas dan predicciones dispersas y de baja
     confianza).
  5. NMS TEMPORAL: una misma seña aparece detectada en varias ventanas
     solapadas consecutivas. Se fusionan con Non-Maximum Suppression sobre
     el eje temporal (IoU de intervalos), conservando la ventana de mayor
     confianza ("frame central" de la seña).
  6. Construir la secuencia final de señas (ordenada por tiempo) por video.
  7. Guardar el detalle de cada ventana en CSV y el resumen por video en TXT.

Como no hay etiqueta "verdadera" por segmento, no se calcula accuracy:
el TXT es un reporte resumen con la secuencia de señas detectadas.

Ejecutar:
    .\.venv\Scripts\python.exe -u piplane_lstm_all_grup_sign.py
"""

import os
import sys
import csv
import time
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
    read_all_frames,
)


# =====================================================================
# Configuración
# =====================================================================
PATH_GRUP_SING = 'dataset_grup_sign'
MODEL_NAME     = 'colsign_lstm_norm_15_154'   # busca automáticamente el `_best`
OUTPUT_CSV     = 'results_piplanes/resuls_colsign_norm_15_154_grup_sign.csv'
OUTPUT_TXT     = 'results_piplanes/accuracy_colsign_norm_15_154_grup_sign.txt'

VIDEO_EXTS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm')

# --- Ventana deslizante (en FRAMES crudos, no en segundos) ---
# Tamaño de la ventana: debe aproximar la duración de UNA seña. Los videos
# de entrenamiento duran ~2-3 s; a ~30 fps reales son ~60-90 frames. La
# ventana se submuestrea internamente a la `sequence_length` del modelo.
WINDOW_SIZE = 45
# Paso entre ventanas (solapamiento = WINDOW_SIZE - STRIDE frames). Un paso
# pequeño da más resolución temporal a costa de más inferencias.
STRIDE = 10

# --- Filtro de confianza ---
# Se ignoran las ventanas cuyo softmax top-1 sea menor a este umbral.
# Las transiciones entre señas suelen caer por debajo.
CONF_THRESHOLD = 0.65

# --- NMS temporal ---
# Dos ventanas se consideran la "misma" detección si su IoU temporal supera
# este umbral; se conserva la de mayor confianza.
IOU_THRESHOLD = 0.30

# Solo para REPORTAR tiempos aproximados en el CSV/TXT (no afecta el corte).
NOMINAL_FPS = 30.0

# Ventanas con menos frames que esto se descartan (cola muy corta que
# no alcanza a contener una seña completa).
MIN_FRAMES_SEGMENT = 15


# =====================================================================
# Utilidades
# =====================================================================

def listar_videos(carpeta):
    """Lista (ordenada) las rutas de video dentro de `carpeta`."""
    if not os.path.isdir(carpeta):
        raise SystemExit(
            f"No existe la carpeta {carpeta!r}. "
            f"Crea la carpeta y coloca allí los videos a procesar."
        )
    nombres = sorted(
        f for f in os.listdir(carpeta)
        if f.lower().endswith(VIDEO_EXTS)
    )
    return [os.path.join(carpeta, n) for n in nombres]


def ventanas_deslizantes(frames, window_size, stride):
    """Genera ventanas deslizantes solapadas sobre la lista de frames.

    Devuelve una lista de tuplas `(win_idx, start_frame, end_frame, sub_frames)`.
    La última ventana se ancla al final del video para no perder la cola.
    """
    window_size = max(1, int(window_size))
    stride = max(1, int(stride))
    n = len(frames)
    ventanas = []
    starts = list(range(0, max(1, n - window_size + 1), stride))
    # Asegurar que la cola del video quede cubierta por una ventana final.
    last_start = max(0, n - window_size)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    for win_idx, ini in enumerate(starts):
        sub = frames[ini:ini + window_size]
        ventanas.append((win_idx, ini, ini + len(sub), sub))
    return ventanas


def temporal_iou(a, b):
    """IoU entre dos intervalos temporales (en frames).

    `a` y `b` son dicts con claves `start_frame` y `end_frame`.
    """
    inter_ini = max(a['start_frame'], b['start_frame'])
    inter_fin = min(a['end_frame'], b['end_frame'])
    inter = max(0, inter_fin - inter_ini)
    len_a = a['end_frame'] - a['start_frame']
    len_b = b['end_frame'] - b['start_frame']
    union = len_a + len_b - inter
    return inter / union if union > 0 else 0.0


def nms_temporal(detecciones, iou_threshold):
    """Non-Maximum Suppression sobre el eje temporal.

    Recorre las detecciones de mayor a menor confianza; al elegir una,
    suprime todas las que se solapen con ella por encima de `iou_threshold`.
    Así una seña detectada en varias ventanas consecutivas se fusiona en una
    sola (la de mayor confianza).

    Args:
        detecciones: lista de dicts con `start_frame`, `end_frame`, `prob`.
        iou_threshold: umbral de solapamiento temporal.

    Returns:
        Lista de índices (sobre `detecciones`) que sobreviven, en el orden
        original de entrada (no por confianza).
    """
    orden = sorted(range(len(detecciones)),
                   key=lambda i: detecciones[i]['prob'], reverse=True)
    suprimido = [False] * len(detecciones)
    conservados = []
    for i in orden:
        if suprimido[i]:
            continue
        conservados.append(i)
        for j in orden:
            if j == i or suprimido[j]:
                continue
            if temporal_iou(detecciones[i], detecciones[j]) > iou_threshold:
                suprimido[j] = True
    return sorted(conservados)


def colapsar_consecutivos(labels):
    """Colapsa labels iguales consecutivas: [A, A, B, B, B, A] -> [A, B, A]."""
    salida = []
    for lab in labels:
        if not salida or salida[-1] != lab:
            salida.append(lab)
    return salida


# =====================================================================
# Main
# =====================================================================

def main():
    t_start = time.time()

    print(f"=== piplane_lstm_all_grup_sign ===")
    print(f"Fecha:        {datetime.now().isoformat(timespec='seconds')}")
    print(f"Carpeta:      {PATH_GRUP_SING}")
    print(f"Modelo:       {MODEL_NAME}  (se prefiere checkpoint _best)")
    print(f"Ventana:      {WINDOW_SIZE} frames | stride {STRIDE} "
          f"(solapamiento {WINDOW_SIZE - STRIDE})")
    print(f"Confianza:    >= {CONF_THRESHOLD}")
    print(f"NMS IoU:      > {IOU_THRESHOLD}")
    print(f"Output CSV:   {OUTPUT_CSV}")
    print(f"Output TXT:   {OUTPUT_TXT}")
    print()

    # 1) Listar videos de la carpeta
    videos = listar_videos(PATH_GRUP_SING)
    print(f"Videos encontrados: {len(videos)}")
    if not videos:
        raise SystemExit(f"No hay videos ({VIDEO_EXTS}) en {PATH_GRUP_SING!r}.")

    # 2) Cargar el modelo (carga perezosa de TensorFlow)
    print(f"Cargando modelo {MODEL_NAME}...")
    t0 = time.time()
    model, info = load_model(MODEL_NAME)
    print(f"  Modelo cargado en {time.time() - t0:.1f}s")
    print(f"  Arquitectura: {info.architecture}")
    print(f"  Input shape:  {info.input_shape}")
    print(f"  Num classes:  {info.num_classes}")
    print(f"  Keras path:   {info.keras_path}")

    # 3) Procesar cada video reutilizando un único Holistic
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    window_rows   = []   # filas para el CSV (una por ventana evaluada)
    resumen_videos = []  # (video, n_ventanas, n_final, secuencia_final)
    total_ventanas = 0
    total_final    = 0
    fallos_proceso = 0

    print("\nProcesando videos...")
    with make_holistic() as holistic:
        for video_path in tqdm(videos, desc='Videos', unit='video'):
            video_name = os.path.basename(video_path)
            try:
                frames, fps = read_all_frames(video_path)
            except Exception as e:  # noqa: BLE001 - resiliente
                fallos_proceso += 1
                tqdm.write(f"  ERROR abriendo {video_name}: {e!r}")
                resumen_videos.append((video_name, 0, 0, []))
                continue

            if not frames:
                resumen_videos.append((video_name, 0, 0, []))
                continue

            ventanas = ventanas_deslizantes(frames, WINDOW_SIZE, STRIDE)

            # --- Inferencia ventana por ventana ---
            detecciones = []  # candidatas que pasan el filtro de confianza
            for win_idx, start_frame, end_frame, sub_frames in ventanas:
                if len(sub_frames) < MIN_FRAMES_SEGMENT:
                    continue
                try:
                    pred = predict(model, info, sub_frames, holistic=holistic)
                    label_pred = pred['label']
                    prob = float(pred['prob'])
                except Exception as e:  # noqa: BLE001
                    label_pred = f'<error:{type(e).__name__}>'
                    prob = 0.0
                    fallos_proceso += 1

                total_ventanas += 1
                passed = prob >= CONF_THRESHOLD and not label_pred.startswith('<error')
                center_frame = (start_frame + end_frame) // 2

                det = {
                    'win_idx':      win_idx,
                    'start_frame':  start_frame,
                    'end_frame':    end_frame,
                    'center_frame': center_frame,
                    'n_frames':     len(sub_frames),
                    'label':        label_pred,
                    'prob':         prob,
                }
                if passed:
                    detecciones.append(det)

                # Guardamos TODAS las ventanas en el CSV para trazabilidad.
                det['_passed'] = passed
                det['_kept']   = False  # se marca tras el NMS
                det['_video']  = video_name
                det['_path']   = video_path
                window_rows.append(det)

            # --- NMS temporal sobre las detecciones que pasaron el filtro ---
            keep_idx = nms_temporal(detecciones, IOU_THRESHOLD)
            kept = [detecciones[i] for i in keep_idx]
            kept.sort(key=lambda d: d['center_frame'])

            # Marcar en el CSV las ventanas conservadas por el NMS.
            kept_keys = {(d['start_frame'], d['end_frame']) for d in kept}
            for row in window_rows:
                if row['_video'] == video_name and row.get('_passed') \
                        and (row['start_frame'], row['end_frame']) in kept_keys:
                    row['_kept'] = True

            secuencia_final = colapsar_consecutivos([d['label'] for d in kept])
            total_final += len(secuencia_final)

            resumen_videos.append((
                video_name,
                len([r for r in window_rows if r['_video'] == video_name]),
                len(secuencia_final),
                secuencia_final,
            ))

    # 4) Guardar CSV (detalle por ventana, con marcas de filtro y NMS)
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            'video', 'path', 'win_idx', 'start_frame', 'end_frame',
            'center_frame', 'approx_start_s', 'n_frames',
            'label_predicha', 'prob', 'passed_conf', 'kept_nms',
        ])
        writer.writeheader()
        for r in window_rows:
            writer.writerow({
                'video':          r['_video'],
                'path':           r['_path'],
                'win_idx':        r['win_idx'],
                'start_frame':    r['start_frame'],
                'end_frame':      r['end_frame'],
                'center_frame':   r['center_frame'],
                'approx_start_s': f"{r['start_frame'] / NOMINAL_FPS:.2f}",
                'n_frames':       r['n_frames'],
                'label_predicha': r['label'],
                'prob':           f"{r['prob']:.4f}",
                'passed_conf':    int(bool(r['_passed'])),
                'kept_nms':       int(bool(r['_kept'])),
            })
    print(f"\nCSV de resultados: {OUTPUT_CSV}  ({len(window_rows)} ventanas)")

    # 5) Guardar TXT (resumen por video con la secuencia final de señas)
    duration = time.time() - t_start
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as fp:
        fp.write(f"# Inferencia ventana deslizante + confianza + NMS  {MODEL_NAME}\n")
        fp.write(f"# Fecha:      {datetime.now().isoformat(timespec='seconds')}\n")
        fp.write(f"# Carpeta:    {PATH_GRUP_SING}\n")
        fp.write(f"# Modelo:     {info.keras_path}\n")
        fp.write(f"# Ventana:    {WINDOW_SIZE} frames | stride {STRIDE} "
                 f"(solapamiento {WINDOW_SIZE - STRIDE})\n")
        fp.write(f"# Confianza:  >= {CONF_THRESHOLD}\n")
        fp.write(f"# NMS IoU:    > {IOU_THRESHOLD}\n")
        fp.write(f"# Duración:   {duration:.1f}s ({duration/60:.2f} min)\n\n")

        fp.write("## Resumen global\n")
        fp.write(f"Videos procesados:    {len(videos)}\n")
        fp.write(f"Ventanas evaluadas:   {total_ventanas}\n")
        fp.write(f"Señas finales (NMS):  {total_final}\n")
        fp.write(f"Fallos de proceso:    {fallos_proceso}\n\n")

        fp.write("## Secuencia final de señas detectadas por video\n")
        for video_name, n_win, n_final, secuencia in resumen_videos:
            fp.write(f"\n- {video_name}  ({n_win} ventanas -> {n_final} señas)\n")
            fp.write(f"    secuencia: {' -> '.join(secuencia) if secuencia else '(vacía)'}\n")

    print(f"Reporte resumen:   {OUTPUT_TXT}")
    print(f"\nVideos: {len(videos)} | Ventanas: {total_ventanas} | "
          f"Señas finales: {total_final} | Fallos: {fallos_proceso}")
    print(f"Duración total: {duration:.1f}s ({duration/60:.2f} min)")


if __name__ == '__main__':
    main()
