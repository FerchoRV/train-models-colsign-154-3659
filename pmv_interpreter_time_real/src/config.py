"""Configuración central del prototipo en tiempo real."""

import os

# --- Rutas ---
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))   # .../pmv_interpreter_time_real/src
PKG_DIR      = os.path.dirname(THIS_DIR)                     # .../pmv_interpreter_time_real
PROJECT_ROOT = os.path.dirname(PKG_DIR)                      # raíz del proyecto

# Carpeta donde se guardan los videos de prueba + CSV de experimentos.
# (el enunciado la nombró "dataset_time_rela"; se usa "dataset_time_real")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset_time_real')
CSV_PATH   = os.path.join(OUTPUT_DIR, 'experiments.csv')

# --- Modelos ---
# Estrategia plana: modelo único de 45 frames, 154 señas (checkpoint _best).
MODEL_NAME      = 'colsign_lstm_norm_45_154'
SEQUENCE_LENGTH = 45

# Estrategia jerárquica (v2): modelo raíz + 4 sub-modelos, igual que
# `piplane_lstm_jerarquico_v2.py`. load_model() resuelve el checkpoint _best.
HIER_ROOT_MODEL = 'colsign_lstm_norm_raiz_45_154_v2'
HIER_SUB_MODELS = {
    'Grupo Estático':                     'colsign_lstm_norm_estatic_45_154_v2',
    'Grupo Dinámico Unimanual':           'colsign_lstm_norm_unimanual_45_154_v2',
    'Grupo Dinámico Bimanual Simétrico':  'colsign_lstm_norm_bi_simetrico_45_154',
    'Grupo Dinámico Bimanual Asimétrico': 'colsign_lstm_norm_bi_asimetrico_45_154',
}

# Sub-modelos individuales (bimanual simétrico / asimétrico) usados como
# estrategias directas de un solo modelo.
SIMETRICO_MODEL  = 'colsign_lstm_norm_bi_simetrico_45_154'
ASIMETRICO_MODEL = 'colsign_lstm_norm_bi_asimetrico_45_154'

# --- Segmentación temporal ---
# Duración de la ventana de análisis. El recuadro alterna de color con esta
# misma cadencia para marcar el inicio de una nueva ventana de seña.
SEGMENT_SECONDS    = 2.0
# Segmentos con menos frames que esto se descartan (ventana demasiado corta).
MIN_SEGMENT_FRAMES = 10
# Umbral de confianza para registrar una seña (predicciones por debajo se ignoran).
MIN_CONFIDENCE     = 0.50
# Cuenta regresiva (s) tras pulsar "Iniciar" antes de empezar a grabar.
COUNTDOWN_SECONDS  = 3

# --- Modo deslizante (solo ColSign 154 por ahora) ---
# Ventana deslizante: cada STRIDE se analizan los últimos WINDOW_SECONDS de
# video (ventanas solapadas) y se colapsan repeticiones consecutivas de la
# misma seña quedándose con la de mayor confianza.
WINDOW_SECONDS     = SEGMENT_SECONDS   # tamaño de la ventana (~45 frames)
STRIDE_SECONDS     = 1.0               # cada cuánto se predice
# Si la misma seña reaparece tras una pausa >= a esto, se cuenta como nueva.
REPEAT_GAP_SECONDS = 3.0
# Se descartan las detecciones de los últimos N segundos antes de detener
# (es cuando el usuario se mueve para dar Enter / "Terminar").
DISCARD_TAIL_SECONDS = 2.0

# --- Cámara ---
CAMERA_INDEX = 0
DEFAULT_FPS  = 20.0   # fallback si la cámara no reporta un fps fiable

# --- Display ---
VIDEO_W = 720
VIDEO_H = 540
DISPLAY_FPS_MS = 33   # ~30 fps de refresco de la previsualización

# Zona de encuadre sugerida (fracciones del frame): recuadro centrado donde
# el usuario debe ubicar cabeza, hombros y ambas manos.
GUIDE_BOX = {'x0': 0.20, 'y0': 0.06, 'x1': 0.80, 'y1': 0.96}

# Colores del recuadro (BGR). El recuadro alterna A<->B cada SEGMENT_SECONDS
# para marcar el inicio de una nueva ventana de seña.
BOX_COLOR_IDLE = (0, 200, 0)     # verde: en espera / cuenta regresiva
BOX_COLOR_A    = (0, 0, 255)     # rojo
BOX_COLOR_B    = (0, 255, 255)   # amarillo
