"""Utilidades de inferencia para los modelos ColSign.

Este módulo expone primitivas reutilizables para:

  1. **Descubrir** modelos entrenados en `models/` usando su metadata
     correspondiente en `info_models/{name}_labels.json`.
  2. **Leer un video** desde múltiples fuentes (ruta, `cv2.VideoCapture`
     ya abierto o lista/ndarray de frames BGR pre-cargados).
  3. **Extraer features** según la arquitectura del modelo:
       - LSTM holístico   → (sequence_length, num_features) keypoints
         normalizados pose+manos (igual que el pipeline de entrenamiento).
       - CNN estático     → crop RGB 128×128 de la mano dominante en el
         frame medoide (igual que `extract_static_hand_crops.py`).
  4. **Predecir** una etiqueta o un top-K usando el modelo cargado.

Pensado como building blocks: el documento de pipeline (jerárquico o
plano) que el usuario escribirá importa de aquí y compone las llamadas.

Ejemplo rápido
--------------
.. code-block:: python

    from src.utils_pipeplanes import (
        get_best_models, load_model, predict, predict_topk,
    )

    # 1) Descubrir el inventario de mejores modelos
    best = get_best_models()
    root_name = best['root']['model_name']           # 'colsign_lstm_norm_raiz_45_154_v2'
    static_name = best['by_root']['Grupo Estático']  # 'colsign_static_cnn_45_154'

    # 2) Predecir
    root_model, root_info = load_model(root_name)
    result = predict(root_model, root_info, 'dataset_videos/a/a_001.mp4')
    print(result['label'], result['prob'])

    top3 = predict_topk(root_model, root_info, 'dataset_videos/a/a_001.mp4', k=3)

Convenciones
------------
- Las "fuentes" de video aceptadas en todas las funciones de extracción son:
    * ``str``/``os.PathLike``  → ruta a un archivo de video.
    * ``cv2.VideoCapture``     → capture ya abierto (no se cierra aquí).
    * ``list[np.ndarray]`` o ``np.ndarray`` con shape ``(T, H, W, 3)``
      BGR uint8: frames pre-cargados.
- Para evitar crear/destruir el modelo de MediaPipe en cada llamada,
  todas las funciones permiten pasar un ``holistic`` instanciado por
  el usuario (typ. con el context manager ``make_holistic()``).
"""

from __future__ import annotations

import os
import re
import json
import glob
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2

try:
    import mediapipe as mp
    _MP_HOLISTIC = mp.solutions.holistic
except Exception as _e:  # pragma: no cover - solo en entornos sin mediapipe
    mp = None
    _MP_HOLISTIC = None

# `src/utils.py` ya tiene la lógica de extracción/normalización del LSTM.
# Reutilizarla garantiza que la inferencia matchea exactamente el
# preprocesamiento de entrenamiento.
from .utils import (
    KP_SIZE_HANDS,
    KP_SIZE_POSE_HANDS,
    extract_keypoints_hands,
    extract_keypoints_pose_hands,
    mediapipe_detection,
    normalize_pose_hands_keypoints,
    compute_target_indices,
)


# =====================================================================
# Rutas por defecto del proyecto
# =====================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR_DEFAULT      = os.path.join(PROJECT_ROOT, 'models')
INFO_MODELS_DIR_DEFAULT = os.path.join(PROJECT_ROOT, 'info_models')

# Sufijo del checkpoint con mejores pesos
BEST_SUFFIX = '_best'

# Mapeo de palabras clave (en `model_name` o `extra.grupo_raiz`) hacia
# las 4 categorías raíz del CSV `etiquetas_modelo_raiz.csv`. Permite
# auto-descubrir qué sub-modelo le toca a cada `etiqueta_raiz`.
ROOT_GROUPS = {
    'estatic':       'Grupo Estático',
    'unimanual':     'Grupo Dinámico Unimanual',
    'bi_simetrico':  'Grupo Dinámico Bimanual Simétrico',
    'bi_asimetrico': 'Grupo Dinámico Bimanual Asimétrico',
}


# =====================================================================
# Metadata de modelos
# =====================================================================

# Identificadores de arquitectura usados internamente.
ARCH_LSTM       = 'lstm_holistic'   # secuencia (T, F) keypoints normalizados
ARCH_CNN_STATIC = 'cnn_static_hand' # imagen 128x128 RGB de mano dominante


@dataclass
class ModelInfo:
    """Metadata mínima necesaria para hacer inferencia con un modelo
    sin necesidad de cargar el `.keras` para preguntarle su input_shape.
    """
    name: str                                # ej. 'colsign_lstm_norm_raiz_45_154_v2'
    keras_path: str                          # ruta absoluta al .keras (idealmente _best)
    labels_json_path: str
    architecture: str                        # ARCH_LSTM o ARCH_CNN_STATIC
    num_classes: int
    id_to_name: dict                         # {int_id: str_label}
    name_to_id: dict                         # {str_label: int_id}
    # Específicos LSTM
    sequence_length:      Optional[int] = None
    num_features:         Optional[int] = None
    normalize_keypoints:  bool          = True
    drop_pose_visibility: bool          = True
    # Específicos CNN
    image_size:    Optional[int] = None
    padding_ratio: float         = 0.25
    # Otros
    role:  Optional[str]  = None    # 'root' | 'sub' | 'flat' | None
    group: Optional[str]  = None    # uno de ROOT_GROUPS.values() si role=='sub'
    raw:   dict           = field(default_factory=dict)  # JSON original sin procesar

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if self.architecture == ARCH_LSTM:
            return (self.sequence_length or 0, self.num_features or 0)
        if self.architecture == ARCH_CNN_STATIC:
            s = self.image_size or 0
            return (s, s, 3)
        return ()

    @property
    def labels_sorted_by_id(self) -> List[str]:
        return [self.id_to_name[i] for i in sorted(self.id_to_name)]


def _detect_architecture(payload: dict) -> str:
    """Decide la arquitectura mirando los campos del labels.json."""
    arch_raw = (payload.get('architecture') or '').lower()
    if 'mobilenet' in arch_raw or 'cnn' in arch_raw:
        return ARCH_CNN_STATIC
    if 'image_size' in payload or (
        isinstance(payload.get('input_shape'), list) and len(payload['input_shape']) == 3
    ):
        return ARCH_CNN_STATIC
    if 'sequence_length' in payload and 'num_features' in payload:
        return ARCH_LSTM
    raise ValueError(
        f"No se pudo detectar la arquitectura del modelo "
        f"'{payload.get('model_name')}' a partir del labels.json"
    )


def _detect_role_and_group(name: str, payload: dict) -> Tuple[Optional[str], Optional[str]]:
    """Determina si el modelo es raíz/sub/plano y, si es sub, a qué grupo
    `etiqueta_raiz` corresponde.
    """
    lower = name.lower()
    if 'raiz' in lower:
        return ('root', None)

    # Si el JSON declara explícitamente el grupo (lo hace train_lstm_cluster_labels.py)
    extra = payload.get('extra', {}) or {}
    group_explicit = extra.get('grupo_raiz')
    if group_explicit:
        return ('sub', group_explicit)

    # CNN estático va al Grupo Estático aunque no lo diga explícito
    if _detect_architecture(payload) == ARCH_CNN_STATIC:
        return ('sub', ROOT_GROUPS['estatic'])

    for kw, gname in ROOT_GROUPS.items():
        if kw in lower:
            return ('sub', gname)

    # No matchea ningún grupo y no es raíz → es modelo plano (todas las clases)
    return ('flat', None)


def _build_id_to_name(payload: dict) -> Tuple[dict, dict]:
    """Devuelve `(id_to_name, name_to_id)` con `id` como int."""
    raw_id_to_name = payload.get('id_to_name') or {}
    id_to_name = {int(k): v for k, v in raw_id_to_name.items()}
    name_to_id = payload.get('name_to_id') or {v: k for k, v in id_to_name.items()}
    return id_to_name, name_to_id


def _info_from_payload(
    payload: dict,
    keras_path: str,
    labels_json_path: str,
) -> ModelInfo:
    arch = _detect_architecture(payload)
    id_to_name, name_to_id = _build_id_to_name(payload)
    role, group = _detect_role_and_group(payload.get('model_name', ''), payload)

    info = ModelInfo(
        name             = payload['model_name'],
        keras_path       = keras_path,
        labels_json_path = labels_json_path,
        architecture     = arch,
        num_classes      = int(payload.get('num_classes', len(id_to_name))),
        id_to_name       = id_to_name,
        name_to_id       = name_to_id,
        role             = role,
        group            = group,
        raw              = payload,
    )

    if arch == ARCH_LSTM:
        info.sequence_length      = int(payload.get('sequence_length', 45))
        info.num_features         = int(payload.get('num_features', 225))
        info.normalize_keypoints  = bool(payload.get('normalize_keypoints', True))
        info.drop_pose_visibility = bool(payload.get('drop_pose_visibility', True))
    else:  # CNN
        info.image_size = int(payload.get('image_size', 128))
        preproc = payload.get('preprocessing', {}) or {}
        info.padding_ratio = float(preproc.get('padding_ratio', 0.25))

    return info


# =====================================================================
# Descubrimiento de modelos
# =====================================================================

def _list_label_json_files(info_dir: str) -> List[str]:
    pattern = os.path.join(info_dir, '*_labels.json')
    return sorted(glob.glob(pattern))


def _candidate_keras_paths(model_name: str, models_dir: str) -> List[str]:
    """Devuelve, en orden de preferencia, los `.keras` que podrían
    representar al `model_name`: primero el `_best` (pesos restaurados al
    mejor val_loss), luego el final.
    """
    best  = os.path.join(models_dir, f"{model_name}{BEST_SUFFIX}.keras")
    final = os.path.join(models_dir, f"{model_name}.keras")
    return [best, final]


def list_available_models(
    models_dir: str = MODELS_DIR_DEFAULT,
    info_dir: str   = INFO_MODELS_DIR_DEFAULT,
    prefer_best: bool = True,
) -> List[ModelInfo]:
    """Devuelve todos los modelos cuyo `.keras` está presente en `models_dir`
    y cuyo `_labels.json` está en `info_dir`.

    Si `prefer_best=True` (default), elige el checkpoint con sufijo
    `_best` cuando exista; si no, cae al final de la corrida.
    """
    out: List[ModelInfo] = []
    for json_path in _list_label_json_files(info_dir):
        with open(json_path, 'r', encoding='utf-8') as fp:
            payload = json.load(fp)
        name = payload.get('model_name')
        if not name:
            continue
        candidates = _candidate_keras_paths(name, models_dir)
        if not prefer_best:
            candidates = list(reversed(candidates))
        keras_path = next((c for c in candidates if os.path.exists(c)), None)
        if keras_path is None:
            continue
        out.append(_info_from_payload(payload, keras_path, json_path))
    return out


# Regla de "mejor versión": el sufijo `_v` con número más alto gana.
# Ej. `colsign_lstm_norm_raiz_45_154_v2` > `colsign_lstm_norm_raiz_45_154`.
_VERSION_RE = re.compile(r'(.*?)(?:_v(\d+))?$')

def _version_key(name: str) -> Tuple[str, int]:
    m = _VERSION_RE.match(name)
    base = m.group(1)
    version = int(m.group(2)) if m.group(2) else 1
    return base, version


def _pick_latest_version(models: Sequence[ModelInfo]) -> ModelInfo:
    """De varios modelos que comparten el mismo `base` (sin sufijo _vN),
    devuelve el de mayor versión.
    """
    return max(models, key=lambda m: _version_key(m.name)[1])


def get_best_models(
    models_dir: str = MODELS_DIR_DEFAULT,
    info_dir:   str = INFO_MODELS_DIR_DEFAULT,
) -> dict:
    """Inventario del *mejor candidato por rol* para construir pipelines.

    Estrategia:
      - Para cada grupo (raíz / cada sub-grupo / plano), se filtra la
        familia de modelos correspondientes y se elige el de mayor `_vN`.
      - Para el **Grupo Estático**, si existe un modelo CNN
        (`colsign_static_cnn_*`), se prefiere por encima del LSTM
        equivalente, porque empíricamente da mejor accuracy.
      - El `.keras` resuelto ya es el `_best` cuando existe.

    Returns:
        {
            'root':    ModelInfo o None,
            'flat':    ModelInfo o None,
            'by_root': { 'Grupo Estático': ModelInfo, ... },
        }
    """
    available = list_available_models(models_dir, info_dir, prefer_best=True)

    roots = [m for m in available if m.role == 'root']
    flats = [m for m in available if m.role == 'flat']
    subs  = [m for m in available if m.role == 'sub']

    by_root: dict = {}
    for group_name in ROOT_GROUPS.values():
        candidates = [m for m in subs if m.group == group_name]
        if not candidates:
            by_root[group_name] = None
            continue
        # Para "Grupo Estático" prefiere CNN sobre LSTM si existe
        if group_name == ROOT_GROUPS['estatic']:
            cnns = [c for c in candidates if c.architecture == ARCH_CNN_STATIC]
            if cnns:
                by_root[group_name] = _pick_latest_version(cnns)
                continue
        # Si no, el de mayor versión dentro de la misma familia
        by_root[group_name] = _pick_latest_version(candidates)

    return {
        'root':    _pick_latest_version(roots) if roots else None,
        'flat':    _pick_latest_version(flats) if flats else None,
        'by_root': by_root,
    }


# =====================================================================
# Carga de modelos Keras (con caché en proceso)
# =====================================================================

_MODEL_CACHE: dict = {}  # name -> (keras.Model, ModelInfo)


def _import_keras():
    """Import perezoso para que `utils_pipeplanes` se pueda usar para
    descubrir modelos sin que TensorFlow se cargue."""
    import tensorflow as tf  # noqa: WPS433
    return tf.keras


def load_model(
    name_or_info,
    models_dir: str = MODELS_DIR_DEFAULT,
    info_dir:   str = INFO_MODELS_DIR_DEFAULT,
    use_cache:  bool = True,
):
    """Carga un modelo Keras a partir de su nombre, `ModelInfo` o ruta.

    Args:
        name_or_info: 
            - ``str`` igual al ``model_name`` (ej. ``'colsign_static_cnn_45_154'``).
            - ``str`` con ruta directa al ``.keras``.
            - ``ModelInfo`` ya construido.
        models_dir, info_dir: dónde buscar si se pasó solo el nombre.
        use_cache: cachea instancias en memoria para evitar recargar.

    Returns:
        ``(keras_model, ModelInfo)``
    """
    # Resolver ModelInfo
    if isinstance(name_or_info, ModelInfo):
        info = name_or_info
    elif isinstance(name_or_info, str) and name_or_info.endswith('.keras'):
        # Ruta directa al .keras → derivamos el labels.json a partir del nombre
        keras_path = name_or_info
        base = os.path.basename(keras_path).replace('.keras', '')
        # Quitar sufijo _best si lo tiene para mapear al labels.json
        model_name = base[:-len(BEST_SUFFIX)] if base.endswith(BEST_SUFFIX) else base
        labels_json_path = os.path.join(info_dir, f"{model_name}_labels.json")
        if not os.path.exists(labels_json_path):
            raise FileNotFoundError(f"No existe el labels.json: {labels_json_path}")
        with open(labels_json_path, 'r', encoding='utf-8') as fp:
            payload = json.load(fp)
        info = _info_from_payload(payload, keras_path, labels_json_path)
    else:
        model_name = name_or_info
        labels_json_path = os.path.join(info_dir, f"{model_name}_labels.json")
        if not os.path.exists(labels_json_path):
            raise FileNotFoundError(f"No existe {labels_json_path}")
        with open(labels_json_path, 'r', encoding='utf-8') as fp:
            payload = json.load(fp)
        candidates = _candidate_keras_paths(model_name, models_dir)
        keras_path = next((c for c in candidates if os.path.exists(c)), None)
        if keras_path is None:
            raise FileNotFoundError(
                f"No se encontró el .keras para '{model_name}'. Buscado en: {candidates}"
            )
        info = _info_from_payload(payload, keras_path, labels_json_path)

    cache_key = info.keras_path
    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    keras = _import_keras()
    model = keras.models.load_model(info.keras_path, compile=False)

    if use_cache:
        _MODEL_CACHE[cache_key] = (model, info)
    return model, info


def clear_model_cache() -> None:
    """Vacía la caché de modelos cargados (útil entre experimentos)."""
    _MODEL_CACHE.clear()


# =====================================================================
# Lectura uniforme de video (soporta path / VideoCapture / frames)
# =====================================================================

VideoSource = Union[str, os.PathLike, 'cv2.VideoCapture', np.ndarray, List[np.ndarray]]


def _is_video_capture(obj) -> bool:
    return hasattr(obj, 'read') and hasattr(obj, 'release') \
        and hasattr(obj, 'get') and hasattr(obj, 'isOpened')


def read_all_frames(source: VideoSource) -> Tuple[List[np.ndarray], float]:
    """Devuelve `(frames_bgr_list, declared_fps)` a partir de cualquier
    fuente soportada. Cuando la fuente es un `VideoCapture` ya abierto,
    se consume hasta el final (NO se hace seek); el caller decide si
    libera el capture.

    Para casos donde el video es muy grande, considera procesar por
    frame en streaming en vez de cargarlo todo. Para los videos típicos
    del proyecto (1-5 s) esto es perfectamente aceptable.
    """
    # 1) Frames pre-cargados (ndarray (T,H,W,3))
    if isinstance(source, np.ndarray):
        if source.ndim != 4 or source.shape[-1] != 3:
            raise ValueError(
                f"Si pasas ndarray debe ser (T,H,W,3) BGR uint8, llegó {source.shape}"
            )
        return [source[i] for i in range(source.shape[0])], 0.0

    # 2) Lista de frames
    if isinstance(source, list):
        if not source:
            return [], 0.0
        if not isinstance(source[0], np.ndarray):
            raise ValueError("La lista debe contener ndarray BGR.")
        return list(source), 0.0

    # 3) VideoCapture ya abierto
    if _is_video_capture(source):
        cap = source
        declared_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames, declared_fps

    # 4) Path
    path = os.fspath(source)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f"No se pudo abrir el video: {path}")
    declared_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, declared_fps


def trim_tail_frames(
    frames: Sequence[np.ndarray],
    fps: float,
    trim_threshold_s: float = 3.0,
    trim_tail_s: float = 1.0,
) -> List[np.ndarray]:
    """Replica el recorte temporal que usó el pipeline de entrenamiento:
    si el video dura más de `trim_threshold_s`, se omite el último
    `trim_tail_s`. Si no hay `fps` confiable, se devuelve la lista tal cual.
    """
    if fps <= 0 or trim_threshold_s <= 0 or trim_tail_s <= 0:
        return list(frames)
    duration_s = len(frames) / fps
    if duration_s <= trim_threshold_s:
        return list(frames)
    tail_frames = int(trim_tail_s * fps)
    tail_frames = min(tail_frames, len(frames) // 2)
    return list(frames[: max(1, len(frames) - tail_frames)])


# =====================================================================
# MediaPipe Holistic helpers
# =====================================================================

class HolisticContext:
    """Context manager fino sobre `mp.solutions.holistic.Holistic`.

    Permite reusar la misma instancia entre muchas predicciones sin
    pagar el coste de inicialización en cada llamada.

    IMPORTANTE — defaults idénticos al entrenamiento
    ------------------------------------------------
    Los modelos de este proyecto fueron entrenados con keypoints
    extraídos por `mediapipe_point_extraction.py`, que crea Holistic con:

        static_image_mode=True
        model_complexity=1
        enable_segmentation=False
        min_detection_confidence=0.3

    El motivo es que durante la extracción NO se procesan frames
    consecutivos: se muestrean ``sequence_length`` frames a lo largo del
    video con ``np.linspace``. Si ``static_image_mode=False``, MediaPipe
    asume continuidad temporal entre llamadas y usa tracking +
    ``smooth_landmarks`` para filtrar entre frames "adyacentes" — pero
    como en realidad están separados por varios segundos del video real,
    el tracker se confunde y produce keypoints muy distintos a los que
    se guardaron en el HDF5. Empíricamente: el error L2 promedio entre
    una extracción "video mode" y la del HDF5 fue ~32, vs ~0.0 con
    ``static_image_mode=True``.

    Por esto el default aquí es ``static_image_mode=True``. Puedes
    pasar ``static_image_mode=False`` si vas a procesar TODOS los frames
    consecutivos de un stream en vivo (caso de uso distinto).

    Ejemplo::

        with make_holistic() as holistic:                  # ← entrenamiento-compat
            for path in test_paths:
                result = predict(model, info, path, holistic=holistic)
    """

    def __init__(self, **kwargs):
        if _MP_HOLISTIC is None:
            raise RuntimeError(
                "MediaPipe no está instalado. Instala `mediapipe` para usar este modulo."
            )
        # Defaults alineados con `mediapipe_point_extraction.py`.
        # NO incluimos `smooth_landmarks` ni `min_tracking_confidence` porque
        # ambos solo aplican cuando `static_image_mode=False`.
        self._kwargs = {
            'static_image_mode': True,
            'model_complexity': 1,
            'enable_segmentation': False,
            'min_detection_confidence': 0.3,
            **kwargs,
        }
        self._holistic = None

    def __enter__(self):
        self._holistic = _MP_HOLISTIC.Holistic(**self._kwargs)
        return self._holistic

    def __exit__(self, exc_type, exc, tb):
        if self._holistic is not None:
            self._holistic.close()
            self._holistic = None


def make_holistic(**kwargs) -> HolisticContext:
    """Factory para usar como context manager (ver `HolisticContext`)."""
    return HolisticContext(**kwargs)


# =====================================================================
# Extracción de features: LSTM holístico
# =====================================================================

def _select_indices(n_total: int, sequence_length: int) -> np.ndarray:
    """Mismo muestreo uniforme que el entrenamiento (`compute_target_indices`)."""
    return compute_target_indices(n_total, sequence_length)


def extract_lstm_features(
    source: VideoSource,
    sequence_length: int = 45,
    type_extract: str   = 'pose_hands',
    normalize: bool     = True,
    drop_pose_visibility: bool = True,
    holistic = None,
    trim_threshold_s: float = 3.0,
    trim_tail_s: float = 1.0,
) -> np.ndarray:
    """Procesa un video y devuelve la matriz `(sequence_length, F)` que
    espera un LSTM holístico de este proyecto.

    Igual que el pipeline de entrenamiento (`mediapipe_point_extraction.py`
    + `normalize_pose_hands_keypoints`): muestrea `sequence_length` frames
    uniformemente, ejecuta Holistic, concatena keypoints pose+manos y
    normaliza por hombros si se pide.

    Args:
        source: ver `read_all_frames`.
        sequence_length: T del vector resultante (default 45).
        type_extract: ``'pose_hands'`` (258 features raw) o ``'hands'`` (126).
        normalize: si True, aplica `normalize_pose_hands_keypoints`.
        drop_pose_visibility: si True y normalize, la salida queda en
            (T, 225) en vez de (T, 258).
        holistic: instancia de MediaPipe Holistic reutilizable. Si es
            None, se crea (y libera) una local.

    Returns:
        ndarray float32 de shape (sequence_length, F) lista para
        ``model.predict``. Si el video no produce keypoints, devuelve ceros.
    """
    frames, fps = read_all_frames(source)
    frames = trim_tail_frames(frames, fps, trim_threshold_s, trim_tail_s)

    F_raw = KP_SIZE_HANDS if type_extract == 'hands' else KP_SIZE_POSE_HANDS

    if not frames:
        return _zeros_lstm_output(sequence_length, F_raw, type_extract,
                                  normalize, drop_pose_visibility)

    indices = _select_indices(len(frames), sequence_length)
    indices_set = set(int(i) for i in indices)

    # No podemos garantizar que `holistic` haya sido creado con
    # static_image_mode=True, pero como los frames vienen ordenados
    # incluso si saltamos algunos, suele funcionar bien.
    own_holistic = holistic is None
    if own_holistic:
        if _MP_HOLISTIC is None:
            raise RuntimeError("MediaPipe no está instalado")
        holistic = _MP_HOLISTIC.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.3,
        )
    try:
        kps_by_idx: dict = {}
        for src_i in sorted(indices_set):
            if src_i >= len(frames):
                continue
            _, results = mediapipe_detection(frames[src_i], holistic)
            if type_extract == 'hands':
                kps_by_idx[src_i] = extract_keypoints_hands(results)
            else:
                kps_by_idx[src_i] = extract_keypoints_pose_hands(results)
    finally:
        if own_holistic:
            holistic.close()

    zero_kp = np.zeros(F_raw, dtype=np.float32)
    raw = np.empty((sequence_length, F_raw), dtype=np.float32)
    for out_i, src_i in enumerate(indices):
        raw[out_i] = kps_by_idx.get(int(src_i), zero_kp)

    if normalize and type_extract == 'pose_hands':
        return normalize_pose_hands_keypoints(
            raw, drop_pose_visibility=drop_pose_visibility,
        )
    return raw


def _zeros_lstm_output(
    T: int, F_raw: int, type_extract: str,
    normalize: bool, drop_pose_visibility: bool,
) -> np.ndarray:
    """Output "todo cero" con la forma final esperada."""
    if not (normalize and type_extract == 'pose_hands'):
        return np.zeros((T, F_raw), dtype=np.float32)
    F_out = 225 if drop_pose_visibility else KP_SIZE_POSE_HANDS
    return np.zeros((T, F_out), dtype=np.float32)


# =====================================================================
# Extracción de features: CNN estático (crop de mano dominante)
# =====================================================================

WRIST_IDX = 0  # punto 0 = muñeca (MediaPipe Hands)


def _hand_landmarks_to_xy_array(landmarks) -> np.ndarray:
    """Convierte un `mp.solutions.holistic` hand_landmarks a (21, 2)."""
    return np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=np.float32)


def extract_static_hand_crop(
    source: VideoSource,
    crop_size: int = 128,
    padding_ratio: float = 0.25,
    sequence_length: int = 45,
    holistic = None,
    return_info: bool = False,
) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], dict]]:
    """Replica de `extract_static_hand_crops.py` para inferencia.

    Procesa hasta `sequence_length` frames uniformemente muestreados,
    detecta la mano dominante (la que aparece más veces) y devuelve el
    crop RGB ``crop_size × crop_size`` BGR del frame **medoide** (pose
    más estable) con `padding_ratio` alrededor del bbox de keypoints.

    NO espeja la mano izquierda a la derecha (el CNN actual fue
    entrenado con `random_flip_horizontal` y soporta ambas orientaciones).

    Args:
        source: ver `read_all_frames`.
        crop_size: lado del crop final cuadrado.
        padding_ratio: 25% por defecto, idéntico al entrenamiento.
        sequence_length: cuántos frames muestrear para encontrar el medoide.
        holistic: instancia Holistic reutilizable.
        return_info: si True, devuelve ``(crop, info_dict)``.

    Returns:
        crop ``ndarray (crop_size, crop_size, 3)`` uint8 BGR o ``None``
        si no se detectó mano. Si `return_info=True`, devuelve también un
        dict con ``{'frame_idx', 'hand_side', 'n_left', 'n_right'}``.
    """
    frames, _ = read_all_frames(source)
    if not frames:
        return (None, {}) if return_info else None

    indices = _select_indices(len(frames), sequence_length)
    indices_set = set(int(i) for i in indices)

    own_holistic = holistic is None
    if own_holistic:
        if _MP_HOLISTIC is None:
            raise RuntimeError("MediaPipe no está instalado")
        holistic = _MP_HOLISTIC.Holistic(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.3,
        )

    try:
        frames_imgs: dict      = {}
        frames_kp_left: dict   = {}
        frames_kp_right: dict  = {}
        for src_i in sorted(indices_set):
            if src_i >= len(frames):
                continue
            frame_bgr = frames[src_i]
            _, results = mediapipe_detection(frame_bgr, holistic)
            lh = _hand_landmarks_to_xy_array(results.left_hand_landmarks)  \
                if results.left_hand_landmarks else None
            rh = _hand_landmarks_to_xy_array(results.right_hand_landmarks) \
                if results.right_hand_landmarks else None
            frames_imgs[src_i]     = frame_bgr
            frames_kp_left[src_i]  = lh
            frames_kp_right[src_i] = rh
    finally:
        if own_holistic:
            holistic.close()

    if not frames_imgs:
        return (None, {}) if return_info else None

    n_left  = sum(1 for v in frames_kp_left.values()  if v is not None)
    n_right = sum(1 for v in frames_kp_right.values() if v is not None)
    if n_left == 0 and n_right == 0:
        return (None, {'n_left': 0, 'n_right': 0}) if return_info else None

    use_left = n_left > n_right
    kp_source = frames_kp_left if use_left else frames_kp_right
    valid = [(fi, kp_source[fi]) for fi in sorted(frames_imgs.keys()) if kp_source[fi] is not None]
    if not valid:
        return (None, {}) if return_info else None

    # Frame medoide entre los keypoints normalizados (wrist + scale)
    normed = []
    for _, kp in valid:
        p = kp - kp[WRIST_IDX]
        scale = float(np.linalg.norm(p[1:], axis=1).mean())
        if scale > 1e-6:
            p = p / scale
        normed.append(p)
    normed = np.array(normed)
    centroid = normed.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(normed - centroid, axis=(1, 2))
    medoid_idx = int(np.argmin(dists))

    chosen_fi, chosen_kp = valid[medoid_idx]
    img_bgr = frames_imgs[chosen_fi]
    H, W = img_bgr.shape[:2]

    xs = chosen_kp[:, 0] * W
    ys = chosen_kp[:, 1] * H
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    half = max(x_max - x_min, y_max - y_min) * 0.5 * (1.0 + padding_ratio)

    if half < 10:
        return (None, {}) if return_info else None

    x0 = int(max(0, round(cx - half)))
    y0 = int(max(0, round(cy - half)))
    x1 = int(min(W, round(cx + half)))
    y1 = int(min(H, round(cy + half)))
    if (x1 - x0) < 10 or (y1 - y0) < 10:
        return (None, {}) if return_info else None

    crop = img_bgr[y0:y1, x0:x1]
    crop_resized = cv2.resize(crop, (crop_size, crop_size),
                              interpolation=cv2.INTER_AREA)
    if return_info:
        return crop_resized, {
            'frame_idx': chosen_fi,
            'hand_side': 'left' if use_left else 'right',
            'n_left':    n_left,
            'n_right':   n_right,
            'bbox':      (x0, y0, x1, y1),
        }
    return crop_resized


# =====================================================================
# Predicción
# =====================================================================

def _preprocess_cnn_input(crop_bgr: np.ndarray) -> np.ndarray:
    """Convierte el crop BGR uint8 al tensor (1, H, W, 3) que espera el
    CNN actual: RGB float32. NO aplica `mobilenet_v2.preprocess_input`
    porque el modelo `colsign_static_cnn_45_154` lo incluye internamente
    como capa (ver `train_static_cnn.py`)."""
    if crop_bgr.dtype != np.uint8:
        crop_bgr = crop_bgr.astype(np.uint8)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)[None, ...]


def predict_proba(
    model,
    info: ModelInfo,
    source: VideoSource,
    holistic = None,
    trim_threshold_s: float = 3.0,
    trim_tail_s: float = 1.0,
) -> np.ndarray:
    """Devuelve el vector softmax `(num_classes,)` que el modelo asigna
    a un video. Resuelve automáticamente la arquitectura para aplicar el
    preprocesamiento correcto.

    Levanta `RuntimeError` si la arquitectura del modelo es desconocida.
    """
    if info.architecture == ARCH_LSTM:
        x = extract_lstm_features(
            source,
            sequence_length=info.sequence_length or 45,
            type_extract='pose_hands',
            normalize=info.normalize_keypoints,
            drop_pose_visibility=info.drop_pose_visibility,
            holistic=holistic,
            trim_threshold_s=trim_threshold_s,
            trim_tail_s=trim_tail_s,
        )
        x = x[None, ...]
        proba = model.predict(x, verbose=0)[0]
        return np.asarray(proba, dtype=np.float32)

    if info.architecture == ARCH_CNN_STATIC:
        crop = extract_static_hand_crop(
            source,
            crop_size=info.image_size or 128,
            padding_ratio=info.padding_ratio,
            holistic=holistic,
        )
        if crop is None:
            # Sin mano detectada → distribución uniforme (deja que el
            # caller decida cómo manejar el "no_hand").
            return np.full(info.num_classes, 1.0 / info.num_classes, dtype=np.float32)
        x = _preprocess_cnn_input(crop)
        proba = model.predict(x, verbose=0)[0]
        return np.asarray(proba, dtype=np.float32)

    raise RuntimeError(f"Arquitectura desconocida: {info.architecture}")


def predict(
    model,
    info: ModelInfo,
    source: VideoSource,
    holistic = None,
    **kwargs,
) -> dict:
    """Versión "top-1" amigable. Devuelve un dict::

        {
            'label':  str,    # nombre de la clase con mayor probabilidad
            'id':     int,    # id correspondiente
            'prob':   float,  # probabilidad asignada
            'proba':  np.ndarray,  # vector completo (num_classes,)
        }
    """
    proba = predict_proba(model, info, source, holistic=holistic, **kwargs)
    idx = int(np.argmax(proba))
    return {
        'label': info.id_to_name.get(idx, f"<id={idx}>"),
        'id':    idx,
        'prob':  float(proba[idx]),
        'proba': proba,
    }


def predict_topk(
    model,
    info: ModelInfo,
    source: VideoSource,
    k: int = 5,
    holistic = None,
    **kwargs,
) -> List[dict]:
    """Devuelve una lista de ``{'label', 'id', 'prob'}`` ordenada de
    mayor a menor probabilidad, con ``k`` elementos.
    """
    proba = predict_proba(model, info, source, holistic=holistic, **kwargs)
    k = min(k, len(proba))
    top = np.argsort(-proba)[:k]
    return [
        {
            'label': info.id_to_name.get(int(i), f"<id={int(i)}>"),
            'id':    int(i),
            'prob':  float(proba[int(i)]),
        }
        for i in top
    ]


# =====================================================================
# Conveniencia: cargar TODO el set para pipeline jerárquico
# =====================================================================

def load_hierarchical_set(
    models_dir: str = MODELS_DIR_DEFAULT,
    info_dir:   str = INFO_MODELS_DIR_DEFAULT,
) -> dict:
    """Carga en memoria los modelos necesarios para un pipeline
    jerárquico: el raíz y un modelo por cada `etiqueta_raiz`.

    Returns:
        {
            'root':    (keras_model, ModelInfo),
            'by_root': {
                'Grupo Estático':                      (model, info),
                'Grupo Dinámico Unimanual':           (model, info),
                'Grupo Dinámico Bimanual Simétrico':  (model, info),
                'Grupo Dinámico Bimanual Asimétrico': (model, info),
            },
        }

    Si falta algún sub-modelo, queda como ``None`` en el dict.
    """
    best = get_best_models(models_dir, info_dir)
    if best['root'] is None:
        raise FileNotFoundError("No se encontró ningún modelo raíz disponible.")
    out = {
        'root': load_model(best['root'], models_dir, info_dir),
        'by_root': {},
    }
    for group, info in best['by_root'].items():
        out['by_root'][group] = load_model(info, models_dir, info_dir) if info else None
    return out


def load_flat_model(
    models_dir: str = MODELS_DIR_DEFAULT,
    info_dir:   str = INFO_MODELS_DIR_DEFAULT,
):
    """Carga el modelo "plano" (todas las 154 clases en una sola red)."""
    best = get_best_models(models_dir, info_dir)
    if best['flat'] is None:
        raise FileNotFoundError("No se encontró ningún modelo plano (154 clases).")
    return load_model(best['flat'], models_dir, info_dir)


# =====================================================================
# Resumen para CLI / debug
# =====================================================================

def describe_available_models(
    models_dir: str = MODELS_DIR_DEFAULT,
    info_dir:   str = INFO_MODELS_DIR_DEFAULT,
) -> str:
    """Devuelve un texto humano-legible con el inventario detectado.

    Útil para imprimir desde un script o sanity-check rápido en consola.
    """
    available = list_available_models(models_dir, info_dir)
    best      = get_best_models(models_dir, info_dir)

    lines = []
    lines.append(f"=== utils_pipeplanes: inventario de modelos ===")
    lines.append(f"models_dir: {models_dir}")
    lines.append(f"info_dir:   {info_dir}")
    lines.append(f"Total modelos detectados: {len(available)}")
    lines.append("")
    lines.append("--- Todos los modelos ---")
    for m in available:
        ckpt = '_best' if m.keras_path.endswith(f'{BEST_SUFFIX}.keras') else 'final'
        lines.append(
            f"  [{m.architecture:<16}] {m.name:<48} "
            f"role={m.role or '-':<5} group={m.group or '-':<35} "
            f"K={m.num_classes:<4} ckpt={ckpt}"
        )
    lines.append("")
    lines.append("--- Mejores candidatos por rol ---")
    if best['root']:
        lines.append(f"  root:  {best['root'].name}  (K={best['root'].num_classes})")
    if best['flat']:
        lines.append(f"  flat:  {best['flat'].name}  (K={best['flat'].num_classes})")
    for g, m in best['by_root'].items():
        if m is None:
            lines.append(f"  sub[{g}]: <faltante>")
        else:
            lines.append(
                f"  sub[{g}]: {m.name}  ({m.architecture}, K={m.num_classes})"
            )
    return '\n'.join(lines)


__all__ = [
    # Constantes
    'ARCH_LSTM', 'ARCH_CNN_STATIC', 'ROOT_GROUPS', 'BEST_SUFFIX',
    'MODELS_DIR_DEFAULT', 'INFO_MODELS_DIR_DEFAULT',
    # Tipos
    'ModelInfo', 'VideoSource',
    # Descubrimiento / carga
    'list_available_models', 'get_best_models',
    'load_model', 'clear_model_cache',
    'load_hierarchical_set', 'load_flat_model',
    # Video / MediaPipe
    'read_all_frames', 'trim_tail_frames',
    'make_holistic', 'HolisticContext',
    # Features
    'extract_lstm_features', 'extract_static_hand_crop',
    # Predicción
    'predict_proba', 'predict', 'predict_topk',
    # Debug
    'describe_available_models',
]
