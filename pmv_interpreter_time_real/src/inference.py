"""Predictores de señas para el prototipo en tiempo real.

Dos estrategias, ambas con la MISMA interfaz para que la app las use de
forma intercambiable:

  - ``FlatPredictor``         → modelo único 45/154 (`colsign_lstm_norm_45_154`).
  - ``HierarchicalPredictor`` → modelo raíz + 4 sub-modelos, replicando la
    lógica de `piplane_lstm_jerarquico_v2.py`: se extraen los keypoints UNA
    sola vez por segmento, el raíz decide el grupo y el sub-modelo
    correspondiente produce la etiqueta final.

Interfaz común:
    .strategy        -> str para el CSV ('colsign 154' | 'colsign jerarquico')
    .display_name    -> str para la UI
    .detail          -> str descriptivo para la UI
    .new_holistic()  -> context manager de MediaPipe (defaults = entrenamiento)
    .load()          -> carga el/los modelo(s) (dispara TensorFlow)
    .predict_segment(frames, holistic) -> {'label', 'prob', 'group'}

Se reutiliza `src.utils_pipeplanes` para que la extracción + normalización
sea idéntica a la del entrenamiento/pipelines.
"""

import sys

from . import config

if config.PROJECT_ROOT not in sys.path:
    sys.path.insert(0, config.PROJECT_ROOT)

import numpy as np
from src.utils_pipeplanes import load_model, make_holistic, extract_lstm_features


def _extract(frames, holistic, info):
    """Extrae los keypoints normalizados (sequence_length, num_features)."""
    return extract_lstm_features(
        frames,
        sequence_length=info.sequence_length or config.SEQUENCE_LENGTH,
        type_extract='pose_hands',
        normalize=info.normalize_keypoints,
        drop_pose_visibility=info.drop_pose_visibility,
        holistic=holistic,
    )


def _predict_from_features(model, info, x):
    """Top-1 desde features ya extraídas. Devuelve (label, prob)."""
    proba = model.predict(x[None, ...], verbose=0)[0]
    idx = int(np.argmax(proba))
    return info.id_to_name.get(idx, f"<id={idx}>"), float(proba[idx])


class FlatPredictor:
    strategy = 'colsign 154'
    display_name = 'ColSign 154'
    detail = 'Modelo único · 45 frames · 154 señas'
    streaming = True   # ventana deslizante + colapso de repeticiones

    def __init__(self, model_name=config.MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.info = None

    @staticmethod
    def new_holistic():
        return make_holistic()

    def load(self):
        self.model, self.info = load_model(self.model_name)
        return self.info

    def predict_segment(self, frames, holistic):
        x = _extract(frames, holistic, self.info)
        label, prob = _predict_from_features(self.model, self.info, x)
        return {'label': label, 'prob': prob, 'group': None}


class SingleModelPredictor:
    """Predictor de un único sub-modelo LSTM (p. ej. bimanual simétrico).

    Usa ventanas fijas (no deslizante) y expone la lista de señas del modelo
    para mostrarla de forma informativa en la interfaz.
    """
    streaming = True   # ventana deslizante + colapso de repeticiones
    show_labels = True

    def __init__(self, model_name, display_name, detail, strategy):
        self.model_name = model_name
        self.display_name = display_name
        self.detail = detail
        self.strategy = strategy
        self.model = None
        self.info = None
        self.labels = []

    @staticmethod
    def new_holistic():
        return make_holistic()

    def load(self):
        self.model, self.info = load_model(self.model_name)
        self.labels = [self.info.id_to_name[i]
                       for i in sorted(self.info.id_to_name)]
        return self.info

    def predict_segment(self, frames, holistic):
        x = _extract(frames, holistic, self.info)
        label, prob = _predict_from_features(self.model, self.info, x)
        return {'label': label, 'prob': prob, 'group': None}


class HierarchicalPredictor:
    strategy = 'colsign jerarquico'
    display_name = 'ColSign Jerárquico'
    detail = 'Raíz + 4 sub-modelos (v2)'
    streaming = True   # ventana deslizante + colapso de repeticiones

    ROOT_MODEL_NAME = config.HIER_ROOT_MODEL
    SUB_MODEL_NAMES = config.HIER_SUB_MODELS

    def __init__(self):
        self.root_model = None
        self.root_info = None
        self.sub_assets = {}   # grupo -> (model, info)

    @staticmethod
    def new_holistic():
        return make_holistic()

    def load(self):
        self.root_model, self.root_info = load_model(self.ROOT_MODEL_NAME)
        self.sub_assets = {}
        for group, name in self.SUB_MODEL_NAMES.items():
            self.sub_assets[group] = load_model(name)
        return self.root_info

    def predict_segment(self, frames, holistic):
        # Una sola extracción reutilizada por raíz y sub-modelo (todos los
        # LSTM comparten input shape).
        x = _extract(frames, holistic, self.root_info)
        pred_root, _ = _predict_from_features(self.root_model, self.root_info, x)
        if pred_root in self.sub_assets:
            sub_model, sub_info = self.sub_assets[pred_root]
            label, prob = _predict_from_features(sub_model, sub_info, x)
        else:
            label, prob = f"<sin sub-modelo:{pred_root}>", 0.0
        return {'label': label, 'prob': prob, 'group': pred_root}
