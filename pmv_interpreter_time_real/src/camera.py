"""Captura de cámara en un hilo aparte + grabación + overlay de encuadre.

El hilo de captura mantiene el último frame disponible para la GUI y, cuando
la grabación está activa, escribe cada frame CRUDO al video y lo acumula en el
segmento actual (la ventana de SEGMENT_SECONDS). El overlay de guía/HUD se
dibuja solo sobre la copia de previsualización, nunca sobre el video guardado.
"""

import threading
import time
from collections import deque

import cv2

from . import config


class CameraStream:
    """Lee frames de la webcam en un hilo y expone el último frame.

    Además gestiona la grabación a disco y la acumulación de frames del
    segmento temporal en curso, de forma thread-safe.
    """

    def __init__(self, index=config.CAMERA_INDEX):
        self.index = index
        self.cap = None
        self.fps = config.DEFAULT_FPS
        self.frame_size = None

        self._latest = None
        self._lock = threading.Lock()

        self._running = False
        self._thread = None

        # Grabación / segmento
        self._rec_lock = threading.Lock()
        self._recording = False
        self._writer = None
        self._segment = []          # ventanas fijas (modo no solapado)
        self._window = None         # buffer rodante (modo deslizante)

    # ------------------------------------------------------------------
    def start(self):
        # En Windows, CAP_DSHOW abre la cámara más rápido y estable.
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"No se pudo abrir la cámara (index={self.index}).")
        self.cap = cap

        reported = self.cap.get(cv2.CAP_PROP_FPS)
        if reported and 5 <= reported <= 120:
            self.fps = float(reported)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.frame_size = (w, h)

        # Buffer rodante dimensionado a WINDOW_SECONDS según el fps real.
        win_max = max(1, int(round(config.WINDOW_SECONDS * self.fps)))
        self._window = deque(maxlen=win_max)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame
            with self._rec_lock:
                if self._recording:
                    if self._writer is not None:
                        self._writer.write(frame)
                    self._segment.append(frame)
                    if self._window is not None:
                        self._window.append(frame)

    # ------------------------------------------------------------------
    def read_latest(self):
        """Devuelve una copia del último frame BGR, o None si aún no hay."""
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def start_recording(self, video_path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        with self._rec_lock:
            self._writer = cv2.VideoWriter(
                video_path, fourcc, self.fps, self.frame_size)
            self._segment = []
            if self._window is not None:
                self._window.clear()
            self._recording = True

    def pop_segment(self):
        """Devuelve los frames acumulados desde el último pop y reinicia.

        Usado por el modo de ventanas fijas (no solapadas).
        """
        with self._rec_lock:
            seg = self._segment
            self._segment = []
            return seg

    def get_window(self):
        """Devuelve una copia de los últimos WINDOW_SECONDS de frames.

        Usado por el modo deslizante: NO vacía el buffer (las ventanas se
        solapan entre llamadas consecutivas).
        """
        with self._rec_lock:
            return [] if self._window is None else list(self._window)

    def stop_recording(self):
        """Detiene la grabación, cierra el writer y devuelve el último segmento."""
        with self._rec_lock:
            self._recording = False
            seg = self._segment
            self._segment = []
            if self._window is not None:
                self._window.clear()
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            return seg

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._rec_lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def draw_overlay(frame, box_color, status_text='', last_sign='',
                 recording=False, countdown=None):
    """Dibuja la zona de encuadre + HUD sobre una copia de previsualización.

    `box_color` es el color (BGR) del recuadro. `countdown`, si no es None,
    dibuja un número grande centrado (cuenta regresiva antes de grabar).
    No modifica el frame original que se graba a disco.
    """
    h, w = frame.shape[:2]
    g = config.GUIDE_BOX
    x0, y0 = int(g['x0'] * w), int(g['y0'] * h)
    x1, y1 = int(g['x1'] * w), int(g['y1'] * h)

    cv2.rectangle(frame, (x0, y0), (x1, y1), box_color, 3)
    cv2.putText(frame, "Ubiquese dentro del recuadro (cabeza, hombros y manos)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

    if recording:
        cv2.circle(frame, (w - 24, 24), 9, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if countdown is not None:
        text = str(countdown)
        scale, thick = 5.0, 8
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cx, cy = (w - tw) // 2, (h + th) // 2
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), thick)

    if status_text:
        cv2.putText(frame, status_text, (10, h - 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if last_sign:
        cv2.putText(frame, last_sign, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame
