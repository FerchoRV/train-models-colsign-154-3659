"""Interfaz del intérprete en tiempo real (customtkinter).

Flujo:
  1. Pantalla de selección de estrategia: "ColSign 154" o "ColSign Jerárquico".
  2. Tras elegir, se cargan el/los modelo(s) y se muestra el intérprete:
     - previsualización con zona de encuadre,
     - cuenta regresiva de 3 s antes de grabar,
     - cada SEGMENT_SECONDS arma una secuencia y la envía al worker,
     - el recuadro alterna rojo/amarillo por ventana de seña,
     - solo se registran predicciones con confianza >= MIN_CONFIDENCE,
     - al terminar guarda el video y una fila en el CSV (incluye 'modelo').
"""

import os
import queue
import threading
import time

import cv2
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk

from . import config
from . import session
from .camera import CameraStream, draw_overlay
from .inference import FlatPredictor, HierarchicalPredictor, SingleModelPredictor


class InterpreterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("ColSign · Intérprete en tiempo real")
        self.geometry("1180x700")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Estado ---
        self.predictor = None
        self.camera = None
        self.model_ready = False

        self.translating = False
        self.counting_down = False
        self.signs = []                 # lista de {'label': str, 'prob': float}
        self._sign_index = 0
        self.exp_id = None
        self.video_path = None
        self._seg_job = None
        self._color_job = None
        self._countdown_job = None
        self._countdown_value = None
        self._box_color = config.BOX_COLOR_IDLE

        # Modo deslizante (colapso de repeticiones).
        self._capture_times = {}        # idx -> tiempo de captura (monotonic)
        self._last_label_time = 0.0
        self._stop_time = None          # marca para descartar la cola final

        self._pending = 0
        self._finalize_pending = False

        # Colas hilo <-> GUI (se recrean en cada sesión)
        self._seg_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._worker_running = False
        self._worker = None

        # Generación de sesión: invalida los bucles after() de sesiones
        # anteriores al cambiar de modelo (evita loops duplicados).
        self._gen = 0

        self._imgtk = None
        self.selection_frame = None
        self.video_frame = None
        self.panel = None
        self.labels_box = None

        self._build_selection_screen()

    # ==================================================================
    # Pantalla 1: selección de estrategia
    # ==================================================================
    def _build_selection_screen(self):
        self.selection_frame = ctk.CTkFrame(self, corner_radius=0,
                                             fg_color="transparent")
        self.selection_frame.pack(expand=True, fill="both")

        ctk.CTkLabel(self.selection_frame, text="Intérprete ColSign",
                     font=ctk.CTkFont(size=34, weight="bold")).pack(pady=(120, 6))
        ctk.CTkLabel(self.selection_frame,
                     text="Elige la estrategia de traducción en tiempo real",
                     font=ctk.CTkFont(size=16), text_color="#9aa0a6").pack(pady=(0, 40))

        btns = ctk.CTkFrame(self.selection_frame, fg_color="transparent")
        btns.pack()

        ctk.CTkButton(
            btns, text="ColSign 154", width=230, height=100,
            font=ctk.CTkFont(size=20, weight="bold"),
            command=lambda: self._choose_strategy(FlatPredictor()),
        ).grid(row=0, column=0, padx=18, pady=12)
        ctk.CTkButton(
            btns, text="ColSign Jerárquico", width=230, height=100,
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color="#2e7d32", hover_color="#1b5e20",
            command=lambda: self._choose_strategy(HierarchicalPredictor()),
        ).grid(row=0, column=1, padx=18, pady=12)
        ctk.CTkButton(
            btns, text="ColSign Simétrico", width=230, height=100,
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color="#6a1b9a", hover_color="#4a148c",
            command=lambda: self._choose_strategy(SingleModelPredictor(
                model_name=config.SIMETRICO_MODEL,
                display_name='ColSign Simétrico',
                detail='Sub-modelo bimanual simétrico (jerárquico)',
                strategy='colsign simetrico')),
        ).grid(row=1, column=0, padx=18, pady=12)
        ctk.CTkButton(
            btns, text="ColSign Asimétrico", width=230, height=100,
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color="#ad1457", hover_color="#880e4f",
            command=lambda: self._choose_strategy(SingleModelPredictor(
                model_name=config.ASIMETRICO_MODEL,
                display_name='ColSign Asimétrico',
                detail='Sub-modelo bimanual asimétrico (jerárquico)',
                strategy='colsign asimetrico')),
        ).grid(row=1, column=1, padx=18, pady=12)

        ctk.CTkLabel(
            self.selection_frame,
            text="ColSign 154: modelo único (45 frames · 154 señas)\n"
                 "ColSign Jerárquico: modelo raíz + 4 sub-modelos (v2)\n"
                 "ColSign Simétrico / Asimétrico: sub-modelo individual con su "
                 "lista de señas",
            text_color="#9aa0a6").pack(pady=(30, 0))

    def _choose_strategy(self, predictor):
        self.predictor = predictor
        self.model_ready = False
        if self.selection_frame is not None:
            self.selection_frame.destroy()
            self.selection_frame = None

        # Sesión nueva: colas limpias y nueva generación de bucles.
        self._seg_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._gen += 1
        gen = self._gen

        self.camera = CameraStream()
        self._build_interpreter_screen()

        try:
            self.camera.start()
        except Exception as e:  # noqa: BLE001
            self.status_var.set(f"Error de cámara: {e}")

        self._worker_running = True
        self._worker = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker.start()

        self.bind('<Return>', self._on_enter)
        self.after(100, self._load_model_async)
        self.after(config.DISPLAY_FPS_MS, lambda: self._update_frame(gen))
        self.after(80, lambda: self._poll_results(gen))

    # ==================================================================
    # Pantalla 2: intérprete
    # ==================================================================
    def _build_interpreter_screen(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # ---- Video (izquierda) ----
        self.video_frame = ctk.CTkFrame(self, corner_radius=12)
        self.video_frame.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        self.video_label = tk.Label(self.video_frame, bg="#101010")
        self.video_label.pack(expand=True, fill="both", padx=8, pady=8)

        # ---- Control (derecha) ----
        panel = ctk.CTkFrame(self, corner_radius=12)
        self.panel = panel
        panel.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)

        self.change_btn = ctk.CTkButton(
            panel, text="← Cambiar modelo", width=160, height=28,
            fg_color="transparent", border_width=1, text_color="#9aa0a6",
            command=self._back_to_selection)
        self.change_btn.pack(anchor="w", padx=12, pady=(12, 0))

        ctk.CTkLabel(panel, text=self.predictor.display_name,
                     font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(8, 2))
        ctk.CTkLabel(panel, text=self.predictor.detail,
                     text_color="#9aa0a6").pack(pady=(0, 12))

        btns = ctk.CTkFrame(panel, fg_color="transparent")
        btns.pack(pady=6, fill="x", padx=16)
        btns.grid_columnconfigure((0, 1), weight=1)
        self.start_btn = ctk.CTkButton(btns, text="Iniciar traducción",
                                       command=self.start_translation, state="disabled")
        self.start_btn.grid(row=0, column=0, padx=4, sticky="ew")
        self.stop_btn = ctk.CTkButton(btns, text="Terminar", fg_color="#b3261e",
                                      hover_color="#8c1d18",
                                      command=self.stop_translation, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=4, sticky="ew")

        self.counter_var = tk.StringVar(value="Señas: 0")
        ctk.CTkLabel(panel, textvariable=self.counter_var,
                     font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(14, 2))

        self.last_sign_var = tk.StringVar(value="—")
        ctk.CTkLabel(panel, textvariable=self.last_sign_var,
                     font=ctk.CTkFont(size=16), text_color="#00d2c6").pack(pady=(0, 8))

        show_labels = getattr(self.predictor, 'show_labels', False)

        ctk.CTkLabel(panel, text="Secuencia detectada:",
                     anchor="w").pack(fill="x", padx=16)
        self.signs_text = ctk.CTkTextbox(panel, height=160 if show_labels else 320)
        self.signs_text.pack(expand=True, fill="both", padx=16, pady=(4, 8))
        self.signs_text.configure(state="disabled")

        # Panel informativo con las señas que el modelo puede reconocer.
        self.labels_box = None
        if show_labels:
            self.labels_header_var = tk.StringVar(value="Señas disponibles en el modelo:")
            ctk.CTkLabel(panel, textvariable=self.labels_header_var,
                         anchor="w").pack(fill="x", padx=16)
            self.labels_box = ctk.CTkTextbox(panel, height=170)
            self.labels_box.pack(expand=True, fill="both", padx=16, pady=(4, 8))
            self.labels_box.insert("end", "(cargando modelo...)")
            self.labels_box.configure(state="disabled")

        self.status_var = tk.StringVar(value="Iniciando...")
        ctk.CTkLabel(panel, textvariable=self.status_var,
                     text_color="#9aa0a6", wraplength=380).pack(pady=(0, 14), padx=16)

    # ==================================================================
    # Carga del/los modelo(s) en hilo
    # ==================================================================
    def _load_model_async(self):
        self.status_var.set(f"Cargando {self.predictor.display_name} (TensorFlow)...")
        predictor = self.predictor
        rq = self._result_queue   # captura la cola de ESTA sesión

        def task():
            try:
                predictor.load()
                rq.put(('model_loaded', None))
            except Exception as e:  # noqa: BLE001
                rq.put(('error', f"Error cargando modelo: {e}"))

        threading.Thread(target=task, daemon=True).start()

    # ==================================================================
    # Worker de inferencia
    # ==================================================================
    def _inference_worker(self):
        predictor = self.predictor       # fijo durante toda la sesión
        seg_queue = self._seg_queue
        result_queue = self._result_queue
        cm = predictor.new_holistic()
        holistic = cm.__enter__()
        try:
            while self._worker_running:
                try:
                    item = seg_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                idx, frames = item
                try:
                    r = predictor.predict_segment(frames, holistic)
                    result_queue.put(('sign', idx, r['label'], r['prob']))
                except Exception as e:  # noqa: BLE001
                    result_queue.put(('sign', idx, f"<error:{type(e).__name__}>", 0.0))
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    # ==================================================================
    # Previsualización
    # ==================================================================
    def _update_frame(self, gen):
        if gen != self._gen:
            return   # sesión obsoleta: detener este bucle
        if self.camera is not None:
            frame = self.camera.read_latest()
            if frame is not None:
                disp = draw_overlay(frame, self._box_color,
                                    status_text=self.status_var.get(),
                                    last_sign=self.last_sign_var.get()
                                    if self.translating else '',
                                    recording=self.translating,
                                    countdown=self._countdown_value)
                rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb).resize((config.VIDEO_W, config.VIDEO_H))
                self._imgtk = ImageTk.PhotoImage(img)
                self.video_label.configure(image=self._imgtk)
        self.after(config.DISPLAY_FPS_MS, lambda: self._update_frame(gen))

    # ==================================================================
    # Alternancia de color del recuadro (cada SEGMENT_SECONDS, ambos modos)
    # ==================================================================
    def _color_tick(self):
        if not self.translating:
            return
        self._box_color = (config.BOX_COLOR_B
                           if self._box_color == config.BOX_COLOR_A
                           else config.BOX_COLOR_A)
        self._color_job = self.after(int(config.SEGMENT_SECONDS * 1000),
                                     self._color_tick)

    # ==================================================================
    # Inferencia: ventana FIJA (jerárquico) o DESLIZANTE (ColSign 154)
    # ==================================================================
    def _enqueue_segment(self, frames):
        if len(frames) >= config.MIN_SEGMENT_FRAMES:
            self._sign_index += 1
            self._pending += 1
            if self.predictor.streaming:
                self._capture_times[self._sign_index] = time.monotonic()
            self._seg_queue.put((self._sign_index, frames))

    def _segment_tick(self):
        """Modo no solapado: cada SEGMENT_SECONDS toma y vacía el segmento."""
        if not self.translating:
            return
        self._enqueue_segment(self.camera.pop_segment())
        self._seg_job = self.after(int(config.SEGMENT_SECONDS * 1000),
                                   self._segment_tick)

    def _window_tick(self):
        """Modo deslizante: cada STRIDE analiza los últimos WINDOW_SECONDS."""
        if not self.translating:
            return
        self._enqueue_segment(self.camera.get_window())
        self._seg_job = self.after(int(config.STRIDE_SECONDS * 1000),
                                   self._window_tick)

    # ==================================================================
    # Resultados (hilo -> GUI)
    # ==================================================================
    def _poll_results(self, gen):
        if gen != self._gen:
            return   # sesión obsoleta: detener este bucle
        try:
            while True:
                msg = self._result_queue.get_nowait()
                kind = msg[0]
                if kind == 'model_loaded':
                    self.model_ready = True
                    self.start_btn.configure(state="normal")
                    self._populate_labels()
                    self.status_var.set("Listo. Ubíquese en el recuadro e inicie la traducción.")
                elif kind == 'error':
                    self.status_var.set(msg[1])
                elif kind == 'sign':
                    _, idx, label, prob = msg
                    self._pending = max(0, self._pending - 1)
                    t = self._capture_times.pop(idx, time.monotonic())
                    # Descartar ventanas que caen en los últimos segundos antes
                    # de detener (movimiento para dar Enter / "Terminar").
                    in_tail = (self._stop_time is not None
                               and t >= self._stop_time - config.DISCARD_TAIL_SECONDS)
                    if prob >= config.MIN_CONFIDENCE and not in_tail:
                        if self.predictor.streaming:
                            self._handle_sliding_prediction(label, prob, t)
                        else:
                            self._handle_fixed_prediction(label, prob)
                    self._maybe_finalize()
        except queue.Empty:
            pass
        self.after(80, lambda: self._poll_results(gen))

    def _handle_fixed_prediction(self, label, prob):
        """Ventanas fijas: cada predicción válida es una seña nueva."""
        self.signs.append({'label': label, 'prob': prob})
        self._render_signs()

    def _handle_sliding_prediction(self, label, prob, t):
        """Ventana deslizante: colapsa repeticiones consecutivas de la misma
        seña quedándose con la de mayor confianza. Una pausa >= REPEAT_GAP
        hace que la misma seña vuelva a contar como nueva."""
        last = self.signs[-1] if self.signs else None
        same_run = (last is not None and last['label'] == label
                    and (t - self._last_label_time) < config.REPEAT_GAP_SECONDS)
        if same_run:
            if prob > last['prob']:
                last['prob'] = prob
        else:
            self.signs.append({'label': label, 'prob': prob, 't_start': t})
        self._last_label_time = t
        self._render_signs()

    def _populate_labels(self):
        """Llena el panel informativo con las señas que reconoce el modelo."""
        if self.labels_box is None:
            return
        labels = getattr(self.predictor, 'labels', []) or []
        self.labels_header_var.set(f"Señas disponibles en el modelo ({len(labels)}):")
        self.labels_box.configure(state="normal")
        self.labels_box.delete("1.0", "end")
        self.labels_box.insert("end", "\n".join(f"•  {lbl}" for lbl in labels))
        self.labels_box.configure(state="disabled")

    def _render_signs(self):
        n = len(self.signs)
        self.counter_var.set(f"Señas: {n}")
        if self.signs:
            d = self.signs[-1]
            self.last_sign_var.set(f"Seña {n}: {d['label']}  ({d['prob']*100:.0f}%)")
        else:
            self.last_sign_var.set("—")
        self.signs_text.configure(state="normal")
        self.signs_text.delete("1.0", "end")
        for i, d in enumerate(self.signs, 1):
            self.signs_text.insert("end", f"{i:>2}. {d['label']}   ({d['prob']*100:.0f}%)\n")
        self.signs_text.see("end")
        self.signs_text.configure(state="disabled")

    # ==================================================================
    # Control de traducción
    # ==================================================================
    def start_translation(self):
        if self.translating or self.counting_down or not self.model_ready:
            return
        if self.camera is None or self.camera.frame_size is None:
            self.status_var.set("La cámara aún no está lista.")
            return
        # Reset del experimento.
        self.signs = []
        self._sign_index = 0
        self._pending = 0
        self._finalize_pending = False
        self._capture_times = {}
        self._last_label_time = 0.0
        self._stop_time = None
        self.signs_text.configure(state="normal")
        self.signs_text.delete("1.0", "end")
        self.signs_text.configure(state="disabled")
        self.counter_var.set("Señas: 0")
        self.last_sign_var.set("—")
        self._box_color = config.BOX_COLOR_IDLE

        # Cuenta regresiva antes de empezar a grabar.
        self.counting_down = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")   # permite cancelar la cuenta
        self.change_btn.configure(state="disabled")
        self._tick_countdown(config.COUNTDOWN_SECONDS)

    def _tick_countdown(self, n):
        if not self.counting_down:
            return
        if n > 0:
            self._countdown_value = n
            self.status_var.set(f"Prepárese... la grabación inicia en {n}")
            self._countdown_job = self.after(1000, lambda: self._tick_countdown(n - 1))
        else:
            self._countdown_value = None
            self._countdown_job = None
            self._begin_recording()

    def _begin_recording(self):
        self.counting_down = False
        self.exp_id, self.video_path = session.new_experiment_paths()
        self.camera.start_recording(self.video_path)
        self.translating = True
        self._last_label_time = time.monotonic()
        self._box_color = config.BOX_COLOR_A
        self.status_var.set("Traduciendo... (Enter o 'Terminar' para parar)")
        # Alternancia de color del recuadro (independiente del análisis).
        self._color_job = self.after(int(config.SEGMENT_SECONDS * 1000),
                                     self._color_tick)
        # Análisis: deslizante (ColSign 154) o por ventanas fijas (jerárquico).
        if self.predictor.streaming:
            self._seg_job = self.after(int(config.STRIDE_SECONDS * 1000),
                                       self._window_tick)
        else:
            self._seg_job = self.after(int(config.SEGMENT_SECONDS * 1000),
                                       self._segment_tick)

    def stop_translation(self):
        # Cancelar durante la cuenta regresiva (aún no se graba nada).
        if self.counting_down:
            self.counting_down = False
            if self._countdown_job is not None:
                self.after_cancel(self._countdown_job)
                self._countdown_job = None
            self._countdown_value = None
            self._box_color = config.BOX_COLOR_IDLE
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.change_btn.configure(state="normal")
            self.status_var.set("Cuenta regresiva cancelada.")
            return
        if not self.translating:
            return
        self.translating = False
        self._stop_time = time.monotonic()
        self._box_color = config.BOX_COLOR_IDLE
        for job in (self._seg_job, self._color_job):
            if job is not None:
                self.after_cancel(job)
        self._seg_job = None
        self._color_job = None

        if self.predictor.streaming:
            # No se encola la última ventana: los últimos segundos se descartan
            # (el usuario se mueve para dar Enter / "Terminar").
            self.camera.stop_recording()
        else:
            final = self.camera.stop_recording()
            self._enqueue_segment(final)

        # Quitar las señas que EMPEZARON dentro de la cola final descartada.
        cutoff = self._stop_time - config.DISCARD_TAIL_SECONDS
        while self.signs and self.signs[-1].get('t_start', float('-inf')) >= cutoff:
            self.signs.pop()
        self._render_signs()

        self.stop_btn.configure(state="disabled")
        self.status_var.set("Procesando últimos segmentos...")
        self._finalize_pending = True
        self._maybe_finalize()

    def _maybe_finalize(self):
        """Guarda el CSV cuando ya no quedan segmentos pendientes."""
        if not self._finalize_pending or self._pending > 0:
            return
        self._finalize_pending = False
        labels = [d['label'] for d in self.signs]
        try:
            session.append_experiment_row(
                self.exp_id, self.video_path, labels,
                modelo=self.predictor.strategy)
            csv_msg = f"CSV: {os.path.basename(config.CSV_PATH)}"
        except Exception as e:  # noqa: BLE001
            csv_msg = f"Error guardando CSV: {e}"
        self.status_var.set(
            f"Guardado: {os.path.basename(self.video_path)} · "
            f"{len(self.signs)} señas · {csv_msg}")
        if self.model_ready:
            self.start_btn.configure(state="normal")
        self.change_btn.configure(state="normal")

    # ==================================================================
    # Cambio de modelo / cierre
    # ==================================================================
    def _teardown_session(self):
        """Detiene cámara, worker y bucles de la sesión actual de forma segura.

        Si había una grabación en curso (cambio de modelo a mitad de
        traducción), el video parcial queda en disco SIN fila en el CSV
        (experimento descartado).
        """
        # Invalida los bucles after() de la sesión actual.
        self._gen += 1
        self.translating = False
        self.counting_down = False
        for job in (self._seg_job, self._color_job, self._countdown_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:  # noqa: BLE001
                    pass
        self._seg_job = None
        self._color_job = None
        self._countdown_job = None
        self._countdown_value = None

        # Detener worker.
        self._worker_running = False
        try:
            self._seg_queue.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        if self._worker is not None:
            self._worker.join(timeout=1.5)
            self._worker = None

        # Detener cámara (libera el VideoWriter si seguía grabando).
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:  # noqa: BLE001
                pass
            self.camera = None

        # Reset de estado.
        self.predictor = None
        self.model_ready = False
        self._pending = 0
        self._finalize_pending = False
        self._box_color = config.BOX_COLOR_IDLE

    def _back_to_selection(self):
        self.unbind('<Return>')
        self._teardown_session()

        # Desmontar la pantalla del intérprete y volver a la de selección.
        for frame in (self.video_frame, self.panel):
            if frame is not None:
                frame.destroy()
        self.video_frame = None
        self.panel = None
        self.labels_box = None
        self._imgtk = None
        self._build_selection_screen()

    def _on_enter(self, _event=None):
        if self.translating or self.counting_down:
            self.stop_translation()

    def _on_close(self):
        self._teardown_session()
        self.destroy()


def run():
    app = InterpreterApp()
    app.mainloop()
