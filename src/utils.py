import os
import numpy as np
import cv2
import mediapipe as mp


try:
    # cargar modelos de mediapipe
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
except Exception as e:
    print(f"Error al importar MediaPipe: {e}")
    mp_holistic = None
    mp_drawing = None


# tamaños esperados de los vectores de keypoints
KP_SIZE_HANDS = 21 * 3 + 21 * 3            # 126
KP_SIZE_POSE_HANDS = 33 * 4 + 21 * 3 + 21 * 3  # 258

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def extract_keypoints_hands(results):
    
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def extract_keypoints_pose_hands(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh, rh])

def process_video_sign(type_extract, url_video):
    if mp_holistic is None:
        raise RuntimeError("MediaPipe no está disponible. Asegúrate de que la biblioteca esté instalada correctamente.")
    
    cap = cv2.VideoCapture(url_video) # Usa url_video directamente
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video desde la URL: {url_video}")
        return None # O lanzar una excepción específica

    sequence_keypoints = []

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Sale del bucle cuando no hay más frames o hay un error de lectura

            image, results = mediapipe_detection(frame, holistic)
            
            if type_extract == 'hands':
                keypoints = extract_keypoints_hands(results)
            elif type_extract == 'pose_hands':
                keypoints = extract_keypoints_pose_hands(results)
            else:
                # Esto ya se valida antes en el endpoint, pero es una buena práctica aquí también
                raise ValueError("El tipo de extracción enviado no existe debe ser 'hands' o 'pose_hands'.") 
            
            sequence_keypoints.append(keypoints)
    
    cap.release() # Cierra el objeto de captura de video
    cv2.destroyAllWindows() # Cierra las ventanas de OpenCV, si se abrieron

    return sequence_keypoints


def _extract_by_type(results, type_extract):
    if type_extract == 'hands':
        return extract_keypoints_hands(results)
    if type_extract == 'pose_hands':
        return extract_keypoints_pose_hands(results)
    raise ValueError("El tipo de extracción enviado no existe debe ser 'hands' o 'pose_hands'.")


def _zero_keypoints(type_extract):
    if type_extract == 'hands':
        return np.zeros(KP_SIZE_HANDS)
    return np.zeros(KP_SIZE_POSE_HANDS)


def extract_video_keypoints_with_holistic(
    holistic,
    url_video,
    sequence_length=45,
    type_extract='pose_hands',
    trim_threshold_s=3.0,
    trim_tail_s=1.0,
):
    """Extrae keypoints de un video y devuelve un ndarray (sequence_length, F).

    Es la variante "in-memory" de `process_video_to_sequence_with_holistic`:
    en vez de escribir `.npy` al disco, retorna la matriz lista para que el
    proceso que la llamó decida qué hacer con ella (por ejemplo, escribirla
    a un HDF5 desde el proceso principal en un pipeline con multiprocessing).

    Aplica la misma estrategia de muestreo uniforme con `np.linspace` y solo
    procesa con MediaPipe los frames objetivo (no todo el video).

    Recorte temporal opcional: si el video dura más de `trim_threshold_s`
    segundos, se ignora el último `trim_tail_s` del final, para evitar
    frames de "pantalla negra" o quietud al final del clip (típico en
    videos de 4-5 s donde la seña ocurre solo en los primeros 2-3 s).
    Si `trim_threshold_s <= 0` o el FPS no es fiable, no se recorta.

    NOTA: la instancia `holistic` DEBE crearse con `static_image_mode=True`
    porque se saltan frames no consecutivos.

    Returns:
        ndarray de shape (sequence_length, F) dtype float32. Si el video
        no se puede abrir o no tiene frames procesables, retorna un array
        de ceros con la forma esperada.
    """
    F = KP_SIZE_HANDS if type_extract == 'hands' else KP_SIZE_POSE_HANDS

    cap = cv2.VideoCapture(url_video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video desde la URL: {url_video}")
        return np.zeros((sequence_length, F), dtype=np.float32)

    declared_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    # PASADA 1: contar frames REALES recorriendo el video con grab() (que
    # decodifica solo el header de cada paquete, mucho más rápido que read()).
    # Es necesario porque CAP_PROP_FRAME_COUNT no es confiable: algunos
    # contenedores (p. ej. videos VP8/.mp4 grabados en móvil) reportan un
    # frame_count inflado (ej. 2976 frames declarados pero solo 90 reales).
    real_frames = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        real_frames += 1
    cap.release()

    if real_frames == 0:
        print(f"Advertencia: video sin frames procesables: {url_video}")
        return np.zeros((sequence_length, F), dtype=np.float32)

    # Recorte temporal: usamos la duración derivada del conteo real y del
    # FPS declarado (la razón sigue siendo coherente con el playback aunque
    # frame_count mienta). Limitamos el recorte para no destruir videos cortos.
    if (
        trim_threshold_s
        and trim_threshold_s > 0
        and trim_tail_s
        and trim_tail_s > 0
        and declared_fps > 0
    ):
        duration_s = real_frames / declared_fps
        if duration_s > trim_threshold_s:
            tail_frames = int(trim_tail_s * declared_fps)
            tail_frames = min(tail_frames, real_frames // 2)
            effective_frames = max(1, real_frames - tail_frames)
        else:
            effective_frames = real_frames
    else:
        effective_frames = real_frames

    target_indices = compute_target_indices(effective_frames, sequence_length)
    target_set = set(int(i) for i in target_indices.tolist())
    max_needed = max(target_set)

    # PASADA 2: re-abrimos el video y decodificamos solo los frames target.
    cap = cv2.VideoCapture(url_video)
    if not cap.isOpened():
        print(f"Error: No se pudo reabrir el video en pasada 2: {url_video}")
        return np.zeros((sequence_length, F), dtype=np.float32)

    keypoints_by_frame = {}
    current_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_set:
            _, results = mediapipe_detection(frame, holistic)
            keypoints_by_frame[current_idx] = _extract_by_type(results, type_extract)
        if current_idx >= max_needed:
            break
        current_idx += 1
    cap.release()

    if not keypoints_by_frame:
        print(f"Advertencia: no se pudo procesar ningún frame: {url_video}")
        return np.zeros((sequence_length, F), dtype=np.float32)

    zero_kp = _zero_keypoints(type_extract)
    out = np.empty((sequence_length, F), dtype=np.float32)
    for out_i, src_i in enumerate(target_indices):
        out[out_i] = keypoints_by_frame.get(int(src_i), zero_kp)
    return out


def normalize_pose_hands_keypoints(sequence, drop_pose_visibility=True, eps=1e-6):
    """Normaliza keypoints pose+manos para hacerlos invariantes a cámara.

    Entrada esperada: (T, 258)
      - pose: 33 landmarks * (x, y, z, visibility)
      - mano izquierda: 21 landmarks * (x, y, z)
      - mano derecha: 21 landmarks * (x, y, z)

    Normalización:
      - Centro: punto medio entre hombro izquierdo (11) y derecho (12).
      - Escala: distancia entre hombros en XY.
      - Landmarks no detectados (todo cero) se preservan como cero.

    Si `drop_pose_visibility=True`, la salida queda en (T, 225):
      pose 33*3 + manos 21*3 + 21*3.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != KP_SIZE_POSE_HANDS:
        raise ValueError(
            f"normalize_pose_hands_keypoints espera shape (T, {KP_SIZE_POSE_HANDS}), "
            f"recibió {sequence.shape}"
        )

    T = sequence.shape[0]
    pose = sequence[:, :33 * 4].reshape(T, 33, 4)
    lh = sequence[:, 33 * 4:33 * 4 + 21 * 3].reshape(T, 21, 3)
    rh = sequence[:, 33 * 4 + 21 * 3:].reshape(T, 21, 3)

    pose_xyz = pose[:, :, :3].copy()
    pose_vis = pose[:, :, 3:4].copy()
    lh_xyz = lh.copy()
    rh_xyz = rh.copy()

    left_shoulder = pose_xyz[:, 11, :]
    right_shoulder = pose_xyz[:, 12, :]
    shoulder_valid = (
        (np.linalg.norm(left_shoulder, axis=1) > eps)
        & (np.linalg.norm(right_shoulder, axis=1) > eps)
    )

    centers = (left_shoulder + right_shoulder) / 2.0
    scales = np.linalg.norm(left_shoulder[:, :2] - right_shoulder[:, :2], axis=1)

    valid_scales = scales[shoulder_valid & (scales > eps)]
    fallback_scale = float(np.median(valid_scales)) if len(valid_scales) else 1.0
    scales = np.where(scales > eps, scales, fallback_scale).astype(np.float32)

    # Si algún frame no tiene hombros detectados, usamos el centro válido más
    # cercano de la secuencia; si no hay ninguno, dejamos centro en cero.
    if not np.all(shoulder_valid):
        valid_indices = np.flatnonzero(shoulder_valid)
        if len(valid_indices):
            for t in np.flatnonzero(~shoulder_valid):
                nearest = valid_indices[np.argmin(np.abs(valid_indices - t))]
                centers[t] = centers[nearest]
        else:
            centers[:] = 0.0

    centers = centers[:, None, :]
    scales = scales[:, None, None]

    def _normalize_points(points):
        out = points.copy()
        detected = np.linalg.norm(points, axis=2, keepdims=True) > eps
        out = np.where(detected, (out - centers) / scales, 0.0)
        return out

    pose_xyz = _normalize_points(pose_xyz)
    lh_xyz = _normalize_points(lh_xyz)
    rh_xyz = _normalize_points(rh_xyz)

    if drop_pose_visibility:
        return np.concatenate(
            [
                pose_xyz.reshape(T, 33 * 3),
                lh_xyz.reshape(T, 21 * 3),
                rh_xyz.reshape(T, 21 * 3),
            ],
            axis=1,
        ).astype(np.float32)

    pose_norm = np.concatenate([pose_xyz, pose_vis], axis=2).reshape(T, 33 * 4)
    return np.concatenate(
        [
            pose_norm,
            lh_xyz.reshape(T, 21 * 3),
            rh_xyz.reshape(T, 21 * 3),
        ],
        axis=1,
    ).astype(np.float32)


def load_hdf5_dataset(
    hdf5_path,
    sequence_length=None,
    num_features=None,
    sanitize=lambda name: name,
    normalize=False,
    drop_pose_visibility=True,
):
    """Carga un dataset HDF5 jerárquico (grupos por etiqueta) y devuelve
    `(X, y, label_names)` listo para entrenar una LSTM.

    Estructura HDF5 esperada::

        f/
          "A veces"/
              "0"   shape (T, F)
              "1"   shape (T, F)
              ...
          "Abandonar"/
              "0"
              ...

    Args:
        hdf5_path: ruta al archivo `.h5`.
        sequence_length: si se da, se valida que todos los datasets tengan
            ese número de frames.
        num_features: igual, para la dimensión de features antes de aplicar
            transformaciones.
        sanitize: función opcional para normalizar nombres de etiqueta
            (por defecto identidad).
        normalize: si True, normaliza keypoints pose+manos centrando en
            hombros y escalando por distancia entre hombros.
        drop_pose_visibility: si `normalize=True`, elimina la dimensión
            visibility de pose (258 -> 225 features).

    Returns:
        X: ndarray (N, T, F) float32.
        y: ndarray (N,) int64 con el índice de la etiqueta en `label_names`.
        label_names: lista de strings con los nombres de las etiquetas,
            ordenadas alfabéticamente. `label_names[i]` es la etiqueta de
            clase `i`.
    """
    import h5py
    X_list = []
    y_list = []
    label_names = []

    with h5py.File(hdf5_path, 'r') as f:
        label_names = sorted([sanitize(k) for k in f.keys()])
        label_to_idx = {name: i for i, name in enumerate(label_names)}
        for label in label_names:
            grupo = f[label]
            for ds_name in grupo:
                arr = grupo[ds_name][...]
                if sequence_length is not None and arr.shape[0] != sequence_length:
                    raise ValueError(
                        f"Shape inesperado en {label}/{ds_name}: {arr.shape}, "
                        f"se esperaba primera dim = {sequence_length}"
                    )
                if num_features is not None and arr.shape[1] != num_features:
                    raise ValueError(
                        f"Shape inesperado en {label}/{ds_name}: {arr.shape}, "
                        f"se esperaba segunda dim = {num_features}"
                    )
                arr = arr.astype(np.float32, copy=False)
                if normalize:
                    arr = normalize_pose_hands_keypoints(
                        arr,
                        drop_pose_visibility=drop_pose_visibility,
                    )
                X_list.append(arr)
                y_list.append(label_to_idx[label])

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, label_names


def compute_target_indices(total_frames, sequence_length):
    """Devuelve los `sequence_length` índices de frame que se deben procesar.

    Estrategia única para downsampling y upsampling: muestreo uniforme con
    `np.linspace`. Esto evita padding con ceros (que confunde a la LSTM)
    y evita pegar todo el padding al final (que sesga el estado final del
    LSTM hacia "movimiento congelado").

      - Video largo:  `linspace` salta uniformemente -> downsampling.
      - Video corto:  `linspace` repite frames distribuidos uniformemente
                      a lo largo de la secuencia -> "cámara lenta".
    """
    return np.linspace(0, total_frames - 1, sequence_length).astype(int)


def process_video_to_sequence_with_holistic(
    holistic,
    url_video,
    output_dir,
    sequence_length=120,
    type_extract='pose_hands',
    overwrite=False,
):
    """Procesa un video usando una instancia Holistic ya creada.

    Optimización clave: solo procesa con MediaPipe los `sequence_length`
    frames que realmente van a guardarse. Si un video tiene 2960 frames
    y `sequence_length=120`, se calculan 120 índices objetivo con
    `np.linspace` y se procesan únicamente esos, descartando los demás
    durante la lectura.

    Si el video tiene menos frames que `sequence_length`, también se usa
    `np.linspace`: los frames se repiten distribuidos uniformemente a lo
    largo de la secuencia (efecto de "cámara lenta"). NO se hace padding
    con ceros porque eso introduce un salto artificial al origen que la
    LSTM interpreta como movimiento súbito y degrada el aprendizaje.

    NOTA importante: como aquí se saltan frames (no consecutivos), la
    instancia `holistic` que se pase DEBE crearse con
    `static_image_mode=True`. Si se crea con `static_image_mode=False`
    el tracking interno se confunde porque asume movimiento suave entre
    frames consecutivos.

    Solo en caso degenerado (video corrupto, sin frames procesables) se
    rellena con ceros para mantener la forma del dataset.

    Es la versión "low-level" pensada para reutilizar la misma instancia
    de MediaPipe entre muchos videos (por ejemplo en un worker de
    `multiprocessing.Pool`).
    """
    os.makedirs(output_dir, exist_ok=True)

    if not overwrite:
        existentes = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
        if len(existentes) >= sequence_length:
            return True

    cap = cv2.VideoCapture(url_video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video desde la URL: {url_video}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Fallback: si el contenedor no reporta frame_count fiable, leemos
    # todos los frames procesándolos y muestreamos desde la lista resultante.
    if total_frames <= 0:
        sequence_keypoints = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, results = mediapipe_detection(frame, holistic)
            sequence_keypoints.append(_extract_by_type(results, type_extract))
        cap.release()
        total_frames = len(sequence_keypoints)
        if total_frames == 0:
            print(f"Advertencia: video sin frames procesables: {url_video}")
            zero_kp = _zero_keypoints(type_extract)
            for out_i in range(sequence_length):
                np.save(os.path.join(output_dir, str(out_i)), zero_kp)
            return False
        indices = compute_target_indices(total_frames, sequence_length)
        for out_i, src_i in enumerate(indices):
            np.save(os.path.join(output_dir, str(out_i)), sequence_keypoints[int(src_i)])
        return True

    # Caso normal: conocemos el total y procesamos SOLO los frames muestreados.
    target_indices = compute_target_indices(total_frames, sequence_length)
    target_set = set(int(i) for i in target_indices.tolist())
    max_needed = max(target_set)

    keypoints_by_frame = {}
    current_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in target_set:
            _, results = mediapipe_detection(frame, holistic)
            keypoints_by_frame[current_idx] = _extract_by_type(results, type_extract)
        if current_idx >= max_needed:
            break
        current_idx += 1
    cap.release()

    if not keypoints_by_frame:
        print(f"Advertencia: no se pudo procesar ningún frame: {url_video}")
        zero_kp = _zero_keypoints(type_extract)
        for out_i in range(sequence_length):
            np.save(os.path.join(output_dir, str(out_i)), zero_kp)
        return False

    zero_kp = _zero_keypoints(type_extract)
    for out_i, src_i in enumerate(target_indices):
        kp = keypoints_by_frame.get(int(src_i), zero_kp)
        np.save(os.path.join(output_dir, str(out_i)), kp)

    return True


def process_video_to_sequence(
    url_video,
    output_dir,
    sequence_length=120,
    type_extract='pose_hands',
    overwrite=False,
):
    """Procesa un video y guarda sus keypoints frame a frame en `output_dir`.

    Lee el video, extrae keypoints con MediaPipe Holistic y normaliza la
    duración a exactamente `sequence_length` frames usando muestreo
    uniforme con `np.linspace`:
      - si el video tiene más frames -> downsampling uniforme.
      - si tiene menos -> los frames se repiten distribuidos uniformemente
        ("cámara lenta"), sin padding con ceros al final.

    Cada frame de la secuencia se guarda como `0.npy`, `1.npy`, ...,
    `{sequence_length-1}.npy` dentro de `output_dir`, replicando la
    estructura del notebook original.

    Args:
        url_video: ruta al archivo de video.
        output_dir: carpeta destino para los `.npy` (se crea si no existe).
        sequence_length: número fijo de frames a guardar (por defecto 120).
        type_extract: `'hands'` o `'pose_hands'`.
        overwrite: si es False y `output_dir` ya contiene
            `sequence_length` `.npy`, se omite el procesamiento.

    Returns:
        True si se generó/ya existía la secuencia, False si hubo error
        leyendo el video.
    """
    if mp_holistic is None:
        raise RuntimeError("MediaPipe no está disponible. Asegúrate de que la biblioteca esté instalada correctamente.")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        return process_video_to_sequence_with_holistic(
            holistic=holistic,
            url_video=url_video,
            output_dir=output_dir,
            sequence_length=sequence_length,
            type_extract=type_extract,
            overwrite=overwrite,
        )


def process_dataset_videos(
    videos_root='dataset_videos',
    sequences_root='dataset_sequences',
    sequence_length=120,
    type_extract='pose_hands',
    video_extensions=('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm'),
    overwrite=False,
    verbose=True,
):
    """Recorre `videos_root` y genera secuencias `.npy` en `sequences_root`.

    Estructura esperada de entrada::

        videos_root/
            etiqueta_1/
                video_a.mp4
                video_b.m4v
                ...
            etiqueta_2/
                ...

    Estructura generada de salida::

        sequences_root/
            etiqueta_1/
                0/   -> 0.npy, 1.npy, ..., 119.npy
                1/   -> 0.npy, 1.npy, ..., 119.npy
                ...
            etiqueta_2/
                ...

    Args:
        videos_root: carpeta raíz con los videos organizados por etiqueta.
        sequences_root: carpeta raíz donde se guardarán las secuencias.
        sequence_length: cantidad fija de frames por video (default 120).
        type_extract: `'hands'` o `'pose_hands'`.
        video_extensions: extensiones consideradas como video.
        overwrite: si False, omite videos cuya secuencia ya está completa.
        verbose: imprime progreso por etiqueta y video.

    Returns:
        Dict con un resumen por etiqueta: `{etiqueta: {'ok': int, 'fail': int}}`.
    """
    if not os.path.isdir(videos_root):
        raise FileNotFoundError(f"No existe la carpeta de videos: {videos_root}")

    etiquetas = sorted([
        d for d in os.listdir(videos_root)
        if os.path.isdir(os.path.join(videos_root, d))
    ])

    resumen = {}
    for etiqueta in etiquetas:
        carpeta_etiqueta = os.path.join(videos_root, etiqueta)
        videos = sorted([
            v for v in os.listdir(carpeta_etiqueta)
            if v.lower().endswith(tuple(ext.lower() for ext in video_extensions))
        ])

        if verbose:
            print(f"\n[{etiqueta}] {len(videos)} videos a procesar")

        ok, fail = 0, 0
        for seq_idx, video_name in enumerate(videos):
            video_path = os.path.join(carpeta_etiqueta, video_name)
            output_dir = os.path.join(sequences_root, etiqueta, str(seq_idx))
            exito = process_video_to_sequence(
                url_video=video_path,
                output_dir=output_dir,
                sequence_length=sequence_length,
                type_extract=type_extract,
                overwrite=overwrite,
            )
            if exito:
                ok += 1
            else:
                fail += 1
            if verbose:
                estado = "OK" if exito else "FALLO"
                print(f"  [{seq_idx:>3}] {estado}: {video_name}")

        resumen[etiqueta] = {'ok': ok, 'fail': fail}

    return resumen