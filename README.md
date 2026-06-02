# Entrenamiento de modelos de reconocimiento de Lengua de Señas Colombiana (LSC)

Repositorio de entrenamiento de los modelos utilizados por **Colsign**, un
prototipo de intérprete bidireccional de Lengua de Señas Colombiana.
Aquí se procesa el dataset de videos `Colsign-LSC-154`, se extraen
keypoints con MediaPipe Holistic y se entrenan dos familias de modelos:

- **Clasificador plano LSTM** sobre keypoints, capaz de reconocer las 154 señas en un único modelo.
- **Clasificador jerárquico** (LSTM raíz + 4 sub‑modelos) que separa las señas por *grupo de articulación* y aprovecha una **CNN especializada** para el alfabeto estático.

Aplicación final: https://www.colsign.com.co/

---

## 1. Descripción del dataset

### 1.1 Dataset de videos crudos (`dataset_videos/`)

El dataset fuente es **Colsign‑LSC‑154** (154 etiquetas de la LSC).
Está publicado en Zenodo: `<10.5281/zenodo.20501599>`, https://zenodo.org/records/20501599.

| Característica | Valor |
|---|---|
| Total de etiquetas | **154** |
| Videos por etiqueta | 17 – 30 |
| Total aproximado de videos | ~3.900 |
| Resolución típica | 480p – 720p (variable) |
| Codec | VP8 / H.264 según fuente |
| Duración | 2 – 4 s |
| FPS típico | 25 – 30 |

**Estructura en disco:**

```
dataset_videos/
├── a/                         (letra estática)
│   ├── a_<uuid>.mp4
│   └── ...
├── A veces/                   (sena dinámica bimanual asimétrica)
│   └── A veces_<uuid>.mp4
├── Abandonar/
└── ... (154 carpetas en total)
```

Cada carpeta es **una etiqueta** (la letra/seña) y contiene entre 17 y 30
videos de la misma seña ejecutada por personas distintas.

### 1.2 Mapa jerárquico de etiquetas (`etiquetas_modelo_raiz.csv`)

Las 154 etiquetas están agrupadas en **4 grupos por articulación**, con
los que se entrena el pipeline jerárquico:

| `etiqueta_raiz` | # clases | Ejemplos |
|---|---|---|
| Grupo Estático | ~27 | letras `a`, `b`, `c`, `ñ`, ... (manos quietas) |
| Grupo Dinámico Unimanual | ~50 | `Hola`, `Yo`, `Decir`, `Ayer`, ... |
| Grupo Dinámico Bimanual Simétrico | ~30 | `Abandonar`, `Amigo`, `Cuánto`, ... |
| Grupo Dinámico Bimanual Asimétrico | ~47 | `A veces`, `Ayudar`, `Gracias`, ... |

Formato: CSV con separador `;` y columnas
`nombre;etiqueta_raiz;videos;Descargado`.

### 1.3 Dataset procesado de keypoints (`dataset_colsign_45_154.h5`)

Generado por [`mediapipe_point_extraction.py`](mediapipe_point_extraction.py).
Es el dataset principal usado por **todos los modelos LSTM**.

| Atributo | Valor |
|---|---|
| Formato | HDF5 + gzip nivel 4 |
| Backbone | MediaPipe Holistic (`model_complexity=1`, `static_image_mode=True`, `min_detection_confidence=0.3`) |
| Frames por video | **45** (muestreados uniformemente con `np.linspace`) |
| Recorte temporal | si `duración > 3s` → se descarta el último 1s |
| Tipo de extracción | `pose_hands` → **258 features raw** (33×4 pose + 21×3 mano izq + 21×3 mano der) |
| Normalización al cargar | centra en hombros + escala por distancia inter‑hombros + descarta visibility → **225 features finales** |
| dtype | float32 |
| Multiproceso | 8 workers (Pool) |
| Reanudable | sí (skip de videos ya escritos) |

**Estructura interna del HDF5:**

```
dataset_colsign_45_154.h5
├── attrs: sequence_length=45, num_features=258, type_extract='pose_hands',
│          trim_threshold_s=3.0, trim_tail_s=1.0
├── "a"/
│     ├── "0"   shape (45, 258), float32, gzip   ← un video
│     ├── "1"
│     └── ...
├── "A veces"/
└── ...
```

> El archivo `.h5` está en `.gitignore` por su tamaño (~varios GB).
> Genera el tuyo a partir de los videos con `python mediapipe_point_extraction.py`.

### 1.4 Dataset de recortes de mano para el modelo estático

El grupo estático (alfabeto) inicialmente alcanzaba sólo ~60% de accuracy
con el LSTM. Para resolverlo se entrena una **CNN sobre recortes RGB de la
mano dominante** en lugar de keypoints. Los recortes se generan así:

- `extract_static_hand_crops.py` → crops 224×224 del frame medoide de cada video, centrados en la muñeca, con la mano izquierda espejada para canonicalizar orientación.
- `expand_static_dataset.py` → toma un *golden set* manualmente curado (149 imágenes limpias) y expande automáticamente a ~4.400 imágenes buscando frames similares en los videos originales.

| Carpeta | Origen | Tamaño aprox. |
|---|---|---|
| `dataset_static_crops/` | crops automáticos (sin curar) | ~750 imágenes |
| `dataset_static_crops_expanded/` | golden set + matching de frames | ~4.400 imágenes (usado por la CNN) |

---

## 2. Modelos entrenados

Todos los modelos finales viven en `models/` (`.keras`) y sus
métricas/labels en `info_models/`.

### 2.1 Clasificador plano (1 modelo, 154 clases)

| Modelo | Arquitectura | Val. accuracy | Test pipeline |
|---|---|---|---|
| `colsign_lstm_norm_45_154` | LSTM(128)→LSTM(64)→Dense | **85.66%** (val) | **96.33%** sobre 1.172 videos de test |

Entrenado con [`train_lstm.py`](train_lstm.py).

### 2.2 Pipeline jerárquico (5 modelos)

Primero el **modelo raíz** decide a qué grupo pertenece la seña, luego
el video se enruta al **sub‑modelo** correspondiente.

```
                                 ┌──► colsign_lstm_norm_estatic_45_154     (27 clases, alfabeto)
                                 ├──► colsign_lstm_norm_unimanual_45_154   (~50 clases)
video → keypoints → modelo_raiz ─┤
                                 ├──► colsign_lstm_norm_bi_simetrico_45_154  (~30 clases)
                                 └──► colsign_lstm_norm_bi_asimetrico_45_154 (~47 clases)
```

| Modelo | Val. accuracy |
|---|---|
| `colsign_lstm_norm_raiz_45_154` (4 grupos) | **94.40%** |
| `colsign_lstm_norm_estatic_45_154` | 59.60% (descartado en favor de la CNN) |
| `colsign_lstm_norm_unimanual_45_154` | 88.21% |
| `colsign_lstm_norm_bi_simetrico_45_154` | 94.96% |
| `colsign_lstm_norm_bi_asimetrico_45_154` | 92.27% |
| **Pipeline jerárquico v1 (extremo a extremo, sobre 1.172 videos)** | **89.68%** |

Entrenado con [`train_lstm_cluster_labels.py`](train_lstm_cluster_labels.py).

### 2.3 CNN especializada para el grupo estático

Sustituye al LSTM del grupo estático en la versión `_v2` del pipeline.

| Modelo | Arquitectura | Val. accuracy |
|---|---|---|
| `colsign_static_cnn_45_154` | MobileNetV2 + head, fine‑tuning 2 fases, 224×224 RGB | **95.26%** (val) |

Entrenado con [`train_static_cnn.py`](train_static_cnn.py).

### 2.4 Pipeline jerárquico v2 (recomendado)

Usa la CNN para el grupo estático y los sub‑modelos `_v2` re‑entrenados con
el mapeo actualizado de etiquetas (`etiquetas_modelo_raiz_v2.csv`).

| Métrica | Resultado |
|---|---|
| Accuracy modelo raíz | **98.89%** |
| **Accuracy del pipeline (1.172 videos de test)** | **94.28%** |

---

## 3. Flujo de trabajo del repositorio

```
┌──────────────────┐    mediapipe_point_extraction.py
│ dataset_videos/  │ ────────────────────────────────────► dataset_colsign_45_154.h5
└──────────────────┘                                                  │
                                                                      ▼
                                ┌──────────────── train_lstm.py ──────────────► colsign_lstm_norm_45_154.keras
                                │
                                └──────────── train_lstm_cluster_labels.py ───► raíz + 4 sub‑modelos LSTM

dataset_videos/ ──► extract_static_hand_crops.py ──► dataset_static_crops/
                                                       │ (curación manual: golden set)
                                                       ▼
dataset_videos/ ──► expand_static_dataset.py ──► dataset_static_crops_expanded/
                                                       │
                                                       └──► train_static_cnn.py ──► colsign_static_cnn_45_154.keras

selecction_video_test_piplanes.py ──► video_path_test_piplane.csv
                                                       │
                                                       ▼
                                          ┌────► piplane_lstm_all.py        (evalúa modelo plano)
                                          ├────► piplane_lstm_jerarquico.py (v1)
                                          └────► piplane_lstm_jerarquico_v2.py (v2 con CNN)
                                                       │
                                                       ▼
                                            results_piplanes/*.csv + *.txt
```

### Estructura del repositorio

```
.
├── dataset_videos/                  # input: 154 carpetas de videos (no versionado)
├── dataset_colsign_45_154.h5        # keypoints procesados (no versionado)
├── dataset_static_crops/            # crops de mano (golden set, no versionado)
├── dataset_static_crops_expanded/   # dataset CNN expandido (no versionado)
├── models/                          # *.keras finales y checkpoints _best
├── info_models/                     # logs de entrenamiento + *_labels.json
├── graphics/                        # curvas de loss/accuracy (jpeg)
├── results_piplanes/                # salidas de los pipelines de test
├── src/
│   ├── utils.py                     # extracción + normalización + HDF5 loader
│   └── utils_pipeplanes.py          # utilidades de inferencia (load_model, predict, ...)
├── etiquetas_modelo_raiz.csv        # mapa jerárquico de etiquetas (v1)
├── etiquetas_modelo_raiz_v2.csv     # mapa jerárquico v2
├── video_path_test_piplane.csv      # 30% estratificado para test de pipelines
├── mediapipe_point_extraction.py    # genera el HDF5
├── train_lstm.py                    # modelo plano
├── train_lstm_cluster_labels.py     # modelo raíz + 4 sub-modelos
├── extract_static_hand_crops.py     # crops de mano por video (medoide)
├── expand_static_dataset.py         # golden set → dataset expandido
├── train_static_cnn.py              # CNN para grupo estático
├── selecction_video_test_piplanes.py# genera el CSV de test
├── piplane_lstm_all.py              # evalúa modelo plano
├── piplane_lstm_jerarquico.py       # evalúa pipeline jerárquico v1
├── piplane_lstm_jerarquico_v2.py    # evalúa pipeline jerárquico v2 (con CNN)
└── requirements.txt
```

---

## 4. Decisiones técnicas clave

Estas decisiones fueron las que más impacto tuvieron en el accuracy
durante el desarrollo (de 11% inicial a 85%+ en validación).

1. **`static_image_mode=True` en MediaPipe.** Como muestreamos 45 frames
   no consecutivos, el tracker en modo video se confunde y produce
   keypoints muy ruidosos. Forzar modo imagen estática lo arregla.
   *Esta configuración debe replicarse exactamente en inferencia (ver
   `src/utils_pipeplanes.make_holistic`).*
2. **Conteo de frames a dos pasadas.** Los videos VP8 reportan mal
   `CAP_PROP_FRAME_COUNT`. Se cuenta primero con `cap.grab()` y luego se
   re‑abre el video.
3. **Recorte de la cola del video** (`>3s → −1s`) para descartar pantalla
   negra o señas ya terminadas.
4. **Normalización por hombros + descarte de `visibility`.** Centrar
   los keypoints en el punto medio entre hombros y escalarlos por la
   distancia inter‑hombros elimina la dependencia de la posición/tamaño
   de la persona en la cámara. Subir de ~18% a ~85% de accuracy fue
   gracias a esto.
5. **Bajar `min_detection_confidence` a 0.3.** El default 0.5 era
   demasiado estricto para videos a baja resolución y descartaba
   demasiados frames.
6. **Golden set + expansión automática** para el grupo estático: una
   pequeña curación manual (~149 imágenes) seguida de matching por
   keypoints contra los videos originales lleva la CNN del 40% al 95%
   de accuracy.

---

## 5. Cómo reproducir

```powershell
# 1. Entorno
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Descargar dataset_videos.zip desde Zenodo y descomprimirlo en la raíz

# 3. Generar el HDF5 de keypoints (~1-2 h con 8 workers)
python mediapipe_point_extraction.py

# 4. Entrenar el modelo plano (~1 h)
python train_lstm.py

# 5. Entrenar el pipeline jerárquico (raíz + 4 sub-modelos, ~3-4 h)
python train_lstm_cluster_labels.py

# 6. (Opcional) Reentrenar el grupo estático con CNN
python extract_static_hand_crops.py
# (curar manualmente dataset_static_crops/ para construir el golden set)
python expand_static_dataset.py
python train_static_cnn.py

# 7. Generar test set y evaluar pipelines
python selecction_video_test_piplanes.py
python piplane_lstm_all.py
python piplane_lstm_jerarquico.py
python piplane_lstm_jerarquico_v2.py
```

Para usar los modelos en otro proyecto basta con cargarlos con
`src.utils_pipeplanes.load_model(...)` o directamente con `tf.keras.models.load_model(...)`.

---

## 6. Créditos y enlaces

- Dataset Colsign‑LSC‑154 en Zenodo: `<DOI pendiente>`
- Aplicación: https://www.colsign.com.co/
- Stack: TensorFlow/Keras 2.17, MediaPipe 0.10, OpenCV 4.10, h5py 3.16. Ver [`requirements.txt`](requirements.txt).
