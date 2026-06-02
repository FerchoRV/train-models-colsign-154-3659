"""Pipeline de evaluación jerárquica (modelo raíz + 4 sub-modelos v1).

Flujo
-----
1. Lee los videos a evaluar desde `video_path_test_piplane.csv`.
2. Carga la jerarquía real label → `etiqueta_raiz` desde
   `etiquetas_modelo_raiz.csv` (solo se usa para diagnóstico).
3. Para cada video:
     a. Extrae UNA SOLA VEZ los keypoints normalizados (45, 225) con
        MediaPipe Holistic. Este paso es el cuello de botella; lo
        reutilizamos entre el modelo raíz y el sub-modelo porque todos
        los LSTM del proyecto comparten el mismo input shape.
     b. Predice el `grupo_raiz` con el modelo raíz.
     c. Enruta los keypoints al sub-modelo correspondiente y obtiene
        la etiqueta final.
4. Calcula la accuracy SOLO al final (comparando `label_predicha` con
   `label`), igual que la pipeline plana anterior. Añade un bloque
   diagnóstico con accuracy del modelo raíz aparte, sin reemplazar el
   formato base.

Modelos usados (versión inicial v1, sin sufijo `_v2`)
-----------------------------------------------------
- Raíz : colsign_lstm_norm_raiz_45_154
- Sub  : colsign_lstm_norm_estatic_45_154
         colsign_lstm_norm_unimanual_45_154
         colsign_lstm_norm_bi_simetrico_45_154
         colsign_lstm_norm_bi_asimetrico_45_154

Para cada modelo se carga automáticamente el checkpoint `_best.keras`
si existe (resolución manejada por `utils_pipeplanes.load_model`).

Salida (carpeta `results_piplanes/`)
------------------------------------
- `resuls_consign_norm_45_154_jerarquico_v1.csv`
    columnas: label, path, label_predicha
- `accuracy_consign_norm_45_154_jerarquico_v1.txt`
    resumen global + accuracy por clase + accuracy del paso raíz.

Ejecutar:
    .\.venv\Scripts\python.exe -u piplane_lstm_jerarquico.py
"""

import os
import sys
import csv
import time
from collections import defaultdict
from datetime import datetime

# UTF-8 en consola (consistente con el resto del proyecto en Windows)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils_pipeplanes import (
    ARCH_LSTM,
    load_model,
    make_holistic,
    extract_lstm_features,
)


# =====================================================================
# Configuración
# =====================================================================

INPUT_CSV    = 'video_path_test_piplane.csv'
ROOT_CSV     = 'etiquetas_modelo_raiz.csv'

RESULTS_DIR  = 'results_piplanes'
RUN_NAME     = 'consign_norm_45_154_jerarquico_v1'

OUTPUT_CSV   = os.path.join(RESULTS_DIR, f"resuls_{RUN_NAME}.csv")
OUTPUT_TXT   = os.path.join(RESULTS_DIR, f"accuracy_{RUN_NAME}.txt")

# Modelos v1 (sin sufijo _v2). load_model() resolverá automáticamente al
# checkpoint `_best.keras` si está disponible.
ROOT_MODEL_NAME = 'colsign_lstm_norm_raiz_45_154'
SUB_MODEL_NAMES = {
    'Grupo Estático':                       'colsign_lstm_norm_estatic_45_154',
    'Grupo Dinámico Unimanual':             'colsign_lstm_norm_unimanual_45_154',
    'Grupo Dinámico Bimanual Simétrico':    'colsign_lstm_norm_bi_simetrico_45_154',
    'Grupo Dinámico Bimanual Asimétrico':   'colsign_lstm_norm_bi_asimetrico_45_154',
}


# =====================================================================
# Utilidades internas
# =====================================================================

def load_label_to_root(csv_path: str) -> dict:
    """Lee `etiquetas_modelo_raiz.csv` y devuelve `{label: etiqueta_raiz}`.

    El CSV usa separador `;`. Hacemos `strip` defensivo para evitar
    desencuentros con espacios o BOM en valores.
    """
    df = pd.read_csv(csv_path, sep=';', encoding='utf-8')
    df['nombre']        = df['nombre'].astype(str).str.strip()
    df['etiqueta_raiz'] = df['etiqueta_raiz'].astype(str).str.strip()
    return dict(zip(df['nombre'], df['etiqueta_raiz']))


def predict_lstm_from_features(model, info, x_features) -> tuple:
    """Predice top-1 desde features ya extraídas, evitando la doble
    pasada de MediaPipe. Devuelve `(label, prob)`.
    """
    proba = model.predict(x_features[None, ...], verbose=0)[0]
    idx = int(np.argmax(proba))
    return info.id_to_name.get(idx, f"<id={idx}>"), float(proba[idx])


# =====================================================================
# Main
# =====================================================================

def main():
    t_start = time.time()

    print(f"=== piplane_lstm_jerarquico ===")
    print(f"Fecha:       {datetime.now().isoformat(timespec='seconds')}")
    print(f"Run name:    {RUN_NAME}")
    print(f"Input CSV:   {INPUT_CSV}")
    print(f"Raíz CSV:    {ROOT_CSV}")
    print(f"Results dir: {RESULTS_DIR}")
    print()

    # Validaciones de existencia
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"No se encontró {INPUT_CSV}")
    if not os.path.exists(ROOT_CSV):
        raise SystemExit(f"No se encontró {ROOT_CSV}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Conjunto de evaluación
    with open(INPUT_CSV, 'r', encoding='utf-8', newline='') as fp:
        rows = list(csv.DictReader(fp))
    print(f"Videos a evaluar: {len(rows)}")

    # 2) Jerarquía label -> etiqueta_raiz (verdad de tierra para diagnosticar
    #    los aciertos del modelo raíz; NO se usa para decidir, eso lo hace el
    #    modelo raíz.)
    label_to_root = load_label_to_root(ROOT_CSV)
    print(f"Mapeo label→grupo_raíz cargado: {len(label_to_root)} etiquetas")

    # 3) Cargar todos los modelos. Usamos load_model con el nombre
    #    canónico (sin sufijo _best) y la utilidad ya prefiere el _best.
    print(f"\nCargando modelos (esto carga TF en frío)...")
    t0 = time.time()
    root_model, root_info = load_model(ROOT_MODEL_NAME)
    print(f"  RAIZ:  {root_info.name}  K={root_info.num_classes}  "
          f"input={root_info.input_shape}")

    sub_assets = {}
    for group_name, model_name in SUB_MODEL_NAMES.items():
        m, info = load_model(model_name)
        sub_assets[group_name] = (m, info)
        print(f"  SUB    [{group_name}]: {info.name}  "
              f"K={info.num_classes}  input={info.input_shape}")
    print(f"  Carga total: {time.time() - t0:.1f}s")

    # 4) Sanity checks
    # Todos los LSTM deben compartir el mismo input shape para poder
    # reutilizar la matriz de features entre raíz y sub-modelo.
    expected_input = root_info.input_shape
    expected_seq   = root_info.sequence_length
    expected_feat  = root_info.num_features

    for group_name, (_, info) in sub_assets.items():
        if info.architecture != ARCH_LSTM:
            raise SystemExit(
                f"Sub-modelo de '{group_name}' no es LSTM "
                f"(architecture={info.architecture}). Este pipeline solo "
                f"soporta LSTM para los sub-modelos."
            )
        if info.input_shape != expected_input:
            raise SystemExit(
                f"Sub-modelo de '{group_name}' tiene input_shape {info.input_shape} "
                f"distinto al raíz {expected_input}. No es seguro reutilizar features."
            )

    # El modelo raíz tiene que predecir las claves de SUB_MODEL_NAMES.
    root_vocab = set(root_info.name_to_id.keys())
    missing_in_root = set(SUB_MODEL_NAMES) - root_vocab
    if missing_in_root:
        raise SystemExit(
            f"El modelo raíz no conoce los grupos: {missing_in_root}. "
            f"Vocabulario raíz: {sorted(root_vocab)}"
        )

    # Advertencia si hay etiquetas reales del CSV que ningún sub-modelo conoce
    sub_vocabs = {g: set(info.name_to_id.keys()) for g, (_, info) in sub_assets.items()}
    real_labels = sorted({r['label'] for r in rows})
    unknown_in_subs = []
    for lbl in real_labels:
        true_root = label_to_root.get(lbl)
        if true_root is None:
            unknown_in_subs.append((lbl, '<sin grupo en CSV>'))
            continue
        if true_root in sub_vocabs and lbl not in sub_vocabs[true_root]:
            unknown_in_subs.append((lbl, true_root))
    if unknown_in_subs:
        print(f"\nADVERTENCIA: {len(unknown_in_subs)} etiquetas no están en "
              f"el vocabulario del sub-modelo que les correspondería. "
              f"Ejemplos: {unknown_in_subs[:5]}")

    # 5) Iteración con un único Holistic compartido
    results       = []
    correct       = 0   # acierto FINAL (label_predicha == label)
    root_correct  = 0   # acierto del primer paso (modelo raíz)
    routed_ok     = 0   # llegamos a un sub-modelo válido (grupo predicho en SUB_MODEL_NAMES)
    failed        = 0   # excepción en cualquier paso

    # Confusión de grupos a nivel raíz para diagnosticar
    root_confusion = defaultdict(lambda: defaultdict(int))  # real -> pred -> count

    print(f"\nProcesando {len(rows)} videos...\n")
    with make_holistic() as holistic:
        for row in tqdm(rows, desc='Hierarchical', unit='video'):
            label_real = row['label']
            video_path = row['path']
            real_root  = label_to_root.get(label_real, '<desconocido>')

            try:
                # 5a) Una sola extracción de features para todo el video
                features = extract_lstm_features(
                    video_path,
                    sequence_length=expected_seq,
                    type_extract='pose_hands',
                    normalize=True,
                    drop_pose_visibility=True,
                    holistic=holistic,
                )

                # 5b) Predicción raíz
                pred_root, _ = predict_lstm_from_features(
                    root_model, root_info, features,
                )

                # 5c) Predicción sub-modelo (si el grupo predicho tiene
                # uno disponible). En condiciones normales SIEMPRE existe
                # porque ya validamos missing_in_root arriba, pero
                # mantenemos el fallback para no romper la corrida.
                if pred_root in sub_assets:
                    sub_model, sub_info = sub_assets[pred_root]
                    final_label, _ = predict_lstm_from_features(
                        sub_model, sub_info, features,
                    )
                    routed_ok += 1
                else:
                    final_label = f'<sin sub-modelo:{pred_root}>'
            except Exception as e:  # noqa: BLE001 - resiliente a errores individuales
                pred_root   = f'<error:{type(e).__name__}>'
                final_label = f'<error:{type(e).__name__}>'
                failed += 1

            # Contabilidad
            if pred_root == real_root:
                root_correct += 1
            if final_label == label_real:
                correct += 1

            root_confusion[real_root][pred_root] += 1

            results.append({
                'label':                label_real,
                'path':                 video_path,
                'label_predicha':       final_label,
                'grupo_raiz_real':      real_root,
                'grupo_raiz_predicho':  pred_root,
            })

    # 6) CSV de resultados — el orden de columnas mantiene `label, path,
    # label_predicha` al frente (estructura igual al pipeline plano) y
    # añade dos columnas extra de diagnóstico jerárquico, que NO
    # requieren cambios en consumidores que solo lean las 3 primeras.
    fieldnames = ['label', 'path', 'label_predicha',
                  'grupo_raiz_real', 'grupo_raiz_predicho']
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV de resultados: {OUTPUT_CSV}  ({len(results)} filas)")

    # 7) Métricas
    total = len(results)
    accuracy      = correct / total      if total else 0.0
    accuracy_root = root_correct / total if total else 0.0
    duration      = time.time() - t_start

    # accuracy por clase (sobre label final)
    per_class_ok    = defaultdict(int)
    per_class_total = defaultdict(int)
    for r in results:
        per_class_total[r['label']] += 1
        if r['label'] == r['label_predicha']:
            per_class_ok[r['label']] += 1

    # accuracy por grupo (sobre la decisión del root model)
    per_group_ok    = defaultdict(int)
    per_group_total = defaultdict(int)
    for r in results:
        per_group_total[r['grupo_raiz_real']] += 1
        if r['grupo_raiz_real'] == r['grupo_raiz_predicho']:
            per_group_ok[r['grupo_raiz_real']] += 1

    # 8) Reporte
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as fp:
        fp.write(f"# Evaluación jerárquica {RUN_NAME}\n")
        fp.write(f"# Fecha:    {datetime.now().isoformat(timespec='seconds')}\n")
        fp.write(f"# Input:    {INPUT_CSV}\n")
        fp.write(f"# Mapping:  {ROOT_CSV}\n")
        fp.write(f"# Modelo raíz: {root_info.keras_path}\n")
        for group_name, (_, info) in sub_assets.items():
            fp.write(f"# Sub [{group_name}]: {info.keras_path}\n")
        fp.write(f"# Duración: {duration:.1f}s ({duration/60:.2f} min)\n\n")

        fp.write("## Resumen global (decisión final del pipeline)\n")
        fp.write(f"Total videos:        {total}\n")
        fp.write(f"Aciertos finales:    {correct}\n")
        fp.write(f"Errores finales:     {total - correct}\n")
        fp.write(f"Fallos de proceso:   {failed}\n")
        fp.write(f"Ruteos válidos:      {routed_ok} (pred_root reconocido)\n")
        fp.write(f"Accuracy final:      {accuracy:.4f}  ({accuracy*100:.2f}%)\n\n")

        fp.write("## Diagnóstico paso raíz (clasificación de grupo)\n")
        fp.write(f"Accuracy modelo raíz: {accuracy_root:.4f}  ({accuracy_root*100:.2f}%)\n")
        fp.write(f"{'grupo_real':<40} {'ok':>5} {'total':>5} {'acc':>7}\n")
        fp.write('-' * 60 + '\n')
        for g in sorted(per_group_total):
            n  = per_group_total[g]
            ok = per_group_ok[g]
            acc = ok / n if n else 0.0
            fp.write(f"{g:<40} {ok:>5} {n:>5} {acc*100:>6.2f}%\n")
        fp.write('\n')

        fp.write("## Matriz de confusión del modelo raíz\n")
        fp.write("(filas = grupo real, columnas = grupo predicho)\n\n")
        groups = sorted({r['grupo_raiz_real'] for r in results}
                        | {r['grupo_raiz_predicho'] for r in results})
        col_w = max(8, max(len(g) for g in groups) + 1)
        fp.write(f"{'':<40} ")
        for g in groups:
            fp.write(f"{g[:col_w-1]:>{col_w}}")
        fp.write('\n')
        for gr in sorted(root_confusion):
            fp.write(f"{gr:<40} ")
            for gp in groups:
                fp.write(f"{root_confusion[gr].get(gp, 0):>{col_w}}")
            fp.write('\n')
        fp.write('\n')

        fp.write("## Accuracy por clase (label final del pipeline)\n")
        fp.write(f"{'label':<35} {'ok':>5} {'total':>5} {'acc':>7}\n")
        fp.write('-' * 56 + '\n')
        for label in sorted(per_class_total):
            n  = per_class_total[label]
            ok = per_class_ok[label]
            acc = ok / n if n else 0.0
            fp.write(f"{label:<35} {ok:>5} {n:>5} {acc*100:>6.2f}%\n")
        fp.write('-' * 56 + '\n')
        fp.write(f"{'TOTAL':<35} {correct:>5} {total:>5} {accuracy*100:>6.2f}%\n")

    print(f"Reporte de accuracy: {OUTPUT_TXT}")
    print(f"\nAccuracy FINAL:       {accuracy*100:.2f}%  ({correct}/{total})")
    print(f"Accuracy modelo raíz: {accuracy_root*100:.2f}%  ({root_correct}/{total})")
    print(f"Duración total:       {duration:.1f}s ({duration/60:.2f} min)")


if __name__ == '__main__':
    main()
