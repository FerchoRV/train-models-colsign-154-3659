"""Evaluación sobre `dataset_videos_evaluate` de:

  1. La ESTRATEGIA JERÁRQUICA v2 completa (modelo raíz -> sub-modelo).
  2. Cada uno de los 4 SUB-MODELOS de forma INDEPENDIENTE.

Parte de la misma dinámica que `evaluate_models_lstm.py`: la etiqueta real
es el nombre de la subcarpeta que contiene el video; se calcula accuracy
global y por clase y se guardan un CSV y un TXT por modelo.

Filtrado por sub-modelo
-----------------------
La carpeta de evaluación NO contiene las 154 señas. Para evaluar un
sub-modelo de forma independiente y justa, SOLO se le pasan los videos cuya
etiqueta (carpeta) pertenece al vocabulario de ese sub-modelo. Ejemplo: la
carpeta "a" es una seña estática, así que solo se evalúa contra el
sub-modelo estático; nunca contra unimanual / simétrico / asimétrico.

La estrategia jerárquica SÍ procesa todos los videos (el modelo raíz decide
el grupo y rutea), igual que en producción.

Eficiencia
----------
MediaPipe es el cuello de botella, así que los keypoints (45, F) se extraen
UNA sola vez por video y se reutilizan para el raíz y para todos los
sub-modelos (todos los LSTM comparten input shape).

Modelos (v2)
------------
- Raíz : colsign_lstm_norm_raiz_45_154_v2
- Sub  : colsign_lstm_norm_estatic_45_154_v2
         colsign_lstm_norm_unimanual_45_154_v2
         colsign_lstm_norm_bi_simetrico_45_154
         colsign_lstm_norm_bi_asimetrico_45_154

Salida (carpeta `results_evaluate/`)
------------------------------------
- Jerárquico:
    resuls_colsign_norm_45_154_jerarquico_v2.csv   (label, path, label_predicha, prob, grupo_raiz_predicho)
    accuracy_colsign_norm_45_154_jerarquico_v2.txt
- Por sub-modelo (uno por cada uno):
    resuls_<model_name>.csv    (label, path, label_predicha, prob)
    accuracy_<model_name>.txt

Ejecutar:
    .\.venv\Scripts\python.exe -u evaluate_models_lstm_jerarquico.py
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
PATH_DATASET_EVALUATE = 'dataset_videos_evaluate'
RESULTS_DIR           = 'results_evaluate'
ROOT_CSV              = 'etiquetas_modelo_raiz_v2.csv'

VIDEO_EXTS = ('.mp4', '.m4v', '.avi', '.mov', '.mkv', '.webm')

ROOT_MODEL_NAME = 'colsign_lstm_norm_raiz_45_154_v2'
SUB_MODEL_NAMES = {
    'Grupo Estático':                       'colsign_lstm_norm_estatic_45_154_v2',
    'Grupo Dinámico Unimanual':             'colsign_lstm_norm_unimanual_45_154_v2',
    'Grupo Dinámico Bimanual Simétrico':    'colsign_lstm_norm_bi_simetrico_45_154',
    'Grupo Dinámico Bimanual Asimétrico':   'colsign_lstm_norm_bi_asimetrico_45_154',
}

HIER_RUN_NAME = 'colsign_norm_45_154_jerarquico_v2'


# =====================================================================
# Utilidades
# =====================================================================
def construir_items(carpeta):
    """Recorre `carpeta` y devuelve [(label, video_path)] (label = subcarpeta)."""
    if not os.path.isdir(carpeta):
        raise SystemExit(
            f"No existe la carpeta {carpeta!r}. Crea la carpeta con "
            f"subcarpetas por etiqueta y coloca los videos dentro."
        )
    items = []
    for label in sorted(os.listdir(carpeta)):
        label_dir = os.path.join(carpeta, label)
        if not os.path.isdir(label_dir):
            continue
        for nombre in sorted(os.listdir(label_dir)):
            if nombre.lower().endswith(VIDEO_EXTS):
                items.append((label, os.path.join(label_dir, nombre)))
    return items


def load_label_to_root(csv_path):
    """Lee el CSV de jerarquía y devuelve {label: etiqueta_raiz}."""
    if not os.path.exists(csv_path):
        raise SystemExit(f"No se encontró el CSV de jerarquía: {csv_path}")
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as fp:
        reader = csv.DictReader(fp, delimiter=';')
        for row in reader:
            label = str(row.get('nombre', '')).strip()
            root = str(row.get('etiqueta_raiz', '')).strip()
            if label and root:
                mapping[label] = root
    return mapping


def predict_from_features(model, info, x_features):
    """Top-1 desde features ya extraídas. Devuelve (label, prob)."""
    proba = model.predict(x_features[None, ...], verbose=0)[0]
    idx = int(np.argmax(proba))
    return info.id_to_name.get(idx, f"<id={idx}>"), float(proba[idx])


def extraer_features(items, holistic, seq_len):
    """Extrae UNA vez los keypoints (seq_len, F) por video.

    Devuelve `{path: ndarray|None}` (None si falló la extracción).
    """
    feats = {}
    for _label, path in tqdm(items, desc='Extraer keypoints', unit='video'):
        try:
            feats[path] = extract_lstm_features(
                path,
                sequence_length=seq_len,
                type_extract='pose_hands',
                normalize=True,
                drop_pose_visibility=True,
                holistic=holistic,
            )
        except Exception:  # noqa: BLE001 - resiliente a videos corruptos
            feats[path] = None
    return feats


def calcular_metricas(results):
    """Devuelve (total, correct, per_class_ok, per_class_total)."""
    correct = sum(1 for r in results if r['label'] == r['label_predicha'])
    per_class_ok    = defaultdict(int)
    per_class_total = defaultdict(int)
    for r in results:
        per_class_total[r['label']] += 1
        if r['label'] == r['label_predicha']:
            per_class_ok[r['label']] += 1
    return len(results), correct, per_class_ok, per_class_total


def escribir_csv(output_csv, results, fieldnames):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)


def escribir_reporte(output_txt, encabezado_lineas, results, failed,
                     labels_presentes, extra_secciones=None):
    """Escribe el TXT con el mismo formato base que `evaluate_models_lstm.py`."""
    total, correct, per_class_ok, per_class_total = calcular_metricas(results)
    accuracy = correct / total if total else 0.0

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as fp:
        for linea in encabezado_lineas:
            fp.write(linea + "\n")
        fp.write("\n")

        fp.write("## Resumen global\n")
        fp.write(f"Etiquetas:         {len(labels_presentes)}\n")
        fp.write(f"Total videos:      {total}\n")
        fp.write(f"Aciertos:          {correct}\n")
        fp.write(f"Errores:           {total - correct}\n")
        fp.write(f"Fallos de proceso: {failed}\n")
        fp.write(f"Accuracy:          {accuracy:.4f}  ({accuracy*100:.2f}%)\n\n")

        if extra_secciones:
            for seccion in extra_secciones:
                fp.write(seccion)
                if not seccion.endswith("\n"):
                    fp.write("\n")
            fp.write("\n")

        fp.write("## Accuracy por clase\n")
        fp.write(f"{'label':<35} {'ok':>5} {'total':>5} {'acc':>7}\n")
        fp.write('-' * 56 + '\n')
        for label in sorted(per_class_total.keys()):
            n = per_class_total[label]
            ok = per_class_ok[label]
            acc = ok / n if n else 0.0
            fp.write(f"{label:<35} {ok:>5} {n:>5} {acc*100:>6.2f}%\n")
        fp.write('-' * 56 + '\n')
        fp.write(f"{'TOTAL':<35} {correct:>5} {total:>5} {accuracy*100:>6.2f}%\n")

    return total, correct, accuracy


# =====================================================================
# Evaluación: estrategia jerárquica v2
# =====================================================================
def evaluar_jerarquico(items, feats, root_assets, sub_assets, label_to_root):
    root_model, root_info = root_assets

    results = []
    failed = 0
    root_group_pred = defaultdict(int)   # diagnóstico: cuántas a cada grupo
    root_correct = 0
    root_known = 0
    root_per_group_ok = defaultdict(int)
    root_per_group_total = defaultdict(int)

    for label_real, path in tqdm(items, desc='Jerárquico', unit='video'):
        real_root = label_to_root.get(label_real, '<desconocido>')
        x = feats.get(path)
        if x is None:
            results.append({'label': label_real, 'path': path,
                            'label_predicha': '<error:extraccion>',
                            'prob': '0.0000',
                            'grupo_raiz_real': real_root,
                            'grupo_raiz_predicho': '<error>'})
            failed += 1
            continue

        pred_root, _ = predict_from_features(root_model, root_info, x)
        root_group_pred[pred_root] += 1
        if real_root != '<desconocido>':
            root_known += 1
            root_per_group_total[real_root] += 1
            if pred_root == real_root:
                root_correct += 1
                root_per_group_ok[real_root] += 1

        if pred_root in sub_assets:
            sub_model, sub_info = sub_assets[pred_root]
            final_label, prob = predict_from_features(sub_model, sub_info, x)
        else:
            final_label, prob = f'<sin sub-modelo:{pred_root}>', 0.0

        results.append({
            'label':               label_real,
            'path':                path,
            'label_predicha':      final_label,
            'prob':                f'{prob:.4f}',
            'grupo_raiz_real':     real_root,
            'grupo_raiz_predicho': pred_root,
        })

    # Sección extra: accuracy y distribución del modelo raíz
    root_acc = root_correct / root_known if root_known else 0.0
    diag = [
        "## Diagnóstico del modelo raíz\n",
        f"Videos con grupo real conocido: {root_known}\n",
        f"Aciertos del raíz:             {root_correct}\n",
        f"Accuracy modelo raíz:          {root_acc:.4f}  ({root_acc*100:.2f}%)\n\n",
        "### Accuracy del raíz por grupo real\n",
        f"{'grupo_real':<40} {'ok':>5} {'total':>5} {'acc':>7}\n",
        '-' * 60 + "\n",
    ]
    for g in sorted(root_per_group_total):
        n = root_per_group_total[g]
        ok = root_per_group_ok[g]
        acc = ok / n if n else 0.0
        diag.append(f"{g:<40} {ok:>5} {n:>5} {acc*100:>6.2f}%\n")
    diag.append("\n### Distribución de grupo predicho por el modelo raíz\n")
    for g in sorted(root_group_pred, key=lambda k: -root_group_pred[k]):
        diag.append(f"  {g:<40} {root_group_pred[g]:>4}\n")
    return results, failed, [''.join(diag)]


# =====================================================================
# Evaluación: un sub-modelo independiente (solo sus etiquetas)
# =====================================================================
def evaluar_submodelo(group_name, model, info, items, feats):
    vocab = set(info.name_to_id.keys())
    # Solo videos cuya etiqueta (carpeta) conoce este sub-modelo.
    sub_items = [(lab, path) for (lab, path) in items if lab in vocab]

    results = []
    failed = 0
    for label_real, path in tqdm(sub_items, desc=f'Sub[{group_name}]', unit='video'):
        x = feats.get(path)
        if x is None:
            results.append({'label': label_real, 'path': path,
                            'label_predicha': '<error:extraccion>', 'prob': '0.0000'})
            failed += 1
            continue
        label_pred, prob = predict_from_features(model, info, x)
        results.append({'label': label_real, 'path': path,
                        'label_predicha': label_pred, 'prob': f'{prob:.4f}'})

    labels_presentes = sorted({lab for lab, _ in sub_items})
    return results, failed, labels_presentes, vocab


# =====================================================================
# Main
# =====================================================================
def main():
    t_start = time.time()

    print("=== evaluate_models_lstm_jerarquico ===")
    print(f"Fecha:    {datetime.now().isoformat(timespec='seconds')}")
    print(f"Carpeta:  {PATH_DATASET_EVALUATE}")
    print(f"Results:  {RESULTS_DIR}")
    print()

    # 1) Listar videos
    items = construir_items(PATH_DATASET_EVALUATE)
    labels_presentes = sorted({lab for lab, _ in items})
    print(f"Etiquetas encontradas: {len(labels_presentes)}")
    print(f"Videos a evaluar:      {len(items)}")
    if not items:
        raise SystemExit(f"No hay videos ({VIDEO_EXTS}) en {PATH_DATASET_EVALUATE!r}.")

    # 2) Mapeo etiqueta -> grupo raíz (verdad de terreno del paso raíz)
    label_to_root = load_label_to_root(ROOT_CSV)
    missing_root = sorted(set(labels_presentes) - set(label_to_root))
    print(f"Mapeo label→grupo raíz: {len(label_to_root)} etiquetas ({ROOT_CSV})")
    if missing_root:
        print(f"  ADVERTENCIA: {len(missing_root)} etiquetas de evaluación no están "
              f"en {ROOT_CSV}: {missing_root[:5]}")

    # 3) Cargar modelos (raíz + sub-modelos)
    print(f"\nCargando modelos (carga perezosa de TensorFlow)...")
    t0 = time.time()
    root_model, root_info = load_model(ROOT_MODEL_NAME)
    print(f"  RAIZ:  {root_info.name}  K={root_info.num_classes}  input={root_info.input_shape}")

    sub_assets = {}
    for group_name, model_name in SUB_MODEL_NAMES.items():
        m, info = load_model(model_name)
        if info.architecture != ARCH_LSTM:
            raise SystemExit(f"Sub-modelo '{group_name}' no es LSTM ({info.architecture}).")
        if info.input_shape != root_info.input_shape:
            raise SystemExit(
                f"Sub-modelo '{group_name}' input {info.input_shape} != raíz "
                f"{root_info.input_shape}; no es seguro reutilizar features.")
        sub_assets[group_name] = (m, info)
        print(f"  SUB [{group_name}]: {info.name}  K={info.num_classes}")
    print(f"  Carga total: {time.time() - t0:.1f}s")

    seq_len = root_info.sequence_length or 45

    # 4) Extraer features UNA sola vez por video
    print(f"\nExtrayendo keypoints (una pasada de MediaPipe por video)...")
    with make_holistic() as holistic:
        feats = extraer_features(items, holistic, seq_len)
    n_fail_feat = sum(1 for v in feats.values() if v is None)
    print(f"  Extracciones fallidas: {n_fail_feat}/{len(items)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 5) Estrategia jerárquica
    print(f"\n[1/5] Evaluando estrategia JERÁRQUICA v2...")
    results_h, failed_h, extra_h = evaluar_jerarquico(
        items, feats, (root_model, root_info), sub_assets, label_to_root)
    out_csv_h = os.path.join(RESULTS_DIR, f"resuls_{HIER_RUN_NAME}.csv")
    out_txt_h = os.path.join(RESULTS_DIR, f"accuracy_{HIER_RUN_NAME}.txt")
    escribir_csv(out_csv_h, results_h,
                 ['label', 'path', 'label_predicha', 'prob',
                  'grupo_raiz_real', 'grupo_raiz_predicho'])
    encabezado_h = [
        f"# Evaluación ESTRATEGIA JERÁRQUICA v2 ({HIER_RUN_NAME})",
        f"# Fecha:    {datetime.now().isoformat(timespec='seconds')}",
        f"# Carpeta:  {PATH_DATASET_EVALUATE}",
        f"# Mapping:  {ROOT_CSV}",
        f"# Raíz:     {root_info.keras_path}",
    ]
    for g, (_, info) in sub_assets.items():
        encabezado_h.append(f"# Sub [{g}]: {info.keras_path}")
    total_h, correct_h, acc_h = escribir_reporte(
        out_txt_h, encabezado_h, results_h, failed_h, labels_presentes, extra_h)
    print(f"      Accuracy jerárquica: {acc_h*100:.2f}%  ({correct_h}/{total_h})")
    print(f"      CSV: {out_csv_h}")
    print(f"      TXT: {out_txt_h}")

    # 6) Cada sub-modelo de forma independiente (solo sus etiquetas)
    resumen_sub = []
    for i, (group_name, (m, info)) in enumerate(sub_assets.items(), start=2):
        print(f"\n[{i}/5] Evaluando SUB-MODELO independiente: {group_name} ({info.name})...")
        results_s, failed_s, labels_s, vocab = evaluar_submodelo(
            group_name, m, info, items, feats)
        if not results_s:
            print(f"      (sin videos de evaluación con etiquetas de este sub-modelo)")
        out_csv_s = os.path.join(RESULTS_DIR, f"resuls_{info.name}.csv")
        out_txt_s = os.path.join(RESULTS_DIR, f"accuracy_{info.name}.txt")
        escribir_csv(out_csv_s, results_s, ['label', 'path', 'label_predicha', 'prob'])
        encabezado_s = [
            f"# Evaluación SUB-MODELO independiente: {info.name}",
            f"# Grupo:    {group_name}",
            f"# Fecha:    {datetime.now().isoformat(timespec='seconds')}",
            f"# Carpeta:  {PATH_DATASET_EVALUATE}",
            f"# Modelo:   {info.keras_path}",
            f"# Nota: solo se evalúan los videos cuya etiqueta pertenece al "
            f"vocabulario de este sub-modelo ({info.num_classes} señas).",
        ]
        total_s, correct_s, acc_s = escribir_reporte(
            out_txt_s, encabezado_s, results_s, failed_s, labels_s)
        print(f"      Etiquetas presentes: {len(labels_s)} | videos: {total_s}")
        print(f"      Accuracy: {acc_s*100:.2f}%  ({correct_s}/{total_s})")
        print(f"      CSV: {out_csv_s}")
        print(f"      TXT: {out_txt_s}")
        resumen_sub.append((group_name, info.name, total_s, correct_s, acc_s))

    # 6) Resumen final en consola
    duration = time.time() - t_start
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Jerárquica v2:           {acc_h*100:6.2f}%  ({correct_h}/{total_h})")
    for group_name, name, total_s, correct_s, acc_s in resumen_sub:
        etiqueta = f"{group_name}"
        print(f"  {etiqueta:<38} {acc_s*100:6.2f}%  ({correct_s}/{total_s})")
    print(f"\nDuración total: {duration:.1f}s ({duration/60:.2f} min)")


if __name__ == '__main__':
    main()
