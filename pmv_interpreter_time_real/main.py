"""Punto de entrada del prototipo de intérprete ColSign en tiempo real.

Abre la cámara, muestra la zona de encuadre y, al iniciar la traducción,
arma una secuencia de frames cada 2 segundos, la pasa por el modelo LSTM
(45 frames · 154 señas) y va listando las señas detectadas. Al terminar
(botón "Terminar" o tecla Enter) guarda el video del experimento en
`dataset_time_real/` y registra la fila correspondiente en `experiments.csv`.

Ejecutar (desde la raíz del proyecto):
    .\.venv\Scripts\python.exe -m pmv_interpreter_time_real.main
  o:
    .\.venv\Scripts\python.exe pmv_interpreter_time_real\main.py
"""

import os
import sys

# La raíz del proyecto debe ir en sys.path para poder importar el `src` raíz
# (src.utils_pipeplanes). Además quitamos el directorio de este script del
# path para que el `src` INTERNO del prototipo no tape al `src` raíz.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)

sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or os.getcwd()) != _THIS_DIR]
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pmv_interpreter_time_real.src.app import run


if __name__ == '__main__':
    run()
