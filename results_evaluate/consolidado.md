# Consolidado de Evaluación en `dataset_videos_evaluate`

> Nota sobre "accuracy entrenamiento": se usa la **accuracy de validación/test**
> reportada en cada `*_train_log.txt` (conjunto held-out). La evaluación
> corresponde a `dataset_videos_evaluate` (37 videos / 22 carpetas).

## Tabla Principal

| Modelo o estrategia | Acc. entrenamiento (val) | Acc. evaluación | Lista labels evaluación | Labels sin aciertos (0%) |
|---|---:|---:|---|---|
| **ColSign 154** (plano, 45f) | 85.66% | 10.81% (4/37) | A veces, Abrazar, Adiós, Allá, Aprender, Ayer, Buscar, Desmayarse, Doler, Dudar, El, Esperar, Gracias, Interesar, Molestar, Otra vez, Radiografía, a, e, i, t, v | A veces, Adiós, Allá, Aprender, Ayer, Buscar, Doler, Dudar, El, Esperar, Gracias, Molestar, Otra vez, a, e, i, t, v |
| **ColSign 154 aug** (45f) | 85.11% | 13.51% (5/37) | A veces, Abrazar, Adiós, Allá, Aprender, Ayer, Buscar, Desmayarse, Doler, Dudar, El, Esperar, Gracias, Interesar, Molestar, Otra vez, Radiografía, a, e, i, t, v | Adiós, Allá, Aprender, Ayer, Buscar, Doler, Dudar, El, Esperar, Gracias, Interesar, Molestar, a, e, i, t, v |
| **ColSign 154** (15f) | 86.34% | 18.92% (7/37) | A veces, Abrazar, Adiós, Allá, Aprender, Ayer, Buscar, Desmayarse, Doler, Dudar, El, Esperar, Gracias, Interesar, Molestar, Otra vez, Radiografía, a, e, i, t, v | Adiós, Allá, Aprender, Ayer, Buscar, Desmayarse, Doler, Esperar, Gracias, Molestar, a, e, i, t, v |
| **Jerárquica v2** (estrategia completa) | compuesta (ver nota) | 16.22% (6/37) | A veces, Abrazar, Adiós, Allá, Aprender, Ayer, Buscar, Desmayarse, Doler, Dudar, El, Esperar, Gracias, Interesar, Molestar, Otra vez, Radiografía, a, e, i, t, v | Adiós, Allá, Aprender, Ayer, Buscar, Doler, El, Gracias, Interesar, Molestar, Otra vez, a, e, i, t, v |
| **Sub Estático v2** | 64.41% | 10.00% (1/10) | a, e, i, t, v | a, i, t, v |
| **Sub Unimanual v2** | 86.49% | 14.29% (2/14) | Adiós, Allá, Ayer, Buscar, Desmayarse, Doler, Dudar, El, Interesar | Adiós, Allá, Ayer, Buscar, Doler, El, Interesar |
| **Sub Bimanual Simétrico** | 94.96% | 60.00% (3/5) | Abrazar, Esperar, Radiografía | ninguna |
| **Sub Bimanual Asimétrico** | 92.27% | 12.50% (1/8) | A veces, Aprender, Gracias, Molestar, Otra vez | Aprender, Gracias, Molestar, Otra vez |

## Resumen de Accuracies

| Modelo / estrategia | Train (val) | Eval | Caída |
|---|---:|---:|---:|
| ColSign 154 (45f) | 85.66% | 10.81% | -74.85 pp |
| ColSign 154 aug (45f) | 85.11% | 13.51% | -71.60 pp |
| ColSign 154 (15f) | 86.34% | 18.92% | -67.42 pp |
| Jerárquica v2 | — | 16.22% | — |
| Sub Estático v2 | 64.41% | 10.00% | -54.41 pp |
| Sub Unimanual v2 | 86.49% | 14.29% | -72.20 pp |
| Sub Bimanual Simétrico | 94.96% | 60.00% | -34.96 pp |
| Sub Bimanual Asimétrico | 92.27% | 12.50% | -79.77 pp |

## Estrategia Jerárquica v2

La estrategia jerárquica no tiene un único número de entrenamiento porque
encadena varios modelos. Sus componentes reportaron estas accuracies de
validación:

| Componente | Acc. entrenamiento (val) |
|---|---:|
| Modelo raíz v2 (clasifica grupo) | 96.99% |
| Sub Estático v2 | 64.41% |
| Sub Unimanual v2 | 86.49% |
| Sub Bimanual Simétrico | 94.96% |
| Sub Bimanual Asimétrico | 92.27% |

Diagnóstico del paso raíz en evaluación:

| Grupo real | Accuracy raíz |
|---|---:|
| Grupo Dinámico Bimanual Asimétrico | 50.00% (4/8) |
| Grupo Dinámico Bimanual Simétrico | 100.00% (5/5) |
| Grupo Dinámico Unimanual | 71.43% (10/14) |
| Grupo Estático | 30.00% (3/10) |
| **Total raíz** | **59.46% (22/37)** |

## Lectura Rápida

- La caída train→eval es severa en casi todos los modelos; esto coincide con
  el diagnóstico previo de *domain shift* y menor detección de manos por
  MediaPipe en los videos de usuarios nuevos.
- El mejor resultado en evaluación entre los modelos planos sigue siendo
  **ColSign 154 de 15 frames** (18.92%).
- La **estrategia jerárquica v2** mejora frente al plano de 45 frames, pero
  queda por debajo del plano de 15 frames.
- El sub-modelo que mejor se sostiene es **Bimanual Simétrico** (60.00%),
  y también es el grupo donde el modelo raíz acierta el 100% en evaluación.
- Las señas estáticas (`a`, `e`, `i`, `t`, `v`) son de las más afectadas:
  el sub-modelo estático solo alcanza 10.00% y deja sin aciertos a `a`, `i`,
  `t`, `v`.

