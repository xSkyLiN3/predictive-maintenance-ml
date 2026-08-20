# Contrato de datos y evaluación

Este documento fija las reglas antes de observar resultados. Cambiarlas después requiere registrarlo en `docs/DECISIONS.md` y explicar el motivo.

## Fuente y trazabilidad

- Dataset: AI4I 2020 Predictive Maintenance.
- Fuente canónica: UCI Machine Learning Repository, ID 601.
- Licencia: CC BY 4.0.
- La descarga debe guardar URL, fecha, tamaño y SHA-256.
- El archivo original es inmutable; cualquier transformación genera una salida separada.

## Target

`Machine failure`, clasificación binaria.

## Features permitidas inicialmente

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

Estos nombres se confirmaron contra el CSV oficial fijado en M1 y forman la allowlist exacta.

## Columnas prohibidas

- Identificadores: `UDI`, `Product ID`.
- Indicadores de modos de fallo: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

Los indicadores de modos de fallo están vinculados directamente a la definición del target. Incluirlos inflaría artificialmente el desempeño y haría que el proyecto no demostrara inferencia útil desde señales operativas.

## Contrato del snapshot confirmado en M1

El contrato estricto corresponde al dataset fuente completo, no a una futura observación de API:

- 10.000 filas y 14 columnas en el orden oficial;
- strings: `Product ID` y `Type`;
- floats: temperaturas del aire y proceso, y torque;
- enteros: `UDI`, velocidad rotacional, desgaste, target y cinco modos de fallo;
- cero nulos, strings vacíos, infinitos y filas crudas completamente duplicadas;
- `Type` contiene exactamente `L`, `M` y `H`;
- target y modos de fallo son binarios, y el target contiene ambas clases;
- `UDI` es positivo y único; `Product ID` es único, sigue `[LMH]` más cinco dígitos y su
  prefijo coincide con `Type`.

Los guardrails numéricos de regresión son: aire 295–305 K, proceso 305–315 K, velocidad
1.000–3.000 rpm, torque 0–80 Nm y desgaste 0–260 min. Contienen el snapshot observado, pero no
son límites físicos universales ni el contrato de inputs de la API M4.

Los extremos exactos del snapshot completo se inspeccionaron en M1 como control de calidad de la
fuente, pero **no son la procedencia de la regla de abstención**. La envolvente del servicio se
congeló exclusivamente desde `reports/eda/summary.json`, un artefacto versionado que declara
`scope = "training_only"`, `training_rows = 8000` y `holdout_profiled = false`:

| Variable | Mínimo y máximo observados, inclusivos |
|---|---:|
| `Air temperature [K]` | `295.3`–`304.5` K |
| `Process temperature [K]` | `305.7`–`313.8` K |
| `Rotational speed [rpm]` | `1168`–`2886` rpm |
| `Torque [Nm]` | `3.8`–`76.6` Nm |
| `Tool wear [min]` | `0`–`253` min |

Estos son los extremos univariados de **training**, no del holdout. Describen soporte marginal de
datos sintéticos; no son límites industriales, reglas físicas ni evidencia de que toda combinación
interior pertenezca al dominio del generador. La coincidencia de estos cinco pares con los extremos
globales observados durante M1 no cambia su procedencia: el contrato y la regresión automatizada
usan solamente el resumen EDA de training.

## Contrato de inferencia local de M4

La API define un contrato independiente para una observación, no reutiliza el validador del CSV
ni los guardrails de fuente. Exige exactamente estos campos JSON:

- `type`: string exacto `L`, `M` o `H`;
- `air_temperature_k` y `process_temperature_k`: números finitos mayores que cero;
- `rotational_speed_rpm`: entero no negativo;
- `torque_nm`: número finito no negativo;
- `tool_wear_min`: entero no negativo.

No se aceptan campos extra, `null`, booleanos, strings numéricos, `NaN` ni infinitos. Las cinco
restricciones numéricas expresan unidades y signos semánticos, no soporte estadístico ni límites
industriales. Los nombres JSON deben ser únicos: una clave duplicada devuelve `400`. El body tiene
un máximo de `16 KiB` (`16.384` bytes) y excederlo devuelve `413`. Los demás rechazos de validación
y los fallos de inferencia se representan mediante respuestas JSON controladas y serializables.

La UI exige además `Number.isSafeInteger` para velocidad y desgaste, entre `0` y
`9.007.199.254.740.991`, con el único fin de impedir pérdida de precisión al convertir el formulario
a números JavaScript. Esta protección del cliente no es un límite físico ni reemplaza la validación
del servidor.

Una observación que pasa el schema se compara con la envolvente marginal de training registrada
arriba. Los valores están fijados en el código para que inferencia no lea datasets ni reportes en
runtime; `tests/test_reference_envelope.py` demuestra que coinciden exactamente con el resumen EDA
`training_only` y que ese resumen declara no haber perfilado el holdout.
Si todas las variables quedan dentro de sus extremos inclusivos,
`domain_status = "within_reference_envelope"`,
`decision_applicable = true` y el pipeline produce `risk_score` y `predicted_failure` normalmente.
Si cualquier variable queda fuera, la respuesta sigue siendo HTTP `200`, pero el pipeline no se
invoca: `domain_status = "outside_reference_envelope"`, `decision_applicable = false`,
`risk_score = null` y
`predicted_failure = null`. Las advertencias identifican por separado cada campo fuera de rango.

Esta abstención no convierte la API en un detector OOD completo. Solo comprueba cinco proyecciones
univariadas; no valida soporte conjunto, correlaciones, densidad, deriva, orden temporal ni
plausibilidad física. Un input marcado `within_reference_envelope` todavía puede estar fuera de
distribución. Adoptar
esta capa de servicio no modifica el modelo, el umbral ni ninguna métrica M3, y no reutiliza el
holdout para seleccionar o ajustar el sistema.

Además de duplicados crudos, el validador informa observaciones repetidas sobre las seis features
sin rechazarlas: mediciones operativas iguales pueden ser legítimas. En el snapshot fijado ambos
conteos son cero.

El archivo contiene 27 desacuerdos entre `Machine failure` y el OR de los cinco indicadores: 9
positivos sin modo activo y 18 negativos con `RNF = 1`. Se conserva el target original, no se
imputa ni corrige ninguna fila y no se exige que los modos sean mutuamente excluyentes. Esta
anomalía refuerza que los indicadores se auditen pero nunca entren al conjunto de features.

## Partición

1. Separar una vez un 20 % estratificado como test.
2. Conservar el 80 % restante para entrenamiento y validación cruzada.
3. Ajustar preprocesamiento dentro de cada fold mediante `Pipeline`.
4. Elegir modelo y umbral sin consultar test.
5. Ejecutar una evaluación final sobre test y conservarla como resultado del MVP.

La semilla fija, elegida arbitrariamente antes de modelar y sin probar alternativas, es `42`; se
reutiliza en todos los componentes compatibles. La partición se
materializó al inicio de M2, antes de la EDA, mediante `train_test_split` de scikit-learn 1.9.0.
Las filas se ordenan por su posición original dentro de cada partición para obtener CSV canónicos.

Los derivados locales contienen exclusivamente las seis features permitidas y el target. El
manifiesto versionado `data/split_manifest.json` registra hash de fuente, algoritmo, semilla,
proporción, columnas, versiones, tamaños y hashes de ambas particiones, pero ninguna estadística
específica del holdout. La EDA carga y verifica solo `train.csv`; no resuelve `holdout.csv`.
Las versiones del manifiesto registran el entorno que creó el archivo; no son invariantes de
integridad entre revisiones de Python 3.12. Los hashes derivados sí deben coincidir exactamente.

M1 necesariamente inspeccionó conteos y rangos globales para validar el contrato de la fuente.
Desde la materialización de M2, ninguna distribución, ejemplo o resultado específico del holdout
participa en decisiones.

## Validación cruzada y selección

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` compartido por los candidatos.
- Todo preprocesamiento se ajusta dentro de cada fold mediante `Pipeline`.
- La selección usa la media no ponderada de Average Precision en los cinco folds; se publican
  también valores por fold y desviación estándar.
- Average Precision pooled sobre las predicciones OOF se podrá mostrar solo como diagnóstico; no
  sustituye la media de los cinco folds para seleccionar.
- ROC-AUC es secundaria y no desempata la selección.
- `DummyClassifier(strategy="prior")` es el baseline.
- Entre regresión logística y random forest gana la mayor AP media. Un empate numérico dentro de
  `1e-12` favorece regresión logística por simplicidad.
- Si ningún candidato supera al dummy en AP media, el holdout no se abre y M3 se detiene para
  revisión.

## Modelos iniciales

1. `DummyClassifier` como referencia mínima.
2. Regresión logística con preprocesamiento y balance de clases cuando corresponda.
3. Random forest con complejidad controlada.

No se añadirán más algoritmos hasta comprender el error de estos modelos.

La configuración quedó congelada antes de ejecutar M3 y no se hará búsqueda de hiperparámetros:

- los tres estimadores viven dentro de `Pipeline`;
- `Type` se codifica con `OneHotEncoder` usando categorías fijas `L`, `M`, `H`; las cinco
  variables numéricas pasan por `StandardScaler`; el `ColumnTransformer` elimina cualquier otra
  columna;
- dummy: `DummyClassifier(strategy="prior")`;
- logística: L2 mediante `l1_ratio=0`, `C=1`, solver `liblinear`,
  `class_weight="balanced"`, `max_iter=1000` y semilla `42`;
- random forest: 300 árboles, Gini, profundidad máxima 8, `min_samples_split=10`,
  `min_samples_leaf=5`, `max_features="sqrt"`, bootstrap, `class_weight="balanced"`, semilla `42`
  y `n_jobs=1`.

La desviación estándar de CV se calculará con `ddof=0`. La puerta frente al dummy conserva la
lectura literal del protocolo: la AP media del mejor candidato debe ser estrictamente mayor que
la AP media del dummy; la tolerancia `1e-12` solo se usa para el desempate entre logística y
random forest y para empates del umbral.

## Métricas

### Principal

**Average Precision**, adecuada para resumir precision-recall cuando la clase positiva es minoritaria.

### En el umbral elegido

- precision;
- recall;
- F1;
- matriz de confusión;
- cantidad de falsos positivos y falsos negativos.

### Secundaria

- ROC-AUC.

Accuracy se informará solamente acompañada de prevalencia, baseline y las métricas anteriores.

## Umbral

El umbral no será necesariamente 0,5. Después de seleccionar el modelo se generará exactamente
una probabilidad out-of-fold `predict_proba[:, 1]` por fila de training usando los mismos cinco
folds. Se elegirá el umbral que maximice F1 con la regla `score >= threshold`. Los empates dentro
de `1e-12` se resolverán primero
por menor diferencia absoluta entre precision y recall y luego por el menor umbral. Este último
desempate es solo determinista y no representa costos industriales. El valor se congelará antes
de ajustar el pipeline elegido sobre todo training y evaluar el holdout una sola vez.

Precision, recall y F1 OOF se etiquetarán como estimaciones usadas para selección, no como
resultados finales. El reporte mostrará el intercambio completo sin inventar costos de operación.

## Ejecución registrada de M3

El contrato anterior se ejecutó sin tuning ni cambios de features en el run
`b15bab7b54bc2e1f`. Random forest ganó por AP media CV (`0.643812`), por encima de regresión
logística (`0.441433`) y Dummy (`0.033875`). El umbral OOF congelado fue
`0.6965799216184142`; en training OOF produjo precision `0.587879`, recall `0.715867` y F1
`0.645591`.

Después de congelar modelo y umbral, el holdout se leyó una sola vez. El resultado final fue AP
`0.649538`, ROC-AUC `0.965458`, precision `0.588235`, recall `0.735294`, F1 `0.653595` y matriz
`[[1897, 35], [18, 50]]`. Accuracy fue `0.973500`, junto a prevalencia `0.034` y referencia
mayoritaria `0.966`.

Como cierre posterior, sin volver a abrir el holdout ni recalcular predicciones, se añadieron
intervalos Wilson bilaterales del 95 % derivados **solo** de esa matriz versionada. Para precision,
los éxitos son `TP = 50` entre `TP + FP = 85`: IC `0.482010`–`0.686830`. Para recall, los
éxitos son `TP = 50` entre `TP + FN = 68`: IC `0.619923`–`0.825503`. Se usa el cuantil normal
`z = 1.9599639845400536`; la fórmula y sus regresiones viven en
`src/predictive_maintenance/uncertainty.py` y `tests/test_uncertainty.py`.

Estos intervalos cuantifican solamente incertidumbre binomial por soporte finito bajo el holdout
fijado. No corrigen sesgo de selección, naturaleza sintética, shift de distribución, dependencia
entre observaciones ni incertidumbre por predicción. No se usaron para elegir o modificar modelo,
features, umbral o claims.

El recibo y los gráficos están en
`reports/modeling/b15bab7b54bc2e1f/`. Un ledger versionado por SHA-256 en
`reports/holdout_access/` impide que otro run consuma nuevamente ese holdout. Estos scores no se
evaluaron como probabilidades calibradas y el resultado sintético no acredita utilidad industrial.

## Resultado aceptable

No existe una cifra mínima prefijada para el portfolio. Un resultado es válido si:

- supera al baseline en AP media de CV y reporta los deltas por fold, aunque estos no son una
  puerta adicional;
- fue obtenido sin leakage;
- es reproducible;
- se informa completo, incluso si es modesto;
- sus limitaciones quedan claras.
