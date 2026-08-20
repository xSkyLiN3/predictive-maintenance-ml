# EDA de training y protocolo cerrado

> Alcance: este informe usa exclusivamente la partición de **training** después de separar el
> holdout. AI4I 2020 es sintético; las asociaciones observadas no validan uso industrial.

## Protocolo congelado antes de modelar

- Fuente fijada por SHA-256: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.
- Split: 80% training / 20% holdout, estratificado,
  semilla arbitraria predefinida `42` y orden original del snapshot fijado dentro de
  cada partición. La semilla no se optimizó ni se comparó con otras.
- Training analizado: 8,000 filas. El holdout no se carga ni se perfila.
- Validación cruzada de M3: `StratifiedKFold(n_splits=5, shuffle=True,
  random_state=42)`, compartida por todos los candidatos.
- Baseline: `DummyClassifier(strategy="prior")`.
- Selección: mayor AP media de los 5 folds (`average_precision`); ROC-AUC
  (`roc_auc`) será secundaria. AP pooled OOF será solo diagnóstica y no sustituirá la
  media CV. Un empate dentro de `1e-12` entre logística y random forest favorece
  logística por simplicidad. Si ningún candidato supera al dummy en AP media, no se abre holdout.
- Umbral: `maximize_f1_on_out_of_fold_training_predictions` sobre `predict_proba[:, 1]`. Se maximiza F1 con una predicción
  OOF por fila de training y la regla `score >= threshold`; empates dentro de
  `1e-12` se resuelven por menor diferencia absoluta entre precision y recall y
  luego por el menor umbral. El valor se congela antes de evaluar holdout una sola vez.
- Todo preprocesamiento se ajustará dentro de cada fold mediante `Pipeline`.

M1 necesariamente verificó conteos y rangos globales del archivo para validar su contrato. Desde
la creación de esta partición en M2 no se usan estadísticas, ejemplos ni resultados específicos
del holdout para tomar decisiones.

## Calidad y balance en training

- Filas: 8,000.
- Positivos: 271; negativos: 7,729.
- Prevalencia positiva: **3.39%**.
- Celdas ausentes: 0.
- Filas repetidas sobre las seis features: 0.
- Columnas analizadas: únicamente las seis features permitidas y `Machine failure`.

## Type

| Type | Filas | Positivos | Tasa positiva | IC Wilson 95% |
|---|---:|---:|---:|---:|
| L | 4,830 | 197 | 4.08% | 3.56%–4.67% |
| M | 2,381 | 58 | 2.44% | 1.89%–3.14% |
| H | 789 | 16 | 2.03% | 1.25%–3.27% |

La mayor tasa descriptiva aparece en `Type=L`
(4.08%); los intervalos y soportes deben acompañar cualquier lectura.
No se interpreta como efecto causal.

## Perfil numérico de training

| Feature | Min | Q1 | Mediana | Q3 | Max |
|---|---:|---:|---:|---:|---:|
| `Air temperature [K]` | 295.30 | 298.30 | 300.10 | 301.50 | 304.50 |
| `Process temperature [K]` | 305.70 | 308.80 | 310.10 | 311.10 | 313.80 |
| `Rotational speed [rpm]` | 1168.00 | 1422.00 | 1503.00 | 1613.00 | 2886.00 |
| `Torque [Nm]` | 3.80 | 33.20 | 40.10 | 46.80 | 76.60 |
| `Tool wear [min]` | 0.00 | 53.00 | 107.00 | 163.00 | 253.00 |

La asociación monotónica más fuerte entre features numéricas es `Rotational speed [rpm]` frente a
`Torque [Nm]` (Spearman ρ = -0.917). Esto puede importar para la estabilidad
de coeficientes, pero no justifica eliminar variables antes de comparar los pipelines ya
pre-registrados. También existe una asociación alta entre las temperaturas de aire y proceso
(ρ = 0.864).

En training, las observaciones con fallo tienen medianas mayores de torque
(53.2 frente a 39.8 Nm), desgaste
(166 frente a 106 min) y temperatura del aire
(301.6 frente a 300.0 K), y menor velocidad rotacional
(1366 frente a 1507 rpm). Las ECDF muestran amplio
solapamiento y los paneles conjuntos y por quintil no sugieren una única relación lineal uniforme.
Estas observaciones descriptivas respaldan comparar los candidatos logística y random forest ya
registrados; no prueban causalidad ni rendimiento predictivo.

## Visualizaciones

![Conteos de clases y prevalencia de fallos en las 8.000 observaciones de training](figures/01_target_prevalence.png)
![Soporte y tasa de fallos por tipo de producto con intervalos de Wilson](figures/02_type_support_and_rate.png)
![Distribuciones acumuladas de las cinco variables numéricas separadas por target](figures/03_numeric_ecdf_by_target.png)
![Matriz de correlación de Spearman entre las cinco variables numéricas de training](figures/04_spearman_correlation.png)
![Relaciones aire-proceso y velocidad-torque, con densidad de negativos y puntos positivos](figures/05_joint_relationships.png)
![Tasa de fallos por quintil para cada variable numérica, con denominadores e intervalos](figures/06_positive_rate_by_quintile.png)

Los quintiles son solo una ayuda visual calculada sobre training; no se incorporan como
transformación ni crean nuevas features. Las ECDF están normalizadas dentro de cada clase, por lo
que deben leerse junto al gráfico de prevalencia.

## Riesgos y limitaciones

- El split aleatorio estima generalización IID dentro del mismo generador sintético; no mide
  generalización temporal, entre máquinas ni en industria real.
- Las asociaciones con el target son descriptivas y no causales. La EDA no habilita cambiar
  target, features o candidatos sin una nueva decisión explícita.
- Los positivos son escasos; tasas de subgrupos y quintiles tienen incertidumbre visible.
- La ausencia de nulos o duplicados en AI4I no implica que datos reales tendrían esa calidad.
- Los indicadores `TWF`, `HDF`, `PWF`, `OSF` y `RNF`, junto con identificadores, permanecen fuera
  de los derivados y de todas las figuras para evitar leakage.
- Las 27 discrepancias globales entre modos de fallo y target detectadas en M1 no se corrigen; el
  target contractual sigue siendo `Machine failure`.

## Puerta de revisión

Este informe debe revisarse antes de ejecutar M3. Aún no hay modelos entrenados, métricas de
validación ni resultados de holdout.
