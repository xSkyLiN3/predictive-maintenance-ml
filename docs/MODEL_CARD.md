# Model Card — Machine Failure Risk Classifier

> **Modelo educativo.** AI4I 2020 es un dataset sintético. Este modelo no está validado
> para uso industrial, seguridad, mantenimiento real ni decisiones operativas. Su `risk_score`
> no fue evaluado como probabilidad calibrada.

## 1. Identidad y resumen

| Campo | Valor |
|---|---|
| Producto | `Machine Failure Risk Classifier` |
| Run del modelo | `b15bab7b54bc2e1f` |
| Versión del schema de artefactos | `1` |
| Tarea | Clasificación binaria de `Machine failure` para una observación |
| Modelo elegido | `random_forest` (`RandomForestClassifier`) |
| Salida continua | `predict_proba[:, 1]`, expuesta como `risk_score` solo si la decisión aplica |
| Umbral congelado | `0.6965799216184142` |
| Regla de decisión | `risk_score >= threshold` |
| Capa de aplicabilidad | Envolvente marginal derivada solo de training; abstención fuera de ella |
| Estado | Release educativo `1.0.0`; resultado M3 congelado |
| Disponibilidad | Demo pública verificada en `https://ml.nightstrike.cloud` |

El servicio recibe seis variables de una observación operativa. Solo invoca el modelo y devuelve un
score para la clase positiva `Machine failure` cuando las cinco variables numéricas están dentro de
la envolvente marginal obtenida exclusivamente de training AI4I. En ese caso, la clasificación
booleana se deriva aplicando
el umbral elegido con predicciones out-of-fold de training. Fuera de esa referencia el servicio se
abstiene. El producto demuestra un flujo reproducible de ingeniería de machine learning; no
demuestra utilidad sobre maquinaria real.

## 2. Propósito y usos previstos

Usos previstos:

- demostrar en un proyecto de portfolio trazabilidad de datos, prevención de leakage, selección
  por validación cruzada, evaluación final y servicio reproducible de inferencia;
- puntuar observaciones individuales compatibles con el schema y dentro de la referencia marginal,
  y abstenerse explícitamente cuando algún campo queda fuera;
- permitir revisión técnica y experimentación educativa, localmente o mediante una demo pública
  stateless de superficie mínima;
- comparar el resultado real contra un baseline sin seleccionar una cifra por marketing.

Usuarios previstos: personas que revisen o estudien el proyecto y desarrolladores que ejecuten la
demo. No está destinado a operadores industriales.

Usos explícitamente no previstos:

- decidir mantenimiento, seguridad, paradas, inventario o asignación de personal;
- predecir vida útil restante, tiempo hasta el fallo o una secuencia futura;
- inferir la causa o el modo físico de un fallo;
- evaluar maquinaria, plantas, fabricantes o condiciones reales no representadas;
- tratar el score como frecuencia o probabilidad calibrada de un fallo real;
- sustituir diagnóstico, análisis de ingeniería o supervisión humana;
- integración industrial o uso de producción sin una validación independiente.

## 3. Datos

### 3.1 Fuente y licencia

- Dataset: **AI4I 2020 Predictive Maintenance Dataset**.
- Fuente: UCI Machine Learning Repository, dataset ID 601.
- DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
- Licencia: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Atribución y transformaciones: [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
- Tamaño del snapshot: `10000` observaciones.
- Naturaleza: datos sintéticos inspirados en escenarios de mantenimiento predictivo.
- Datos personales: el proyecto documenta que no contiene datos personales.
- SHA-256 del CSV fuente: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

Que el dataset sea sintético limita cualquier conclusión sobre condiciones, ruido, deriva,
dependencias temporales y mecanismos de fallo de equipos reales.

### 3.2 Target y features

Target oficial: `Machine failure`, con valores `0` y `1`. Se conservó la etiqueta del CSV sin
recalcularla desde los modos de fallo.

| Feature sklearn | Campo API | Tipo o unidad |
|---|---|---|
| `Type` | `type` | Categoría `L`, `M` o `H` |
| `Air temperature [K]` | `air_temperature_k` | Kelvin |
| `Process temperature [K]` | `process_temperature_k` | Kelvin |
| `Rotational speed [rpm]` | `rotational_speed_rpm` | Revoluciones por minuto |
| `Torque [Nm]` | `torque_nm` | Newton-metro |
| `Tool wear [min]` | `tool_wear_min` | Minutos |

Columnas excluidas obligatoriamente:

- identificadores `UDI` y `Product ID`, por no representar señales operativas generalizables;
- `TWF`, `HDF`, `PWF`, `OSF` y `RNF`, porque son indicadores de modos de fallo vinculados al
  target y producirían fuga de información.

El snapshot contiene `27` desacuerdos entre `Machine failure` y el OR de los cinco indicadores:
`9` positivos sin modo activo y `18` negativos con `RNF = 1`. El target oficial no se corrigió.
Esta anomalía es una posible fuente de ruido de etiqueta y refuerza la exclusión de los modos.

### 3.3 Partición

- Método: `sklearn.model_selection.train_test_split`.
- Semilla: `42`.
- Estratificación: `Machine failure`.
- Training: `8000` filas (`80 %`).
- Holdout: `2000` filas (`20 %`).
- SHA-256 de training:
  `3b114192f249951632f4c700c07b5edf4306fcff89ac90abe556f15687cf803a`.
- SHA-256 de holdout:
  `50a1c9c07a57afbc6f34dd112852b61a44f81b6a83341241dd1bc7079f3ac4b7`.

El holdout se materializó antes de la EDA, no participó en selección ni ajuste del umbral y fue
leído una sola vez para la evaluación final del run congelado.

## 4. Entrenamiento y selección

### 4.1 Protocolo

- Validación: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Métrica principal: Average Precision media no ponderada de los cinco folds.
- Métrica secundaria: ROC-AUC.
- Desviación estándar de CV: `ddof=0`.
- Baseline: `DummyClassifier(strategy="prior")`.
- Candidatos: regresión logística L2 balanceada y random forest balanceado.
- No se realizó búsqueda de hiperparámetros ni se probaron más algoritmos.
- Los mismos folds se reutilizaron para todos los candidatos.
- Preprocesamiento y estimador permanecieron dentro de un `Pipeline` de scikit-learn.

`Type` se transformó con `OneHotEncoder`, categorías fijas `L`, `M`, `H`,
`handle_unknown="error"` y salida densa. Las cinco variables numéricas pasaron por
`StandardScaler`; `ColumnTransformer(remainder="drop")` eliminó cualquier otra columna.

La puerta frente al baseline exigía literalmente que la AP media del candidato ganador fuera
mayor que la del Dummy. Empates de candidatos dentro de `1e-12` favorecían la regresión
logística. El random forest ganó sin necesidad de ese desempate.

### 4.2 Random forest congelado

| Parámetro | Valor |
|---|---:|
| `n_estimators` | `300` |
| `criterion` | `gini` |
| `max_depth` | `8` |
| `min_samples_split` | `10` |
| `min_samples_leaf` | `5` |
| `max_features` | `sqrt` |
| `bootstrap` | `true` |
| `class_weight` | `balanced` |
| `random_state` | `42` |
| `n_jobs` | `1` |

### 4.3 Selección del umbral

Después de elegir el modelo se generó una predicción `predict_proba[:, 1]` out-of-fold por cada
fila de training. El umbral `0.6965799216184142` maximiza F1 sobre esas predicciones con la
regla inclusiva `score >= threshold`. Los empates dentro de `1e-12` se resolvían por menor
diferencia absoluta entre precision y recall y, después, por el umbral menor. El holdout no se
usó para elegir ni modificar este valor.

## 5. Resultados

Los valores siguientes se reproducen de los recibos JSON versionados. Average Precision es la
métrica principal; accuracy se muestra únicamente con prevalencia y baseline mayoritario.

### 5.1 Validación cruzada sobre training

| Candidato | AP media | AP std (`ddof=0`) | ROC-AUC media |
|---|---:|---:|---:|
| Dummy prior | `0.033875` | `0.0002500000000000002` | `0.5` |
| Regresión logística | `0.44143285508642044` | `0.06850845339545959` | `0.8992749262131948` |
| Random forest | `0.6438124425485383` | `0.02247288575462437` | `0.9699351406866447` |

La mejora de AP media del random forest frente al Dummy fue `0.6099374425485383`. Estas cifras
son estimaciones de selección y no resultados del holdout.

### 5.2 Métricas OOF de selección del umbral

| Métrica | Valor |
|---|---:|
| Average Precision pooled, diagnóstico | `0.6332800463156792` |
| ROC-AUC pooled, diagnóstico | `0.968384753067352` |
| Precision | `0.5878787878787879` |
| Recall | `0.7158671586715867` |
| F1 | `0.6455906821963394` |
| Predicciones positivas | `330` |
| Matriz `[[TN, FP], [FN, TP]]` | `[[7593, 136], [77, 194]]` |

Las métricas pooled se registraron como diagnóstico; la selección del modelo usó la AP media de
los cinco folds, no la AP pooled.

### 5.3 Evaluación final única sobre holdout

| Métrica | Valor |
|---|---:|
| Average Precision | `0.6495379423468456` |
| ROC-AUC | `0.9654579222993546` |
| Precision al umbral | `0.5882352941176471` |
| IC Wilson 95 % de precision (`50/85`) | `0.4820101461448797`–`0.6868299449467584` |
| Recall al umbral | `0.7352941176470589` |
| IC Wilson 95 % de recall (`50/68`) | `0.619922660101109`–`0.825502593301211` |
| F1 al umbral | `0.6535947712418301` |
| Accuracy | `0.9735` |
| Prevalencia positiva | `0.034` |
| Accuracy de clase mayoritaria | `0.966` |
| Predicciones positivas | `85` |
| Verdaderos negativos | `1897` |
| Falsos positivos | `35` |
| Falsos negativos | `18` |
| Verdaderos positivos | `50` |

Matriz de confusión, con filas como clase real y columnas como clase predicha:

|  | Predicho 0 | Predicho 1 |
|---|---:|---:|
| Real 0 | `1897` | `35` |
| Real 1 | `18` | `50` |

El resultado corresponde a `2000` observaciones y al mismo umbral congelado
`0.6965799216184142`. No se modificaron modelo, features ni umbral después de observarlo.

Los intervalos son Wilson bilaterales del 95 % y se calcularon posteriormente solo desde la matriz
final ya versionada: precision usa `TP / (TP + FP) = 50/85` y recall usa
`TP / (TP + FN) = 50/68`. No se accedió de nuevo a observaciones, targets ni scores del holdout.
El endpoint `/model-info` los expone como `precision_wilson_95` y `recall_wilson_95`, derivados de
la misma matriz verificada al cargar la aplicación; no los persiste como una segunda evaluación.

Estos intervalos describen incertidumbre por soporte finito condicionada a este holdout. No miden
incertidumbre por observación, no corrigen el carácter sintético o un cambio de distribución y no
se usaron para seleccionar ni ajustar el sistema.

## 6. Contrato de inferencia de la demo educativa

La aplicación expone:

- `GET /health`: disponibilidad e identidad del modelo cargado;
- `GET /model-info`: modelo, umbral, métricas finales y advertencias;
- `POST /predict`: estado de aplicabilidad y, solo cuando corresponde, score y clasificación;
- `GET /`: interfaz estática.

`POST /predict` exige exactamente:

| Campo | Contrato |
|---|---|
| `type` | String exacto `L`, `M` o `H` |
| `air_temperature_k` | Número finito mayor que `0` |
| `process_temperature_k` | Número finito mayor que `0` |
| `rotational_speed_rpm` | Entero no negativo |
| `torque_nm` | Número finito no negativo |
| `tool_wear_min` | Entero no negativo |

Se rechazan campos extra, `null`, booleanos, strings numéricos, `NaN` e infinitos. Estos límites
validan estructura, unidades y signos; no son límites físicos ni prueban pertenencia al dominio
AI4I. El body JSON admite como máximo `16 KiB` (`16.384` bytes): excederlos devuelve `413`. Los
nombres de miembros JSON deben ser únicos; una clave duplicada devuelve `400` en vez de aplicar la
semántica ambigua de “último valor gana”. Los demás rechazos de schema y fallos de inferencia se
entregan como errores JSON controlados y serializables, sin eco innecesario del input ni detalles de
la excepción interna.

La interfaz web restringe `rotational_speed_rpm` y `tool_wear_min` a enteros seguros de JavaScript
mediante `Number.isSafeInteger` (`0` a `9.007.199.254.740.991`). Este máximo evita pérdida de
precisión en el navegador; no es un límite físico ni la envolvente aplicable al modelo.

Después de validar el schema, el servicio aplica esta referencia marginal de training, con
extremos inclusivos:

| Campo | Envolvente exacta observada en training AI4I |
|---|---:|
| `air_temperature_k` | `295.3`–`304.5` K |
| `process_temperature_k` | `305.7`–`313.8` K |
| `rotational_speed_rpm` | `1168`–`2886` rpm |
| `torque_nm` | `3.8`–`76.6` Nm |
| `tool_wear_min` | `0`–`253` min |

Con todos los campos dentro, la respuesta declara
`domain_status = "within_reference_envelope"` y
`decision_applicable = true`; `risk_score` es el score para la clase positiva y
`predicted_failure` aplica el umbral congelado. Si cualquier campo queda fuera, la request conserva
status HTTP `200`, pero el modelo no se invoca y no se emite decisión:
`domain_status = "outside_reference_envelope"`,
`decision_applicable = false`, `risk_score = null` y `predicted_failure = null`. `warnings`
identifica individualmente cada campo que excede su intervalo.

Los intervalos son mínimos y máximos univariados de las `8000` filas de training, obtenidos del
resumen EDA versionado que declara `scope = "training_only"` y `holdout_profiled = false`. No son
límites físicos ni un detector OOD completo. No evalúan combinaciones entre variables, densidad,
deriva,
secuencia o plausibilidad causal. Por ello, `within_reference_envelope` significa solamente
“dentro de todos los
rangos marginales observados”; una observación así marcada todavía puede ser atípica o irreal. Si
existe score, tampoco es una estimación calibrada de la probabilidad de fallo.

Esta capa de abstención no reentrenó el pipeline, no cambió el run, las features, el umbral o las
métricas M3 y no requirió una nueva lectura del holdout.

El modo local conserva `127.0.0.1:8000` y hosts locales como defaults. M6 permite configurar bind,
puerto y una allowlist explícita de hosts mediante variables de entorno para ejecutarlo detrás de un
proxy; no admite una allowlist `*`. La aplicación carga una vez el pipeline después de validar puntero
activo, manifiestos, hashes, recibo final, ledger, versiones, clases y orden de features. El arranque
de inferencia no debe leer raw, training ni holdout. La disponibilidad pública quedó establecida al
completar las verificaciones de despliegue M6; sigue siendo una demo educativa y no una validación
industrial.

## 7. Limitaciones y consideraciones responsables

### 7.1 Datos y generalización

- AI4I 2020 es sintético y no representa una población comprobada de maquinaria industrial.
- El split aleatorio estima generalización IID dentro del generador; no mide generalización
  temporal, entre máquinas, plantas, fabricantes o regímenes operativos.
- No existe validación externa con datos industriales reales.
- La etiqueta oficial presenta los `27` desacuerdos documentados con los modos de fallo.
- La prevalencia del holdout fue `0.034`; métricas y errores deben interpretarse en ese contexto.

### 7.2 Score, umbral y OOD

- No se evaluaron calibración, error de calibración ni confiabilidad probabilística.
- Optimizar F1 no representa costos reales de falsos positivos y falsos negativos.
- El desempate del umbral es determinista, no una preferencia operacional.
- La API se abstiene fuera de la envolvente marginal de training, pero no detecta OOD conjunto,
  deriva ni inputs estadísticamente atípicos dentro de esos intervalos.
- No se cuantifica incertidumbre por predicción.
- Cada request es una observación independiente; el modelo carece de historia temporal.

### 7.3 Riesgos de uso y ética

- Un falso negativo podría ocultar un fallo en una interpretación indebida; un falso positivo
  podría inducir intervenciones innecesarias. Ninguno de esos costos fue modelado.
- Presentar el score como probabilidad, diagnóstico o recomendación sería engañoso.
- No se evaluaron desempeño por subgrupos operativos, equidad, impactos económicos ni riesgos de
  automatización. La ausencia de datos personales no demuestra ausencia de impacto.
- Cualquier uso real requeriría revisión de ingeniería, seguridad, gobernanza y supervisión
  humana, además de datos representativos del entorno objetivo.
- El binario Joblib solo debe cargarse desde el bundle local verificado: deserializar artefactos
  de procedencia no confiable puede ejecutar código.

## 8. Reproducibilidad y trazabilidad

### 8.1 Identidades criptográficas

| Artefacto o configuración | SHA-256 |
|---|---|
| Configuración M3 | `d880c8048fcb3c09395e38702fd9ca04b1d6e3e0b53fd882e7dd728bdb1b9065` |
| Plan de folds | `2c24c5165e54481a6eb35ac08f579c78601ea191eee8a0d0a76537b034eddf48` |
| Pipeline local | `8f383492fff0a1199a7f62289651a29da39f4c6a149762aa9b75c099efc1568a` |
| Manifiesto del run | `01c3c72a75df64922470ee163166fbb2437fad0b6c279ca54f0b5687ccb02a2a` |
| Resultados CV | `ce9f62b79834c0da8d6a44311ae3714c18922820fc33f61a8ca9b3ab9f9e12f2` |
| Selección del umbral | `9790eedc834f5624029a24c2e64a544392c171995bde51d33f89cf7cbe70d412` |
| Recibo de evaluación final | `f3c947fe38fca0053c3f14e75c01681e5cef1dbcbc09e57fddb15409fd1e26c8` |

El ledger global registra `holdout_evaluation_complete` para el holdout y el run indicados. Una
ejecución posterior debe reutilizar el recibo, no volver a abrir ese holdout.

### 8.2 Entorno registrado por el run

| Componente | Versión |
|---|---|
| Python | `3.12.0` |
| scikit-learn | `1.9.0` |
| pandas | `3.0.5` |
| NumPy | `2.5.2` |
| joblib | `1.5.3` |
| matplotlib | `3.11.1` |

El pipeline exacto evaluado se versiona junto a su manifiesto SHA-256. La decisión evita sustituir
en Linux la serialización Joblib fijada por el recibo final; esos bytes no son portables entre
sistemas operativos aunque el entrenamiento sea determinista. `python -m predictive_maintenance
train` permite reproducir el flujo desde training sin leer holdout, mientras que la demo carga
únicamente el artefacto propio del repositorio después de validar toda la cadena de recibos. La
resolución exacta probada se conserva en `requirements/constraints-win-py312.txt`.

Artefactos versionados principales:

- [manifiesto del pipeline](../artifacts/m3/b15bab7b54bc2e1f/artifact_manifest.json);
- [manifiesto del run](../reports/modeling/b15bab7b54bc2e1f/run_manifest.json);
- [resultados de validación cruzada](../reports/modeling/b15bab7b54bc2e1f/cv_results.json);
- [selección OOF del umbral](../reports/modeling/b15bab7b54bc2e1f/threshold_selection.json);
- [recibo final](../reports/modeling/b15bab7b54bc2e1f/final_evaluation.json);
- [informe M3](../reports/modeling/b15bab7b54bc2e1f/M3_REPORT.md);
- [ledger de acceso al holdout](../reports/holdout_access/50a1c9c07a57afbc6f34dd112852b61a44f81b6a83341241dd1bc7079f3ac4b7.json);
- [contrato de datos y evaluación](DATA_EVALUATION.md);
- [registro de decisiones](DECISIONS.md).

## 9. Mantenimiento y gestión de cambios

No hay reentrenamiento programado, monitorización, telemetría ni detector de deriva. El modelo y
el umbral están congelados para este MVP local.

Reglas de mantenimiento:

- si falta el binario versionado, restaurarlo desde una revisión Git verificada; `train` puede
  reproducir el flujo, pero una reserialización en otro sistema operativo no sustituye
  silenciosamente el artefacto evaluado ni autoriza repetir el holdout consumido;
- validar siempre run, hashes, recibo final, ledger, versiones, clases y orden de features antes
  de servir inferencia;
- tratar cambios de target, features, dataset, split, protocolo, modelos, umbral o contrato API
  como una nueva versión documentada, no como una corrección silenciosa;
- mantener separada la envolvente marginal de aplicabilidad de los límites físicos: cambiarla es
  una decisión del contrato de servicio y no modifica por sí mismo el modelo ni sus métricas;
- no usar las métricas del holdout ya observado para escoger una nueva variante;
- actualizar esta ficha si cambia la identidad del modelo o cualquiera de sus supuestos;
- revisar compatibilidad y volver a ejecutar pruebas ante actualizaciones de Python,
  scikit-learn, joblib, NumPy, pandas, FastAPI o Pydantic;
- tratar la demo pública M6 como educativa y stateless, sin almacenar inputs ni incluir datos o
  particiones; no confundir su disponibilidad con validación industrial o madurez de producción.

Antes de considerar un uso real se necesitarían, como mínimo, datos industriales representativos,
separaciones temporales y por máquina, validación externa, evaluación de calibración, costos de
error, límites operativos definidos por especialistas, detección OOD y deriva, monitorización,
seguridad del servicio y un proceso explícito de gobernanza y retirada.

## 10. Procedencia de esta ficha

Todas las cifras, parámetros e identidades de esta ficha proceden de los artefactos versionados
enlazados arriba y de la documentación aceptada del proyecto. La envolvente procede del resumen EDA
versionado de training y los intervalos Wilson se derivan de la matriz final versionada. No se
consultaron CSV raw, training o holdout, no se recalcularon predicciones y no se produjo una nueva
evaluación para redactarla.
