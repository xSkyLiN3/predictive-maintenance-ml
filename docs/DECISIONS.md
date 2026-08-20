# Registro de decisiones

## D-001 — Desarrollo local antes de publicar

- **Estado:** aceptada.
- **Decisión:** completar y revisar el MVP en local antes de crear un remoto o desplegar.
- **Motivo:** separar aprendizaje y experimentación de cualquier claim público.

## D-002 — Python 3.12 y CPU

- **Estado:** aceptada.
- **Decisión:** usar Python 3.12 en `.venv` y entrenar en CPU.
- **Motivo:** el equipo dispone de Python 3.12 y el dataset tabular es pequeño; una GPU añadiría complejidad sin aportar valor relevante.

## D-003 — Alcance de clasificación por observación

- **Estado:** aceptada.
- **Decisión:** estimar `Machine failure` para una observación, no vida útil restante ni series temporales.
- **Motivo:** es lo que permiten sostener honestamente los datos seleccionados.

## D-004 — Features operativas sin indicadores de fallo

- **Estado:** aceptada.
- **Decisión:** excluir identificadores y `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.
- **Motivo:** evitar señales no generalizables y leakage respecto del target.

## D-005 — Average Precision como métrica principal

- **Estado:** aceptada.
- **Decisión:** usar Average Precision, complementada con precision, recall, F1, matriz de confusión y ROC-AUC.
- **Motivo:** la clase positiva es minoritaria y accuracy aislada sería poco informativa.

## D-006 — Empaquetado mínimo con dependencias por fase

- **Estado:** aceptada.
- **Decisión:** usar `setuptools` con estructura `src/`; mantener `pandas` como única dependencia
  de ejecución durante M1 y aislar pytest y Ruff en el extra `dev`.
- **Motivo:** M1 necesita lectura y validación tabular, mientras que scikit-learn, FastAPI y sus
  dependencias no son necesarios hasta hitos posteriores. Esto mantiene el entorno pequeño y
  evita implementar fases por anticipado.

## D-007 — Snapshot de datos inmutable y verificado

- **Estado:** aceptada.
- **Decisión:** descargar el ZIP oficial con la biblioteca estándar, exigir tamaño y SHA-256
  conocidos, extraer únicamente `ai4i2020.csv` y volver a verificar el CSV. Conservar ambos en
  `data/raw/` sin sustituir silenciosamente archivos existentes. Registrar fecha, fuente, DOI,
  licencia, tamaños y hashes en metadata local.
- **Motivo:** asegura trazabilidad y detecta cambios en la fuente sin añadir una dependencia HTTP
  para una sola descarga. Los datos y la metadata local no se versionan; las identidades esperadas
  sí quedan fijadas en código y documentación.

## D-008 — Conservar el target oficial sin derivarlo de los modos de fallo

- **Estado:** aceptada.
- **Decisión:** usar la columna original `Machine failure` sin corregirla ni recalcularla desde
  `TWF`, `HDF`, `PWF`, `OSF` o `RNF`. El validador informa, pero no rechaza, las 27 filas donde el
  target difiere del OR de esos indicadores.
- **Motivo:** el CSV oficial contiene 9 fallos sin indicador activo y 18 casos con `RNF = 1` y
  target 0. Reescribir el target alteraría la fuente. Los cinco indicadores continúan excluidos
  obligatoriamente de las features por leakage.

## D-009 — Guardrails de fuente separados de límites industriales

- **Estado:** aceptada.
- **Decisión:** validar las variables numéricas contra envolventes redondeadas que contienen el
  snapshot AI4I y registrar por separado los mínimos y máximos observados. No reutilizar estos
  rangos como contrato futuro de la API ni presentarlos como límites físicos universales.
- **Motivo:** M1 necesita detectar archivos corruptos o incompatibles, pero AI4I es sintético y sus
  extremos observados no justifican reglas sobre maquinaria real.

## D-010 — Resolución de dependencias registrada para Windows y Python 3.12

- **Estado:** aceptada.
- **Decisión:** mantener rangos de dependencias directas en `pyproject.toml` y registrar la
  resolución transitiva probada en `requirements/constraints-win-py312.txt`.
- **Motivo:** evita deriva accidental al reconstruir el entorno sin añadir otro gestor de
  dependencias. El archivo registra versiones, pero no promete reproducibilidad binaria entre
  plataformas ni sustituye la verificación funcional con Ruff y pytest.

## D-011 — Sellar el holdout antes de la EDA

- **Estado:** aceptada.
- **Decisión:** materializar al inicio de M2 un split 80/20 estratificado con semilla `42`, antes
  de calcular estadísticas o gráficos orientados a decisiones. La semilla fue una elección
  arbitraria previa a resultados y no se comparará con otras. La EDA carga solamente training.
- **Motivo:** `ROADMAP.md` y el contrato de evaluación ya exigían separar el holdout antes de la
  selección; `TASKS.md` lo ubicaba por error en M3. Mover la tarea a M2 alinea el checklist sin
  cambiar target, features ni proporción. M1 sí verificó agregados globales necesarios para el
  contrato de fuente; la ceguera específica del holdout comienza desde esta materialización.

## D-012 — Derivados sin columnas de leakage

- **Estado:** aceptada.
- **Decisión:** guardar `train.csv` y `holdout.csv` con exactamente las seis features permitidas y
  `Machine failure`; no conservar `UDI`, `Product ID` ni indicadores de modos en los derivados.
  Versionar un manifiesto determinista, no los CSV.
- **Motivo:** una allowlist física reduce el riesgo de que M3 incorpore columnas prohibidas por un
  `drop` incompleto. Los hashes y configuración del manifiesto permiten reconstruir y auditar la
  partición sin publicar datos derivados.

## D-013 — Protocolo de selección y umbral cerrado en M2

- **Estado:** aceptada.
- **Decisión:** usar cinco folds estratificados con shuffle y semilla `42`; seleccionar por AP
  media, con ROC-AUC secundaria; baseline dummy de prior. AP pooled OOF será solo diagnóstica. Un
  empate dentro de `1e-12` entre logística y random forest favorece logística. El umbral
  maximizará F1 sobre `predict_proba[:, 1]` OOF del modelo elegido aplicando
  `score >= threshold`, con el mismo margen de empate y desempate determinista documentado. Si
  ningún candidato supera al dummy en AP media, no se evaluará holdout.
- **Motivo:** cierra las decisiones antes de observar resultados de modelos o holdout, evita
  optimizar una narrativa posterior y no inventa costos industriales inexistentes.

## D-014 — Dependencias mínimas para M2

- **Estado:** aceptada.
- **Decisión:** añadir scikit-learn para el split y futuro modelado, y matplotlib para seis figuras
  reproducibles. No añadir seaborn, statsmodels ni tooling de notebooks.
- **Motivo:** ambas dependencias tienen una función directa en el hito y serán suficientes para la
  EDA. Los intervalos Wilson se calculan con una fórmula pequeña, evitando otra dependencia.

## D-015 — Versiones informativas y hashes invariantes del split

- **Estado:** aceptada.
- **Decisión:** conservar en el manifiesto las versiones exactas del entorno que lo creó, pero no
  exigir que una reconstrucción use el mismo parche de Python 3.12. Fuente, configuración,
  columnas, tamaños y hashes de los CSV sí son invariantes estrictos.
- **Motivo:** `pyproject.toml` admite cualquier Python 3.12.x. Si otra revisión reconstruye bytes
  idénticos, rechazarla solo porque difiere la versión informativa impediría reproducibilidad sin
  mejorar la integridad; una diferencia real del splitter continúa detectándose por los hashes.

## D-016 — Pipelines y complejidad fija antes de M3

- **Estado:** aceptada.
- **Decisión:** comparar exactamente dummy de prior, regresión logística L2 balanceada
  (`l1_ratio=0`, `C=1`, `liblinear`) y random forest balanceado de 300 árboles, profundidad 8 y
  hojas mínimas 5.
  `Type` se codifica con categorías fijas L/M/H y las numéricas se estandarizan dentro de cada
  `Pipeline`. No se hará tuning. Todos los parámetros estocásticos usan semilla `42` y el forest
  usa un solo proceso.
- **Motivo:** son candidatos pequeños, interpretables y suficientemente distintos para el MVP.
  Fijarlos antes de calcular CV impide optimizar la narrativa después de ver resultados y mantiene
  el costo reproducible en CPU.

## D-017 — Operacionalización final de CV y puerta del baseline

- **Estado:** aceptada.
- **Decisión:** materializar una sola tupla de cinco folds y reutilizarla en los tres candidatos y
  las predicciones OOF. Reportar desviación estándar poblacional (`ddof=0`) y deltas pareados por
  fold. La puerta es estricta: `AP_media_ganador > AP_media_dummy`; no se le aplica la tolerancia
  de empates. Reruns reutilizarán resultados finales y no reabrirán holdout.
- **Motivo:** elimina ambigüedades operativas sin cambiar la métrica ni el criterio congelados en
  M2, y reconcilia la evaluación única con una CLI idempotente.

## D-018 — Dependencias directas de los artefactos M3

- **Estado:** aceptada.
- **Decisión:** declarar NumPy y joblib como dependencias directas, aunque scikit-learn también las
  instale transitivamente. NumPy implementa validación y agregación explícitas; joblib serializa el
  pipeline elegido. Registrar también matplotlib en la identidad de versión del run.
- **Motivo:** el código del proyecto importa y usa estas librerías directamente. Declararlas evita
  depender accidentalmente del grafo transitivo y hace auditable la identidad de artefactos y
  figuras.

## D-019 — Ledger global y publicación recuperable de la evaluación

- **Estado:** aceptada.
- **Decisión:** aislar cada corrida en `reports/modeling/<run_id>/` y
  `artifacts/m3/<run_id>/`, ligar pipeline, configuración, folds, recibos y figuras mediante
  SHA-256, y reclamar la evaluación con un ledger exclusivo indexado por el SHA-256 del holdout
  en `reports/holdout_access/`. Publicar primero un bundle local recuperable y el recibo
  versionable al final. Un run distinto, aunque cambie los roots de salida, no puede volver a
  consumir el mismo holdout.
- **Motivo:** una bandera por run no protegía el mismo test frente a cambios de configuración. El
  ledger global preserva la semántica de evaluación única; el bundle permite reparar una
  publicación interrumpida sin segunda lectura. La garantía cubre el flujo secuencial de la
  aplicación desde la raíz, no una lectura manual deliberada del CSV.

## D-020 — Resultado M3 congelado sin iteración post-holdout

- **Estado:** aceptada.
- **Decisión:** conservar el run `b15bab7b54bc2e1f` como resultado M3. Random forest ganó con AP
  media CV `0.643812`; el umbral OOF fue `0.6965799216184142`. En la única evaluación holdout:
  AP `0.649538`, ROC-AUC `0.965458`, precision `0.588235`, recall `0.735294`, F1 `0.653595` y
  matriz `[[1897, 35], [18, 50]]`. No ajustar modelos, features ni umbral después de observar
  estas métricas.
- **Motivo:** publicar el resultado real preserva el protocolo pre-registrado y evita convertir el
  holdout en un conjunto de validación encubierto. AI4I es sintético y estos valores no validan
  desempeño industrial ni calibración probabilística.

## D-021 — Inferencia read-only sobre el run final

- **Estado:** aceptada.
- **Decisión:** cargar para M4 únicamente el pipeline evaluado del run `b15bab7b54bc2e1f` mediante
  un loader de inferencia que valida puntero activo, manifiestos, hashes, recibo final, ledger,
  versiones, clases y orden de features antes de deserializar. Cargarlo una vez en el lifespan de
  FastAPI y no invocar el flujo de evaluación ni resolver archivos de datos.
- **Motivo:** la aplicación debe servir el resultado congelado sin reabrir el holdout, recalcular
  métricas, reparar artefactos ni mezclar una selección distinta. Fallar cerrado ante una
  inconsistencia es preferible a servir un modelo cuya identidad no pueda demostrarse.

## D-022 — Schema API semántico sin aparentar soporte industrial

- **Estado:** aceptada.
- **Decisión:** separar nombres públicos de las columnas sklearn y exigir JSON estricto con
  categoría L/M/H, números finitos, temperaturas mayores que cero kelvin, y velocidad, torque y
  desgaste no negativos; velocidad y desgaste son enteros. No imponer como límites API las
  envolventes 295–305/305–315/1.000–3.000/0–80/0–260 usadas por M1 para validar la fuente.
- **Motivo:** conserva D-009 y evita presentar extremos de un generador sintético como límites
  físicos. Este contrato valida forma, unidades y signos, no detecta out-of-distribution; la API y
  la interfaz advierten que extrapolar puede producir scores poco fiables.
- **Evolución:** D-026 conserva este schema y añade una abstención por soporte marginal sin tratar
  los extremos observados como validación física ni como detector OOD completo.

## D-023 — Aplicación local mínima y superficie cerrada

- **Estado:** aceptada.
- **Decisión:** añadir FastAPI, Pydantic y Uvicorn como dependencias directas; usar `httpx2` solo en
  el extra `dev` para `TestClient`. Servir HTML/CSS/JavaScript estático sin Jinja2 ni multipart,
  enlazar Uvicorn a `127.0.0.1`, aceptar solo hosts locales y aplicar CSP y headers defensivos. No
  habilitar documentación web dependiente de CDN; conservar solamente el schema OpenAPI JSON.
- **Motivo:** cubre la demo y sus pruebas con el menor grafo funcional, evita recursos externos y
  mantiene M4 dentro del alcance local sin autenticación, base de datos, telemetría ni despliegue.

## D-024 — Cierre reproducible del MVP local

- **Estado:** aceptada.
- **Decisión:** cerrar M5 después de reproducir desde una copia limpia las dependencias, los ocho
  outputs EDA, el run `b15bab7b54bc2e1f` y el SHA-256 exacto del pipeline. La prueba ejecutó
  `174/174` tests y arrancó la aplicación con el holdout inaccesible, sin invocar otra evaluación.
- **Motivo:** una reproducción documentada ofrece evidencia más fuerte que repetir comandos sobre
  el entorno de desarrollo y preserva el contrato de evaluación única.

## D-025 — Separar atribución de datos, licencia del código y autorizaciones

- **Estado:** aceptada.
- **Decisión:** documentar AI4I y sus transformaciones bajo `CC BY 4.0` sin aplicar esa licencia
  automáticamente al código. La licencia del código queda pendiente de una elección del usuario.
  Publicar en GitHub requerirá además un commit curado, revisión final de secretos y autorización
  explícita. Desplegar será una decisión posterior y separada.
- **Motivo:** atribución, licencia de código, publicación y operación son permisos distintos. M5
  puede cerrar el producto local sin presumir ninguno de ellos.
- **Evolución:** D-029 registra la elección posterior de MIT y la autorización separada de M6.

## D-026 — Abstención marginal y endurecimiento adversarial del contrato API

- **Estado:** aceptada.
- **Decisión:** mantener el schema semántico de D-022 y añadir antes de inferencia una comprobación
  inclusiva contra los extremos exactos observados en AI4I: aire `295.3`–`304.5` K, proceso
  `305.7`–`313.8` K, velocidad `1168`–`2886` rpm, torque `3.8`–`76.6` Nm y desgaste `0`–`253` min.
  Una observación interior conserva `domain_status = "within_reference_envelope"`,
  `decision_applicable = true` y la
  inferencia normal. Si uno o más campos quedan fuera, responder `200` sin invocar el modelo ni
  decidir: `domain_status = "outside_reference_envelope"`, `decision_applicable = false`,
  `risk_score = null` y
  `predicted_failure = null`, con una advertencia por cada campo infractor. Esta envolvente es una
  referencia educativa de soporte marginal, no un límite físico ni detección OOD conjunta. No
  contradice D-009: una observación exterior sigue siendo válida para el schema y recibe una
  abstención explícita, no un rechazo presentado como regla industrial.

  Endurecer además el transporte: rechazar claves JSON duplicadas con `400`, limitar el body a
  `16 KiB` (`16.384` bytes) con `413`, devolver errores de validación e inferencia como JSON
  controlado y exigir en la UI enteros seguros de JavaScript para velocidad y desgaste
  (`Number.isSafeInteger`, de `0` a `9.007.199.254.740.991`). Este último máximo protege la
  serialización del navegador y no expresa soporte industrial.
- **Motivo:** no asignar score ni decisión a extrapolaciones marginales evidentes y cerrar casos
  adversariales que podían producir ambigüedad, pérdida de precisión o errores no controlados. Es
  un cambio del contrato de servicio: no cambia features, target, pipeline, run, umbral ni métricas,
  no reabre el holdout y no autoriza publicación o despliegue.

## D-027 — Procedencia de la envolvente exclusivamente desde training

- **Estado:** aceptada; precisa la procedencia declarada en D-026 sin cambiar sus valores.
- **Decisión:** definir la envolvente de abstención desde
  `reports/eda/summary.json`, que declara `scope = "training_only"`, `training_rows = 8000` y
  `holdout_profiled = false`. Mantener los cinco pares ya publicados porque coinciden exactamente
  con esos extremos de training. Fijarlos en el código para que inferencia continúe sin leer datos
  ni reportes, y añadir una regresión que los compara con el resumen versionado de training.
- **Motivo:** D-026 describía los valores como extremos de AI4I sin separar con suficiente claridad
  su procedencia del snapshot completo. Aunque en este split los extremos de training coinciden con
  los globales, una regla de aplicabilidad no debe obtener información del holdout. La corrección no
  reabre ningún CSV, no cambia respuestas, pipeline, run, umbral o métricas y no convierte la regla
  marginal en un detector OOD.

## D-028 — Intervalos Wilson derivados de la matriz final congelada

- **Estado:** aceptada.
- **Decisión:** acompañar precision y recall con intervalos Wilson bilaterales del 95 %, calculados
  exclusivamente desde `[[1897, 35], [18, 50]]`, la matriz del recibo final versionado. Precision
  usa `50/85` y obtiene `0.4820101461448797`–`0.6868299449467584`; recall usa `50/68` y obtiene
  `0.619922660101109`–`0.825502593301211`. `/model-info` los deriva en memoria de la matriz cuya
  integridad ya valida el loader; el recibo final y su ledger permanecen inmutables.
- **Motivo:** el holdout contiene solo `68` positivos y `85` predicciones positivas. Mostrar la
  incertidumbre por soporte finito evita una lectura excesivamente precisa de las estimaciones sin
  volver a abrir observaciones o scores. Los intervalos no corrigen sesgos, shift, dependencia ni
  la naturaleza sintética; no son incertidumbre por predicción y no se usan para elegir o modificar
  modelo, features, umbral o claims.

## D-029 — Licencia y autorización separada para M6

- **Estado:** aceptada el `2026-08-19`; evoluciona D-025.
- **Decisión:** licenciar el código bajo MIT, manteniendo separada la atribución y licencia
  `CC BY 4.0` de AI4I. Preparar y ejecutar la publicación en GitHub y una demo pública educativa,
  stateless y de superficie mínima. No incluir datasets, particiones o inputs almacenados en la
  imagen ni presentar el despliegue como un sistema industrial o de producción.
- **Motivo:** el usuario autorizó expresamente ambas acciones después del cierre local. La licencia,
  la publicación y el despliegue siguen siendo decisiones conceptualmente separadas; la
  autorización no descongela modelo, features, split, umbral o métricas y no permite reabrir el
  holdout. La demo solo se declarará disponible después de verificar el endpoint desplegado.
