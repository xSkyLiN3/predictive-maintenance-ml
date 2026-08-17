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

Los nombres exactos se validarán contra el archivo oficial antes de codificarlos como contrato definitivo.

## Columnas prohibidas

- Identificadores: `UDI`, `Product ID`.
- Indicadores de modos de fallo: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

Los indicadores de modos de fallo están vinculados directamente a la definición del target. Incluirlos inflaría artificialmente el desempeño y haría que el proyecto no demostrara inferencia útil desde señales operativas.

## Partición

1. Separar una vez un 20 % estratificado como test.
2. Conservar el 80 % restante para entrenamiento y validación cruzada.
3. Ajustar preprocesamiento dentro de cada fold mediante `Pipeline`.
4. Elegir modelo y umbral sin consultar test.
5. Ejecutar una evaluación final sobre test y conservarla como resultado del MVP.

La semilla se definirá en configuración y se reutilizará en todos los componentes compatibles.

## Modelos iniciales

1. `DummyClassifier` como referencia mínima.
2. Regresión logística con preprocesamiento y balance de clases cuando corresponda.
3. Random forest con complejidad controlada.

No se añadirán más algoritmos hasta comprender el error de estos modelos.

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

El umbral no será necesariamente 0,5. Se elegirá sobre validación o predicciones out-of-fold según un criterio declarado antes de tocar test. El MVP no asumirá costos industriales inventados; mostrará el intercambio entre recall y precision y justificará un punto educativo.

## Resultado aceptable

No existe una cifra mínima prefijada para el portfolio. Un resultado es válido si:

- supera de forma consistente al baseline;
- fue obtenido sin leakage;
- es reproducible;
- se informa completo, incluso si es modesto;
- sus limitaciones quedan claras.
