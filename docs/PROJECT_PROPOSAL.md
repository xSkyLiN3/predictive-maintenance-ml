# Propuesta del proyecto

## Resumen ejecutivo

`Machine Failure Risk Classifier` será un proyecto local de clasificación binaria que estima el riesgo de fallo asociado a una observación de operación. Su valor para el portfolio no estará en presentar un modelo como producto industrial, sino en demostrar un proceso completo: datos trazables, prevención de leakage, baseline, evaluación reproducible, servicio de inferencia y documentación honesta.

## Objetivo profesional

Crear la primera evidencia pública potencial de Cristóbal en machine learning sin presentarlo todavía como AI/ML Engineer. El proyecto debe permitir que una persona externa compruebe:

- que entiende el problema y los límites del dataset;
- que puede preparar datos sin contaminar la evaluación;
- que compara un modelo contra un baseline;
- que usa métricas adecuadas para una clase minoritaria;
- que convierte el modelo en software probado y ejecutable;
- que comunica resultados y limitaciones con precisión.

## Definición del problema

### Entrada

Una observación con:

- tipo de producto (`L`, `M` o `H`);
- temperatura del aire;
- temperatura del proceso;
- velocidad rotacional;
- torque;
- desgaste de herramienta.

### Salida

- score de riesgo de `Machine failure` sin afirmar calibración probabilística;
- clasificación según un umbral documentado;
- versión del modelo y advertencia de uso educativo.

### Lo que no predice

- tiempo restante hasta una avería;
- una secuencia futura de estados;
- causa real de un fallo;
- desempeño sobre maquinaria industrial real.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**, UCI Machine Learning Repository, ID 601.

- 10.000 observaciones.
- Datos sintéticos inspirados en escenarios de mantenimiento predictivo.
- Target principal: `Machine failure`.
- Licencia: CC BY 4.0.
- Sin datos personales.

Fuente oficial: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>

Los identificadores `UDI` y `Product ID` no aportan señal generalizable. Los indicadores `TWF`, `HDF`, `PWF`, `OSF` y `RNF` describen modos de fallo relacionados con el target y se excluirán para evitar fuga de información.

## MVP local

1. Descarga y trazabilidad automatizadas del dataset.
2. Validación explícita del esquema.
3. EDA breve y reproducible.
4. Holdout estratificado y protocolo de evaluación cerrado.
5. `DummyClassifier` como baseline.
6. Regresión logística y random forest dentro de pipelines.
7. Selección mediante validación cruzada solo sobre training.
8. Métricas y artefacto del modelo guardados de forma reproducible.
9. API FastAPI local con endpoints de salud, información y predicción.
10. Página HTML/CSS local para probar observaciones.
11. Pruebas automáticas y análisis estático.
12. README, ficha del modelo, resultados y limitaciones.

## Fuera del alcance inicial

- publicación en GitHub;
- VPS, dominio y demo pública;
- Docker;
- autenticación, cuentas o base de datos;
- monitorización y reentrenamiento;
- MLflow, DVC, Airflow o Kubernetes;
- deep learning y uso de GPU;
- LLM, chatbot o generación de explicaciones;
- SHAP por predicción;
- búsqueda extensa de hiperparámetros;
- predicción temporal o vida útil restante.

## Arquitectura implementada

```text
predictive-maintenance-ml/
├── src/predictive_maintenance/
│   ├── artifact_io.py
│   ├── config.py
│   ├── dataset.py
│   ├── validation.py
│   ├── splitting.py
│   ├── eda.py
│   ├── modeling.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── schemas.py
│   ├── api.py
│   └── web/
├── tests/
├── notebooks/
├── data/
├── artifacts/
└── reports/
```

La EDA se implementó como módulo y CLI reproducibles; `notebooks/` queda reservado para vistas
exploratorias opcionales. Toda la lógica reutilizable vive en `src/`.

## Estimación realista

- Preparación: 2–3 horas.
- Datos y auditoría: 4–5 horas.
- Modelado y evaluación: 6–8 horas.
- API e interfaz local: 4–6 horas.
- Pruebas y documentación: 4–6 horas.

Total estimado: **20–28 horas**, distribuido en una o dos semanas de trabajo enfocado.

## Criterios de aceptación del MVP local

- El entorno se reconstruye desde cero siguiendo el README.
- El dataset se obtiene y valida sin pasos manuales.
- Ninguna columna de resultado entra como feature.
- El entrenamiento es reproducible.
- Existe una comparación explícita con baseline.
- El holdout no participa en decisiones de modelo o umbral.
- Se publican métricas reales, no una cifra seleccionada por marketing.
- La API rechaza inputs inválidos.
- La interfaz funciona en localhost.
- Ruff y pytest pasan.
- Las limitaciones son visibles.
- No hay secretos, datos personales ni afirmaciones de producción.

Superar estos criterios habilita una revisión separada para decidir si conviene publicar y desplegar.
