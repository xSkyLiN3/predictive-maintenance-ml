# Instrucciones del proyecto para Codex

## Misión

Construir un proyecto pequeño pero completo de ingeniería de machine learning con el dataset AI4I 2020. El resultado debe ser comprensible, reproducible y defendible en una entrevista técnica.

## Alcance vigente

- Trabajar únicamente en local hasta que el usuario autorice expresamente publicar o desplegar.
- No crear repositorios remotos, hacer push, modificar el VPS ni configurar dominios.
- No añadir Docker, WSL, servicios cloud, bases de datos, autenticación, MLflow, DVC, Airflow, Kubernetes, LLM ni GPU durante el MVP local.
- Implementar un hito cada vez y verificarlo antes de continuar.

## Descripción honesta

- Llamar al producto `Machine Failure Risk Classifier` o clasificador de riesgo de fallo por observación.
- No llamarlo plataforma industrial ni sistema que predice la vida útil restante.
- Indicar siempre que AI4I 2020 es sintético y que los resultados no validan uso industrial real.
- No inventar métricas, resultados, usuarios ni impacto.

## Integridad de datos y ML

- Target: `Machine failure`.
- Features candidatas: tipo de producto, temperatura del aire, temperatura del proceso, velocidad rotacional, torque y desgaste de herramienta.
- Excluir identificadores: `UDI` y `Product ID`.
- Excluir obligatoriamente `TWF`, `HDF`, `PWF`, `OSF` y `RNF`: son indicadores de modos de fallo y producirían fuga de información respecto del target.
- Separar un holdout estratificado del 20 % antes de seleccionar modelos o umbrales.
- Usar solamente entrenamiento y validación cruzada para comparar modelos.
- Encapsular transformaciones y modelo en un `Pipeline` de scikit-learn.
- Mantener el test sin consultar hasta la evaluación final del modelo elegido.
- Elegir el umbral con validación o predicciones out-of-fold, nunca mirando el test.
- Usar semillas fijas y registrar versiones, parámetros y resultados.
- No aplicar SMOTE en el primer MVP; comenzar con modelos comprensibles y `class_weight` cuando corresponda.

## Evaluación

- Baseline obligatorio: `DummyClassifier`.
- Modelos iniciales: regresión logística y random forest.
- Métrica principal: Average Precision.
- En el umbral declarado: precision, recall y F1.
- Reportar también matriz de confusión y ROC-AUC como métrica secundaria.
- Accuracy puede mostrarse solo con contexto; nunca será la métrica principal.
- No fijar por adelantado una cifra objetivo atractiva. Publicar el resultado real y compararlo con el baseline.

## Ingeniería

- Usar Python 3.12 mediante `.venv`; no depender del `python` o `pip` global.
- Mantener estructura `src/` y pruebas en `tests/`.
- Preferir módulos pequeños, type hints y funciones puras cuando sea razonable.
- Validar explícitamente el esquema, tipos, categorías, rangos y target.
- Ruff debe pasar y pytest debe cubrir al menos descarga/validación, entrenamiento mínimo e inputs de API cuando exista.
- Los datos descargados, artefactos binarios, secretos y entornos virtuales no se versionan.
- Los resultados reproducibles, gráficos finales y documentación sí se versionan.
- Justificar cualquier nueva dependencia antes de incorporarla.

## Flujo de trabajo

1. Leer `README.md`, `docs/PROJECT_PROPOSAL.md`, `docs/DATA_EVALUATION.md`, `docs/ROADMAP.md` y `TASKS.md`.
2. Confirmar el hito activo y no implementar fases posteriores por anticipado.
3. Ejecutar las comprobaciones relevantes después de cada cambio.
4. Actualizar `TASKS.md` y `docs/DECISIONS.md` cuando cambie el estado o se tome una decisión material.
5. Al terminar un hito, informar archivos modificados, comandos ejecutados, resultados, limitaciones y siguiente paso recomendado.

## Regla de parada

Ante una decisión que altere el problema, las features, el target, el protocolo de evaluación o el alcance de publicación, detenerse y pedir confirmación. Los errores ordinarios de implementación deben resolverse y verificarse sin trasladarlos al usuario prematuramente.
