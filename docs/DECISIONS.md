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
