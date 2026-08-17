# Tareas

## M0 — Preparación local

- [ ] Confirmar Python 3.12 y crear `.venv`.
- [ ] Crear `pyproject.toml` con dependencias mínimas.
- [ ] Crear estructura `src/`, `tests/`, `notebooks/`, `data/`, `reports/` y `artifacts/`.
- [ ] Configurar Ruff y pytest.
- [ ] Documentar comandos reproducibles para Windows PowerShell.

## M1 — Ingesta y validación

- [ ] Descargar AI4I 2020 mediante código desde una fuente oficial.
- [ ] Registrar URL, licencia, fecha de obtención y checksum.
- [ ] Validar nombres, tipos, categorías, nulos, duplicados y target.
- [ ] Prohibir identificadores y columnas de modos de fallo como features.
- [ ] Añadir pruebas automatizadas de datos.

## M2 — Auditoría y EDA

- [ ] Crear un análisis breve y reproducible.
- [ ] Medir prevalencia de la clase positiva.
- [ ] Generar únicamente gráficos que aporten a una decisión.
- [ ] Documentar riesgos de leakage y sesgos del dataset sintético.

## M3 — Modelado y evaluación

- [ ] Reservar holdout estratificado antes de seleccionar modelos.
- [ ] Entrenar `DummyClassifier`.
- [ ] Entrenar regresión logística y random forest mediante pipelines.
- [ ] Seleccionar con validación cruzada sobre training.
- [ ] Elegir umbral sin consultar test.
- [ ] Evaluar una sola vez sobre holdout y guardar resultados.

## M4 — Aplicación local

- [ ] Crear `GET /health`, `GET /model-info` y `POST /predict`.
- [ ] Añadir validación estricta de inputs.
- [ ] Crear formulario HTML/CSS local con advertencia educativa.
- [ ] Añadir pruebas de API e inferencia.

## M5 — Cierre del MVP local

- [ ] Crear ficha del modelo y resultados reproducibles.
- [ ] Completar README con arquitectura y limitaciones.
- [ ] Ejecutar el proyecto desde cero siguiendo solo la documentación.
- [ ] Revisar secretos, licencias, datos y artefactos antes de considerar publicación.
- [ ] Solicitar autorización separada para GitHub o despliegue.
