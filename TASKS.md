# Tareas

## M0 — Preparación local

- [x] Confirmar Python 3.12 y crear `.venv`.
- [x] Crear `pyproject.toml` con dependencias mínimas.
- [x] Crear estructura `src/`, `tests/`, `notebooks/`, `data/`, `reports/` y `artifacts/`.
- [x] Configurar Ruff y pytest.
- [x] Documentar comandos reproducibles para Windows PowerShell.

## M1 — Ingesta y validación

- [x] Descargar AI4I 2020 mediante código desde una fuente oficial.
- [x] Registrar URL, licencia, fecha de obtención y checksum.
- [x] Validar nombres, tipos, categorías, nulos, duplicados y target.
- [x] Prohibir identificadores y columnas de modos de fallo como features.
- [x] Añadir pruebas automatizadas de datos.

## M2 — Auditoría y EDA

- [x] Materializar y verificar el holdout estratificado antes de la EDA.
- [x] Crear un análisis breve y reproducible usando solamente training.
- [x] Medir prevalencia y rangos en training.
- [x] Generar seis gráficos que aportan a decisiones de modelado.
- [x] Documentar riesgos de leakage y sesgos del dataset sintético.
- [x] Congelar semilla, validación cruzada, métricas y estrategia de umbral OOF.

## M3 — Modelado y evaluación

- [x] Entrenar `DummyClassifier`.
- [x] Entrenar regresión logística y random forest mediante pipelines.
- [x] Seleccionar con validación cruzada sobre training.
- [x] Elegir umbral sin consultar test.
- [x] Evaluar una sola vez sobre holdout y guardar resultados.

## M4 — Aplicación local

- [x] Crear `GET /health`, `GET /model-info` y `POST /predict`.
- [x] Añadir validación estricta de inputs.
- [x] Crear formulario HTML/CSS local con advertencia educativa.
- [x] Añadir pruebas de API e inferencia.
- [x] Abstenerse de puntuar fuera de la envolvente marginal AI4I y explicarlo en la respuesta.
- [x] Endurecer entradas adversariales, transporte JSON y enteros del navegador.

## M5 — Cierre del MVP local

- [x] Crear ficha del modelo y resultados reproducibles.
- [x] Completar README con arquitectura y limitaciones.
- [x] Ejecutar el proyecto desde cero siguiendo solo la documentación.
- [x] Revisar secretos, licencias, datos y artefactos antes de considerar publicación.
- [x] Dejar GitHub y despliegue sujetos a autorizaciones explícitas y separadas.

## M6 — Publicación y demo educativa

- [x] Registrar la autorización explícita de publicación y despliegue.
- [x] Adoptar licencia MIT y preparar metadata semántica `1.0.0`.
- [x] Cerrar los matices metodológicos sin volver a leer el holdout.
- [ ] Verificar CI en Windows/Linux y un contenedor runtime mínimo, no root y sin datos.
- [x] Preparar README público, captura y changelog.
- [ ] Auditar el snapshot staged, dependencias, secretos y artefactos.
- [ ] Publicar repositorio y release `v1.0.0` en GitHub.
- [ ] Desplegar la demo en `ml.nightstrike.cloud` y ejecutar smoke tests externos.
- [ ] Añadir el proyecto verificado al portfolio personal.
