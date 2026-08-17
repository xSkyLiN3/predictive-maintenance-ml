# Hoja de ruta local

Cada hito tiene una puerta de revisión. No se debe comenzar el siguiente solo porque queden tokens o tiempo disponibles.

## M0 — Entorno y estructura

**Resultado:** proyecto Python 3.12 reproducible, todavía sin modelado.

- Crear `.venv` explícitamente con Python 3.12.
- Definir `pyproject.toml` y dependencias mínimas.
- Crear estructura `src/`, `tests/`, `data/`, `notebooks/`, `reports/` y `artifacts/`.
- Configurar Ruff y pytest.
- Documentar comandos PowerShell.

**Puerta:** una instalación limpia importa el paquete y ejecuta una prueba mínima.

## M1 — Ingesta y contrato de datos

**Resultado:** dataset oficial descargado y validado mediante código.

- Descargar desde UCI.
- Guardar metadatos y checksum.
- Verificar esquema, tipos, categorías, nulos, duplicados y target.
- Implementar lista explícita de features permitidas y columnas prohibidas.
- Crear pruebas de regresión del esquema.

**Puerta:** Ruff y pytest pasan; ninguna columna de leakage puede entrar accidentalmente al entrenamiento.

## M2 — EDA y protocolo cerrado

**Resultado:** informe breve que sustenta las decisiones de modelado.

- Medir prevalencia y revisar rangos.
- Crear 5–7 visualizaciones relevantes.
- Identificar anomalías y limitaciones.
- Materializar la partición estratificada reproducible.
- Confirmar por escrito métricas y estrategia de umbral.

**Puerta:** revisión humana del informe antes de entrenar modelos candidatos.

## M3 — Baseline, modelos y evaluación

**Resultado:** modelo seleccionado y evaluado limpiamente.

- Entrenar Dummy, logística y random forest.
- Comparar mediante validación cruzada solo en training.
- Elegir umbral sin consultar test.
- Evaluar una vez en holdout.
- Guardar pipeline, configuración, métricas y gráficos.

**Puerta:** ejecución repetida con la misma configuración reproduce los resultados dentro de tolerancias declaradas.

## M4 — API e interfaz local

**Resultado:** demo funcional en localhost.

- Implementar `/health`, `/model-info` y `/predict`.
- Validar inputs con esquemas estrictos.
- Crear interfaz HTML/CSS sencilla servida localmente.
- Mostrar probabilidad, decisión y advertencia educativa.
- Añadir pruebas de API e inferencia.

**Puerta:** un usuario puede levantar la aplicación siguiendo el README y completar el flujo principal.

## M5 — Cierre y revisión de publicación

**Resultado:** candidato a portfolio, aún local.

- Completar README, ficha del modelo y arquitectura.
- Verificar instalación desde cero.
- Ejecutar suite completa y revisión de secretos/licencias.
- Revisar accesibilidad y claridad de la interfaz.
- Comparar claims del README con evidencia generada.

**Puerta:** decisión explícita del usuario sobre publicación en GitHub y, por separado, despliegue.
