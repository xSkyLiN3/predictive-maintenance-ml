# Hoja de ruta

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

- Materializar primero la partición estratificada reproducible y sellar el holdout.
- Medir prevalencia y revisar rangos.
- Crear 5–7 visualizaciones relevantes.
- Identificar anomalías y limitaciones.
- Confirmar por escrito métricas y estrategia de umbral.

**Puerta:** revisión humana del informe antes de entrenar modelos candidatos.

## M3 — Baseline, modelos y evaluación

**Estado:** completado y verificado en el run `b15bab7b54bc2e1f`.

**Resultado:** modelo seleccionado y evaluado limpiamente.

- Entrenar Dummy, logística y random forest.
- Comparar mediante validación cruzada solo en training.
- Elegir umbral sin consultar test.
- Evaluar una vez en holdout.
- Guardar pipeline, configuración, métricas y gráficos.

**Puerta:** ejecución repetida con la misma configuración reproduce los resultados dentro de tolerancias declaradas.

## M4 — API e interfaz local

**Estado:** completado y verificado sobre el run `b15bab7b54bc2e1f`.

**Resultado:** demo funcional en localhost.

- Implementar `/health`, `/model-info` y `/predict`.
- Validar inputs con esquemas estrictos.
- Crear interfaz HTML/CSS sencilla servida localmente.
- Mostrar score, decisión y advertencia educativa sin afirmar calibración.
- Añadir pruebas de API e inferencia.
- Abstenerse explícitamente fuera de la envolvente marginal educativa, sin presentarla como límite
  físico ni detector OOD completo.
- Rechazar de forma controlada JSON ambiguo o excesivo y evitar pérdida de precisión en la UI.

**Puerta:** un usuario puede levantar la aplicación siguiendo el README, completar el flujo
principal y obtener respuestas controladas ante entradas adversariales.

## M5 — Cierre y revisión de publicación

**Estado:** completado y verificado en local.

**Resultado:** candidato técnico aprobado para preparar publicación.

- Completar README, ficha del modelo y arquitectura.
- Verificar instalación desde cero.
- Ejecutar suite completa y revisión de secretos/licencias.
- Revisar accesibilidad y claridad de la interfaz.
- Comparar claims del README con evidencia generada.

**Puerta:** el MVP local pasó la revisión. La licencia, publicación y demo fueron autorizadas para
M6 el 2026-08-19 y posteriormente superaron la auditoría del snapshot e historial.

## M6 — Publicación y demo educativa

**Estado:** completado y verificado.

**Resultado:** release público `v1.0.0`, CI verificable y demo stateless detrás de HTTPS.

- Adoptar MIT sin mezclarla con la licencia `CC BY 4.0` del dataset.
- Mantener modelo, features, umbral y holdout congelados; corregir solo la trazabilidad de la
  envolvente y contextualizar precision/recall desde la matriz ya versionada.
- Verificar el paquete en Windows y Linux, y construir un runtime no root sin particiones de datos.
- Curar README, changelog, captura, metadata y commits honestos.
- Auditar secretos, dependencias, archivos staged y tamaños antes del push.
- Publicar el repositorio y release en GitHub.
- Desplegar una demo educativa con TLS, límites de recursos y sin persistencia de inputs.
- Integrar el enlace únicamente después de smoke tests externos satisfactorios.

**Puerta superada:** CI verde, release verificable, demo saludable y claims públicos conciliados
con la evidencia versionada. Un proyecto ML posterior será un hito distinto, no una expansión de
este MVP.
