# Artefactos del modelo

Esta carpeta contiene pipelines y metadatos generados por entrenamiento. Los artefactos de runs
experimentales permanecen fuera de Git durante el desarrollo local.

M3 usa `artifacts/m3/<run_id>/` para el pipeline, su manifiesto y un bundle local de recuperación.
Para el run público congelado `b15bab7b54bc2e1f` se versionan únicamente `pipeline.joblib` y
`artifact_manifest.json`; el bundle local de recuperación continúa ignorado. El pipeline exacto
es necesario porque una reserialización Joblib en otro sistema operativo no conserva su SHA-256.
`train` sigue permitiendo reproducir el flujo sin reconstruir el bundle final. Perder ese bundle
no autoriza otra lectura del holdout: el recibo y el ledger versionados siguen siendo la fuente de
verdad.

Los resultados JSON, gráficos y documentación revisables viven en `reports/modeling/` y sí se
versionan. El ledger que impide volver a abrir el mismo holdout está separado del run, indexado
por el SHA-256 del holdout y versionado en `reports/holdout_access/`.
