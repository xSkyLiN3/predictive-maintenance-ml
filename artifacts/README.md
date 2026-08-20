# Artefactos locales

Esta carpeta contiene pipelines y metadatos locales generados por entrenamiento. Los binarios del
modelo permanecen fuera de Git durante el desarrollo local.

M3 usa `artifacts/m3/<run_id>/` para el pipeline, su manifiesto y un bundle local de recuperación.
Todo ese contenido se ignora en Git. `train` reconstruye el pipeline y su manifiesto bajo el mismo
entorno; no reconstruye el bundle final. Perder ese bundle no autoriza otra lectura del holdout:
el recibo y el ledger versionados siguen siendo la fuente de verdad.

Los resultados JSON, gráficos y documentación revisables viven en `reports/modeling/` y sí se
versionan. El ledger que impide volver a abrir el mismo holdout está separado del run, indexado
por el SHA-256 del holdout y versionado en `reports/holdout_access/`.
