# Informes

Este directorio contiene resultados, gráficos y documentación reproducibles y revisados.
Los archivos temporales deben guardarse en `reports/tmp/`, que Git ignora.

La EDA de M2 está en [eda/EDA_REPORT.md](eda/EDA_REPORT.md), con un resumen JSON y seis figuras
generadas exclusivamente desde la partición de training.

Se regenera desde la raíz, después de materializar el split, mediante:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance eda
```

M3 está en [modeling/b15bab7b54bc2e1f/M3_REPORT.md](modeling/b15bab7b54bc2e1f/M3_REPORT.md).
Incluye los recibos de CV y umbral OOF, dos figuras de selección y la matriz de confusión final.
El run eligió random forest, obtuvo AP media CV `0.643812` y AP final `0.649538`.

`holdout_access/<sha256>.json` es el ledger global y versionable de la única evaluación. No debe
eliminarse ni editarse: bloquea que otro run o directorio de artefactos vuelva a abrir el mismo
holdout. Los reruns compatibles validan y reutilizan el recibo final.
