# Revisión de portfolio y publicación

**Fecha:** 2026-08-17
**Actualización adversarial:** 2026-08-18
**Actualización de release:** 2026-08-19
**Alcance:** evidencia del cierre M5 y puerta técnica para el release educativo M6.

## Actualización M6

El usuario autorizó de forma explícita preparar y ejecutar la publicación y el despliegue. Se
adoptó MIT para el código, separada de la `CC BY 4.0` del dataset, y se fijó la versión pública
`1.0.0`. El modelo, las features, el split, el umbral y la evaluación permanecen congelados.

M6 añade CI Windows/Linux, empaquetado, un contenedor runtime sin datos y una demo educativa
stateless. La envolvente de abstención quedó ligada al resumen versionado `training_only`; los
intervalos Wilson de precision y recall se derivan de la matriz final ya registrada. Ninguno de
estos cambios vuelve a leer el holdout ni modifica el resultado M3.

La primera validación Linux demostró que los PNG de Matplotlib y la serialización Joblib no son
idénticos byte a byte entre Windows y Linux. El runtime no relaja los hashes ni reemplaza el
modelo: distribuye el pipeline exacto evaluado de 1,25 MB, comprueba su SHA-256 antes de cargarlo y
mantiene dataset, particiones y artefactos aportados por usuarios fuera de la imagen.

Las conclusiones de autorización/estado escritas durante M5 y conservadas más abajo son evidencia
histórica de aquella puerta; quedan evolucionadas por D-029 y por la auditoría final de M6.

Evidencia local de release sobre el snapshot M6:

- `pip check`, Ruff y formato pasan; Ruff verificó `46` archivos.
- pytest pasa `227/227` pruebas.
- wheel y sdist construidos con `setuptools 84.0.0` pasan `227/227` pruebas cada uno.
- OSV no reporta advisories activos en los `45` pins auditados; los pins de PyPI no están yanked.
- Gitleaks `8.30.1`, descargado desde el release oficial y verificado por SHA-256, no encontró
  secretos en el snapshot ni en todo el historial alcanzable.
- La captura pública es un PNG `1440 × 1100`, sin metadata ni strings sensibles detectados.
- GitHub Actions pasó calidad en Windows/Linux, instalación de wheel/sdist, build Docker, Trivy
  sin vulnerabilidades `HIGH`/`CRITICAL` corregibles y smoke endurecido del runtime.
- El repositorio es público y la demo HTTPS fue verificada por IPv4 e IPv6, con health, inferencia,
  abstención, límites de cuerpo/frecuencia y regresión de los servicios existentes.

## Veredicto de cierre M5

El MVP local es reproducible y apto para revisión técnica. La documentación, los recibos, las
métricas y la aplicación concilian con el run congelado `b15bab7b54bc2e1f`. No se volvió a
evaluar el holdout durante M5.

En el cierre de M5, el repositorio todavía no estaba autorizado ni listo para publicación: faltaba
elegir licencia y curar el commit del MVP. Esas condiciones motivaron M6; no describen su estado
actual.

## Reconstrucción limpia

Se creó una copia temporal aislada sin `.venv`, cachés, datos locales ni artefactos binarios. La
secuencia se ejecutó siguiendo solamente el README con CPython `3.12.0` y pip `26.2.1`:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade "pip==26.2.1"
.\.venv\Scripts\python.exe -m pip install -c requirements\constraints-win-py312.txt -e ".[dev]"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m predictive_maintenance download
.\.venv\Scripts\python.exe -m predictive_maintenance validate
.\.venv\Scripts\python.exe -m predictive_maintenance split
.\.venv\Scripts\python.exe -m predictive_maintenance eda
.\.venv\Scripts\python.exe -m predictive_maintenance train
```

Resultados:

- las `40` versiones del archivo de constraints coincidieron y `pip check` no halló conflictos;
- Ruff y el control de formato pasaron;
- pytest pasó `174/174` pruebas;
- los ocho outputs de EDA fueron byte a byte idénticos;
- se reprodujo el run `b15bab7b54bc2e1f`;
- se reprodujo exactamente el pipeline SHA-256
  `8f383492fff0a1199a7f62289651a29da39f4c6a149762aa9b75c099efc1568a`;
- los reportes versionados, el puntero activo y el ledger permanecieron byte a byte idénticos;
- Uvicorn arrancó con el holdout inaccesible para la aplicación y `/health`, `/model-info` y
  `/predict` respondieron `200` con identidad y umbral correctos.

No se ejecutó `evaluate-holdout`: el recibo y el ledger existentes siguieron siendo la fuente de
verdad. El archivo de holdout de la copia limpia se mantuvo bajo un bloqueo exclusivo durante
training y el smoke de la aplicación.

Tiempos orientativos de esa máquina, no benchmarks: creación de venv `8,472 s`, instalación
`88,836 s`, pytest `27,769 s`, descarga `3,340 s`, validación `1,993 s`, split `2,067 s`, EDA
`4,025 s` y training `13,114 s`.

Identidades principales:

| Elemento | SHA-256 |
|---|---|
| CSV fuente | `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e` |
| Training | `3b114192f249951632f4c700c07b5edf4306fcff89ac90abe556f15687cf803a` |
| Holdout | `50a1c9c07a57afbc6f34dd112852b61a44f81b6a83341241dd1bc7079f3ac4b7` |
| Pipeline | `8f383492fff0a1199a7f62289651a29da39f4c6a149762aa9b75c099efc1568a` |

## Endurecimiento adversarial posterior

El 2026-08-18 se revisaron entradas inesperadas sin abrir datos, reentrenar ni volver a evaluar el
holdout. La API local pasó inicialmente a la versión `0.2.0`; M6 la fija como `1.0.0`. Las
nuevas regresiones cubren extremos marginales inclusivos, abstención sin invocar el pipeline,
enteros de tamaño arbitrario, claves JSON duplicadas, cuerpos mayores que `16 KiB`, valores no
finitos, excepciones internas, matrices de probabilidad inválidas y cabeceras defensivas.

El smoke con el artefacto real confirmó respuestas controladas para todos esos casos: las
observaciones fuera de la envolvente marginal devuelven `200` con score y decisión nulos; schema
inválido devuelve `422`, JSON ambiguo `400` y body excesivo `413`, siempre como JSON y sin exponer
tracebacks. La UI se verificó contra la API real en los estados aplicable/no aplicable y sus assets
usan versión en la URL para evitar mezclar JavaScript antiguo con un contrato nuevo. Esta revisión
no alteró el run, modelo, features, umbral ni métricas M3.

## Claims y evidencia

Se contrastaron README y Model Card con los JSON, manifiestos, figuras y ledger versionados. El
modelo, umbral, CV, OOF, métricas finales y matriz de confusión coinciden. Los textos declaran que
AI4I es sintético, el score no fue evaluado como probabilidad calibrada y el resultado no acredita
uso industrial, RUL ni causalidad.

La atribución del dataset registra título, UCI, DOI, `CC BY 4.0`, cita recomendada y las
transformaciones realizadas en [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). El CSV original
permanece fuera de Git. El pipeline exacto evaluado sí está versionado en
`artifacts/m3/b15bab7b54bc2e1f/pipeline.joblib`, ligado a su manifiesto y verificado por SHA-256.

## Interfaz y accesibilidad

Se revisó el flujo real en el navegador local tanto en escritorio como en un viewport móvil de
`390 × 844`:

- predicción válida, score, umbral, decisión y advertencias visibles;
- validación inválida con foco en el resumen y `aria-invalid` en el campo;
- sin overflow horizontal ni errores de consola;
- estructura con skip link, landmarks, headings, labels y región live;
- foco visible, targets táctiles y soporte de movimiento reducido;
- contrastes muestreados entre `5,14:1` y `17,56:1`.

Esta revisión no es una certificación WCAG. No se ejecutó una sesión completa con lector de
pantalla ni una auditoría manual a zoom de `200 %`; son verificaciones recomendables antes de una
exposición pública. Los textos alternativos de las figuras EDA se hicieron descriptivos en M5.

## Higiene, secretos y artefactos

- Durante M5 aún no había remoto ni publicación; M6 configuró el repositorio público y completó el
  release `v1.0.0` después de las verificaciones registradas al inicio de este documento.
- La búsqueda estática inicial no encontró claves, tokens, credenciales, correos ni rutas
  personales. Antes de publicar, Gitleaks revisó el snapshot y todo el historial alcanzable sin
  detectar secretos.
- `.gitignore` excluye `.venv`, cachés, datos raw/processed, metadata local y bundles generados.
- `*.joblib`, `*.pkl` y `*.pickle` permanecen ignorados globalmente; solo la ruta nominal del
  pipeline evaluado está permitida de forma explícita.
- Reportes y figuras curados son versionables; suman aproximadamente `990 KB` y el mayor PNG es
  de aproximadamente `247 KB`.
- No se detectaron binarios grandes o inesperados candidatos a Git.

## Licencias

El dataset usa `CC BY 4.0`; su atribución y los cambios derivados están documentados por separado.
Las expresiones siguientes se revisaron en la metadata instalada de las dependencias directas:

| Dependencia directa | Licencia declarada |
|---|---|
| FastAPI | MIT |
| Pydantic | MIT |
| Uvicorn | BSD-3-Clause |
| scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| joblib | BSD-3-Clause |
| httpx2 (dev) | BSD-3-Clause |
| pytest (dev) | MIT |
| Ruff (dev) | MIT |
| Matplotlib | licencia permisiva propia Matplotlib/PSF |
| NumPy | expresión compuesta de licencias permisivas según su metadata |

No se observó una incompatibilidad evidente; esto no es asesoría legal. M6 repite la revisión sobre
el entorno efectivo de release. El proyecto usa MIT, declarada en `LICENSE` y `pyproject.toml`, sin
sustituir la `CC BY 4.0` de los datos.

## Puerta de publicación

Estado final M6: **GO completado y publicado**. El modelo permanece congelado y las cuatro puertas
definidas durante M5 quedaron satisfechas:

1. suite limpia y build de paquete en Windows/Linux;
2. auditoría del snapshot, secretos, dependencias y artefactos;
3. CI verde sobre el commit público;
4. smoke tests HTTPS antes de crear el tag `v1.0.0`.

La evidencia se resume al inicio del documento y permanece accesible en el
[release `v1.0.0`](https://github.com/xSkyLiN3/predictive-maintenance-ml/releases/tag/v1.0.0) y la
[demo educativa](https://ml.nightstrike.cloud). Esta publicación no convierte el sistema en un
despliegue industrial ni valida su uso en producción.
