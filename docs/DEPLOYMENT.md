# Contenedor y despliegue de la demo

Este documento cubre la imagen de la demo `v1.0.0`. No registra un despliegue ya realizado ni
convierte el clasificador educativo en un servicio industrial.

## Garantías del build

El `Dockerfile` usa tres etapas:

1. construye el wheel del paquete;
2. descarga el snapshot AI4I fijado, reproduce la partición y ejecuta solamente `train` con
   Python `3.12.0` y versiones fijadas;
3. instala el wheel y copia el pipeline verificado a una imagen de runtime separada.

El build **no ejecuta** `evaluate-holdout`. `split` materializa temporalmente `train.csv` y
`holdout.csv` en la etapa de construcción porque ambos hashes forman parte del contrato, pero el
comando de entrenamiento solo resuelve `train.csv`. Los diagnósticos regenerados se escriben en
un directorio temporal: los PNG de Matplotlib no son byte a byte portables entre Windows y Linux.
Después, el loader valida el pipeline reconstruido contra los recibos versionados de selección y
evaluación final; los diagnósticos temporales se descartan. Ningún CSV raw, de training o de
holdout se copia a la imagen final.

Las dos imágenes base están fijadas por digest. La etapa histórica de entrenamiento usa Python
`3.12.0` porque la versión de parche forma parte del identificador del run congelado; el runtime
usa una imagen Python 3.12 más reciente y fijada por separado. Las dependencias Python se fijan en
`requirements/constraints-py312.txt`. Esto busca una reconstrucción controlada por versiones, no
promete que dos builds produzcan una imagen byte a byte idéntica.

El build necesita acceso saliente a Docker Hub, PyPI y la fuente de UCI. Si la fuente no está
disponible o cambia su checksum, falla cerrado.

## Construir y ejecutar localmente

Desde la raíz del repositorio:

```bash
docker build --tag machine-failure-risk-classifier:1.0.0 .
docker run --rm \
  --name machine-failure-demo \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --publish 127.0.0.1:8000:8000 \
  machine-failure-risk-classifier:1.0.0
```

Abrir <http://127.0.0.1:8000>. La imagen se ejecuta como UID/GID `10001`, declara un healthcheck
y no necesita un volumen escribible.

Comprobaciones útiles:

```bash
docker inspect --format '{{.State.Health.Status}}' machine-failure-demo
docker exec machine-failure-demo id
docker exec machine-failure-demo test ! -e /app/data
```

## Variables de entorno

| Variable | Valor en la imagen | Propósito |
|---|---|---|
| `MACHINE_FAILURE_HOST` | `0.0.0.0` | Dirección en la que escucha Uvicorn dentro del contenedor. |
| `MACHINE_FAILURE_PORT` | `8000` | Puerto interno, entre `1` y `65535`. |
| `MACHINE_FAILURE_ALLOWED_HOSTS` | `127.0.0.1,localhost` | Lista explícita de valores `Host` aceptados. |

Para usar otro puerto interno:

```bash
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --publish 127.0.0.1:8080:8080 \
  --env MACHINE_FAILURE_PORT=8080 \
  machine-failure-risk-classifier:1.0.0
```

`MACHINE_FAILURE_ALLOWED_HOSTS` no acepta `*`, esquemas ni rutas. En un proxy para
`ml.nightstrike.cloud`, por ejemplo, debe incluir el dominio:

```bash
--env MACHINE_FAILURE_ALLOWED_HOSTS=ml.nightstrike.cloud,127.0.0.1,localhost
```

El proxy debe preservar `Host`, terminar HTTPS y aplicar límites de frecuencia y tamaño. La API
no tiene autenticación, persistencia ni rate limiting propio; por ello el contenedor no debe
publicarse directamente en Internet. La aplicación no guarda los inputs, pero los logs del proxy
y de la plataforma deben configurarse de manera coherente con esa política.

## CI

`.github/workflows/ci.yml` ejecuta:

- Ruff, formato, `pip check` y pytest en Python 3.12 sobre Ubuntu y Windows;
- build de `sdist` y `wheel`, instalación aislada y suite completa sobre cada distribución;
- build real de la imagen y escaneo Trivy de vulnerabilidades `HIGH`/`CRITICAL` con corrección
  disponible;
- espera del healthcheck, smoke de `/health`, comprobación de usuario no root y ausencia de los
  CSV en runtime.

Las Actions oficiales están fijadas por SHA y anotadas con su versión. El workflow solo solicita
`contents: read`; no publica paquetes, imágenes ni despliega infraestructura.

## Límites operacionales

- Es una demo educativa con un dataset sintético, no un sistema de mantenimiento.
- No hay garantía de disponibilidad, calibración, detección conjunta OOD ni monitorización de
  deriva.
- Un fallo de arranque indica que el pipeline o alguno de sus recibos no pasó la validación de
  integridad; no se debe desactivar ese fallo cerrado.
- Cambiar dependencias de NumPy, pandas, scikit-learn o joblib exige reconstruir y revisar de forma
  explícita la compatibilidad del artefacto.
