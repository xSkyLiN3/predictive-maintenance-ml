# Contenedor y despliegue de la demo

Este documento cubre la imagen de la demo `v1.0.0`. No registra un despliegue ya realizado ni
convierte el clasificador educativo en un servicio industrial.

## Garantías del build

El `Dockerfile` usa dos etapas:

1. construye el wheel del paquete;
2. instala el wheel y el pipeline exacto evaluado en una imagen de runtime separada.

El build no descarga el dataset, no ejecuta `split`, `train` ni `evaluate-holdout`, y no contiene
ningún CSV. Joblib y los PNG de Matplotlib no son byte a byte portables entre Windows y Linux;
reentrenar dentro de la imagen produciría una serialización distinta de la que fijó el recibo
final. Por ello se versionan únicamente el pipeline evaluado de 1,25 MB y su manifiesto. Antes de
cargarlo, el loader reconcilia su SHA-256, run, versiones, recibos de selección, evaluación final y
ledger global. La API no acepta artefactos aportados por usuarios.

Las dos imágenes base están fijadas por digest y las dependencias Python por versión en
`requirements/constraints-py312.txt`. El build necesita acceso saliente a Docker Hub y PyPI, pero
no a UCI. `pip` y `setuptools` se retiran después de instalar y verificar el wheel porque el
servicio no necesita gestores de paquetes en runtime. La reproducción de training sigue
disponible como flujo separado y nunca vuelve a abrir el holdout consumido.

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
