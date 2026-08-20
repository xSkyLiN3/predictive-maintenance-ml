# Datos

Los datos no se incorporan manualmente ni se versionan.

- `raw/`: copia inmutable obtenida desde UCI mediante el comando implementado en M1.
- `processed/`: `train.csv` y `holdout.csv`, derivados reproducibles e ignorados por Git. Contienen
  únicamente las seis features permitidas y el target.

El proceso de descarga debe registrar fuente, licencia, fecha, tamaño y SHA-256.

## Snapshot oficial fijado en M1

- Página UCI: <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- DOI: <https://doi.org/10.24432/C5HS5C>
- Licencia de datos: CC BY 4.0.
- Cita recomendada: *AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine
  Learning Repository.* <https://doi.org/10.24432/C5HS5C>.
- ZIP: 522.170 bytes; SHA-256
  `f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252`.
- CSV extraído: 522.048 bytes; SHA-256
  `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

La fecha real de obtención y estos mismos datos de procedencia quedan en
`data/raw/download_metadata.json`. Ese archivo, el ZIP y el CSV son locales y están ignorados
por Git. Las constantes versionadas viven en `predictive_maintenance.dataset`.

Desde la raíz del repositorio, descarga, extracción segura y validación del contrato M1 para el
snapshot fijado:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance download
```

Validación posterior completamente offline:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance validate
```

El código nunca sustituye silenciosamente un snapshot local: si tamaño o checksum no
coinciden, se detiene y exige una revisión explícita.

El proyecto no modifica ni versiona el CSV original. Sí crea particiones locales, perfiles,
figuras, métricas y modelos derivados; estos cambios y la atribución completa se documentan en
[../docs/DATA_ATTRIBUTION.md](../docs/DATA_ATTRIBUTION.md).

## Partición M2

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance split
```

La partición usa 80 % training y 20 % holdout estratificado con semilla fija `42`. El manifiesto
versionado [split_manifest.json](split_manifest.json) permite reconstruir y verificar ambos CSV.
El holdout permaneció local, ignorado y sin perfilar desde la partición hasta la evaluación final
de M3. Esa evaluación ya se consumió una vez; el recibo está en `reports/modeling/` y el ledger
global que impide otro consumo está en `reports/holdout_access/`.

La EDA posterior verifica y carga solo `train.csv`:

```powershell
.\.venv\Scripts\python.exe -m predictive_maintenance eda
```

Sus artefactos versionables se describen en [../reports/README.md](../reports/README.md).
