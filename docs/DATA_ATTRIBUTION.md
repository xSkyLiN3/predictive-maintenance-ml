# Atribución de datos y transformaciones

## Dataset fuente

- Título: **AI4I 2020 Predictive Maintenance Dataset**.
- Fuente: [UCI Machine Learning Repository, dataset 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maint).
- DOI: <https://doi.org/10.24432/C5HS5C>.
- Licencia: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
  (`CC BY 4.0`).
- Artículo introductorio indicado por UCI: S. Matzka (2020), *Explainable Artificial
  Intelligence for Predictive Maintenance Applications*.

Cita recomendada por UCI:

> AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository.
> <https://doi.org/10.24432/C5HS5C>.

AI4I es un dataset sintético. La atribución a UCI y la licencia anterior corresponden al dataset,
no constituyen una validación de este proyecto ni de sus resultados.

## Cambios y material derivado

Este proyecto no modifica ni redistribuye el CSV original en Git. El comando de ingesta conserva
una copia local inmutable y verifica su identidad mediante tamaño y SHA-256. A partir de esa copia
se realizan estas transformaciones:

- validación del esquema y perfil descriptivo;
- selección mediante allowlist de seis variables y exclusión de identificadores/modos de fallo;
- partición estratificada reproducible 80/20 en archivos locales;
- estadísticas y figuras de EDA calculadas únicamente sobre training;
- validación cruzada, predicciones OOF, selección de modelo y umbral;
- métricas y figuras de una evaluación final única del holdout;
- entrenamiento de un pipeline y exposición mediante una demo educativa.

Los reportes, figuras y código producidos por este proyecto se publican bajo la licencia MIT. El
pipeline binario se reconstruye localmente y no se versiona. La licencia MIT del proyecto es
independiente de la licencia `CC BY 4.0` que UCI declara para el dataset AI4I.

## Snapshot utilizado

- ZIP: 522.170 bytes; SHA-256
  `f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252`.
- CSV: 522.048 bytes; SHA-256
  `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

Los archivos originales permanecen ignorados por Git. Los detalles técnicos de descarga y
verificación están en [../data/README.md](../data/README.md).
