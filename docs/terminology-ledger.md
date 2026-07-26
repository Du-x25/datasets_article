# Terminology ledger

| Canonical term | First-use definition | Do not substitute |
|---|---|---|
| M-SWIR-RGB | paired multiband short-wave infrared and RGB dataset | large-scale dataset |
| source scene | scene-level acquisition group and split unit | patch, independent image |
| RGB | red–green–blue visible image | optical image |
| SWIR | short-wave infrared | generic infrared |
| B1–B5 | five nominal SWIR passbands | channels without wavelength context |
| B124 | channel stack ordered as B1, B2, B4 | three-band SWIR without order |
| five-band input | channel stack ordered as B1–B5 | hyperspectral cube |
| scene-disjoint split | no source scene shared across train, validation, and test | random image split |
| LOSO | leave-one-scene-out evaluation | leave-one-sample-out |
| registration diagnostic | reproducible image-feature residual | manual-checkpoint RMSE |
| GrayRep | deterministic grayscale replication into three RGB channels | learned baseline |
| data usability benchmark | computational evidence that released pairs are learnable | SOTA benchmark |

Nominal wavelength ranges remain B1 1000–1050 nm, B2 1050–1100 nm,
B3 1225–1275 nm, B4 1525–1575 nm, and B5 1625–1675 nm.

