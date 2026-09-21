# M-SWIR-RGB Data Descriptor

Independent bilingual LaTeX manuscript project for a ten-scene paired RGB and
five-band SWIR dataset.

## Authoritative manuscript entries

- `main_en.tex`: English submission manuscript.
- `main_zh.tex`: complete Chinese review manuscript.
- `generated/dataset_macros.tex`: facts generated from the validated data.
- `figures/source_data/`: clean source tables behind quantitative figures.
- `docs/terminology-ledger.md`: canonical terminology for both languages.
- `docs/figure-contracts.md`: conclusion, evidence, integrity, and export rules.

## Build figures

```powershell
conda activate mswir-paper
$env:PYTHONUTF8="1"
python scripts\generate_dataset_figures.py
python scripts\generate_benchmark_figures.py --benchmark-dir D:\A_projects\swir_rgb_dataset_benchmark\artifacts\benchmarks\m_swir_v3dataset_20260921 --stages input model scene
```

`generate_dataset_figures.py` reads the v3 release snapshot
(`D:\Data_dir\SWIR-RGB\datasets_article_auto_v3`) and the v3 audit bundle
(`D:\A_projects\swir_rgb_dataset_pipeline\artifacts\audit_v3`), and writes
`generated/dataset_macros.tex`, the acquisition tables and all dataset figures.
`PYTHONUTF8=1` is required on Windows so that the Chinese table fragments are
written as UTF-8.

## Build manuscripts

```powershell
xelatex -interaction=nonstopmode -output-directory=build\en main_en.tex
bibtex build\en\main_en
xelatex -interaction=nonstopmode -output-directory=build\en main_en.tex
xelatex -interaction=nonstopmode -output-directory=build\en main_en.tex
xelatex -interaction=nonstopmode -output-directory=build\zh main_zh.tex
bibtex build\zh\main_zh
xelatex -interaction=nonstopmode -output-directory=build\zh main_zh.tex
xelatex -interaction=nonstopmode -output-directory=build\zh main_zh.tex
python scripts\qa_manuscripts.py
```

Run `bibtex` from the repository root so that `refs.bib` resolves; the two
extra `xelatex` passes settle cross-references and the bibliography.

Dataset DOI, public repository URL, license, final authorship, funding, and
author confirmation remain explicit submission gates. The manuscript benchmark
values are generated from the audited 69-run result bundle
(`m_swir_v3dataset_20260921`, trained on the v3 task tree).
