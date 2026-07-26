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
python scripts\generate_dataset_figures.py
python scripts\generate_benchmark_figures.py --benchmark-dir D:\A_projects\swir_rgb_dataset_benchmark\artifacts\benchmarks\m_swir_v3
```

## Build manuscripts

```powershell
python C:\Users\dx254\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py main_en.tex --compiler texlive --engine xelatex --output-directory build\en
python C:\Users\dx254\.codex\plugins\cache\openai-bundled\latex\0.2.4\scripts\compile_latex.py main_zh.tex --compiler texlive --engine xelatex --output-directory build\zh
```

Dataset DOI, public repository URL, license, final authorship, funding, and
author confirmation remain explicit submission gates. The manuscript benchmark
values are generated from the audited 69-run result bundle.
