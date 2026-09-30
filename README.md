# M-SWIR-RGB — current v3 manuscript

Current revision: 2026-09-24. `main_en.tex` / `main_zh.tex` are synchronized protocol drafts, **not submission-ready manuscripts**. Ten physical sites are author-confirmed; independent annotation, final pools/supports, budgets and formal results remain pending.

Read `docs/最新协议_20260924.md`, `docs/图表与证据合同.md`, and `docs/任务清单与证据缺口.md`. The illustrated operation guide is in `D:/USTC/Gen/outputs/v3_标注手册与最新方案同步_20260924/`.

Compile each TeX file with XeLaTeX + BibTeX (or latexmk -xelatex), writing to `build/en` and `build/zh`. Run `scripts/qa_manuscripts.py` afterward. Table topics 1–8 remain comparison, hardware, bands, model energy, acquisition, inventory, folds and metrics. New human-evidence figures/tables have HAV identifiers pending final assembly.

Historical generated benchmark tables/macros, figure source data and patch manifests are retained for traceability and are not included by the current manuscripts. Do not invoke legacy generators into the active tree: use an isolated historical copy. This synchronization does not change TIFFs, raw inputs, existing annotations or the user-modified benchmark figure generator.

Backups and per-file synchronization hashes are in `D:/USTC/Gen/outputs/v3_标注手册与最新方案同步_20260924/`. Publication still requires author fact verification, real evidence, privacy/license review, and working repository/DOI links.
