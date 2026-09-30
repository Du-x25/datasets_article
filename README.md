# M-SWIR-RGB — current v3 manuscript

Current revision: 2026-09-30. `main_en.tex` / `main_zh.tex` are synchronized protocol drafts, **not submission-ready manuscripts**. Ten physical sites are author-confirmed; independent annotation, final pools/supports, budgets and formal results remain pending.

Read `docs/最新协议_20260924.md`, `docs/图表与证据合同.md`, and `docs/任务清单与证据缺口.md`. The illustrated operation guide is in `D:/USTC/Gen/outputs/v3_标注手册与最新方案同步_20260924/`.

## Build and QA

```powershell
conda activate mswir-paper
python scripts\qa_manuscripts.py      # protocol split, blank-page, inventory-vs-provenance and citation-key checks
latexmk -xelatex -bibtex -interaction=nonstopmode -outdir=build/en main_en.tex
latexmk -xelatex -bibtex -interaction=nonstopmode -outdir=build/zh main_zh.tex
```

`PYTHONUTF8=1` may be required on Windows for scripts writing Chinese table fragments. `scripts/qa_manuscripts.py` needs `pypdf` (installed in `mswir-paper`); the base environment will fail with `ModuleNotFoundError`.

## Data release consistency

The release snapshot lives at `D:/Data_dir/SWIR-RGB/datasets_article_auto_v3`. Its metadata is audited by `D:/USTC/Gen/v3_pipeline/release_audit/audit_release_metadata.py` (run with the `mswir-data` environment, which has `tifffile`); the audit verifies hashes, per-scene canvases and provenance invariants. `figures/source_data/acquisition_provenance.csv` is a copy of the snapshot metadata and must be re-copied whenever the snapshot metadata changes — `qa_manuscripts.py` fails if the inventory tables and this copy disagree. On 2026-09-30 the C2/C5 provenance rows were corrected from `work/current_v3/{C2,C5}/geometry/geometry.json` and the audit passes; the remaining WARNs (acquisition dates inferred from file mtimes, no capture logs) await author verification.

`figures/fig1_scene_modality_overview.*` is regenerated from the 60 released TIFFs by calling `figure_scene_plate()` from `scripts/generate_dataset_figures.py` (import the module and call the one function; do not run its `main()`, which is intentionally blocked in the active tree). It was last regenerated on 2026-09-30 from the repaired C2/C5 mosaics.

Historical generated benchmark tables/macros, figure source data and patch manifests are retained for traceability and are not included by the current manuscripts. Do not invoke legacy generators into the active tree: use an isolated historical copy. This synchronization does not change TIFFs, raw inputs, existing annotations or the user-modified benchmark figure generator.

Backups and per-file synchronization hashes are in `D:/USTC/Gen/outputs/v3_标注手册与最新方案同步_20260924/`. Publication still requires author fact verification, real evidence, privacy/license review, and working repository/DOI links.
