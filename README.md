# M-SWIR-RGB 数据集论文

本仓库保存《面向跨光谱图像生成的场景级 RGB 与五波段短波红外配对数据集》的论文源稿、必要图表、参考文献、图表统计来源和相关脚本。

2026-11-09 中文稿已整合最新正式实验记录，更新摘要、实验章节、图表和结论，并修复图号引用与图表超页。新版 PDF 共 19 页。主线保持数据集描述与使用边界：225 次地点隔离训练、45 次随机窗口对照，以及具有明确保留评价区域的目标地点配对颜色校准。英文稿尚未同步本次结果更新。

## 稿件与图表

- [中文主稿](main_zh.tex)：当前主要修改入口。
- [新版中文 PDF](output/pdf/多波段SWIR_RGB数据集_中文稿_实验整合_20261109.pdf)：经编译及逐页视觉检查的交付版本。
- [英文主稿](main_en.tex)：保留现有版本，后续同步。
- [参考文献](refs.bib)、[生成的表格与数值宏](generated/)。
- [中文图表](figures/zh/)：从本地交付包复制的图表及其统计来源，正文直接引用这里；无需交付包即可编译。
- [其他图表及统计来源](figures/)：英文稿、既有图表及补充材料。
- [协议及工作记录](docs/)、[图表与检查脚本](scripts/)。
- [本次证据筛选与解释边界](docs/revision_20261109.md)、[验证记录](docs/verification_20261109.json)。

图表保留 PDF、SVG 和 PNG，以及用于说明数值和图像选择的 CSV/JSON。高体积 TIFF 导出副本不进入仓库。图表中的缩略图和局部对照属于论文插图，不包含完整原始数据集。

## 编译

使用包含 XeLaTeX、latexmk、BibTeX 和 ctex 的 TeX Live。英文稿使用 fontspec 默认字体；中文字体由 ctex 配置选择。本机使用 TeX Live 2026。

在仓库根目录运行：

```powershell
latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/zh main_zh.tex
latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/en main_en.tex
```

输出为 `build/zh/main_zh.pdf` 与 `build/en/main_en.pdf`。编译只读取本仓库的源稿、文献和图表，不需要训练数据或模型权重。`build/` 和编译临时文件不提交。

## 检查与复算的区别

Python 检查与绘图依赖见 [environment.yml](environment.yml)。

```powershell
conda env create -f environment.yml
conda activate mswir-paper
python scripts/check_tex_structure.py
python scripts/check_repository.py
```

新版中文结果使用本仓库内的最小证据包，可直接复算导出记录并重建图表：

```powershell
python scripts/build_results_20261109.py
latexmk -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/zh main_zh.tex
python scripts/qa_manuscripts.py --language zh
```

绘图在本机使用 Microsoft YaHei；其他平台需安装该字体或在绘图脚本中选择等价中文字体。版面检查需要 Poppler 的 `pdftotext`。只验证数值、不重新绘图可运行 `python scripts/build_results_20261109.py --check`。

本次核验包括 270 项训练完成记录、仅使用验证集的输入选择、逐地点重聚合、校准评价样本数、图板位置与输入身份、全部图表引用、参考文献键和编译版面。最终编译无未定义引用、超页浮动图或 overfull box，并已逐页查看 19 页 PDF。它验证保存记录与文稿的一致性，没有重新训练、推理或从完整预测重算所有像素指标。

数值证据、来源哈希和必要原图板位于 [results_20261109](figures/source_data/results_20261109/)。其中 `evidence_code/` 仅是所审查算法的来源快照，不构成可独立训练的完整工程。`docs/current_protocol.json`、旧结果宏及其他历史脚本仍属于旧版/英文稿证据；英文检查与历史图像审计可能依赖本地研究工作区。本仓库不是数据和模型的完整复现包。

## Git 同步范围

远程目标为 GitHub 私有仓库 `Du-x25/datasets_article`，日常同步 `main` 分支。

以下内容仅保留在本地，不提交或上传：

- 原始图像、完整成图数据、训练/测试数据、保存预测和模型权重。
- `automation_completion_20261015/`、`delivery_zh_20261015/` 及压缩交付包。
- TIFF 导出副本、临时预览、环境目录和编译缓存。

新版交付 PDF 是体积较小的论文成果，保存在 `output/pdf/` 并纳入 Git；`build/` 中的编译缓存仍忽略。

首次同步前，原始 Git 历史和修改前的工作区差异已备份到本地同级目录 `datasets_article_git_backup_20261109/`。待上传的 `main` 历史移除了 `figures/*.tiff`；原来的提交哈希因此变化，其他历史内容保留。历史分支 `legacy-69run` 仅留本地，不推送。

后续更新可执行 `git add`、`git commit`、`git push`，不要使用 `git push --all` 或 `--mirror` 上传本地历史分支与备份引用。提交前运行 `python scripts/check_repository.py` 检查暂存文件与编译依赖。

## 当前证据状态

中文初稿、既有保存预测的指标复核、图表和本地候选发布包已经形成；独立人工几何验证、质量与隐私验收、外部复现及最终数据许可/归档仍按既有记录推进。私有论文仓库同步不代表数据集已公开，也不代表论文已经投稿。

整理前的说明保留在 [历史 README](docs/project_history_before_github_20261109.md)，其中交付包链接、页数与“最新”措辞属于历史记录，以本页入口为准。
