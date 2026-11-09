# M-SWIR-RGB 数据集论文

本仓库保存《面向跨光谱图像生成的场景级 RGB 与五波段短波红外配对数据集》的论文源稿、必要图表、参考文献、图表统计来源和相关脚本。

2026-11-09 整理版本管理。当前中文正文仍为 2026-10-15 晚的文风修订稿（20 页）；本次仅整理编译路径与仓库范围，不改动论文结果或研究结论。英文稿尚未同步最新中文表达。11 月的新实验报告尚未整合进正文。

## 稿件与图表

- [中文主稿](main_zh.tex)：当前主要修改入口。
- [英文主稿](main_en.tex)：保留现有版本，后续同步。
- [参考文献](refs.bib)、[生成的表格与数值宏](generated/)。
- [中文图表](figures/zh/)：从本地交付包复制的图表及其统计来源，正文直接引用这里；无需交付包即可编译。
- [其他图表及统计来源](figures/)：英文稿、既有图表及补充材料。
- [协议及工作记录](docs/)、[图表与检查脚本](scripts/)。

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

`scripts/qa_manuscripts.py --language zh` 是完整的本地证据核验入口，需要相邻研究工作区 `D:/USTC/Gen` 中的既有结果、支持域与掩膜。绘图、图像筛选及其他历史审计脚本也可能依赖本地实验路径。它们作为研究记录保留；本仓库不是数据与模型的完整复现包。JSON/CSV 中的原路径和哈希保留历史含义。

2026-11-09 已从仅包含暂存文件的独立目录编译中英文稿，输出分别为 20 页与 21 页。中文逐页文本与整理前 PDF 一致；英文按当前已有源稿和图表重新编译，未改写英文正文，旧英文 PDF 仅作为本地历史文件保留。中文稿原有的 `fig:local-modalities-zh` 未定义引用和局部对照图浮动体超页警告仍保留，本次未改写正文或调整版面。

## Git 同步范围

远程目标为 GitHub 私有仓库 `Du-x25/datasets_article`，日常同步 `main` 分支。

以下内容仅保留在本地，不提交或上传：

- 原始图像、完整成图数据、训练/测试数据、保存预测和模型权重。
- `automation_completion_20261015/`、`delivery_zh_20261015/` 及压缩交付包。
- TIFF 导出副本、临时预览、环境目录和编译缓存。

首次同步前，原始 Git 历史和修改前的工作区差异已备份到本地同级目录 `datasets_article_git_backup_20261109/`。待上传的 `main` 历史移除了 `figures/*.tiff`；原来的提交哈希因此变化，其他历史内容保留。历史分支 `legacy-69run` 仅留本地，不推送。

后续更新可执行 `git add`、`git commit`、`git push`，不要使用 `git push --all` 或 `--mirror` 上传本地历史分支与备份引用。提交前运行 `python scripts/check_repository.py` 检查暂存文件与编译依赖。

## 当前证据状态

中文初稿、既有保存预测的指标复核、图表和本地候选发布包已经形成；独立人工几何验证、质量与隐私验收、外部复现及最终数据许可/归档仍按既有记录推进。私有论文仓库同步不代表数据集已公开，也不代表论文已经投稿。

整理前的说明保留在 [历史 README](docs/project_history_before_github_20261109.md)，其中交付包链接、页数与“最新”措辞属于历史记录，以本页入口为准。
