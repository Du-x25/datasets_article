# 项目使用与维护指南

[返回论文主页](../README.md) · [最新版中文 PDF](../output/pdf/main_zh_latest.pdf)

本指南说明在另一台电脑上拉取论文项目、安装环境、编译中文稿，以及修改后同步回 GitHub 的完整流程。所有终端命令都在仓库根目录执行。

**在新电脑上编译论文只需要本仓库和 LaTeX 工具链，不需要原始图像、模型权重、GPU、PyTorch 或原电脑的 `D:` 盘目录。** Python 环境用于数值核验、PDF 检查及可选的图表重建。以下以 Windows PowerShell 为主，Git、Conda 和编译命令也可在 macOS/Linux 终端运行。

## 日常方式：本地交付，云端复核

每次修改论文正文、参考文献或图表后，**在本地重新编译和检查，将最新版 PDF 与源稿一起提交并推送**。固定 PDF 入口为 [`output/pdf/main_zh_latest.pdf`](../output/pdf/main_zh_latest.pdf)，可直接在 GitHub 文件页面预览或下载，也会随 `git pull` 一起取得。

GitHub Actions 在推送成功后，使用 GitHub 托管的 Ubuntu 云端运行器独立复核。它不占用本地电脑的编译资源；推送后可关闭本地电脑，也可以继续推送或拉取。检查失败会显示红色状态，不会撤销已经成功的推送。当前工作流只有仓库读取权限，没有修改分支的步骤；也未为本流程设置要求 CI 通过的分支规则。

项目操作约定保存在 [AGENTS.md](../AGENTS.md)。

## 后台自动检查与备用 PDF

[Chinese paper PDF 工作流](https://github.com/Du-x25/datasets_article/actions/workflows/chinese-paper.yml) 会在每次推送和 Pull Request 更新时运行，也可在该页面点击 **Run workflow** 手动触发。

工作流使用 Ubuntu 24.04、Python 3.11 和 TeX Live，从当前提交编译 `main_zh.tex`，然后检查：

- 未定义的图表引用和文献引用、重复标签、仍需重跑的交叉引用及缺失字形。
- `Float too large`、横向/纵向 `Overfull` 警告，以及 PDF 文字是否越过页面边界。
- 已有的源稿结构、文件范围、数值证据和表格一致性。

任何编译或检查失败都会使工作流失败，**只有全部通过才上传云端备用 PDF 产物**，不影响已提交的本地 PDF。自动检查能发现上述编译及边界问题，图内文字重叠、字号和整体视觉质量仍需人工看 PDF。

下载步骤：

1. 打开仓库的 **Actions → Chinese paper PDF**。
2. 选择对应提交且带绿色勾的运行记录。
3. 点击摘要中的 **Download the checked PDF**，或在 **Artifacts** 下载 `chinese-paper-提交SHA`。
4. 解压后得到 `main_zh.pdf` 和 `checks.json` 检查报告。

PDF 产物保留 **30 天**，构建诊断日志 `chinese-paper-diagnostics-运行编号-尝试次数` 保留 **14 天**。失败时可查看步骤日志或下载已有诊断文件。下载 Actions 产物需要登录 GitHub，详见[官方说明](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts)。过期后可重新运行工作流；需要长期归档时，将确认后的版本另存到 `output/pdf/`。

工作流只读取仓库并上传 Actions 产物，不会把编译缓存或 PDF 自动提交回仓库，也不会启动训练。云端使用 Linux 中文字体，页数可能与 Windows 归档版不同。配置位于 [.github/workflows/chinese-paper.yml](../.github/workflows/chinese-paper.yml)。

## 1. 安装工具

| 工具 | 用途与安装方式 |
|---|---|
| Git | Windows 使用 [Git for Windows](https://gitforwindows.org/)，保留 Git Credential Manager；其他系统使用 [Git 官方安装说明](https://git-scm.com/downloads) |
| TeX Live / MacTeX | 提供 XeLaTeX、BibTeX 和 latexmk。Windows/Linux 使用 [TeX Live](https://tug.org/texlive/)，macOS 使用 [MacTeX](https://tug.org/mactex/)；初次安装建议完整方案 |
| Conda | 已有 Anaconda/Miniconda 可直接使用；新安装见 [Conda 官方说明](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) |
| Poppler | 提供 PDF 检查需要的 `pdftotext` 和渲染预览的 `pdftoppm`，随下方 Conda 环境安装 |
| 中文绘图字体 | 仅重新出图时需要。Windows 通常可用微软雅黑；其他系统可安装 [Noto Sans SC / Noto CJK](https://github.com/notofonts/noto-cjk) |

TeX Live 和 Conda 是两套独立工具，安装 Python 环境不会安装 LaTeX。安装后重新打开终端检查：

```powershell
git --version
xelatex --version
latexmk -v
bibtex --version
```

项目使用 `ctexart`、`xeCJK`、`fontspec`、`geometry`、`graphicx`、`booktabs`、`longtable`、`tabularx`、`amsmath`、`xcolor`、`hyperref`、`caption/subcaption`、`float`、`placeins` 和 `natbib`。完整 TeX Live/MacTeX 提供这些组件；精简安装可用环境检查脚本定位缺项。本文采用 **XeLaTeX + BibTeX**，不要改用 pdfLaTeX 或 Biber。

## 2. 在另一台电脑克隆

打开终端，进入准备存放项目的父目录，然后运行：

```powershell
git clone https://github.com/Du-x25/datasets_article.git
cd datasets_article
git switch main
git remote -v
```

公开仓库可以匿名克隆和拉取。此后所有命令均在包含 `main_zh.tex` 的仓库根目录执行。已有副本时无需再次克隆，按第 6 节拉取即可。

## 3. 安装 Python 环境

推荐使用 Conda：

```powershell
conda env create -f environment.yml
conda activate mswir-paper
python -m pip check
python scripts/check_environment.py
```

[environment.yml](../environment.yml) 通过 conda-forge 安装 Python 3.11、pip 和 Poppler，再读取 [requirements.txt](../requirements.txt) 安装 NumPy、pandas、Matplotlib、Pillow 和 pypdf。Python 依赖统一在 `requirements.txt` 维护，其版本范围用于兼容安装，不是原始训练环境的锁文件。Conda 的创建与更新方式见[官方说明](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)。

同名环境已经存在时：

```powershell
conda env update -n mswir-paper -f environment.yml --prune
conda activate mswir-paper
python -m pip check
```

PowerShell 无法执行 `conda activate` 时，可以使用 Anaconda/Miniconda Prompt；或运行一次 `conda init powershell`，关闭终端后重新打开。

### 不使用 Conda 的替代方式

先安装 Python 3.11，并自行安装 Poppler、将 `pdftotext` 加入 PATH。Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/check_environment.py
```

之后将文中 `python` 替换为 `.venv\Scripts\python.exe` 即可，无需更改 PowerShell 执行策略。macOS/Linux 使用 `python3.11 -m venv .venv`，再执行 `source .venv/bin/activate` 和 `python -m pip install -r requirements.txt`。pip 不会安装 TeX Live 或 Poppler。

历史脚本需要的 seaborn、PyYAML、pytest 在 [requirements-legacy.txt](../requirements-legacy.txt) 中；仅在需要时执行 `python -m pip install -r requirements-legacy.txt`。安装这些库不会补齐历史脚本需要的外部实验数据。

## 4. 编译中文论文并检查

现成图表已在仓库中，日常修改文字后直接编译：

```powershell
latexmk -norc -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/zh main_zh.tex
```

latexmk 会按需运行 XeLaTeX、BibTeX 并更新交叉引用。PDF 位于 **`build/zh/main_zh.pdf`**，日志位于 `build/zh/main_zh.log`。Windows 可用 `Start-Process "build/zh/main_zh.pdf"` 打开。

编译后检查：

```powershell
python scripts/check_tex_structure.py
python scripts/qa_manuscripts.py --language zh
```

检查覆盖数值证据、验证集选择、表格、图号和文献键、PDF 是否新于源文件，以及未定义引用与超页/溢出信息。还应打开 PDF 查看修改页的图表、图注和分页；自动检查不能替代目视检查。单纯编译可以跳过 Python 环境及检查；已装 Python 时可用 `python scripts/check_environment.py --compile-only` 单独检查 TeX 工具链。

`build/` 是临时目录，不提交。每次论文修改通过检查后，都更新固定的最新版 PDF：

```powershell
Copy-Item -LiteralPath "build/zh/main_zh.pdf" -Destination "output/pdf/main_zh_latest.pdf"
```

将 `output/pdf/main_zh_latest.pdf` 与本次源稿修改一起提交。已归档的 `20261109` PDF 和验证记录继续保留为历史版本；需要额外里程碑归档时再创建日期命名副本。不同系统字体可能改变分页，19 页是当前 Windows/TeX Live 2026 归档版本的页数。

## 5. 可选：复算结果与重新出图

只核对现有数值，不写入图表：

```powershell
python scripts/build_results_20261109.py --check
```

需要重新出图时：

```powershell
python scripts/check_environment.py --figures
python scripts/build_results_20261109.py
latexmk -norc -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/zh main_zh.tex
python scripts/qa_manuscripts.py --language zh
```

绘图会重建 `figures/zh/results_20261109/` 中的 PDF/SVG/PNG、`generated/` 中对应的表格和宏，以及数值汇总。生成的表格与宏应从脚本更新，避免手工改数后与证据不一致。普通文字修改无需重新出图。

脚本自动查找微软雅黑、Noto Sans SC/CJK SC、思源黑体等已安装字体，找不到时明确报错。也可在当前终端指定字体族或字体文件：

```powershell
$env:MSWIR_PAPER_FONT = "Noto Sans SC"
python scripts/check_environment.py --figures
```

macOS/Linux 使用 `export MSWIR_PAPER_FONT="Noto Sans SC"`。这个变量只影响 Python 图表，正文中文字体由 `ctex` 选择。直接编译现有 PDF 插图不需要安装原绘图字体。

## 6. 修改后同步回 GitHub

### 首次配置身份和登录

在仓库内设置自己的提交身份，替换为实际姓名及 GitHub 已验证邮箱或 GitHub 提供的 noreply 邮箱：

```powershell
git config user.name "你的姓名"
git config user.email "你的GitHub提交邮箱"
```

提交身份不等于登录凭据。推送到本仓库需要 `Du-x25` 账号或被授予写入权限的账号；没有写权限的读者可以 Fork 后向自己的仓库推送。HTTPS 推荐使用 Git Credential Manager，首次推送按提示在浏览器登录。也可使用 GitHub CLI 的 `gh auth login`，选择 HTTPS 并允许配置 Git 凭据。GitHub 不接受账号密码作为 HTTPS Git 密码，详见[官方凭据说明](https://docs.github.com/en/get-started/git-basics/caching-your-github-credentials-in-git)。

### 每次开始修改前

先检查有无未提交修改；工作区干净时拉取最新版本：

```powershell
git status
git switch main
git pull --ff-only origin main
conda activate mswir-paper
```

已有本地修改时先提交，或用 `git stash push -u -m "local work before pull"` 暂存，拉取后用 `git stash pop` 恢复。依赖文件更新后重新执行第 3 节的环境更新命令。

### 修改、检查、提交和推送

主要修改入口是 `main_zh.tex`、`refs.bib` 和图表脚本。编译、PDF 检查通过并更新 `output/pdf/main_zh_latest.pdf` 后，将源稿和 PDF 一起暂存，例如：

```powershell
git diff --stat
git diff -- main_zh.tex refs.bib
git add -- main_zh.tex refs.bib output/pdf/main_zh_latest.pdf
# 如果修改了 README 或重新出图，再逐项 git add 对应文件。
git diff --cached --stat
python scripts/check_repository.py
git diff --cached --check
git commit -m "Update Chinese manuscript and figures"
git push origin main
git status
```

`check_repository.py` 检查 **Git 暂存区中的文件**，所以在 `git add` 之后运行。它会拒绝原始数据、权重、压缩包等禁入类型和大于 20 MiB 的单个文件，并核对主要编译依赖是否已纳入 Git。

若另一台电脑已经推送新提交，当前推送可能提示远程领先。先确保本地修改已提交，再运行：

```powershell
git pull --rebase origin main
```

冲突时按 `git status` 提示编辑冲突文件，解决后运行 `git add -- 冲突文件路径` 和 `git rebase --continue`；用 `git rebase --abort` 可取消本次变基。变基完成后重新编译、检查，再推送。日常同步无需强制推送，也不要使用 `git push --all` 或 `--mirror`，本地可能保留历史分支和备份引用。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| 找不到 `xelatex` / `latexmk` | 检查 TeX Live/MacTeX 安装及 PATH，重新打开终端；Windows 可用 TeX Live 自带终端确认 |
| 缺少 `ctexart.cls`、`placeins.sty` 等 | 运行环境检查定位缺项，用发行版的包管理器补齐，或使用完整 TeX Live/MacTeX |
| 找不到 `pdftotext` | 激活 `mswir-paper`，按环境文件更新；该工具不是 pip 包 |
| 中文绘图字体缺失或出现方框 | 安装 Noto Sans SC/CJK SC 或指定 `MSWIR_PAPER_FONT` 后出图；现有 PDF 插图可直接编译 |
| 图号或文献显示 `??` | 查看编译日志并使用完整 latexmk 命令；仍失败时检查文献键和 `\label/\ref` |
| PDF 检查提示文件比源码旧 | 重新编译；拉取或重新出图之后也要编译 |
| 输出 PDF 无法覆盖 | 关闭可能锁定文件的阅读器后重试 |
| 推送出现 403 / permission denied | 检查 `git remote -v` 和登录账号的写权限；公开可读不等于任何账号都可写 |

## 8. 仓库范围与证据状态

仓库保留源稿、必要图表、CSV/JSON 证据和较小的归档 PDF。[.gitignore](../.gitignore) 排除原始图像、完整成图数据、训练/测试数据、完整预测、模型权重、TIFF 副本、大交付包、环境和编译缓存。不要用 `git add -f` 绕过规则。证据快照通过 [.gitattributes](../.gitattributes) 保留原始字节，使不同电脑上的 SHA-256 校验一致。

[results_20261109](../figures/source_data/results_20261109/) 是当前中文稿的最小证据包，`evidence_code/` 仅是来源代码快照，不是完整训练工程。`docs/current_protocol.json`、旧结果宏和部分历史脚本仍属于旧版/英文稿，可能依赖原电脑的实验目录。新版中文稿的编译、记录复算和 PDF 检查不依赖这些外部目录。

核验基于保存记录，不包含重新训练、生成预测或从完整浮点预测重算全部指标。独立人工几何验证、质量与隐私验收、外部复现及最终数据许可/归档仍待完成。论文仓库公开不代表原始数据集已经公开或论文已经投稿。

旧版本说明见[历史 README](../docs/project_history_before_github_20261109.md)，其中旧交付包路径、页数和“最新”措辞不代表当前入口。
