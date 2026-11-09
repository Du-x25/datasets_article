# 论文项目工作约定

用户已确定以下日常流程：本地编译并交付最新 PDF，源码与 PDF 一起同步到 GitHub；GitHub Actions 保留为云端后台复核。

- 主要修改中文稿 `main_zh.tex`；英文稿除非用户要求，不自动同步内容。
- 修改正文、参考文献、被引用表格/数值宏或插图后，在本地重新编译中文稿并运行检查：

  ```text
  latexmk -norc -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/zh main_zh.tex
  python scripts/check_tex_structure.py
  python scripts/qa_manuscripts.py --language zh
  ```

- 检查修改页的版面。通过后把 `build/zh/main_zh.pdf` 复制到固定入口 `output/pdf/main_zh_latest.pdf`。源稿与这个 PDF 必须在同一提交中同步；新图表、文献和必要的数值证据也要纳入提交。
- 本地编译或检查失败时先修复，不用旧 PDF 冒充最新版。日期命名的历史 PDF 保留，不用每次覆盖。
- 按用户已授权的日常方式，将完成的论文修改提交并推送到已有的 `origin/main`。暂存后运行 `python scripts/check_repository.py`，只推送当前工作分支，不推送本地历史分支或备份引用。
- 原始数据、完整预测、模型权重、大交付包和编译缓存继续排除在 Git 之外。
- 保留现有 Actions 后台检查及临时产物。工作流只读仓库，不自动提交文件；不要为了这个流程添加必须等待 CI 的分支保护或合并规则。

安装环境、出图和 Git 操作细节见 `README.md`。
