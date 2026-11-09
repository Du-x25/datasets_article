# 新电脑使用流程验证（2026-11-09）

本次验证在 Windows 上进行。使用仅由 Git 暂存文件导出的干净目录及新建的 Conda 环境，不使用原 Python 环境的 site-packages，不读取训练数据或模型权重。TeX 工具链使用本机已有的 TeX Live 2026。

## 实际环境版本

| 组件 | 实际版本 |
|---|---|
| Python | 3.11.18 |
| NumPy | 2.4.6 |
| pandas | 2.3.3 |
| Matplotlib | 3.11.2 |
| Pillow | 11.3.0 |
| pypdf | 5.9.0 |
| Poppler | 26.10.0 |
| TeX Live | 2026 |

这些版本记录本次验证环境；依赖文件采用兼容范围，不锁定整个操作系统和字体环境。

## 完成的检查

- 按 `environment.yml` 从 conda-forge 安装 Python 和 Poppler，再通过 `requirements.txt` 安装 Python 库，安装成功。
- `python -m pip check` 无依赖冲突；`check_environment.py --figures` 找到全部 18 项 TeX 文件、环境内的 `pdftotext` 和中文字体。
- 干净目录直接编译现成图表成功；`qa_manuscripts.py --language zh` 通过，PDF 为 19 页、22 个标签、29 个被引文献键，无未定义引用、超页浮动体或 overfull box。
- 在新环境完整运行图表重建、编译及 PDF 核验，全部通过；五张 PNG 与原归档图逐像素一致。
- 默认微软雅黑选择、显式选择 Noto Sans SC、拒绝不含所需汉字的字体均通过检查。
- README 的本地文件链接、Git 暂存区文件范围和源证据哈希通过检查。

本次未在 macOS/Linux 上实际运行，跨系统字体及分页仍需在目标机器检查。测试使用的编译文件位于本地忽略目录，归档论文 PDF、实验数值和论文正文未因这次环境整理而修改。
