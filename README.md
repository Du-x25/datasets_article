# M-SWIR-RGB：RGB 与五波段短波红外配对数据集

**中文 Data Descriptor：** [中文主文 PDF](output/pdf/main_zh_latest.pdf) · [中文补充材料 PDF](output/pdf/supplementary_zh_latest.pdf) · [中文源稿](main_zh.tex)  
**English Data Descriptor (*Scientific Data* format):** [Main Manuscript PDF](output/pdf/main_en_latest.pdf) · [Supplementary Information PDF](output/pdf/supplementary_en_latest.pdf) · [Cover Letter](docs/cover_letter_scientific_data.md)  
**证据与工具：** [窗口级基准证据](figures/source_data/bench_20261124/) · [五折基准证据](figures/source_data/results_20261109/) · [发布包组装脚本](scripts/package_dataset_release.py) · [源帧隐私筛查工具](scripts/prepare_raw_privacy_review.py) · [人工控制点校验工具](scripts/evaluate_tiepoints.py) · [Zenodo 上传指南](docs/zenodo_upload_guide.md)

M-SWIR-RGB 是面向短波红外（SWIR）到 RGB 跨光谱图像生成的场景级多模态配对数据集，包含十个不同物理地点的 RGB 与五个名义 50 nm 窄带 SWIR 子波段配对观测。论文严格按照《Scientific Data》Data Descriptor 规范组织，围绕**数据构建与可追溯记录**、**几何与光谱质量表征**以及**三类定位分离的基准轨道**系统验证数据集的用途与边界（本文不提出新的生成模型）。

## 数据集概览

| 项目 | 当前数据规模与定义 |
|---|---|
| 采集地点 | 10 个不同物理地点（C1–C10；C1–C7 与[前作七场景归档](figures/source_data/bench_20261124/archive_rgb_correspondence.json)存在空间重合，C8–C10 为新增春夏季地点） |
| 重叠采集视野 | 140 个（每位置 1 次 RGB 采集与 5 通道分时 SWIR 曝光） |
| 模态与成图 | RGB + B1–B5 共 6 种模态，每地点 6 幅尺寸对齐的 8 位无压缩 TIFF（共 60 幅） |
| 采集档案与选中源帧 | 1,281 个原始文件（含 C8–C10 每视野 10 帧连拍）；按拉普拉斯方差确定性选出 840 幅源帧（140 RGB + 700 SWIR） |
| 共同有效支持 | 88,861,466 个像素（六模态共同满足 $0 < \mathrm{DN} < 250$ 且不在 C10 隐私脱敏框内） |
| 评价与基准窗口 | 4:1 窗口 3,552 个（1024→256）、1:1 窗口 3,886 个、五折地点隔离合格窗口 5,893 个（256×256、步长 128） |

| SWIR 通道 | 名义通带 / nm | 名义中心波长 / nm | 名义带宽 / nm |
|---|---:|---:|---:|
| B1 | 1000–1050 | 1025 | 50 |
| B2 | 1050–1100 | 1075 | 50 |
| B3 | 1225–1275 | 1250 | 50 |
| B4 | 1525–1575 | 1550 | 50 |
| B5 | 1625–1675 | 1650 | 50 |

波段由滤光片名义规格定义（整机端到端相对光谱响应尚未实测），像素为未辐射定标的 8 位数字量（DN），且未随文件保存原始曝光与增益参数。

<details>
<summary>查看十个地点的 RGB 与五波段 SWIR 成图总览</summary>

![十个地点的 RGB 与五波段 SWIR 成图总览](figures/zh/fig1_scene_modality_overview_zh.png)

每行对应一个地点（C1–C10），每列依次为 RGB、B1、B2、B3、B4 和 B5（仅显示拉伸）。

</details>

## 三类验证轨道与主要发现

1. **几何配准与光谱统计表征：**
   - 流水线特征匹配 RGB–B1 留出 P95 残差为 **0.77–1.92** 预览像素（3.56–12.97 输出像素），五波段 SWIR 之间最大波段内 P95 残差为 **3.48–6.25** 输出像素；独立于 SIFT 的全 15 对模态梯度相位相关亚像素审计（432 有效块 × 15 对 = 6,480 次检测）显示 **SWIR–SWIR 中位残差 0.51 px（91.5% ≤ 2 px）**、**RGB–SWIR 中位残差 2.37 px**；在 4:1 输入尺度上，83.1% 可判定窗口中 **76.6%** 的灰度互相关峰落在零位移 $3\times3$ 邻域内（对照归档 76.1%）。
   - 五个 SWIR 子波段在图像内容相关上聚为 `{B1, B2, B4}`（$r \ge 0.898$）与 `{B3, B5}`（$r \ge 0.967$）两组；受植被在 1000–1100 nm 高反射与可见光暗绿/暗红差异影响，B1/B2/B4 与 RGB 亮度相关随场景变号，而 B3/B5 恒正相关。
2. **轨道 A（4:1 窗口级基准：兼容可用性与重叠泄露诊断）：**
   - 未修改的 `TFD-GAN` 原实现（`pix2pixhd-p-noshuffle`）可在 M-SWIR-RGB 上直接收敛：100 epoch 为 **24.46 ± 1.12 dB**、200 epoch 为 **28.88 ± 0.06 dB**（对照归档为 20.26 ± 1.80 / 22.91 ± 0.18 dB）；同配置下亦兼容 `DRUNet`（36.438 dB）与 `pix2pix`（25.106 dB）。
   - 15 种输入消融中仅 B3、B234、B345 显著掉分（2.56–3.69 dB）；1:1 窗口因 SWIR 放大约 10 倍，比 4:1 掉 6.44–8.16 dB。
   - 25% 训练窗口即可取得与全量相当的分数（24.15 vs 24.46 dB），原因在于 1024×1024 步长 128 窗口重叠 87.5%，**710 个测试窗口中有 701 个完全落入训练窗口并集内**。在同一批测试窗口上轮流留出对应地点后，PSNR 由 **25.40 / 23.62 dB** 骤降至 **7.85 / 7.78 dB**（整折留出 7.64 / 7.84 dB，低于灰度/常数均值基线）。
3. **轨道 B（地点隔离五折基准，225 次训练）与协议对比：**
   - 在均衡的六训/二验/二测五折基准（种子 11、16,680 步、验证集早停与选波段）下，DRUNet、pix2pix 与 TFD-GAN 的十地点等权 PSNR 分别为 **13.619→13.820 dB**、**12.876→12.856 dB** 和 **12.040→12.839 dB**，LPIPS 与 $\Delta E_{00}$ 在三波段上均改善，但模型仍低于训练地点拟合的线性仿射参照（14.136 dB）。
   - 在相同 256×256 步长 128 窗口上对比 `pix2pix`（45 次随机划分），因 **~96.26%** 测试有效像素被训练窗口覆盖，随机划分平均虚高 **+4.435 dB**。
4. **轨道 C（少量目标地点配对校准、DRUNet 少样本微调与空间边界）：**
   - 固定 B1/B3/B4 仿射映射形式，改用目标地点 10 个配对图块（4.02%–11.27% 有效面积）拟合后，保留区域 PSNR 由 **14.113 dB** 升至 **16.502 dB**（$\Delta E_{00}$ 由 19.719 降至 12.864），二次映射达 16.715 dB；半区迁移测试显示对侧半区比同侧低 **1.845 dB**。
   - 在完全相同的目标 10 块与 50 万保留像素上，DRUNet 全参数少样本微调较零样本提升 **+1.070 dB**（13.378 → **14.448 dB**，$\Delta E_{00}$ 23.505 → 17.808），在阴影/接缝显著的 C9 上反超二次映射（16.504 vs 14.260 dB），但受限于 32.6M 参数对 10 块的空间过拟合，在其余 9 个地点仍落后于低自由度参数化颜色校准逾 2 dB。

## 当前版本与数据发布边界

- **完整数据集已存缴于 Zenodo（DOI：`https://doi.org/10.5281/zenodo.24575028`，v1.0.0；三个模块化 ZIP 压缩包及顶层校验清单合计 775 MiB，解压全量 1,109 个 SHA-256 校验文件约 1.26 GiB，数据采用 CC BY 4.0 许可，配套代码采用 MIT 许可）。**本 GitHub 仓库公开中英文论文源码、图表、最小数值汇总及复算/校验脚本，不含完整影像归档与模型权重。
- C10 的七个车牌已在 60 幅成图中采用 `border_median_constant` 直接脱敏并写入隐私掩膜；全部 840 幅公开源帧亦已完成发布前隐私核查（C1–C9 全部 125 个视野目视确认无车牌/人脸，C10 全部 15 个视野的 15 幅 RGB 与 75 幅 SWIR 均已对全部 8 辆可见车辆完成 `border_median_constant` 脱敏并复核通过）。独立盲法人工控制点验证模板随数据集一并发布。

## 编译、检查与维护

```powershell
conda activate mswir-paper
latexmk -norc -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/zh main_zh.tex
latexmk -norc -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/zh supplementary_zh.tex
latexmk -norc -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/en main_en.tex
latexmk -norc -xelatex -bibtex -interaction=nonstopmode -halt-on-error -outdir=build/en supplementary_en.tex
python scripts/check_tex_structure.py
python scripts/qa_manuscripts.py --language both
```

详细上传指南与仓库维护规则见 [docs/zenodo_upload_guide.md](docs/zenodo_upload_guide.md) 与 [AGENTS.md](AGENTS.md)。

[![Chinese paper PDF](https://github.com/Du-x25/datasets_article/actions/workflows/chinese-paper.yml/badge.svg?branch=main)](https://github.com/Du-x25/datasets_article/actions/workflows/chinese-paper.yml)
