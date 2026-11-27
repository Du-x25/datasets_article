# M-SWIR-RGB Zenodo 数据集上传、预留 DOI 与草稿占位完整操作指南

本指南对应已经打包并全量校验完成的上传包目录：
- `D:\Data_dir\SWIR-RGB\zenodo_upload_v1.0.0\`（共 **7 个文件**，总计 **775.04 MiB / 0.757 GiB**；解压还原全部 **1,109 个文件 ≈ 1.26 GiB**）

> **当前状态（已完成草稿占位与论文回填）**：已在 Zenodo 上传完整 7 个文件、填写全部元数据并预留正式 DOI **`https://doi.org/10.5281/zenodo.24575028`**（保存为 Draft，尚未点击 `Publish`）。论文中英文主文、`refs.bib`、`README.md` 与投稿信均已同步回填该 DOI。正式投稿前只需执行文末「正式投稿前最终发布 Zenodo 的操作提醒」。

---

## 第一步：登录并新建上传页面

1. 浏览器打开 https://zenodo.org 并登录账号（建议绑定你的 ORCID：`0009-0006-1742-2455`）。
2. 点击页面右上角的 **`+`** 号 $\to$ 选择 **`New upload`**，进入新建记录表单页。

---

## 第二步：上传 7 个文件（Files 区域）

打开本地目录 `D:\Data_dir\SWIR-RGB\zenodo_upload_v1.0.0\`，**全选全部 7 个文件**，一次性拖入 Zenodo 页面的 **Files** 上传框（或点击 **`Upload files`** 选择）：

| 序号 | 文件名 | 实测大小 | 说明 |
|---:|---|---:|---|
| 1 | `README.md` | 3.76 KiB | 数据集目录结构、隐私脱敏与校验说明（在线免下载预览） |
| 2 | `SHA256SUMS.txt` | 109.85 KiB | 解压后全部 1,109 个文件的 SHA-256 校验清单 |
| 3 | `LICENSE_DATA_CC_BY_4.0.txt` | 299 B | 数据 CC BY 4.0 许可协议文本 |
| 4 | `LICENSE_CODE_MIT.txt` | 423 B | 配套代码/配置 MIT 许可协议文本 |
| 5 | `M-SWIR-RGB_v1.0.0_mosaics_60tiffs.zip` | 497.56 MiB | `data/mosaics/`（60 张配准 8-bit TIFF 镶嵌图） |
| 6 | `M-SWIR-RGB_v1.0.0_source_frames_840.zip` | 262.84 MiB | `data/source_frames/`（840 张隐私核查/脱敏后的源帧） |
| 7 | `M-SWIR-RGB_v1.0.0_masks_metadata_support_benchmarks.zip` | 14.53 MiB | `usability_masks/` + `metadata/` + `support/` + `benchmarks/`（206 文件） |

> 等待 7 个文件进度条全部变为绿色勾选完成（总计约 775 MiB）。

---

## 第三步：填写 Basic Information（必填核心信息）

### 1. Digital Object Identifier (DOI) —— 获取预留 DOI
- 问题：`Do you already have a DOI for this upload?`
  - 勾选 **`No`**
  - 点击下方按钮 **`Get a DOI now!`**（部分界面显示 **`Reserve DOI`**）
- 页面会立即弹出一个专属预留 DOI，格式类似：
  `10.5281/zenodo.12345678`
- **务必立刻复制保存这个 DOI**（稍后发回即可同步回填到中英文论文、补充材料与投稿信中）。

### 2. Resource type（资源类型）
- 下拉选择：**`Dataset`**

### 3. Title（标题，直接复制）
```text
M-SWIR-RGB: Scene-Level Paired RGB and Five-Band Short-Wave Infrared Dataset for Cross-Spectral Synthesis
```

### 4. Publication date（发布日期）
- 填写当前计划发布日期（草稿阶段后续发布前可随时修改）：
```text
2026-11-27
```

### 5. Creators（作者列表，按顺序逐一点击 `Add creator` 添加 4 位）
- 每位均选择 **`Person`** 类型：

| 顺序 | Family name（姓） | Given names（名） | Identifiers（ORCID） | Affiliations（单位，粘贴后回车） | Role（可选） |
|---:|---|---|---|---|---|
| 1 | `Du` | `Hu` | `0009-0006-1742-2455` | `University of Science and Technology of China`；再添加一条 `Hefei Institutes of Physical Science, Chinese Academy of Sciences` | `Data curator` |
| 2 | `Liu` | `Xiao` | *(如有可填，无则留空)* | `Anhui Institute of Optics and Fine Mechanics, Hefei Institutes of Physical Science, Chinese Academy of Sciences` | `Supervisor` |
| 3 | `Du` | `Lili` | *(如有可填，无则留空)* | `Anhui Institute of Optics and Fine Mechanics, Hefei Institutes of Physical Science, Chinese Academy of Sciences` | `Data curator` |
| 4 | `Cui` | `Hu` | *(如有可填，无则留空)* | `Hefei Institutes of Physical Science, Chinese Academy of Sciences` | `Data curator` |

> 小提示：Zenodo 的 `Affiliations` 支持添加多个单位，也支持直接输入纯文本回车确认（不一定要匹配 ROR 列表）。

### 6. Description（数据集描述，整段直接复制粘贴）
```text
M-SWIR-RGB is a scene-level paired visible (RGB) and five-band short-wave infrared (SWIR, 900–1700 nm) dataset acquired across ten distinct physical ground sites (C1–C10; 140 overlapping camera positions) using a common-aperture dichroic beamsplitting platform that optically suppresses depth-dependent baseline parallax.

Distribution Archives (extract all three ZIP archives into the same root folder as SHA256SUMS.txt):
- M-SWIR-RGB_v1.0.0_mosaics_60tiffs.zip (data/mosaics/, 60 files): 60 uncompressed eight-bit TIFF mosaics (6 per site: RGB plus five nominal 50 nm narrow-band SWIR sub-bands: B1 1000–1050 nm, B2 1050–1100 nm, B3 1225–1275 nm, B4 1525–1575 nm, and B5 1625–1675 nm), co-registered to a B1 anchor and cropped to common valid boundaries.
- M-SWIR-RGB_v1.0.0_source_frames_840.zip (data/source_frames/, 840 files): 840 deterministically selected source frames (140 RGB and 700 SWIR) selected from 1,281 raw acquisition files by greyscale Laplacian variance. All 140 positions completed visual privacy screening prior to release (C1–C9 verified free of readable licence plates and identifiable human faces; all 15 C10 positions redacted across RGB and B1–B5 via border_median_constant).
- M-SWIR-RGB_v1.0.0_masks_metadata_support_benchmarks.zip (206 files):
  * usability_masks/ (80 uint8 PNG masks: per-modality 0 < DN < 250 validity masks, C10 privacy redaction mask, and six-modality joint.png masks totalling 88,861,466 valid pixels);
  * metadata/ (portable SHA-256 manifests dataset_manifest.csv and rawdata_release_manifest.csv, acquisition_provenance.csv, scene_quality_summary.csv, source_provenance.csv, privacy_redaction_layers.json, usability_masks_manifest.json, frame_to_mosaic_coverage.csv, and dynamic_range_usability.csv);
  * support/ (site-disjoint five-fold cross-validation splits over 5,893 stride-128 256×256 windows, geometric/privacy support overlays, archive RGB correspondence check, and blinded tie-point validation template);
  * benchmarks/ (sanitized mosaicking YAML configs c1.yaml–c10.yaml, Python reproduction scripts for all manuscript tables, figures, non-SIFT 15-pair geometric audits, and Track C DRUNet few-shot fine-tuning benchmarks, plus summary source data).
- Top-level files: README.md, SHA256SUMS.txt (1,109 file-level SHA-256 digests), LICENSE_DATA_CC_BY_4.0.txt, and LICENSE_CODE_MIT.txt.

Quality & Technical Validation Highlights:
- Pipeline RGB–B1 feature holdout P95 residuals span 0.77–1.92 preview pixels (3.56–12.97 output pixels); maximum pairwise intra-SWIR P95 residuals span 3.48–6.25 output pixels.
- Independent non-SIFT Sobel-gradient sub-pixel phase-correlation audit across all 15 modality pairs (432 valid 256×256 tiles × 15 pairs = 6,480 checks) yields median residuals of 0.51 output px (91.5% <= 2 px) for intra-SWIR pairs and 2.37 output px for cross-modal RGB–SWIR pairs.
- At the 4:1 network input scale, 76.6% of judgeable windows peak within a 3×3 zero-shift neighbourhood.

Usage & Privacy Notes:
- All imagery consists of uncalibrated 8-bit digital numbers (DN, 0–255) defined by nominal 50 nm filter specifications (end-to-end relative spectral response and absolute radiometric calibration were not measured); do not interpret pixel values as physical surface reflectance.
- Always apply usability_masks/<site>/joint.png during training and scoring, and use the site-disjoint five-fold splits for cross-scene generalization evaluation.
- Privacy redaction: seven vehicle licence plates in site C10 are directly redacted across all six mosaics via border_median_constant and excluded in redaction.png and joint.png. In addition, all 15 C10 source-frame positions (15 RGB + 75 SWIR B1–B5 frames) are directly redacted via border_median_constant covering all eight visible vehicles (the seven mosaic-registered plates plus one right-margin vehicle visible only in uncropped raw positions 07–08) and visually verified clean.

Licensing:
- Dataset files (images, masks, metadata, and evaluation supports) are licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
- Accompanying benchmark and processing scripts are licensed under the MIT License.
```

### 7. Licenses（许可协议）
- 点击 **`Add standard`** $\to$ 搜索 **`Creative Commons Attribution 4.0 International`**（`CC BY 4.0`）并选中添加。
- *(Zenodo 主记录选 `CC BY 4.0` 覆盖数据主体；代码 MIT 许可已在 `Description` 与根目录 `LICENSE_CODE_MIT.txt` 中明确声明。)*

---

## 第四步：填写 Recommended Information（推荐信息）

### 1. Keywords and subjects（关键词）
在输入框中**逐条输入并回车**（共 8 条）：
```text
Short-wave infrared (SWIR)
Cross-spectral image synthesis
SWIR-to-RGB colorization
Multispectral imaging
Common-aperture beamsplitting
Paired image dataset
Site-disjoint cross-validation
Domain adaptation
```

### 2. Languages（语言）
- 搜索并选择：**`English`**

### 3. Version（版本号）
```text
1.0.0
```

### 4. Additional notes（附加说明）
```text
Accompanying Data Descriptor manuscript: "M-SWIR-RGB: Scene-Level Paired RGB and Five-Band Short-Wave Infrared Dataset for Cross-Spectral Synthesis" (submitted to Scientific Data). Full file-level integrity verification for all 1,109 extracted files is provided in SHA256SUMS.txt.
```

### 5. Dates / Publisher / Contributors
- **Publisher**：默认 `Zenodo`（无需修改）
- **Dates / Contributors**：留空即可（作者角色已在 Creators 中填好）

---

## 第五步：Funding 与 Related Works

### 1. Funding（基金资助）
- **留空**（与论文 `Funding` 声明完全一致）。

### 2. Related Works（关联论文代码仓库）
点击 **`Add related work`**，按以下 4 项填写：
- **Relation**：`Is supplemented by`
- **Identifier**：`https://github.com/Du-x25/datasets_article`
- **Scheme**：`URL`
- **Resource type**：`Software`

---

## 第六步：Visibility 与保存草稿（关键！不要点 Publish）

1. 在页面右侧栏 **Visibility（可见性）** 中先保持默认即可（草稿状态对外完全不可见；未来正式发布时再决定选 `Public` 还是 `Restricted` + 审稿人私有链接）：
   - 若未来投稿时想直接公开数据：`Full record: Public` + `Files: Public`
   - 若审稿期间只想对 *Scientific Data* 审稿人开放：`Full record: Public` + `Files: Restricted`（发布后点右上角 **`Share` $\to$ `Links` $\to$ `Create a new link`** 生成 `Can view` 私有链接填入投稿系统）
2. **现在点击右侧栏的蓝色/灰色按钮：`Save draft`**（保存草稿）。
   - **千万不要点击红色的 `Publish` 按钮**！
   - 保存成功后，页面顶部会显示 `Draft` 标签，且可在 Zenodo 顶部菜单 **`My dashboard` $\to$ `Uploads`** 中随时重新打开编辑或补充文件。
3. 核对并复制你刚预留到的 DOI（格式：`10.5281/zenodo.XXXXXXX`）。

---

## 第七步：拿到预留 DOI 后的收尾工作

把预留 DOI（例如 `10.5281/zenodo.12345678`）直接发回，我会自动一次性完成以下 5 处回填与验证：
1. `refs.bib`：将 `publisher` 更新为 `Zenodo`，将 `10.XXXX/xxxxx` 替换为正式预留 DOI；
2. `main_zh.tex`：回填正式预留 DOI、更新打包体积（`775 MiB` 压缩包 / `1.26 GiB` 解压全量）与 C10 源帧脱敏已完成状态；
3. `main_en.tex`：回填正式预留 DOI、同步修正体积（替换旧的 `≈11 GB`）与 C10 源帧脱敏已完成状态；
4. `README.md` 与 `docs/cover_letter_scientific_data.md`：回填正式预留 DOI、Zenodo 名称与实际体积；
5. 重编全部 4 份 PDF（中文主文/补充材料、英文主文/补充材料）并跑通 `check_tex_structure.py` 与 `qa_manuscripts.py`。

---

## 附：正式投稿前（或录用前）最终发布 Zenodo 的操作提醒

当你确认论文定稿并准备正式向 *Scientific Data* 提交时：
1. 回到 Zenodo `My dashboard` $\to$ 打开该 Draft；
2. 更新 `Publication date` 为当天实际日期；
3. 确认 `Visibility`（直接公开选 `Public`；仅对审稿人开放选 `Files: Restricted`）；
4. 点击 **`Publish`** 完成 DOI 正式激活（若选了 `Restricted`，随后点右上角 **`Share` $\to$ `Links` $\to$ `Create a new link`** 把私有预览链接粘贴到投稿系统 Data Availability 栏）。
5. *(强烈建议同步操作)* 在 Zenodo 头像 $\to$ **`GitHub`** 开启 `Du-x25/datasets_article` 同步，并在 GitHub 创建 `v1.0.0` Release，自动获得永久 **Code DOI**。
