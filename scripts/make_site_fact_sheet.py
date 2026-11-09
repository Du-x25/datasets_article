#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成逐地点事实表（site fact sheet）。

设计原则：不引入场景类别体系。
输出的是可核验的采集事实（日期、视野数、画幅、窗口数、日期证据状态、
隐私遮蔽状态、跨模态残差），外加一列中性的画面构成描述。

画面构成描述由作者在 2026-10-14 依据 scene_crop_overview.png 与各场景固定
窗口图核对后确认，用于说明十个地点并不相同；它不是分类体系，也不互斥。
数据全部来自已落盘的发布元数据，未重新训练、未载入权重复演。
"""
import csv
import io
import json
from pathlib import Path

ROOT = Path(r"D:\USTC\Articles\datasets_article")
META = Path(r"D:\Data_dir\SWIR-RGB\datasets_article_auto_v3\metadata")
COUNTS = Path(r"D:\USTC\Gen\outputs\crop_comparison_20261011\current_crop_counts.csv")

# 画面构成描述：2026-10-15 依据各场景**全幅RGB 成图**（scene{N}.tif）核对，
# 辅以 scene_crop_overview.png 与固定窗口图。措辞只描述画面内可见内容，不做场景归类。
# 修订记录：首版误用固定窗口图判读，导致 C1/C2/C7/C9 描述与全图不符，已更正。
CONTENT = {
    "C1": "现代建筑群，蓝灰玻璃幕墙与密集窗格，前景红叶树与天空",
    "C2": "在建工地，蓝色安全网与红色横幅，脚手架与塔吊",
    "C3": "城市道路，斑马线与车道标线，刷白行道树，两侧停放车辆与秋色植被",
    "C4": "俯视沥青路面，黄色虚线与菱形标记，右侧绿化带",
    "C5": "长条形多层建筑群，前景为大片枯黄裸土地面",
    "C6": "高层建筑群，玻璃幕墙反光明显，底部绿化带与晴空",
    "C7": "枯黄收割后农田与灌渠，道路两侧路灯阵列，画幅比约八比一",
    "C8": "白色多层建筑与平屋顶砖房，大片草坪与密集乔木",
    "C9": "俯视弯道，两侧浓密植被与红叶植物，远处湖面山体，含少量车辆",
    "C10": "俯视停车场，一排轿车，车牌已遮蔽，上缘绿化带",
}


# 第 1 版描述（2026-10-14，依据固定窗口图，其中四处有误，保留供对照）
OLD_CONTENT = {
    "C1": "多层建筑立面，窗格密集，外墙为粉棕色",
    "C2": "板式建筑，横向带状构图，含水平檐带与窗列",
    "C3": "城市道路，含斑马线、车道标线、路边树木与停放车辆",
    "C4": "沥青路面与车道线，旁侧绿篱与车辆",
    "C5": "长条形多层建筑群，窗格阵列规律，画幅比约五比一",
    "C6": "高层建筑群，玻璃幕墙反光明显",
    "C7": "远距离地平线，植被与低对比天空",
    "C8": "白色多层建筑，前景草坪与树木，含天空",
    "C9": "极暗弱纹理表面，整体接近黑场",
    "C10": "俯视视角铺装地面，多辆车排列",
}

# 各场景 RGB 全幅中位亮度（DN），用于辅助判断画面明暗
MEDIAN_LUMA = {
    "C1": 158, "C2": 39, "C3": 45, "C4": 82, "C5": 93,
    "C6": 135, "C7": 113, "C8": 109, "C9": 25, "C10": 52,
}


def read_csv(p):
    with io.open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    prov = read_csv(META / "acquisition_provenance.csv")
    qual = {r["scene"]: r for r in read_csv(META / "scene_quality_summary.csv")}
    counts = {r["scene"]: r for r in read_csv(COUNTS)}
    privacy = json.load(io.open(META / "privacy_redaction_layers.json", encoding="utf-8"))
    priv = {p["scene"]: p for p in privacy["per_scene"]}

    rows = []
    for r in prov:
        s = r["scene"]
        w, h = int(r["canvas_width"]), int(r["canvas_height"])
        # 日期证据状态：C8 为协议接受值，其余为文件时间证据
        if s == "C8":
            date_ev = "作者确认，2026-10-15"
        else:
            date_ev = "只读源文件时间"
        p = priv[s]
        rows.append({
            "scene": s,
            "date": r["acquisition_date"],
            "date_evidence": date_ev,
            "positions": int(r["source_positions"]),
            "rgb_frames": int(r["rgb_source_frames"]),
            "rgb_per_pos": r["rgb_frames_per_position"],
            "width": w,
            "height": h,
            "ratio": w / h,
            "windows": int(counts[s]["s128_end_common_ge50"]),
            "holdout_p95": float(qual[s]["rgb_b1_holdout_p95_preview_px"]),
            "intra_p95": float(qual[s]["maximum_intra_swir_output_p95_px"]),
            "privacy_boxes": p["privacy_boxes"],
            "content": CONTENT[s],
            "median": MEDIAN_LUMA[s],
        })

    # --- CSV ---
    csv_path = ROOT / "figures" / "source_data" / "site_fact_sheet.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    # --- LaTeX (中文) ---
    zh = ROOT / "generated" / "site_fact_sheet_zh.tex"
    lines = []
    lines.append("% 自动生成，勿手工编辑；由 scripts/make_site_fact_sheet.py 生成")
    lines.append("% 数据源：datasets_article_auto_v3/metadata + crop_comparison_20261011")
    lines.append("\\begin{tabularx}{\\textwidth}{@{}llrrlr>{\\raggedright\\arraybackslash}X@{}}")
    lines.append("\\toprule")
    lines.append("地点 & 采集日期 & 视野数 & 档案RGB & 画幅（宽$\\times$高） & 合格窗口 & 画面构成 \\\\")
    lines.append("\\midrule")
    for r in rows:
        note = f"（每视野{r['rgb_per_pos']}帧）" if r["rgb_per_pos"] != "1" else ""
        lines.append(
            f"{r['scene']} & {r['date']} & {r['positions']} & {r['rgb_frames']} & "
            f"{r['width']}$\\times${r['height']} & {r['windows']} & {r['content']}{note} \\\\"
        )
    lines.append("\\midrule")
    tot_pos = sum(r["positions"] for r in rows)
    tot_rgb = sum(r["rgb_frames"] for r in rows)
    tot_win = sum(r["windows"] for r in rows)
    lines.append(f"合计 & --- & {tot_pos} & {tot_rgb} & 十个不同尺寸 & {tot_win} & --- \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    io.open(zh, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # --- LaTeX (English) ---
    en = ROOT / "generated" / "site_fact_sheet_en.tex"
    CONTENT_EN = {
        "C1": "modern building blocks, blue-grey glass curtain wall with dense windows, red-leaved trees in front, sky",
        "C2": "construction site, blue safety netting and red banners, scaffolding and tower crane",
        "C3": "urban road with crosswalk and lane markings, white-painted roadside trees, parked vehicles, autumn vegetation",
        "C4": "top-down asphalt carriageway, yellow dashed line and diamond marking, verge on the right",
        "C5": "long multi-storey building block, large area of dry bare ground in the foreground",
        "C6": "high-rise blocks with reflective glass curtain wall, greenery at the base, clear sky",
        "C7": "harvested dry field and irrigation channel, rows of street lights, aspect ratio near eight to one",
        "C8": "white multi-storey building with flat-roofed brick houses, large lawn and dense trees",
        "C9": "top-down curved road, dense vegetation and red-leaved plants on both sides, distant lake and hills, a few vehicles",
        "C10": "top-down car park, one row of sedans with plates redacted, greenery at the upper edge",
    }
    lines = []
    lines.append("% auto-generated; do not edit by hand")
    lines.append("\\begin{tabularx}{\\textwidth}{@{}llrrlr>{\\raggedright\\arraybackslash}X@{}}")
    lines.append("\\toprule")
    lines.append("Site & Date & Views & Archive RGB & Canvas (W$\\times$H) & Windows & Scene content \\\\")
    lines.append("\\midrule")
    for r in rows:
        note = f" ({r['rgb_per_pos']} frames per view)" if r["rgb_per_pos"] != "1" else ""
        lines.append(
            f"{r['scene']} & {r['date']} & {r['positions']} & {r['rgb_frames']} & "
            f"{r['width']}$\\times${r['height']} & {r['windows']} & {CONTENT_EN[r['scene']]}{note} \\\\"
        )
    lines.append("\\midrule")
    lines.append(f"Total & --- & {tot_pos} & {tot_rgb} & ten distinct sizes & {tot_win} & --- \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    io.open(en, "w", encoding="utf-8").write("\n".join(lines) + "\n")

    # --- 事实核查说明（供作者确认） ---
    md = Path("D:/USTC/Gen/outputs/paper_writing_advice_20261014/地点事实表_已确认_20261015.md")
    mlines = [
        "# 逐地点事实表 · 已确认",
        "",
        "生成日期：2026-10-14；**画面构成于 2026-10-15 由作者逐条确认**。\n"
        "数据全部来自已发布元数据，未重训、未改数据。",
        "",
        "## 为什么要这张表",
        "",
        "原正文写“四类”，但 `acquisition_provenance.csv` 的 45 列里没有任何类别字段，",
        "`v3_scene_manifest.json` 只有路径与哈希——**十个场景从未被正式分类过**。",
        "旧稿的“三类”属于 S01–S07 版本，不能套到当前 C 编号。",
        "",
        "两天内做完整分类体系不现实（需要判定规则、单/多标签约定、每场景归属记录），",
        "因此改为**不声明类别数，只陈述事实**。下面这张表可完全由现有数据生成。",
        "",
        "## 逐地点事实",
        "",
        "| 地点 | 采集日期 | 日期证据 | 视野 | 源RGB | SWIR | 画幅 | 画幅比 | 合格窗口 | 跨模态P95 | 隐私框 |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        swir = r["positions"] * 5
        mlines.append(
            f"| {r['scene']} | {r['date']} | {r['date_evidence']} | {r['positions']} | "
            f"{r['rgb_frames']} | {swir} | "
            f"{r['width']}×{r['height']} | {r['ratio']:.2f} | {r['windows']} | "
            f"{r['holdout_p95']:.2f} | {r['privacy_boxes']} |"
        )
    mlines += [
        "",
        f"合计：{tot_pos} 视野、{tot_rgb} RGB 源帧、{tot_win} 个合格窗口。",
        "",
        "## 三个计数单位是什么",
        "",
        "| 单位 | 含义 | 为什么和别的数不一样 |",
        "|---|---|---|",
        "| **采集视野** | 一次相机与滤光片轮位置切换下拍得的一组同名模态，是采集动作的最小单位 | C1–C7 每视野 1 幅 RGB；C8–C10 每视野 10 帧连拍取最清晰一幅 |",
        "| **源 RGB** | 参与拼接的 RGB 原图幅数 | C8–C10 因连拍而多于视野数（23 视野→230 帧）|",
        "| **合格窗口** | 256×256、步长 128、整窗几何与隐私有效、窗内 ≥50% 像素落在六模态共同动态范围的裁块 | 这是模型实际读取的单位，不是原始文件计数 |",
        "",
        "换算关系：SWIR 源图 = 视野数 × 5（每视野五个波段各一幅）；成图 TIFF = 10 地点 × 6 = 60 幅。",
        "三者不可相加，也不能与 TIFF 数量混算。",
        "",
        "## 画面构成（第 2 版，作者已确认）",
        "",
        "**首版有误。** 我最初依据 `scene_crop_overview.png` 与固定窗口图判读，",
        "而固定窗口只是整幅成图的 256×256 局部，落在暗部时被误当成全图，",
        "导致 **C1、C2、C7、C9 四处描述与全图实质不符**；其余六处也补全了首版遗漏的内容。",
        "",
        "第 2 版改为逐一读取各场景**全幅 RGB 成图**（`datasets_article_auto_v3/C*/scene*.tif`）核对，",
        "并用亮度分位数辅助判断：",
        "",
        "| 地点 | 中位亮度 | 第 1 版描述（错） | 第 2 版描述（依据全图） |",
        "|---|---:|---|---|",
    ]
    for r in rows:
        old_desc = OLD_CONTENT[r["scene"]]
        mark = " ⚠" if old_desc.strip() != r["content"].strip() else ""
        mlines.append(f"| {r['scene']}{mark} | {r['median']} | {old_desc} | {r['content']} |")
    mlines += [
        "",
        "标 ⚠ 表示描述已改写（C1、C2、C7、C9 为实质更正，其余为措辞补全）。**全部十条已由作者确认。**",
        "",
        "| 地点 | 第 2 版描述 | 作者确认 |",
        "|---|---|---|",
    ]
    for r in rows:
        mlines.append(f"| {r['scene']} | {r['content']} | 已确认 2026-10-15 |")
    mlines += [
        "",
        "## 关于你问的 C9",
        "",
        "你说得对，C9 不是“极暗弱纹理表面”。全图是**俯视的弯道**，",
        "两侧浓密植被（含红叶植物），远处有湖面与山体，画面内有少量车辆。",
        "",
        "它看起来偏暗是有客观依据的：C9 的中位亮度只有 25（C2 是 39、C3 是 45），",
        "暗部占比确实最高——大量深色树冠阴影。但 p95 亮度达到 255，说明高光区域",
        "（天空、水面）依然存在，不能概括为“接近黑场”。",
        "",
        "C7 同理：不是“远距离地平线低对比”，而是**枯黄收割后农田与灌渠**，",
        "画幅 7.6:1，两侧有路灯阵列。",
        "",
        "## 三点设计说明",
        "",
        "1. **不报类别数**。正文只说“十个不同物理地点”，不写“四类”。逻辑自洽，可直接投稿。",
        "2. **画面构成不是分类**。它只说明十个地点互不相同，不互斥、不构成类别体系，",
        "   因此不会出现“C3 是道路还是城市”这类无法判定的争论。",
        "3. **C8 的日期已由作者确认为 2025-04-30**（2026-10-15）；其余九个场景仍为只读源文件时间推断。",
        "",
        "## 若你仍想要类别（不推荐在两天内做）",
        "",
        "从全幅画面看，C3/C4/C9 同属道路与绿化组合，C1/C2/C5/C6/C8 同属建筑场景，",
        "C7 为农田与灌渠，C10 为停车场。但 C3（道路+车辆+行道树）与 C9（弯道+植被+远山）",
        "天然是多标签的——强行单标签会失真，多标签又需要每标签的判定标准。",
        "两天内做不扎实，反而会成为审稿靶子。",
        "",
        "## 产出文件",
        "",
        "- `figures/source_data/site_fact_sheet.csv`：逐地点事实（机读）",
        "- `generated/site_fact_sheet_zh.tex` / `_en.tex`：LaTeX 表格片段",
        "",
        "改完画面构成列后重跑脚本即可，LaTeX 会同步更新。",
    ]
    md.parent.mkdir(parents=True, exist_ok=True)
    io.open(md, "w", encoding="utf-8").write("\n".join(mlines) + "\n")

    print("已生成：")
    for p in (csv_path, zh, en, md):
        print("  ", p)
    print()
    print(f"合计：{tot_pos} 视野 / {tot_rgb} RGB / {tot_win} 窗口")


if __name__ == "__main__":
    main()