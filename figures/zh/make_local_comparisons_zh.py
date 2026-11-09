"""Translate labels only; retain the already-verified crop and pixel workflow."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import warnings

OUT=Path(__file__).resolve().parent
PAPER=OUT.parents[1]
SOURCE_SCRIPT=PAPER/'scripts/make_local_comparisons_20261015.py'
spec=importlib.util.spec_from_file_location('saved_local_figures',SOURCE_SCRIPT)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT=OUT
base.OUT=OUT
base.SOURCE=OUT/'source_data'
base.plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei'],
                         'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})

TRANSLATE={
    'RGB target':'RGB真值','B1 input':'B1输入',
    'U-Net (5 bands)':'U-Net（五波段）','Restormer (5 bands)':'Restormer（五波段）',
    'pix2pix (5 bands)':'pix2pix（五波段）','64 px':'64像素',
    '256 × 256 output pixels per panel; identical coordinates and display; illustrative fixed-window subset.':
        '每格256×256输出像素；相同坐标与显示范围；仅为既有固定窗口的说明性子集。',
    'Fixed support-only selection; 0–255 display, no enhancement; gray = excluded support. No statistical or geometry-accuracy claim.':
        '仅按支持选择的固定窗口；0–255显示，无增强；灰色为排除区域。不代表统计推断或独立几何精度。',
}

def translate(text):
    if text in TRANSLATE:return TRANSLATE[text]
    text=text.replace(' | test fold ',' | 测试折 ').replace(' | saved seed-11 test predictions | fold ',' | 已保存的种子11测试预测 | 折 ')
    text=text.replace('256 × 256 output pixels','256×256输出像素')
    text=text.replace('RGB target','RGB真值').replace('B1 input','B1输入')
    text=text.replace(' / B1-5',' / 五波段').replace(' / B1,2,4',' / B1、B2、B4')
    return text

def translate_figure(fig):
    for item in fig.texts:item.set_text(translate(item.get_text()))
    for ax in fig.axes:
        for item in [ax.title,ax._left_title,ax._right_title,*ax.texts]:
            item.set_text(translate(item.get_text()))

def destination(path):
    if not isinstance(path,(str,Path)):return path
    p=Path(path)
    if p.name=='supp_local_comparisons_20261015.pdf':return p.with_name('supp_local_comparisons_zh.pdf')
    if p.name.startswith('fig_local_') and p.stem.endswith('_20261015'):return p.with_name(p.stem+'_zh'+p.suffix)
    return path

old_savefig=base.plt.Figure.savefig
def savefig(self,path,*args,**kwargs):
    translate_figure(self)
    return old_savefig(self,destination(path),*args,**kwargs)
base.plt.Figure.savefig=savefig
old_pdf_pages=base.PdfPages
base.PdfPages=lambda path,*args,**kwargs:old_pdf_pages(destination(path),*args,**kwargs)

with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    base.main()
    missing=[str(w.message)for w in captured if 'Glyph' in str(w.message) and 'missing' in str(w.message)]
    assert not missing,missing

trace_path=base.SOURCE/'local_comparisons_20261015.json'
trace=json.loads(trace_path.read_text(encoding='utf-8'))
trace['language']='zh-CN'
trace['label_only_adaptation']={'original_script':str(SOURCE_SCRIPT),'original_script_sha256':base.sha(SOURCE_SCRIPT),
    'translated_script':str(Path(__file__)),'translated_script_sha256':base.sha(Path(__file__)),
    'changes':'仅翻译文字、使用Microsoft YaHei字体及更换独立输出目录。裁剪、显示数值、坐标、图像布局、支持和预测来源保持不变。',
    'missing_glyph_warnings':len(missing)}
trace['exports']={str(p.relative_to(OUT)):{'sha256':base.sha(p),'bytes':p.stat().st_size}
    for pattern in ['fig_local_*_zh.*','supp_local_comparisons_zh.pdf']for p in OUT.glob(pattern)}
trace_path.write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
print('Chinese exports complete. Missing glyph warnings:',len(missing),flush=True)
