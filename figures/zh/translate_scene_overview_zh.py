"""Replace one title only; preserve the existing image objects and placement."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
from collections import Counter
import contextlib,io,re,shutil
import pymupdf
from pypdf import PdfReader

OUT=Path(__file__).resolve().parent
SRC=OUT.parents[1]/'figures'
EN='Ten paired scenes with RGB and five SWIR bands'
ZH='十个地点的RGB与五波段SWIR成图'
FONT=Path(r'C:\Windows\Fonts\msyhbd.ttc')
source=SRC/'fig1_scene_modality_overview.pdf'
target=OUT/'fig1_scene_modality_overview_zh.pdf'
before_fix=OUT/'tmp/fig1_scene_modality_overview_zh_before_cmap_fix.pdf'
before_fix.parent.mkdir(exist_ok=True)
if target.exists() and not before_fix.exists():shutil.copy2(target,before_fix)
doc=pymupdf.open(source);page=doc[0]
spans=[s for b in page.get_text('dict')['blocks']if 'lines'in b for l in b['lines']for s in l['spans']if s['text']==EN]
assert len(spans)==1
s=spans[0];bbox=pymupdf.Rect(s['bbox']);origin=s['origin'];size=s['size']
# A narrow strip intersects title glyphs only; the full font box also overlaps
# the B1–B3 column-label boxes and must not be used for redaction.
title_strip=pymupdf.Rect(bbox.x0,bbox.y0,bbox.x1,bbox.y0+2)
page.add_redact_annot(title_strip,fill=(1,1,1),cross_out=False)
page.apply_redactions(images=0,graphics=0,text=0)
font=pymupdf.Font(fontfile=str(FONT))
x=page.rect.width/2-font.text_length(ZH,fontsize=size)/2
page.insert_text((x,origin[1]-3),ZH,fontname='ChineseOverviewTitle',fontfile=str(FONT),fontsize=size,color=(0,0,0))
doc.subset_fonts()
# MuPDF's full-font ToUnicode for this TTC includes unused astral codepoints as
# odd-length hex strings. Bind only the title CIDs to valid UTF-16BE strings.
# This changes character extraction only; glyph IDs, outlines and positions stay.
title_font=next(item for item in page.get_fonts(full=True)if item[4]=='ChineseOverviewTitle')
cmap_ref=int(doc.xref_get_key(title_font[0],'ToUnicode')[1].split()[0])
mapping={font.has_glyph(ord(ch)):ch for ch in ZH}
assert 0 not in mapping and all(0<=cid<=0xffff for cid in mapping)
mapping_lines=[f'<{cid:04X}> <{ch.encode("utf-16-be").hex().upper()}>'for cid,ch in sorted(mapping.items())]
cmap=('''/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /ChineseOverviewTitle-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
'''+f'{len(mapping_lines)} beginbfchar\n'+'\n'.join(mapping_lines)+'''
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
''').encode('ascii')
doc.update_stream(cmap_ref,cmap)
doc.save(target,garbage=4,deflate=True);doc.close()

svg_source=SRC/'fig1_scene_modality_overview.svg'
tree=ET.parse(svg_source);changed=0
for e in tree.iter():
    if e.text==EN:
        e.text=ZH
        style=e.get('style','')
        e.set('style',style+"; font-family: 'Microsoft YaHei'; font-weight: bold")
        e.set('y',str(float(e.get('y'))-3))
        changed+=1
assert changed==1,changed
tree.write(OUT/'fig1_scene_modality_overview_zh.svg',encoding='utf-8',xml_declaration=True)

old,new=pymupdf.open(source),pymupdf.open(target)
assert old[0].rect==new[0].rect
def image_records(document):
    page=document[0]
    records=[]
    for item in page.get_images():
        pix=pymupdf.Pixmap(document,item[0])
        records.append({'shape':[pix.width,pix.height,pix.n],
            'pixels_sha256':hashlib.sha256(pix.samples).hexdigest(),
            'placements':[[list(rect),list(matrix)]for rect,matrix in page.get_image_rects(item[0],transform=True)]})
    return records
a,b=image_records(old),image_records(new)
assert a==b and len(a)==60,'Image pixels or placements changed'
assert EN not in new[0].get_text() and ZH in new[0].get_text()
expected_text=Counter(old[0].get_text().splitlines());expected_text[EN]-=1;expected_text[ZH]+=1
assert +expected_text==Counter(new[0].get_text().splitlines()),'A non-title label changed'
parse_stderr=io.StringIO()
with contextlib.redirect_stderr(parse_stderr):
    parsed_text=PdfReader(target,strict=True).pages[0].extract_text()
assert parse_stderr.getvalue()=='',parse_stderr.getvalue()
assert ZH in parsed_text
prior=pymupdf.open(before_fix)
assert prior[0].get_pixmap(dpi=150).samples==new[0].get_pixmap(dpi=150).samples,'CMap repair changed the rendered page'
new[0].get_pixmap(dpi=220).save(OUT/'fig1_scene_modality_overview_zh.png')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
csv_source=SRC/'source_data/fig1_display_provenance.csv'
csv_target=OUT/'source_data/fig1_display_provenance_zh.csv'
csv_target.write_bytes(csv_source.read_bytes())
report={'operation':'只替换英文标题及标题字体；保留现有全部图像对象、显示分位、顺序与放置矩阵',
    'source_pdf':str(source),'source_pdf_sha256':sha(source),'source_svg_sha256':sha(svg_source),
    'translated_title':ZH,'font':str(FONT),'embedded_images':60,'all_image_arrays_and_placement_matrices_identical':True,
    'all_non_title_text_preserved':True,'title_baseline_adjustment_pt':-3,
    'ToUnicode_repair':'Only actual title CIDs mapped to valid UTF-16BE strings; unused invalid full-font entries removed',
    'ToUnicode_mapping_count':len(mapping),'pypdf_strict_parse_stderr':'',
    'page_render_identical_to_before_CMap_fix':True,
    'page_rect':list(new[0].rect),'percentile_provenance_csv_sha256':sha(csv_target),
    'csv_identical_to_original':sha(csv_target)==sha(csv_source),
    'outputs':{p.name:sha(p)for p in [target,OUT/'fig1_scene_modality_overview_zh.svg',OUT/'fig1_scene_modality_overview_zh.png']}}
(OUT/'source_data/fig1_overview_zh_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: 60 image arrays and placement matrices unchanged; title translated; percentile provenance identical.')
