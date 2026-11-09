from pathlib import Path
import json
import hashlib
import fitz
from PIL import Image, ImageOps, ImageDraw

OUT = Path(__file__).resolve().parent
QA = OUT / 'qa_six_figures'
QA.mkdir(exist_ok=True)
stems = ['figV9_generalization_gap_zh', 'figV10_baseline_comparison_zh',
         'figV11_misalignment_sensitivity_zh', 'figV12_aggregation_zh',
         'v3_inventory_folds_zh', 'fig_training_monitoring_zh']
notes = []
tiles = []
for stem in stems:
    doc = fitz.open(OUT / f'{stem}.pdf')
    assert len(doc) == 1
    page = doc[0]
    txt = page.get_text()
    assert '\ufffd' not in txt
    assert any('\u4e00' <= x <= '\u9fff' for x in txt)
    pix = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
    pix.save(QA / f'{stem}.png')
    (QA / f'{stem}.txt').write_text(txt, encoding='utf-8')
    tile = Image.new('RGB', (1000, 820), 'white')
    preview = Image.open(QA / f'{stem}.png').convert('RGB')
    preview.thumbnail((970,770))
    tile.paste(preview, ((1000-preview.width)//2,30))
    ImageDraw.Draw(tile).text((15,10),stem,fill='black')
    tiles.append(tile)
    notes.append({'figure': stem, 'pages':len(doc), 'pdf_chinese_text':True,
                  'bounds_pt':list(page.rect), 'font_records':page.get_fonts()})
    doc.close()
sheet=Image.new('RGB',(2000,2460),'#eeeeee')
for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*1000,(i//2)*820))
sheet.save(QA/'six_figures_contact_sheet.png')

new=json.loads((OUT/'source_data/chinese_result_figures.json').read_text(encoding='utf-8'))
old=json.loads((OUT.parents[1]/'figures/source_data/revision_results_20261015.json').read_text(encoding='utf-8'))
for k in ['generalization','sensitivity','aggregation']:assert new[k]==old[k],k
for a,b in zip(new['baseline']['values'],old['baseline']['values']):
    assert {k:v for k,v in a.items() if k!='label'}=={k:v for k,v in b.items() if k!='label'}
assert new['baseline']['best_learned_psnr']==old['baseline']['best_learned_psnr']
audit=json.loads((OUT/'source_data/chinese_figures_audit.json').read_text(encoding='utf-8'))
for name,v in audit['outputs'].items():
    assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==v['sha256']
(QA/'qa_summary.json').write_text(json.dumps({'numeric_translation_identity':True,
    'output_hashes_valid':True,'six_pdf_files':notes,'visual_review':'Rendered for inspection'},
    ensure_ascii=False,indent=2),encoding='utf-8')
print('Six PDFs rendered; numeric inputs unchanged; all 18 export hashes valid.')
