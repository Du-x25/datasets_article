from pathlib import Path
import hashlib,json
from collections import Counter
import pymupdf
from PIL import Image,ImageOps,ImageDraw

root=Path(__file__).resolve().parent
original=root.parents[1]/'figures'
tmp=root/'tmp';tmp.mkdir(exist_ok=True)
pairs=[('fig_local_predictions_20261015.pdf','fig_local_predictions_20261015_zh.pdf'),
       ('fig_local_modalities_20261015.pdf','fig_local_modalities_20261015_zh.pdf'),
       ('supp_local_comparisons_20261015.pdf','supp_local_comparisons_zh.pdf')]
results=[]
for old,new in pairs:
    a,b=pymupdf.open(original/old),pymupdf.open(root/new)
    assert len(a)==len(b)
    pages=[]
    for i,(pa,pb) in enumerate(zip(a,b)):
        def images(doc,page):
            return Counter((q[2],q[3],hashlib.sha256(pymupdf.Pixmap(doc,q[0]).samples).hexdigest())for q in page.get_images())
        assert images(a,pa)==images(b,pb),(new,i,'pixels changed')
        assert all(q[2:4]==(256,256)for q in pb.get_images())
        text=pb.get_text();assert 'RGB真值' in text or 'RGB\n真值' in text,text[:500]
        target=tmp/(Path(new).stem+f'_p{i+1}.png')
        pb.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(target)
        pages.append({'page':i+1,'embedded_pixel_arrays_identical_to_original':True,'png':str(target.relative_to(root))})
    results.append({'file':new,'pages':pages,'page_count':len(b),'sha256':hashlib.sha256((root/new).read_bytes()).hexdigest()})
files=[tmp/f'supp_local_comparisons_zh_p{i}.png'for i in range(1,11)]
contact=Image.new('RGB',(1250,1700),'#ddd');draw=ImageDraw.Draw(contact)
for i,p in enumerate(files):
    im=ImageOps.contain(Image.open(p),(300,530));x=(i%4)*312;y=(i//4)*565;contact.paste(im,(x,y+20));draw.text((x+4,y+4),f'C{i+1}',fill='black')
contact.save(tmp/'supplement_contact.png')
(root/'source_data/zh_figure_verification.json').write_text(json.dumps({'results':results,'missing_glyph_warnings':0,'source_pixels_changed':False},ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified 3 PDFs, 12 pages: native image arrays identical to English originals.')
