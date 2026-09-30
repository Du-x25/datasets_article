from pathlib import Path
import json, re, hashlib
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[1]
p=json.loads((ROOT/'docs/current_protocol.json').read_text(encoding='utf-8'))
assert p['physical_sites']==10 and p['seeds']==[11,23,37,53,71]
tests=[]
for f in p['folds']:
    a,b,c=map(set,[f['train'],f['validation'],f['test']])
    assert not(a&b or a&c or b&c) and len(a)==6 and len(b)==len(c)==2
    assert len(a|b|c)==10
    tests.extend(c)
assert len(set(tests))==len(tests)==10
for lang in ['en','zh']:
    s=(ROOT/f'main_{lang}.tex').read_text(encoding='utf-8')
    assert 'seven sites' not in s and '七个地点' not in s
    assert not re.search(r'\\input\{generated/(input|model|scene|qualitative)_benchmark_',s)
    assert 'LOSO' not in s and '\\DatasetUsablePatchCount' not in s
    assert '11, 23, 37, 53 and 71' in s or '11、23、37、53、71' in s
    path=ROOT/'build'/lang/f'main_{lang}.pdf'
    reader=PdfReader(path)
    for i,page in enumerate(reader.pages,1):
        text=page.extract_text() or ''
        assert text.strip(),f'blank page {lang}:{i}'
        assert '\ufffd' not in text,f'replacement character {lang}:{i}'
    print(lang,'pages',len(reader.pages),'sha256',hashlib.sha256(path.read_bytes()).hexdigest())

# Regression guard: inventory tables must match the provenance metadata rows,
# and every cited key must exist in refs.bib (the 2026-09-24..30 C2/C5 mismatch class).
import csv
prov={r['scene']:r for r in csv.DictReader((ROOT/'figures/source_data/acquisition_provenance.csv').open(encoding='utf-8-sig'))}
for lang in ['en','zh']:
    inv=(ROOT/f'generated/scene_inventory_{lang}.tex').read_text(encoding='utf-8')
    for m in re.finditer(r'(C\d+) & (\d+)\$\\times\$(\d+) &',inv):
        scene,w,h=m.group(1),int(m.group(2)),int(m.group(3))
        row=prov[scene]
        assert (w,h)==(int(row['canvas_width']),int(row['canvas_height'])),f'inventory {lang} {scene} {w}x{h} != provenance {row["canvas_width"]}x{row["canvas_height"]}'
    assert len(re.findall(r'C\d+ &',inv))==10
bib=(ROOT/'refs.bib').read_text(encoding='utf-8')
bibkeys=set(re.findall(r'@\w+\{([^,]+),',bib))
for lang in ['en','zh']:
    s=(ROOT/f'main_{lang}.tex').read_text(encoding='utf-8')
    cited=set()
    for m in re.finditer(r'\\cite\{([^}]*)\}',s):
        cited|={k.strip() for k in m.group(1).split(',')}
    missing=sorted(cited-bibkeys)
    assert not missing,f'{lang}: cited keys missing from refs.bib: {missing}'
print('Inventory/provenance consistency and citation-key checks passed; pending scientific evidence is not certified by these checks.')
