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
print('Protocol split and manuscript checks passed; pending scientific evidence is not certified by these checks.')
