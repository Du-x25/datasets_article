"""Portable evidence and compiled-PDF checks for the revised Chinese paper."""
from pathlib import Path
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from build_results_20261109 import analyze, ROOT, SRC, MODELS, METRICS, NAMES


def validate():
    d = analyze()
    assert json.loads(json.dumps(d['stats'])) == json.loads((SRC/'summary.json').read_text(encoding='utf-8'))
    dependencies = {ROOT/'refs.bib'}
    def read_tex(path):
        dependencies.add(path)
        s = path.read_text(encoding='utf-8')
        for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', s):
            image = ROOT / name
            assert image.is_file(), name
            dependencies.add(image)
        return re.sub(r'\\input\{([^}]+)\}', lambda m: read_tex(ROOT/(m[1] if Path(m[1]).suffix else m[1]+'.tex')), s)
    source = (ROOT/'main_zh.tex').read_text(encoding='utf-8')
    tex = read_tex(ROOT/'main_zh.tex')
    labels = re.findall(r'\\label\{([^}]+)\}',tex)
    refs = re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',tex)
    assert len(labels)==len(set(labels)), 'Duplicate labels'
    assert not set(refs)-set(labels), set(refs)-set(labels)
    keys=set(re.findall(r'@\w+\s*\{([^,]+),',(ROOT/'refs.bib').read_text(encoding='utf-8')))
    cited={k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}',tex) for k in block.split(',')}
    assert cited <= keys, cited-keys
    for old in ['results_macros_zh.tex','fig:local-modalities-zh','Restormer','GeneralizationGap','固定末步检查点','30次训练','仅扰动B1']:
        assert old not in source, old
    macros=dict(re.findall(r'\\newcommand\{\\([^}]+)\}\{([^}]+)\}',tex))
    for m,prefix in [('drunet','Drunet'),('pix2pix','Pix'),('tfd','Tfd')]:
        for group in ['single','triple']:
            assert macros[prefix+group.title()] == f"{d['selected_mean'].loc[(m,group),'psnr']:.3f}"
        assert macros[prefix+'Gain']==f"{d['gains'][m].mean():.3f}"
    benchmark=(ROOT/'generated/selected_benchmark_20261109_zh.tex').read_text(encoding='utf-8')
    rows=[r for r in benchmark.splitlines() if r.startswith(tuple(NAMES.values()))]
    assert len(rows)==6
    for line,(model,group) in zip(rows,[(m,g) for m in MODELS for g in ['single','triple']]):
        cells=[x.strip() for x in line.removesuffix('\\\\').split('&')]
        assert cells[2:]==[f"{d['selected_mean'].loc[(model,group),metric]:.3f}" for metric in METRICS]
    fixed=(ROOT/'generated/all_inputs_20261109_zh.tex').read_text(encoding='utf-8')
    rows=[r for r in fixed.splitlines() if r.startswith(tuple(NAMES.values()))]
    assert len(rows)==45
    inv={v:k for k,v in NAMES.items()}
    for line in rows:
        cells=[x.strip() for x in line.removesuffix('\\\\').split('&')]
        assert cells[2:]==[f"{d['recomputed'].loc[(inv[cells[0]],cells[1]),metric]:.3f}" for metric in METRICS]
    # The reused image panels must agree with validation selections and frozen ROIs.
    import pandas as pd
    manifest=json.loads((SRC/'figure_selection_manifest.json').read_text())
    for site in manifest['fixed_main_sites']:
        plate=pd.read_csv(SRC/f'plates/planned_F2_validation_selected_drunet_{site}_source.csv')
        roi=next(r for r in manifest['rois'] if r['site']==site)
        for key in ['x','y','width','height','display_min','display_max']:
            assert (plate[key]==roi[key]).all()
        for group in ['single','triple']:
            r=d['selected'][(d['selected'].site==site)&(d['selected'].model=='drunet')&(d['selected'].group==group)].iloc[0]
            assert r.task_id in set(plate.task_id.dropna())
    pdf=ROOT/'build/zh/main_zh.pdf'
    assert pdf.is_file() and pdf.stat().st_mtime >= max(p.stat().st_mtime for p in dependencies)
    log=(pdf.parent/'main_zh.log').read_text(encoding='utf-8',errors='replace')
    issues=re.findall(r'^.*(?:undefined|Overfull \\[hv]box|Float too large|Citation.*Warning|Rerun to get|Missing character).*$' ,log,re.M|re.I)
    assert not issues, issues
    reader=PdfReader(pdf)
    texts=[page.extract_text() for page in reader.pages]
    assert all('??' not in t and '\ufffd' not in t for t in texts)
    assert all(t.strip() for t in texts)
    assert len(reader.pages)>=15
    # Poppler's actual text boxes catch text crossing the physical page boundary.
    poppler=shutil.which('pdftotext')
    assert poppler, 'pdftotext is required for page-boundary verification'
    xml=subprocess.run([poppler,'-bbox',str(pdf),'-'],capture_output=True,check=True).stdout
    tree=ET.fromstring(xml)
    words=0
    for page in tree.findall('.//{*}page'):
        width,height=float(page.attrib['width']),float(page.attrib['height'])
        for word in page.findall('.//{*}word'):
            a=word.attrib
            assert -1<=float(a['xMin'])<=float(a['xMax'])<=width+1
            assert -1<=float(a['yMin'])<=float(a['yMax'])<=height+1
            words+=1
    result={'status':'passed','pages':len(reader.pages),'labels':len(labels),'citations':len(cited),
            'checked_text_boxes':words,'undefined_references':0,'oversized_floats':0,'overfull_boxes':0,
            'completed_runs_verified':270,'scope':'Saved-record, source, table and PDF layout checks; no new inference.'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    validate()
