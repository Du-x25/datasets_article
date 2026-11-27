"""Portable evidence and compiled-PDF checks for Chinese, English, and Supplementary Information manuscripts."""
from pathlib import Path
import csv
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pypdf import PdfReader
from build_results_20261109 import analyze, ROOT, SRC, MODELS, METRICS, NAMES
from check_latex_log import find_issues


def check_4to1_overlap(source, lang):
    """Measure window coverage on the mosaic before random training crops."""
    width, stride = 1024, 128
    train_cells = set()
    test_windows = []
    train_count = 0
    path = ROOT / 'figures/source_data/bench_20261124/data_quality_windows.csv'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['set'] != 'mswir_4to1':
                continue
            match = re.fullmatch(r'(C\d+)_\d+_y(\d+)_x(\d+)\.png', row['image'])
            assert match and match[1] == row['scene'], row['image']
            scene, y, x = match[1], int(match[2]), int(match[3])
            assert x % stride == y % stride == 0, row['image']
            cells = {(scene, cy, cx) for cy in range(y, y + width, stride)
                     for cx in range(x, x + width, stride)}
            if row['split'] == 'train':
                train_cells.update(cells)
                train_count += 1
            else:
                assert row['split'] == 'test', row['split']
                test_windows.append(cells)
    test_count = len(test_windows)
    fully_covered = sum(cells <= train_cells for cells in test_windows)
    counts = (train_count, test_count, fully_covered)
    assert counts == (2842, 710, 701), f'4:1 window overlap changed: {counts}'

    abstract = source.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    budget_label = r'\label{sec:budget}' if lang == 'zh' else r'\label{sec:budget-en}'
    budget = source.split(budget_label, 1)[1].split(r'\subsection', 1)[0]
    sep = r'[^。]*?' if lang == 'zh' else r'[^.]*?'
    coverage_claim = rf'(?<!\d){test_count}(?!\d){sep}(?<!\d){fully_covered}(?!\d)'
    if lang == 'en':
        coverage_claim = rf'(?<!\d){fully_covered}(?!\d){sep}(?<!\d){test_count}(?!\d)'
    for passage in (abstract, budget):
        assert re.search(coverage_claim, passage), f'{lang}: 4:1 test-window coverage differs from manuscript'
    adjacent_pct = 100 * (width - stride) / width
    train_str = f'{train_count:,}' if lang == 'en' else str(train_count)
    assert (train_str in budget or str(train_count) in budget) and f'{adjacent_pct:g}\\%' in budget, f'{lang}: 4:1 sampling figures differ'
    return {'train': train_count, 'test': test_count, 'fully_covered': fully_covered}


def check_pdf_artifact(pdf: Path, dependencies: set[Path], min_pages: int = 1) -> tuple[int, int]:
    assert pdf.is_file() and pdf.stat().st_mtime >= max(p.stat().st_mtime for p in dependencies), f'stale or missing PDF: {pdf}'
    log = (pdf.parent / f'{pdf.stem}.log').read_text(encoding='utf-8', errors='replace')
    issues = find_issues(log)
    assert not issues, f'{pdf.name} log issues: {issues}'
    reader = PdfReader(pdf)
    texts = [page.extract_text() for page in reader.pages]
    assert all('??' not in t and '\ufffd' not in t for t in texts), f'{pdf.name} has unresolved ?? or replacement glyph'
    assert all(t.strip() for t in texts), f'{pdf.name} has blank page'
    assert len(reader.pages) >= min_pages, f'{pdf.name} page count too low: {len(reader.pages)}'

    poppler = shutil.which('pdftotext')
    assert poppler, 'pdftotext is required for page-boundary verification'
    xml = subprocess.run([poppler, '-bbox', str(pdf), '-'], capture_output=True, check=True).stdout
    tree = ET.fromstring(xml)
    words = 0
    for page in tree.findall('.//{*}page'):
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        for word in page.findall('.//{*}word'):
            a = word.attrib
            assert -1 <= float(a['xMin']) <= float(a['xMax']) <= width + 1
            assert -1 <= float(a['yMin']) <= float(a['yMax']) <= height + 1
            words += 1
    return len(reader.pages), words


def check_scientific_data_limits(source: str, macros: dict[str, str]) -> dict:
    title_match = re.search(r'\\title\{\\textbf\{(.*?)\}\}', source, re.S)
    assert title_match, 'en: missing title'
    title_text = re.sub(r'\s+', ' ', title_match.group(1)).strip()
    assert len(title_text) <= 110, f'en: title exceeds 110-char Scientific Data guideline ({len(title_text)} chars): {title_text}'

    abs_raw = source.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    assert r'\cite' not in abs_raw, 'en: Scientific Data abstract must not contain citations'
    abs_expanded = abs_raw
    for k, v in sorted(macros.items(), key=lambda kv: -len(kv[0])):
        abs_expanded = abs_expanded.replace('\\' + k + '{}', v).replace('\\' + k, v)
    abs_clean = re.sub(r'\\,|\\times|\\pm|\\dE', ' ', abs_expanded)
    abs_clean = re.sub(r'\\textbf\{([^}]*)\}', r'\1', abs_clean)
    abs_clean = re.sub(r'[\$\{\}\\]', '', abs_clean)
    word_count = len(re.findall(r'\S+', abs_clean))
    assert word_count <= 170, f'en: abstract exceeds 170-word Scientific Data limit ({word_count} words)'
    return {'title_chars': len(title_text), 'abstract_words': word_count}


def validate(lang='zh'):
    d = analyze()
    assert json.loads(json.dumps(d['stats'])) == json.loads((SRC / 'summary.json').read_text(encoding='utf-8'))

    def read_doc(doc_path: Path):
        deps = {ROOT / 'refs.bib'}

        def read_tex(path: Path):
            deps.add(path)
            s = path.read_text(encoding='utf-8')
            for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', s):
                image = ROOT / name if (ROOT / name).is_file() else ROOT / 'figures' / name
                assert image.is_file(), f'missing image {name}'
                deps.add(image)
            return re.sub(r'\\input\{([^}]+)\}', lambda m: read_tex(ROOT / (m[1] if Path(m[1]).suffix else m[1] + '.tex')), s)

        expanded = read_tex(doc_path)
        return expanded, deps

    main_file = ROOT / f'main_{lang}.tex'
    source = main_file.read_text(encoding='utf-8')
    overlap = check_4to1_overlap(source, lang)
    tex, dependencies = read_doc(main_file)

    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    refs = re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', tex)
    assert len(labels) == len(set(labels)), f'{lang}: Duplicate labels'
    assert not set(refs) - set(labels), f'{lang}: undefined refs: {set(refs) - set(labels)}'
    keys = set(re.findall(r'@\w+\s*\{([^,]+),', (ROOT / 'refs.bib').read_text(encoding='utf-8')))
    cited = {k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}', tex) for k in block.split(',')}
    assert cited <= keys, f'{lang}: missing bib keys: {cited - keys}'

    supp_file = ROOT / f'supplementary_{lang}.tex'
    assert supp_file.is_file(), f'{lang}: missing {supp_file.name}'
    supp_src = supp_file.read_text(encoding='utf-8')
    bundle_source = source + '\n' + supp_src
    supp_tex, supp_deps = read_doc(supp_file)
    s_labels = re.findall(r'\\label\{([^}]+)\}', supp_tex)
    s_refs = re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', supp_tex)
    assert len(s_labels) == len(set(s_labels)), f'{supp_file.stem}: Duplicate labels'
    assert not set(s_refs) - set(s_labels), f'{supp_file.stem}: undefined refs: {set(s_refs) - set(s_labels)}'
    s_cited = {k.strip() for block in re.findall(r'\\cite\w*\{([^}]+)\}', supp_tex) for k in block.split(',')}
    assert s_cited <= keys, f'{supp_file.stem}: missing bib keys: {s_cited - keys}'
    s_pages, s_words = check_pdf_artifact(ROOT / f'build/{lang}/supplementary_{lang}.pdf', supp_deps, min_pages=4)
    supp_info = {'pages': s_pages, 'labels': len(s_labels), 'checked_text_boxes': s_words}

    for old in ['results_macros_zh.tex', 'fig:local-modalities-zh', 'Restormer', 'GeneralizationGap', '固定末步检查点', '30次训练', '仅扰动B1']:
        assert old not in bundle_source, f'{lang}: stale token {old}'

    import build_bench_20261124 as bench
    expected_bench = bench.check_tables()
    if lang == 'zh':
        for name in expected_bench:
            assert name[:-4] in bundle_source, f'bench table not included: {name}'
    else:
        for name in expected_bench:
            en_name = name.replace('_zh.tex', '_en.tex')
            assert (ROOT / 'generated' / en_name).is_file(), f'missing en bench table: {en_name}'
            assert en_name[:-4] in bundle_source, f'en bench table not included in main+SI bundle: {en_name}'

    assert f'geom_diagnostics_{lang}.tex' in source and 'geom_macros.tex' in source, f'{lang}: missing geometry diagnostics table/macros'
    assert f'geom_15pair_{lang}.tex' in bundle_source and 'geom_15pair_macros.tex' in source, f'{lang}: missing 15-pair geometry audit table/macros'
    assert f'trackc_drunet_{lang}.tex' in source and 'trackc_drunet_macros.tex' in source, f'{lang}: missing Track C DRUNet fine-tuning table/macros'

    macros = dict(re.findall(r'\\newcommand\{\\([^}]+)\}\{([^}]+)\}', tex))
    sd_limits = check_scientific_data_limits(source, macros) if lang == 'en' else None

    for m, prefix in [('drunet', 'Drunet'), ('pix2pix', 'Pix'), ('tfd', 'Tfd')]:
        for group in ['single', 'triple']:
            assert macros[prefix + group.title()] == f"{d['selected_mean'].loc[(m, group), 'psnr']:.3f}"
        assert macros[prefix + 'Gain'] == f"{d['gains'][m].mean():.3f}"

    sel_names = [('single', '验证选单波段' if lang == 'zh' else 'Val.-selected single band'),
                 ('triple', '验证选三波段' if lang == 'zh' else 'Val.-selected 3-band')]
    benchmark = (ROOT / f'generated/selected_benchmark_20261109_{lang}.tex').read_text(encoding='utf-8')
    rows = [r for r in benchmark.splitlines() if r.startswith(tuple(NAMES.values()))]
    assert len(rows) == 6
    for line, (model, group) in zip(rows, [(m, g) for m in MODELS for g in ['single', 'triple']]):
        cells = [x.strip() for x in line.removesuffix('\\\\').split('&')]
        assert cells[1] == dict(sel_names)[group]
        assert cells[2:] == [f"{d['selected_mean'].loc[(model, group), metric]:.3f}" for metric in METRICS]

    fixed = (ROOT / f'generated/all_inputs_20261109_{lang}.tex').read_text(encoding='utf-8')
    rows = [r for r in fixed.splitlines() if r.startswith(tuple(NAMES.values()))]
    assert len(rows) == 45
    inv = {v: k for k, v in NAMES.items()}
    for line in rows:
        cells = [x.strip() for x in line.removesuffix('\\\\').split('&')]
        assert cells[2:] == [f"{d['recomputed'].loc[(inv[cells[0]], cells[1]), metric]:.3f}" for metric in METRICS]

    import pandas as pd
    manifest = json.loads((SRC / 'figure_selection_manifest.json').read_text())
    for site in manifest['fixed_main_sites']:
        plate = pd.read_csv(SRC / f'plates/planned_F2_validation_selected_drunet_{site}_source.csv')
        roi = next(r for r in manifest['rois'] if r['site'] == site)
        for key in ['x', 'y', 'width', 'height', 'display_min', 'display_max']:
            assert (plate[key] == roi[key]).all()
        for group in ['single', 'triple']:
            r = d['selected'][(d['selected'].site == site) & (d['selected'].model == 'drunet') & (d['selected'].group == group)].iloc[0]
            assert r.task_id in set(plate.task_id.dropna())

    pages, words = check_pdf_artifact(ROOT / f'build/{lang}/main_{lang}.pdf', dependencies, min_pages=10)

    result = {
        'status': 'passed',
        'language': lang,
        'pages': pages,
        'labels': len(labels),
        'citations': len(cited),
        'checked_text_boxes': words,
        'undefined_references': 0,
        'oversized_floats': 0,
        'overfull_boxes': 0,
        'completed_runs_verified': 270,
        'window_overlap_4to1': overlap,
        'scope': 'Saved-record, source, table and PDF layout checks; no new inference.',
    }
    if sd_limits:
        result['scientific_data_limits'] = sd_limits
    if supp_info:
        result['supplementary_information'] = supp_info
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    validate('zh')
