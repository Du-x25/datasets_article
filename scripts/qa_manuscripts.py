from pathlib import Path
import json, re, hashlib, argparse
from pypdf import PdfReader
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Verify manuscript evidence and compiled assets')
parser.add_argument('--language',choices=['zh','en','both'],default='both')
args=parser.parse_args()
LANGUAGES=['zh','en'] if args.language=='both' else [args.language]
if 'zh' in LANGUAGES and 'results_20261109_macros_zh.tex' in (ROOT/'main_zh.tex').read_text(encoding='utf-8'):
    from qa_results_20261109 import validate
    validate()
    LANGUAGES.remove('zh')
    if not LANGUAGES:
        raise SystemExit(0)
p=json.loads((ROOT/'docs/current_protocol.json').read_text(encoding='utf-8'))
assert p['physical_sites']==10 and p['seeds']==[11] and p['submission_seeds']==[11,23,37]
tests=[]
for f in p['folds']:
    a,b,c=map(set,[f['train'],f['validation'],f['test']])
    assert not(a&b or a&c or b&c) and len(a)==6 and len(b)==len(c)==2
    assert len(a|b|c)==10
    tests.extend(c)
assert len(set(tests))==len(tests)==10
for lang in LANGUAGES:
    s=(ROOT/f'main_{lang}.tex').read_text(encoding='utf-8')
    assert 'seven sites' not in s and '七个地点' not in s
    assert not re.search(r'\\input\{generated/(input|model|scene|qualitative)_benchmark_',s)
    assert 'LOSO' not in s and '\\DatasetUsablePatchCount' not in s
    assert ('seed 11' in s or '种子11' in s)
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', s, re.S)[1]
    assert '11.78' not in abstract and '\\GeneralizationGap' not in abstract
    assert '7125' not in s and '7,125' not in s
    assert 'EX-1' in s and 'EX-2' in s
    assert 'pix2pix-Turbo' not in s
    path=ROOT/'build'/lang/f'main_{lang}.pdf'
    dependencies = [ROOT/f'main_{lang}.tex', ROOT/'refs.bib']
    dependencies += [ROOT / (name if Path(name).suffix else name + '.tex') for name in re.findall(r'\\input\{([^}]+)\}', s)]
    dependencies += [(ROOT/name if (ROOT/name).is_file() else ROOT/'figures'/name) for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', s)]
    for dependency in dependencies:
        assert dependency.is_file(), f'missing dependency: {dependency}'
        assert path.stat().st_mtime >= dependency.stat().st_mtime, f'stale PDF {lang}: newer source {dependency}'
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
for lang in LANGUAGES:
    inv=(ROOT/f'generated/scene_inventory_{lang}.tex').read_text(encoding='utf-8')
    for m in re.finditer(r'(C\d+) & (\d+)\$\\times\$(\d+) &',inv):
        scene,w,h=m.group(1),int(m.group(2)),int(m.group(3))
        row=prov[scene]
        assert (w,h)==(int(row['canvas_width']),int(row['canvas_height'])),f'inventory {lang} {scene} {w}x{h} != provenance {row["canvas_width"]}x{row["canvas_height"]}'
    assert len(re.findall(r'C\d+ &',inv))==10
    assert 'Pending freeze' not in inv and 'Final training pool' not in inv
    assert all(len(line.split('&')) == 3 for line in inv.splitlines() if re.match(r'C\d+ &', line)), f'{lang}: inventory columns differ'
bib=(ROOT/'refs.bib').read_text(encoding='utf-8')
bibkeys=set(re.findall(r'@\w+\{([^,]+),',bib))
for lang in LANGUAGES:
    s=(ROOT/f'main_{lang}.tex').read_text(encoding='utf-8')
    cited=set()
    for m in re.finditer(r'\\cite\{([^}]*)\}',s):
        cited|={k.strip() for k in m.group(1).split(',')}
    missing=sorted(cited-bibkeys)
    assert not missing,f'{lang}: cited keys missing from refs.bib: {missing}'
print('Inventory/provenance consistency and citation-key checks passed; pending scientific evidence is not certified by these checks.')

# Check the published result cells against the saved numerical evidence. This
# catches model/reference macro mix-ups and reversed sensitivity signs that a
# successful TeX build or a citation-key check cannot detect.
SUMMARY = ROOT.parents[1] / 'Gen/v3_pipeline/outputs/summary_v2_20261005'
DIAGNOSTIC = ROOT.parents[1] / 'Gen/outputs/fitting_diagnostic_20261015/fitting_diagnostic.json'
SUPPORT_DIR = SUMMARY.parent / 'draft_support_v2_20261002'
def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

main_rows = {(r['condition'], r['kind']): r for r in csv_rows(SUMMARY / 'main_table.csv')}
macros = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]+)\}',
                        (ROOT / 'generated/results_macros_zh.tex').read_text(encoding='utf-8')))
def number(cell):
    cell = re.sub(r'\\([A-Za-z]+)(?:\{\})?', lambda m: macros[m[1]], cell.strip())
    return float(cell.replace('$', '').strip())

expected_conditions = [
    ('reference_bands1,2,3,4,5', 'linear_reference'),
    ('restormer_bands1,2,3,4,5', 'model'),
    ('tfd_bands1,2,3,4,5', 'model'),
    ('unet_bands1,2,3,4,5', 'model'),
    ('pix2pix_bands1,2,3,4,5', 'model'),
    ('pix2pix_bands1,2,4', 'model'),
    ('pix2pix_bands1', 'model'),
    ('reference_bands1,2,3,4,5', 'channel_reference'),
]
expected_labels = {
    'zh': ['线性参照（五波段）', 'Restormer（五波段）', 'TFD-GAN（五波段）', 'U-Net（五波段）',
           'pix2pix（五波段）', 'pix2pix（B1、B2、B4）', 'pix2pix（B1单波段）', 'B1/B2/B3直映'],
    'en': ['Linear reference (5 bands)', 'Restormer (5 bands)', 'TFD-GAN (5 bands)', 'U-Net (5 bands)',
           'pix2pix (5 bands)', 'pix2pix (B1, B2, B4)', 'pix2pix (B1 only)', 'B1/B2/B3 channel map'],
}
metrics = ['psnr_site_equal', 'ssim_site_equal', 'lpips_rectangle_mean_site_equal',
           'deltaE00_mean_site_equal', 'mae_site_equal']
with (SUMMARY / 'ex3_sensitivity.csv').open(encoding='utf-8-sig', newline='') as stream:
    sensitivity_lines = list(csv.reader(stream))
header_at = next(i for i, r in enumerate(sensitivity_lines) if r[:3] == ['mode', 'magnitude_px', 'site_equal'])
sensitivity = {(r[0], float(r[1])): dict(zip(sensitivity_lines[header_at], r))
               for r in sensitivity_lines[header_at + 1:] if r}

for lang in LANGUAGES:
    text = (ROOT / f'main_{lang}.tex').read_text(encoding='utf-8')
    table = re.search(r'\\label\{tab:baseline-' + lang + r'\}.*?\\midrule(.*?)\\bottomrule', text, re.S)
    assert table, f'{lang}: main result table not found'
    rows = [line.strip().removesuffix('\\\\').split('&') for line in table[1].splitlines() if '&' in line]
    assert len(rows) == len(expected_conditions), f'{lang}: unexpected main-table row count'
    for cells, key, label in zip(rows, expected_conditions, expected_labels[lang]):
        assert len(cells) == 6
        assert cells[0].strip() == label, f'{lang}: unexpected condition label: {cells[0]}'
        for cell, metric in zip(cells[1:], metrics):
            actual, expected = number(cell), float(main_rows[key][metric])
            assert abs(actual - expected) < 0.000051, f'{lang}: {key} {metric}: {actual} != {expected}'
    table = re.search(r'\\label\{tab:sensitivity-' + lang + r'\}.*?\\midrule(.*?)\\bottomrule', text, re.S)
    assert table, f'{lang}: sensitivity table not found'
    rows = [line.strip().removesuffix('\\\\').split('&') for line in table[1].splitlines() if '&' in line]
    assert len(rows) == 4
    assert all(len(cells) == 5 for cells in rows), f'{lang}: expected four sensitivity values per row'
    assert {number(cells[0]) for cells in rows} == {0.5, 1.0, 2.0, 4.0}, f'{lang}: missing or repeated sensitivity magnitude'
    for cells in rows:
        magnitude = number(cells[0])
        expected = [float(sensitivity[(mode, magnitude)][metric]) for mode in ['joint', 'B1_only']
                    for metric in ['psnr_drop', 'deltaE00_rise']]
        actual = [number(c) for c in cells[1:]]
        assert all(abs(a - b) < 0.00051 for a, b in zip(actual, expected)), f'{lang}: sensitivity at {magnitude}: {actual} != {expected}'

diagnostic = json.loads(DIAGNOSTIC.read_text(encoding='utf-8'))
groups = {g: [r for r in diagnostic['per_site'] if r['group'] == g]
          for g in ['train_seen', 'test_heldout']}
means = {g: sum(r['window_mean_psnr'] for r in rows) / len(rows) for g, rows in groups.items()}
for macro, value in [('SeenPsnr', means['train_seen']), ('HeldoutPsnr', means['test_heldout']),
                     ('GeneralizationGap', means['train_seen'] - means['test_heldout']),
                     ('SeenWindowCount', sum(r['n_windows'] for r in groups['train_seen'])),
                     ('HeldoutWindowCount', sum(r['n_windows'] for r in groups['test_heldout']))]:
    assert abs(float(macros[macro]) - value) < 0.0051, f'diagnostic macro mismatch: {macro}'
export = json.loads((SUMMARY / 'export_metadata.json').read_text(encoding='utf-8'))
assert int(macros['TrainRunCount']) == export['runs']
assert int(macros['SensitivityRecordCount']) == export['ex3_condition_files']
support_path = SUMMARY.parent / 'draft_support_v2_20261002/manifest.json'
support = json.loads(support_path.read_text(encoding='utf-8'))
assert hashlib.sha256(support_path.read_bytes()).hexdigest() == export['support_sha256']
assert support['dataset_sha256'] == export['dataset_sha256']
assert support['status'] == 'provisional_after_exploration_not_human_frozen'
assert int(macros['SupportPixelCount']) == sum(s['valid_pixels'] for s in support['scenes'].values())
assert int(macros['QualifiedWindowCount']) == sum(len(s['windows']) for s in support['scenes'].values())
assert int(macros['LpipsRectCount']) == sum(len(s['lpips_rectangles']) for s in support['scenes'].values())
import math
site_rows = csv_rows(SUMMARY / 'site_records.csv')
for row in site_rows:
    assert row['site'] in support['scenes'], f'unknown site: {row["site"]}'
    assert int(row['valid_pixels']) == support['scenes'][row['site']]['valid_pixels'], f'support pixel count mismatch: {row["site"]}'
for macro, condition, kind in [('PooledPsnrLinear', 'reference_bands1,2,3,4,5', 'linear_reference'),
                                ('PooledPsnrRestormer', 'restormer_bands1,2,3,4,5', 'model')]:
    rows = [r for r in site_rows if (r['condition'], r['kind']) == (condition, kind)]
    assert len(rows) == 10 and {r['site'] for r in rows} == set(support['scenes'])
    pooled_mse = sum(10 ** (-float(r['psnr']) / 10) * int(r['valid_pixels']) for r in rows) / sum(int(r['valid_pixels']) for r in rows)
    assert abs(float(macros[macro]) + 10 * math.log10(pooled_mse)) < 0.000051, f'pooled macro mismatch: {macro}'
figure_trace = json.loads((ROOT / 'figures/source_data/revision_results_20261015.json').read_text(encoding='utf-8'))
for source in figure_trace['sources']:
    source_path = Path(source['path'])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source['sha256'], f'figure input changed: {source_path}'
print('Bilingual result cells, sensitivity signs, diagnostic macros and exported counts match saved evidence.')

# The support tables describe the same version as the benchmark, and the new
# image plates must retain the recorded coordinates and untouched exports.
support_report = json.loads((ROOT/'figures/source_data/support_statistics_20261015.json').read_text(encoding='utf-8'))
assert support_report['support_manifest_sha256'] == export['support_sha256']
assert support_report['dataset_manifest_sha256'] == export['dataset_sha256']
assert support_report['support_status'] == support['status']
stats = support_report['sites']
assert len(stats) == 10 and {r['site'] for r in stats} == set(support['scenes'])
for row in stats:
    existing = support['scenes'][row['site']]
    assert row['common_valid_pixels'] == existing['valid_pixels']
    assert row['windows'] == len(existing['windows'])
    assert row['lpips_rectangles'] == len(existing['lpips_rectangles'])
    assert 0 <= row['lpips_unique_pixels'] <= row['common_valid_pixels'] <= row['canvas_pixels']
    assert 0 <= row['ssim_centres'] <= row['common_valid_pixels']
    expected = [f"{row['common_valid_pixels']/1e6:.3f}", f"{row['common_canvas_pct']:.2f}",
                f"{row['ssim_centres']/1e6:.3f}", str(row['windows']), str(row['lpips_rectangles']),
                f"{row['lpips_unique_pixels']/1e6:.3f}", f"{row['lpips_common_pct']:.2f}"]
    for lang in LANGUAGES:
        table = (ROOT/f'generated/support_statistics_{lang}.tex').read_text(encoding='utf-8')
        line = next(line for line in table.splitlines() if line.startswith(row['site'] + ' &'))
        cells = [cell.strip() for cell in line.removesuffix('\\\\').split('&')]
        assert cells[1:] == expected, f'{lang}: support values differ at {row["site"]}'
assert sum(r['common_valid_pixels'] for r in stats) == int(macros['SupportPixelCount'])
assert sum(r['ssim_centres'] for r in stats) == 58532167
assert sum(r['lpips_unique_pixels'] for r in stats) == 23805940
atlas = json.loads((ROOT/'figures/source_data/local_comparisons_20261015.json').read_text(encoding='utf-8'))
windows = atlas['selected_windows']
assert len(windows) == 3 and len({w['scene'] for w in windows}) == 3
for window in windows:
    scene = window['scene']
    assert scene in support['scenes'], f'plate scene not in support: {scene}'
    assert scene in p['folds'][window['fold']-1]['test'], f'plate scene not in its test fold: {scene}'
    x, y, w, h = window['xywh']
    mask = np.asarray(Image.open(SUPPORT_DIR / support['scenes'][scene]['mask'])) > 0
    assert [w, h] == [256, 256], f'plate window size is not 256x256: {scene}'
    assert int(mask[y:y + h, x:x + w].sum()) == w * h, \
        f'plate window is not fully covered by shared support: {scene} {window["xywh"]}'
    assert window['rgb_display_mean'] >= 55, f'plate window is not exposure screened: {scene}'
    assert 'pix2pix / B1-5' in window['psnr_db'] and window['psnr_db']['pix2pix / B1-5'] > 11, \
        f'plate window has no saved-prediction support: {scene}'
    for panel, digest in window['panel_sha256'].items():
        assert len(digest) == 64, f'plate panel hash malformed: {scene} {panel}'
for name, record in atlas['exports'].items():
    export_path = ROOT / 'figures/zh' / name
    assert export_path.is_file(), f'missing local comparison export: {name}'
    assert hashlib.sha256(export_path.read_bytes()).hexdigest() == record['sha256'], \
        f'local comparison export changed: {name}'
for lang in LANGUAGES:
    text = (ROOT/f'main_{lang}.tex').read_text(encoding='utf-8')
    for name in ['support_statistics', 'support_dynamic_range']:
        assert f'\\input{{generated/{name}_{lang}.tex}}' in text
    assert re.search(r'fig_local_predictions_20261015(?:_zh)?\.pdf',text) and re.search(r'fig_local_modalities_20261015(?:_zh)?\.pdf',text)
    assert 'tab:energy' not in text and 'Submission extension seeds' not in text
print('New support tables, exposure-screened plate windows and local-image exports passed checks.')

if 'zh' in LANGUAGES:
    zh=(ROOT/'main_zh.tex').read_text(encoding='utf-8')
    assert '运行目录没有可用于判断收敛过程的逐步验证序列' not in zh
    assert 'fig_training_monitoring_zh.pdf' in zh
    assert '700幅文件的三个通道均逐像素相同' in zh
    assert '同平台学位论文 &' not in zh and '同平台期刊论文 &' not in zh
    print('Chinese-only draft corrections passed; English synchronization is deferred.')
