"""Publish existing fixed crops and saved predictions; no fitting or inference."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parents[1] / 'Gen'
REVIEW = GEN / 'outputs/experiment_review_20261005'
MANIFEST = GEN / 'v3_pipeline/configs/v3_scene_manifest.json'
SUPPORT = GEN / 'v3_pipeline/outputs/draft_support_v2_20261002/manifest.json'
OUT = ROOT / 'figures'
SOURCE = OUT / 'source_data'
MAIN_SITES = ['C1', 'C4', 'C7', 'C10']
CONDITIONS = ['U-Net / B1-5', 'pix2pix / B1-5', 'TFD / B1-5',
              'Restormer / B1-5', 'pix2pix / B1', 'pix2pix / B1,2,4']
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'font.size': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def arr_sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def show_panel(ax, arr, mask, bar=False):
    # One explicit display transform for all panels. No interpolation or local adjustment.
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    shown = arr.copy() if arr.dtype == np.uint8 else np.rint(arr * 255).astype(np.uint8)
    shown[~mask] = 140
    ax.imshow(shown, interpolation='none', vmin=0, vmax=255)
    ax.set_xlim(-.5, 255.5)
    ax.set_ylim(255.5, -.5)
    ax.set_axis_off()
    if bar:
        ax.add_patch(Rectangle((178, 233), 64, 5, fc='white', ec='black', lw=.35))
        ax.text(210, 226, '64 px', ha='center', color='white', fontsize=5.8,
                bbox=dict(facecolor='black', edgecolor='none', pad=.5))


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)


def make_main(cache, selection, modalities=False):
    names = ['RGB', 'B1', 'B2', 'B3', 'B4', 'B5'] if modalities else [
        'RGB', 'B1', 'U-Net / B1-5', 'Restormer / B1-5', 'pix2pix / B1-5']
    headers = ['RGB target', 'B1 input', 'B2', 'B3', 'B4', 'B5'] if modalities else [
        'RGB target', 'B1 input', 'U-Net (5 bands)', 'Restormer (5 bands)', 'pix2pix (5 bands)']
    width = 7.2
    left, right, gap = .08, .08, .055
    panel = (width - left - right - gap * (len(names)-1)) / len(names)
    row_head = .24
    height = .30 + 4 * (panel + row_head) + .16
    fig = plt.figure(figsize=(width, height))
    for row, scene in enumerate(MAIN_SITES):
        x, y, w, h = selection['scenes'][scene]['A']['xywh']
        row_top = height - .30 - row * (panel + row_head)
        fig.text(left/width, (row_top-.04)/height,
                 f"{'abcd'[row]}  {scene}-A | test fold {selection['scenes'][scene]['fold']} | (x, y) = ({x}, {y})",
                 ha='left', va='top', fontsize=7, weight='bold')
        for col, (name, title) in enumerate(zip(names, headers)):
            px = left + col * (panel + gap)
            if row == 0:
                fig.text((px+panel/2)/width, (height-.08)/height, title, ha='center', va='top', fontsize=7.3)
            ax = fig.add_axes([px/width, (row_top-row_head-panel)/height, panel/width, panel/height])
            show_panel(ax, cache[(scene, 'A')][name], cache[(scene, 'A')]['mask'], bar=col == 0)
    fig.text(left/width, .025/height, '256 × 256 output pixels per panel; identical coordinates and display; illustrative fixed-window subset.', fontsize=6.3)
    save(fig, 'fig_local_modalities_20261015' if modalities else 'fig_local_predictions_20261015')


def make_supplement(cache, selection):
    names = ['RGB', 'B1'] + CONDITIONS
    labels = ['RGB target', 'B1 input'] + CONDITIONS
    with PdfPages(OUT / 'supp_local_comparisons_20261015.pdf') as pdf:
        for site_num in range(1, 11):
            scene = f'C{site_num}'
            fig, axes = plt.subplots(4, 4, figsize=(9.0, 10.4))
            fig.subplots_adjust(left=.025, right=.975, top=.902, bottom=.045, hspace=.19, wspace=.055)
            fig.text(.025, .976, f'{scene} | saved seed-11 test predictions | fold {selection["scenes"][scene]["fold"]}', fontsize=12, weight='bold')
            coord = '   '.join(f'{n}: (x, y) = {tuple(selection["scenes"][scene][n]["xywh"][:2])}' for n in ['A', 'B'])
            fig.text(.025, .949, coord + '   | 256 × 256 output pixels', fontsize=9)
            for block, crop in enumerate(['A', 'B']):
                for col, (name, label) in enumerate(zip(names, labels)):
                    ax = axes[block*2+col//4, col % 4]
                    show_panel(ax, cache[(scene, crop)][name], cache[(scene, crop)]['mask'], bar=col == 0)
                    ax.set_title(f'{crop}{col+1}  {label}', fontsize=9, loc='left', pad=4)
            fig.text(.025, .019, 'Fixed support-only selection; 0–255 display, no enhancement; gray = excluded support. No statistical or geometry-accuracy claim.', fontsize=7)
            pdf.savefig(fig, facecolor='white')
            fig.savefig(ROOT / 'tmp' / f'local_comparisons_{scene}_20261015.png', dpi=110, facecolor='white')
            plt.close(fig)


def main():
    SOURCE.mkdir(exist_ok=True)
    (ROOT / 'tmp').mkdir(exist_ok=True)
    selection_path = REVIEW / 'visual_selection.json'
    previous_path = REVIEW / 'visual_provenance.json'
    selection, previous = load_json(selection_path), load_json(previous_path)
    manifest, support = load_json(MANIFEST), load_json(SUPPORT)
    assert sha(MANIFEST) == selection['scene_manifest_sha256']
    assert sha(SUPPORT) == selection['support_manifest_sha256']
    assert sha(selection_path) == previous['selection_sha256']
    trace = {
        'date': '2026-10-15',
        'contract': {
            'claim': 'Same-coordinate local input, target and saved prediction views expose examples of retained structure and color/detail differences; they do not estimate prevalence, alignment accuracy or model superiority.',
            'archetype': 'image plate comparison grid', 'backend': 'Python matplotlib',
            'main_subset': 'A windows for every third site in numerical order: C1, C4, C7, C10. Deterministic illustrative subset of the existing 20 windows; not selected by model score.',
            'all_windows': 'Supplement contains both already-selected windows at each of all 10 sites and all six saved model/input conditions.',
            'selection_timing_boundary': 'The existing retrospective selection was committed before loading predictions in that visual-review procedure; not experiment preregistration.',
            'display': 'RGB and B1–B5 uint8 values unchanged; saved float32 predictions converted with round(255*x) only. No histogram stretch, gamma, sharpening, smoothing or selective contrast. Excluded support is RGB(140,140,140). Native-pixel imshow interpolation=none.',
            'scale': '256×256 output-grid pixels per panel; 64 output-pixel bar, not a physical scale or native sensor-pixel calibration.',
            'statistics': 'No new measurement or statistical comparison; no model execution, fitting or inference.',
            'identity': 'Full source TIFF, mask, result JSON and prediction NPY hashes recomputed. Sampled raw float32 crop hashes also compared with previous review.'},
        'python': sys.version, 'matplotlib': matplotlib.__version__,
        'selection_rule': selection['rule'], 'sources': {}, 'windows': []}
    def record(path, expected=None):
        key = str(path)
        if key not in trace['sources']:
            trace['sources'][key] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        digest = trace['sources'][key]['sha256']
        if expected:
            assert digest == expected, f'Hash mismatch: {path}'
        return digest
    for p in [selection_path, previous_path, MANIFEST, SUPPORT]:
        record(p)
    cache, rows = {}, []
    data_root = Path(manifest['root'])
    for site_num in range(1, 11):
        scene = f'C{site_num}'
        chosen = selection['scenes'][scene]
        mask_path = Path(chosen['mask_path'])
        record(mask_path, chosen['mask_sha256_verified'])
        mask = np.asarray(Image.open(mask_path)) > 0
        for crop in ['A', 'B']:
            x, y, w, h = chosen[crop]['xywh']
            assert [x, y, w, h] in support['scenes'][scene]['windows']
            m = mask[y:y+h, x:x+w]
            assert int(m.sum()) == chosen[crop]['valid_pixels']
            cache[(scene, crop)] = {'mask': m.copy()}
            row = {'scene': scene, 'window': crop, 'fold': chosen['fold'], 'x': x, 'y': y,
                   'width': w, 'height': h, 'valid_pixels': int(m.sum()), 'valid_fraction': float(m.mean()),
                   'main_figure': scene in MAIN_SITES and crop == 'A', 'panels': {}}
            trace['windows'].append(row)
            rows.append({k:v for k,v in row.items() if k != 'panels'})
        for band in ['RGB', 'B1', 'B2', 'B3', 'B4', 'B5']:
            entry = manifest['scenes'][scene][band]
            path = data_root / entry['path']
            record(path, entry['sha256'])
            raw = np.asarray(Image.open(path))
            assert list(raw.shape) == entry['shape'] and raw.dtype == np.uint8
            for crop in ['A', 'B']:
                x,y,w,h = chosen[crop]['xywh']
                arr = raw[y:y+h,x:x+w].copy()
                cache[(scene,crop)][band] = arr
                row = next(q for q in trace['windows'] if q['scene'] == scene and q['window'] == crop)
                row['panels'][band] = {'source': str(path), 'uint8_crop_sha256': arr_sha(arr)}
            del raw
        for label in CONDITIONS:
            prior = previous['scenes'][scene]['predictions'][label]
            result_path, path = Path(prior['result_path']), Path(prior['prediction_path'])
            record(result_path, prior['result_sha256_verified'])
            result = load_json(result_path)
            config = result['config']
            assert config['seed'] == 11 and config['fold'] == chosen['fold'] and scene in config['test_sites']
            assert config['support_sha256'] == selection['support_manifest_sha256']
            assert config['dataset_sha256'] == selection['scene_manifest_sha256']
            record(path, result['artifacts'][path.name])
            pred = np.load(path, mmap_mode='r')
            for crop in ['A', 'B']:
                x,y,w,h = chosen[crop]['xywh']
                arr = np.array(pred[y:y+h,x:x+w], copy=True)
                assert arr.dtype == np.float32 and np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1
                digest = arr_sha(arr)
                assert digest == prior['crops'][crop]['float32_bytes_sha256']
                cache[(scene,crop)][label] = arr
                row = next(q for q in trace['windows'] if q['scene'] == scene and q['window'] == crop)
                row['panels'][label] = {'source': str(path), 'result': str(result_path), 'float32_crop_sha256': digest, 'run': prior['run']}
            del pred
        print(f'Verified {scene}: six source modalities and six saved prediction arrays', flush=True)
    make_main(cache, selection, modalities=False)
    make_main(cache, selection, modalities=True)
    make_supplement(cache, selection)
    record(Path(__file__))
    trace['exports'] = {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                        for pat in ['fig_local_*_20261015.*', 'supp_local_comparisons_20261015.pdf'] for p in OUT.glob(pat)}
    (SOURCE / 'local_comparisons_20261015.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
    with (SOURCE / 'local_comparisons_20261015.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Complete: {len(trace["sources"])} source hashes verified; 20 windows; 2 main plates and 10-page supplement.', flush=True)


if __name__ == '__main__':
    main()
