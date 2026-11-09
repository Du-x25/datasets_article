"""Rebuild the Chinese paper's figures/tables from bundled saved records only.

No training, prediction, or access to image datasets is performed.  --check
validates sources and recomputes all statistics without modifying outputs.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import itertools
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'figures/source_data/results_20261109'
OUT = ROOT / 'figures/zh/results_20261109'
GEN = ROOT / 'generated'
MODELS = ['drunet', 'pix2pix', 'tfd']
NAMES = {'drunet': 'DRUNet', 'pix2pix': 'pix2pix', 'tfd': 'TFD-GAN'}
SITES = [f'C{i}' for i in range(1, 11)]
METRICS = ['psnr', 'ssim', 'lpips', 'deltaE00', 'mae']


def read(name):
    return pd.read_csv(SRC / name)


def analyze():
    provenance = json.loads((SRC / 'provenance.json').read_text(encoding='utf-8'))
    for item in provenance['files']:
        assert hashlib.sha256((SRC / item['file']).read_bytes()).hexdigest() == item['sha256'], item['file']
    protocol = json.loads((SRC / 'protocols.json').read_text())
    assert len(protocol['windows']) == 5893 and protocol['max_validation_per_site'] == 64
    tests = []
    for train, val, test in protocol['folds']:
        assert len(train) == 6 and len(val) == len(test) == 2
        assert len(set(train + val + test)) == 10
        tests.extend(test)
    assert sorted(tests) == list(range(1, 11))
    inventory = json.loads((SRC / 'training_inventory.json').read_text())
    assert len(inventory) == 270 and len({r['task']['task_id'] for r in inventory}) == 270
    for r in inventory:
        p, t = r['scientific_parameters'], r['task']
        assert r['status'] == 'training_complete' and r['updates_done'] == 16680
        assert t['train_seed'] == 11 and p['batch_size'] == 8 and p['max_updates'] == 16680
        assert p['optimizer'] == 'Adam' and p['weight_decay'] == 0
        assert p['checkpoint_selection'] == 'fixed_validation_windows_site_equal_psnr'
        assert p['schedule'] == 'half_constant_half_linear' and p['augmentation'] == 'D4_uniform_8'
        assert p['lr'] == (0.0001 if t['model'] == 'drunet' else 0.0002)
    per = read('per_site_metrics.csv')
    main = per[per.experiment == 'EX2'].copy()
    assert len(main) == 450 and main.task_id.nunique() == 225
    assert main.groupby(['model', 'input_id']).site.nunique().eq(10).all()
    assert not main.duplicated(['model', 'input_id', 'site']).any()
    for r in main.itertuples():
        assert int(r.site[1:]) in protocol['folds'][int(r.fold)-1][2]
    valid = main.groupby('site').valid_pixels.first()
    assert valid.sum() == 88861466
    reported = read('EX2_main_table.csv').set_index(['model', 'input_id'])[METRICS].sort_index()
    recomputed = main.groupby(['model', 'input_id'])[METRICS].mean().sort_index()
    max_error = float(np.abs(reported - recomputed).to_numpy().max())
    assert max_error < 1e-10
    selections = read('validation_selections.csv')
    costs = read('training_cost.csv').set_index('task_id')
    selected_parts = []
    for r in selections.itertuples():
        for group, choice in [('single', r.best_single), ('triple', r.best_triple)]:
            candidates = [x for x in inventory if x['task']['experiment'] == 'EX2'
                          and x['task']['model'] == r.model and x['task']['fold'] == r.fold
                          and x['task']['input_id'].startswith('S' if group == 'single' else 'T')]
            assert len(candidates) == (5 if group == 'single' else 10)
            best = max(candidates, key=lambda x: x['best']['site_equal_psnr'])
            assert best['task']['input_id'] == choice
            assert abs(best['best']['site_equal_psnr'] - costs.loc[best['task']['task_id'], 'best_validation_psnr']) < 1e-10
            rows = main[(main.model == r.model) & (main.fold == r.fold) & (main.input_id == choice)].copy()
            assert len(rows) == 2
            rows['group'] = group
            selected_parts.append(rows)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected_mean = selected.groupby(['model', 'group'])[METRICS].mean()
    gains = selected.pivot(index='site', columns=['model', 'group'], values='psnr').reindex(SITES)
    gains = pd.DataFrame({m: gains[(m, 'triple')] - gains[(m, 'single')] for m in MODELS})
    baselines = read('EX1_EX2A_baselines.csv')
    baseline_mean = baselines[(baselines.kind == 'linear') & baselines.input_id.isin(['B134', 'B12345'])].groupby('input_id')[METRICS].mean()
    assert len(baseline_mean) == 2
    pairs = read('EX3_shared_domain_protocol_pairs.csv')
    assert len(pairs) == 450 and not pairs.duplicated(['input_id', 'split_seed', 'site']).any()
    for metric in METRICS:
        assert np.allclose(pairs['random_' + metric] - pairs['site_isolated_' + metric], pairs['random_minus_isolated_' + metric], atol=1e-10)
    protocol_site = pairs.groupby('site').mean(numeric_only=True).reindex(SITES)
    overlap = []
    for seed, split in protocol['random_splits'].items():
        ids = [set(split[k]) for k in ['train_ids', 'val_ids', 'test_ids']]
        assert list(map(len, ids)) == [4243, 471, 1179] and len(set.union(*ids)) == 5893
        for site, row in split['overlap_by_site'].items():
            overlap.append({'seed': int(seed), 'site': site, **row,
                            'coverage_pct': 100 * row['train_test_overlap_pixels'] / row['test_valid_pixels']})
    overlap = pd.DataFrame(overlap)
    pooled = overlap.groupby('seed')[['train_test_overlap_pixels', 'test_valid_pixels']].sum()
    pooled_coverage = 100 * pooled.train_test_overlap_pixels / pooled.test_valid_pixels
    cal = read('separated_calibration.csv')
    cal = cal[cal.method.isin(['cross_rgb', 'target_rgb', 'target_rgb_quad'])].copy()
    assert len(cal) == 30 and cal.groupby('site').n_eval.nunique().eq(1).all()
    assert cal.groupby('site').calibration_fraction.nunique().eq(1).all()
    cal_mean = cal.groupby('method')[['psnr', 'deltaE00']].mean()
    half = read('disjoint_calibration.csv')
    assert len(half) == 60 and not half.duplicated(['site', 'fit_half', 'method']).any()
    assert half.groupby(['site', 'fit_half']).n_eval.nunique().eq(1).all()
    half_site = half.groupby(['site', 'method'])[['psnr', 'deltaE00']].mean()
    half_mean = half_site.groupby('method').mean()
    curves = read('validation_curves_T124.csv')
    assert len(curves) == 510 and curves.task_id.nunique() == 15
    assert curves.groupby('task_id')['update'].max().eq(16680).all()
    for r in inventory:
        if r['task']['experiment'] == 'EX2' and r['task']['input_id'] == 'T124':
            curve = curves[curves.task_id == r['task']['task_id']]
            assert abs(curve.site_equal_psnr.max() - r['best']['site_equal_psnr']) < 1e-10
    stats = {
        'main_runs': 225, 'random_runs': 45, 'main_table_max_abs_error': max_error,
        'selected': {m: {g: selected_mean.loc[(m, g)].to_dict() for g in ['single', 'triple']} for m in MODELS},
        'triple_minus_single_psnr': gains.mean().to_dict(), 'sites_with_positive_gain': (gains > 0).sum().to_dict(),
        'protocol_difference': protocol_site.random_minus_isolated_psnr.mean(),
        'protocol_site_difference_range': [protocol_site.random_minus_isolated_psnr.min(), protocol_site.random_minus_isolated_psnr.max()],
        'random_test_coverage_pct_by_seed': pooled_coverage.to_dict(),
        'calibration': cal_mean.to_dict(orient='index'),
        'calibration_affine_gain': cal_mean.loc['target_rgb', 'psnr'] - cal_mean.loc['cross_rgb', 'psnr'],
        'calibration_quadratic_increment': cal_mean.loc['target_rgb_quad', 'psnr'] - cal_mean.loc['target_rgb', 'psnr'],
        'calibration_fraction_range': [cal.calibration_fraction.min(), cal.calibration_fraction.max()],
        'half': half_mean.to_dict(orient='index'),
        'within_minus_cross_psnr': half_mean.loc['within', 'psnr'] - half_mean.loc['cross', 'psnr'],
    }
    return locals()


def tex_table(path, header, rows, spec):
    text = '\\begin{tabular}{@{}' + spec + '@{}}\n\\toprule\n'
    text += ' & '.join(header) + '\\\\\n\\midrule\n'
    text += ''.join(' & '.join(map(str, row)) + '\\\\\n' for row in rows)
    text += '\\bottomrule\n\\end{tabular}\n'
    path.write_text(text, encoding='utf-8')


def generate(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from plot_fonts import select_plot_font
    from pypdf import PdfReader
    from PIL import Image
    family, _ = select_plot_font()
    plt.rcParams.update({'font.family': [family, 'DejaVu Sans'], 'font.size': 9,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 0.7, 'lines.linewidth': 1.3, 'savefig.facecolor': 'white'})
    OUT.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=240)
            if ext == 'svg':
                path = OUT / f'{name}.{ext}'
                path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
        plt.close(fig)
    palette = ['#326A91', '#9A5277', '#338373']
    stats, sm = d['stats'], d['selected_mean']
    macros = {'SupportPixelCount': '88861466', 'QualifiedWindowCount': '5893', 'LpipsRectCount': '1541',
              'FormalRunCount': '225', 'RandomRunCount': '45',
              'ProtocolGain': f"{stats['protocol_difference']:.3f}",
              'CalibrationAffineGain': f"{stats['calibration_affine_gain']:.3f}",
              'CalibrationQuadIncrement': f"{stats['calibration_quadratic_increment']:.3f}",
              'CalibrationCross': f"{stats['calibration']['cross_rgb']['psnr']:.3f}",
              'CalibrationTarget': f"{stats['calibration']['target_rgb']['psnr']:.3f}",
              'CalibrationQuad': f"{stats['calibration']['target_rgb_quad']['psnr']:.3f}",
              'HalfWithin': f"{stats['half']['within']['psnr']:.3f}", 'HalfCross': f"{stats['half']['cross']['psnr']:.3f}",
              'HalfDrop': f"{stats['within_minus_cross_psnr']:.3f}"}
    for m, prefix in [('drunet', 'Drunet'), ('pix2pix', 'Pix'), ('tfd', 'Tfd')]:
        macros[prefix+'Gain'] = f"{stats['triple_minus_single_psnr'][m]:.3f}"
        for g in ['single', 'triple']:
            macros[prefix+g.title()] = f"{sm.loc[(m,g), 'psnr']:.3f}"
    (GEN / 'results_20261109_macros_zh.tex').write_text('% Generated from bundled saved records; do not edit.\n' + ''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()), encoding='utf-8')
    rows = []
    for m in MODELS:
        for g, name in [('single', '验证选单波段'), ('triple', '验证选三波段')]:
            rows.append([NAMES[m], name] + [f'{sm.loc[(m,g),v]:.3f}' for v in METRICS])
    tex_table(GEN/'selected_benchmark_20261109_zh.tex', ['模型','输入选择','PSNR↑','SSIM↑','LPIPS↓',r'\dE{}↓','MAE↓'], rows, 'llrrrrr')
    rows = []
    choices = d['selections'].set_index(['model','fold'])
    for fold in range(1,6):
        rows.append([fold] + [choices.loc[(m,fold), key] for m in MODELS for key in ['best_single','best_triple']])
    tex_table(GEN/'band_choices_20261109_zh.tex', ['折','DRUNet单','DRUNet三','pix2pix单','pix2pix三','TFD单','TFD三'], rows, 'ccccccc')
    calnames = {'cross_rgb':'训练地点仿射','target_rgb':'目标地点仿射','target_rgb_quad':'目标地点二次'}
    rows = [[calnames[k]] + [f'{d["cal_mean"].loc[k,m]:.3f}' for m in ['psnr','deltaE00']] for k in calnames]
    tex_table(GEN/'calibration_20261109_zh.tex',['拟合条件','PSNR/dB↑',r'\dE{}↓'],rows,'lrr')
    long = (r'\begin{longtable}{@{}llrrrrr@{}}' + '\n' +
            r'\caption{全部45种固定输入条件的十地点均值。PSNR单位为dB。}\label{tab:all-inputs-zh}\\' + '\n')
    header = r'\toprule 模型 & 输入 & PSNR↑ & SSIM↑ & LPIPS↓ & \dE{}↓ & MAE↓\\\midrule' + '\n'
    long += header + r'\endfirsthead' + '\n' + header + r'\endhead' + '\n'
    long += r'\midrule\multicolumn{7}{r}{续下页}\\\endfoot\bottomrule\endlastfoot' + '\n'
    for m in MODELS:
        for i,r in d['recomputed'].loc[m].iterrows():
            long += ' & '.join([NAMES[m],i] + [f'{r[v]:.3f}' for v in METRICS]) + '\\\\\n'
    long += r'\end{longtable}' + '\n'
    (GEN/'all_inputs_20261109_zh.tex').write_text(long,encoding='utf-8')
    (SRC/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name, frame in [('selected_test_sites',d['selected']),('band_gain_by_site',d['gains']),('protocol_by_site',d['protocol_site']),('random_overlap',d['overlap']),('half_by_site',d['half_site'])]:
        frame.to_csv(SRC/f'{name}.csv', index=name not in ['selected_test_sites','random_overlap'],float_format='%.15g')

    fig, ax = plt.subplots(1,2, figsize=(183/25.4,91/25.4),gridspec_kw={'width_ratios':[1,1.35]},layout='constrained')
    for j,m in enumerate(MODELS):
        ys=[sm.loc[(m,g),'psnr'] for g in ['single','triple']]
        ax[0].plot([0,1],ys,'o-',color=palette[j],label=NAMES[m])
        for x,y in enumerate(ys): ax[0].annotate(f'{y:.3f}',(x,y),xytext=(0,7 if j!=1 else -15),textcoords='offset points',ha='center',fontsize=8)
    ax[0].set(xticks=[0,1],xticklabels=['验证选单波段','验证选三波段'],xlim=(-.2,1.2),ylim=(11.7,14.2),ylabel='地点等权 PSNR / dB',title='a  不同模型的平均收益')
    ax[0].legend(frameon=False,fontsize=8,loc='lower right');ax[0].grid(axis='y',alpha=.15)
    vals=d['gains'][MODELS].T.to_numpy();bound=max(abs(vals.min()),abs(vals.max()))
    im=ax[1].imshow(vals,cmap='RdBu',vmin=-bound,vmax=bound,aspect='auto')
    ax[1].set(xticks=range(10),xticklabels=SITES,yticks=range(3),yticklabels=[NAMES[m] for m in MODELS],title='b  三波段减单波段：逐地点 PSNR')
    ax[1].tick_params(axis='x',labelsize=7)
    for y,x in itertools.product(range(3),range(10)):
        ax[1].text(x,y,f'{vals[y,x]:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(vals[y,x])>bound*.65 else '#202020')
    fig.colorbar(im,ax=ax[1],orientation='horizontal',shrink=.8,pad=.12,label='PSNR 差 / dB（正值表示三波段较高）')
    save(fig,'band_selection')

    fig,ax=plt.subplots(1,2,figsize=(183/25.4,83/25.4),layout='constrained')
    ps=d['protocol_site'];x=np.arange(10)
    ax[0].vlines(x,ps.site_isolated_psnr,ps.random_psnr,color='#BBBBBB',lw=2)
    ax[0].scatter(x,ps.site_isolated_psnr,color=palette[0],label='地点隔离',s=24)
    ax[0].scatter(x,ps.random_psnr,color=palette[2],label='随机窗口',s=24)
    ax[0].set(xticks=x,xticklabels=SITES,ylabel='共同区域 PSNR / dB',title='a  同一测试区域的配对重评分')
    ax[0].legend(frameon=False,fontsize=8);ax[0].grid(axis='y',alpha=.15)
    ov=d['overlap'].groupby('site').coverage_pct.mean().reindex(SITES)
    ax[1].bar(x,ov,color=palette[0],width=.6)
    ax[1].set(xticks=x,xticklabels=SITES,ylim=(0,105),ylabel='测试像素被训练窗口覆盖 / %',title='b  随机窗口的空间重叠')
    for i,v in enumerate(ov):ax[1].text(i,v+1,f'{v:.1f}',ha='center',fontsize=6.7)
    for a in ax:a.tick_params(axis='x',labelsize=7)
    save(fig,'protocol_sensitivity')

    fig,ax=plt.subplots(1,2,figsize=(183/25.4,89/25.4),layout='constrained')
    for a,frame,order,labels in [(ax[0],d['cal'].pivot(index='site',columns='method',values='psnr'),list(calnames),['训练地点\n仿射','目标地点\n仿射','目标地点\n二次']),
                               (ax[1],d['half_site'].psnr.unstack('method'),['within','global','cross'],['同侧半区','全图分散','对侧半区'])]:
        values=frame.reindex(SITES)[order].to_numpy()
        for i in range(10):a.plot(range(3),values[i],'-o',color='#B3BBC2',alpha=.65,lw=.75,ms=2)
        means=values.mean(axis=0);a.plot(range(3),means,'o-',color=palette[0],lw=2,ms=5,label='十地点均值')
        for i,v in enumerate(means):a.annotate(f'{v:.3f}',(i,v),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8,color=palette[0])
        a.set(xticks=range(3),xticklabels=labels,xlim=(-.2,2.2),ylim=(8,24.5),ylabel='PSNR / dB');a.grid(axis='y',alpha=.15)
    ax[0].set_title('a  固定评价样本上的配对颜色校准')
    ax[1].set_title('b  二次映射的半区迁移')
    ax[0].legend(frameon=False,fontsize=8,loc='upper left')
    save(fig,'paired_calibration')

    fig,ax=plt.subplots(3,3,figsize=(183/25.4,173/25.4),layout='constrained')
    roi_manifest=json.loads((SRC/'figure_selection_manifest.json').read_text())
    for row,site in enumerate(['C1','C5','C8']):
        p=SRC/f'plates/planned_F2_validation_selected_drunet_{site}.pdf'
        images=PdfReader(p).pages[0].images
        assert len(images)>=3
        selection=d['selected'][(d['selected'].model=='drunet')&(d['selected'].site==site)].set_index('group')
        roi=next(r for r in roi_manifest['rois'] if r['site']==site)
        for col,title in enumerate(['RGB',f"单波段 {selection.loc['single','input_id']}",f"三波段 {selection.loc['triple','input_id']}"]):
            im=Image.open(io.BytesIO(images[col].data)).convert('RGB')
            assert im.width==im.height==452
            ax[row,col].imshow(im,interpolation='nearest');ax[row,col].set_title(title,fontsize=9)
            ax[row,col].set(xticks=[],yticks=[])
            for spine in ax[row,col].spines.values():spine.set_visible(False)
            if col==0:
                ax[row,col].set_ylabel(f'{site}\n({roi["x"]}, {roi["y"]})',fontsize=9)
                ax[row,col].plot([.68,.93],[.91,.91],transform=ax[row,col].transAxes,color='white',lw=2)
                ax[row,col].text(.8,.86,'32 px',transform=ax[row,col].transAxes,color='white',ha='center',fontsize=7,bbox={'facecolor':'black','alpha':.35,'edgecolor':'none','pad':1})
    save(fig,'validation_selected_examples')

    fig,ax=plt.subplots(1,3,figsize=(183/25.4,68/25.4),sharey=True,layout='constrained')
    colors=['#326A91','#A65775','#338373','#B17A32','#7664A5']
    for a,m in zip(ax,MODELS):
        for fold in range(1,6):
            c=d['curves'][(d['curves'].model==m)&(d['curves'].fold==fold)]
            a.plot(c['update']/1000,c.site_equal_psnr,label=f'折 {fold}',color=colors[fold-1],lw=1)
            r=c.loc[c.site_equal_psnr.idxmax()];a.scatter(r['update']/1000,r.site_equal_psnr,s=15,color=colors[fold-1],zorder=3)
        a.set(title=NAMES[m],xlabel='更新次数 / 千次',xticks=[0,5,10,15]);a.grid(alpha=.15)
    ax[0].set_ylabel('验证地点等权 PSNR / dB');ax[2].legend(fontsize=6,frameon=False,ncol=2,loc='lower right')
    save(fig,'validation_curves')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    d=analyze()
    if args.check:
        saved=json.loads((SRC/'summary.json').read_text(encoding='utf-8'))
        assert json.loads(json.dumps(d['stats']))==saved
        print('PASS: source hashes, 270 completed tasks, validation-only selections, paired supports and manuscript summaries.')
    else:
        generate(d)
        print(json.dumps(d['stats'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
