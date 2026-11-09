"""Chinese labels on saved results, plus saved validation monitoring curves.

Contract: quantitative figures retain the existing data and aggregation; the
new 2x3 monitoring grid describes six conditions across five validation folds.
No training, inference, smoothing, curve selection or new significance tests.
Python matplotlib only; 183 mm-width vector PDF/SVG and 300 dpi PNG previews.
"""
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

OUT=Path(__file__).resolve().parent
PAPER=OUT.parents[1]
PIPE=Path(r'D:\USTC\Gen\v3_pipeline')
DATA=OUT/'source_data'
DATA.mkdir(exist_ok=True)
FONT='Microsoft YaHei'
font_path=font_manager.findfont(FONT, fallback_to_default=False)
SOURCES={}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bind(p):
    p=Path(p)
    SOURCES[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}


def chinese_results():
    original=PAPER/'scripts/make_result_figures_20261014.py'
    bind(original)
    source=original.read_text(encoding='utf-8')
    replacements={
        '["Arial", "Helvetica", "DejaVu Sans"]':'["Microsoft YaHei"]',
        'U-Net (5)':'U-Net（五波段）', 'pix2pix (5)':'pix2pix（五波段）',
        'TFD-GAN (5)':'TFD-GAN（五波段）', 'Restormer (5)':'Restormer（五波段）',
        'pix2pix (B1)':'pix2pix（B1）', 'pix2pix (B124)':'pix2pix（B1/B2/B4）',
        'Linear reference (5)':'线性参照（五波段）',
        'Linear reference (B124)':'线性参照（B1/B2/B4）',
        'Linear reference (B1)':'线性参照（B1）',
        'Linear reference':'线性参照',
        'f"gap {gap:.2f} dB"':'f"落差 {gap:.2f} dB"',
        'Open circles: site means; C1: 1 window\\nBlack marks: B1 grayscale-copy reference':'空心点：地点均值；C1仅1个窗口\\n黑色横线：B1灰度复制参照',
        'Training\\n3 sites / {window_counts[0]} windows':'训练地点\\n3地点 / {window_counts[0]}窗口',
        'Held-out\\n2 sites / {window_counts[1]} windows':'未见地点\\n2地点 / {window_counts[1]}窗口',
        'a  First fold: fixed windows':'a  首折：固定窗口诊断',
        'Held-out site':'留出地点',
        'b  Five folds: full-scene support':'b  五折：整图公共支持',
        'Each site is tested in its assigned fold\\nA different evaluation unit from panel a':'各地点在所属留出折上评价\\n与面板a的评价单位和支持不同',
        'site-equal PSNR (dB)':'地点等权PSNR（dB）',
        'training-only linear reference':'仅训练地点拟合的线性参照',
        'learned model (5 bands)':'学习模型（五波段）',
        'learned model (band ablation)':'学习模型（波段消融）',
        'Dashed line: best learned model, Restormer ({best:.4f} dB)':'虚线：学习模型中最高，Restormer（{best:.4f} dB）',
        'all SWIR bands':'全部SWIR波段', 'B1 only':'仅B1',
        'Displacement (pixels, B1 reference grid)':'位移（B1参考网格像素）',
        'PSNR drop (dB; baseline − perturbed)':'PSNR损失（dB；基准减扰动）',
        'Site-equal\\n(main analysis)':'地点等权\\n（主分析）',
        'Pixel-pooled MSE':'像素汇总MSE',
    }
    for old,new in replacements.items():
        assert old in source, old
        source=source.replace(old,new)
    for stem in ['figV9_generalization_gap','figV10_baseline_comparison',
                 'figV11_misalignment_sensitivity','figV12_aggregation']:
        source=source.replace(stem+'.{ext}',stem+'_zh.{ext}')
    # Chinese condition labels require slightly more width, not different data.
    source=source.replace('left=0.25, right=0.98','left=0.32, right=0.98')
    namespace={'__name__':'translated_saved_results','__file__':str(original)}
    exec(compile(source,str(original), 'exec'),namespace)
    namespace['OUT']=OUT
    rows=namespace['read_rows']()
    namespace['fig_generalization'](rows)
    namespace['fig_baseline'](rows)
    namespace['fig_sensitivity']()
    namespace['fig_aggregation']()
    for p in namespace['SOURCE_PATHS']:bind(p)
    trace=namespace['TRACE']
    trace['translation']='Chinese text only; numerical inputs and calculations retained from original generator'
    trace['font']=FONT
    (DATA/'chinese_result_figures.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')


def chinese_inventory():
    original=PAPER/'scripts/generate_inventory_protocol.py'
    bind(original)
    source=original.read_text(encoding='utf-8')
    replacements={
        '"font.family": "Arial"':'"font.family": "Microsoft YaHei"',
        'labels = ("Tr", "V", "Te")':'labels = ("训", "验", "测")',
        '10 distinct physical sites · 140 acquisition positions':'10个不同物理地点 · 140个采集视野',
        'Inventory and protocol only — no model outcomes':'仅展示采集规模与划分协议，不含模型结果',
        'Acquisition positions':'采集视野数', 'a  Site inventory':'a  逐地点采集规模',
        'f"Fold {fold[\'fold\']}"':'f"第{fold[\'fold\']}折"',
        'b  Site-disjoint five-fold protocol':'b  地点隔离五折协议',
        'Tr = train (6 sites)    V = validation (2 sites)    Te = test (2 sites)':'训：训练（6地点）    验：验证（2地点）    测：测试（2地点）',
        'Reported training seed: {current_seed}':'当前结果的训练种子：{current_seed}',
        'ROOT / f"figures/v3_inventory_folds.{suffix}"':'OUT / f"v3_inventory_folds_zh.{suffix}"',
        'ROOT / "figures/source_data/v3_inventory_folds.csv"':'DATA / "v3_inventory_folds_zh.csv"',
    }
    for old,new in replacements.items():
        assert old in source,old
        source=source.replace(old,new)
    namespace={'__name__':'translated_inventory','__file__':str(original),'OUT':OUT,'DATA':DATA}
    exec(compile(source,str(original),'exec'),namespace)
    namespace['main']()
    for p in [PAPER/'docs/current_protocol.json',PAPER/'figures/source_data/acquisition_provenance.csv']:bind(p)


def monitoring():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[FONT],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':False})
    conditions=[('unet',(1,2,3,4,5),'U-Net（五波段）'),
                ('pix2pix',(1,2,3,4,5),'pix2pix（五波段）'),
                ('tfd',(1,2,3,4,5),'TFD-GAN（五波段）'),
                ('restormer',(1,2,3,4,5),'Restormer（五波段）'),
                ('pix2pix',(1,),'pix2pix（B1）'),
                ('pix2pix',(1,2,4),'pix2pix（B1/B2/B4）')]
    paths={}
    for p in (PIPE/'outputs/draft_runs_v2_20261002').glob('*/result.json'):
        result=json.loads(p.read_text(encoding='utf-8'));cfg=result['config']
        if cfg['seed']==11:paths[(cfg['model'],tuple(cfg['bands']),cfg['fold'])]=(p,result)
    assert len(paths)==30
    points=[]; runs=[]; windows={}; curves={}
    for model,bands,label in conditions:
        for fold in range(1,6):
            p,result=paths[(model,bands,fold)];bind(p)
            c=result['config'];curve=result['curve'];boxes=c['validation_windows']
            assert c['checkpoint_selection']=='last_update_fixed_budget'
            assert len(boxes)==32 and Counter(b[0] for b in boxes)==Counter({s:16 for s in c['val_sites']})
            assert set(c['val_sites']).isdisjoint(c['test_sites'])
            if fold in windows:assert windows[fold]==boxes
            else:windows[fold]=boxes
            assert curve[-1]['update']==c['budget_updates']==result['updates_done']
            assert [x['update'] for x in curve]==list(range(2000,c['budget_updates']+1,2000))
            for point in curve:
                val=point['validation']
                assert val['window_count']==32 and val['use']=='monitoring_only_not_checkpoint_selection'
                assert abs(val['site_equal_psnr']-np.mean(list(val['per_site_psnr'].values())))<1e-10
                points.append({'condition':label,'model':model,'bands':','.join(map(str,bands)),
                    'fold':fold,'seed':11,'update':point['update'],'budget_updates':c['budget_updates'],
                    'validation_sites':','.join(c['val_sites']),'validation_window_count':val['window_count'],
                    'site_equal_psnr':val['site_equal_psnr'],'per_site_psnr':json.dumps(val['per_site_psnr']),
                    'source':str(p),'source_sha256':SOURCES[str(p)]['sha256']})
            runs.append({'condition':label,'fold':fold,'seed':11,'budget_updates':c['budget_updates'],
                'monitoring_points':len(curve),'validation_sites':','.join(c['val_sites']),
                'fixed_windows':32,'first_update':curve[0]['update'],
                'first_psnr':curve[0]['validation']['site_equal_psnr'],
                'last_update':curve[-1]['update'],'last_psnr':curve[-1]['validation']['site_equal_psnr'],
                'checkpoint_selection':c['checkpoint_selection'],'source':str(p)})
            curves[(model,bands,fold)]=curve
    assert len(points)==330 and len(runs)==30
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.3),sharey=True)
    colors=['#264E73','#2B7A78','#B07A2E','#8D6E9F','#7E8E49']
    styles=['-','--','-.',':',(0,(5,1,1,1))]
    markers=['o','s','^','D','v']
    lower=np.floor(min(x['site_equal_psnr'] for x in points))-.25
    upper=np.ceil(max(x['site_equal_psnr'] for x in points))+.25
    for i,((model,bands,label),ax) in enumerate(zip(conditions,axes.flat)):
        for fold in range(1,6):
            curve=curves[(model,bands,fold)]
            xs=np.array([x['update'] for x in curve])/1000
            ys=[x['validation']['site_equal_psnr'] for x in curve]
            ax.plot(xs,ys,color=colors[fold-1],linestyle=styles[fold-1],marker=markers[fold-1],
                    markersize=2.2,linewidth=.95,label=f'第{fold}折')
        budget=curve[-1]['update']//1000
        ax.set_title(f'{"abcdef"[i]}  {label}',loc='left',fontweight='bold',pad=7)
        ax.text(.98,.03,f'预算：{budget}千次',transform=ax.transAxes,ha='right',va='bottom',
                fontsize=6.3,color='#666666',bbox={'facecolor':'white','edgecolor':'none','pad':1})
        ax.set_xlim(0,budget+0.6)
        ax.set_xticks([0,4,8,12] if budget==12 else ([0,5,10,15,20] if budget==20 else [0,10,20,30,40]))
        ax.set_ylim(lower,upper)
        ax.grid(color='#E6E6E6',linewidth=.5)
        ax.spines[['right','top']].set_visible(False)
        ax.set_xlabel('更新步数（千次）')
        if i%3==0:ax.set_ylabel('验证地点等权PSNR（dB）')
    handles=[Line2D([],[],color=colors[i],linestyle=styles[i],marker=markers[i],markersize=3,
                    linewidth=1,label=f'第{i+1}折') for i in range(5)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.99),ncol=5,frameon=False,
               handlelength=2.5,columnspacing=1.7)
    fig.text(.5,.915,'每折固定验证32窗口（2地点各16）；六条件共用该折窗口，种子11',ha='center',fontsize=7)
    fig.text(.5,.025,'曲线为保存记录的逐点连接，用于观察各条件的训练稳定性；主表结果取自预算末步模型的留出地点整图评价。',ha='center',fontsize=6.8)
    fig.subplots_adjust(left=.085,right=.99,bottom=.13,top=.85,hspace=.45,wspace=.25)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'fig_training_monitoring_zh.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    for name,rows in [('training_monitoring_points.csv',points),('training_monitoring_runs.csv',runs)]:
        with (DATA/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    bind(PIPE/'run_formal.py')
    note={
        'contract':'Six-condition, five-fold monitoring grid; direct saved curves without smoothing or selection',
        'counts':{'runs':30,'monitoring_points':330,'folds':5,'seed':11,'windows_per_fold':32,'windows_per_validation_site':16},
        'validation_windows':windows,
        'metric':'Within each validation site, accumulate squared RGB error over valid pixels in its 16 fixed windows, divide by the valid RGB element count, and convert to PSNR. Average the two site PSNRs equally. This is not the mean of individual window PSNRs.',
        'checkpoint':'last_update_fixed_budget; monitoring_only_not_checkpoint_selection',
        'budgets':{'unet':20000,'pix2pix_all_inputs':20000,'tfd':12000,'restormer':40000},
        'display':'Saved points connected directly; each panel uses its actual update range, common y limits, color+dash+marker fold encodings; no zero-step value invented',
        'interpretation':'Validation windows differ from full-scene test support and the first-fold seen/held-out diagnostic. Curves do not establish convergence or a general method ranking.',
        'statistical_tests':'None; no confidence intervals or independent-replicate claim',
        'font':{'family':FONT,'file':font_path},
    }
    (DATA/'training_monitoring_contract.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    chinese_results();chinese_inventory();monitoring();bind(Path(__file__))
    stems=['figV9_generalization_gap_zh','figV10_baseline_comparison_zh',
           'figV11_misalignment_sensitivity_zh','figV12_aggregation_zh',
           'v3_inventory_folds_zh','fig_training_monitoring_zh']
    outputs=[OUT/f'{stem}.{ext}' for stem in stems for ext in ['pdf','png','svg']]
    assert len(outputs)==18 and all(p.is_file() for p in outputs)
    audit={'scope':'Chinese translation of five existing figures plus one saved-curve monitoring figure',
           'actions':{'training':False,'inference':False,'new_statistical_tests':False,'shared_figure_changes':False},
           'font':FONT,'sources':SOURCES,'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in outputs}}
    (DATA/'chinese_figures_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pdf_count':6,'source_files':len(SOURCES),'outputs':sorted(p.name for p in outputs)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
