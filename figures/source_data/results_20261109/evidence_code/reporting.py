"""Automatic paper tables, transparent partial reports and traceable figure sources."""
import csv
import html
import json
import os
from pathlib import Path

import numpy as np

from .data import sha256, stable_hash
from .evaluation import implementation_identity, score_prediction_domains, save_preview
from .protocols import INPUTS, atomic_json, task_protocol

METRICS = ('psnr','ssim','mae','deltaE00','lpips')
DIRECTION = {'psnr':1,'ssim':1,'mae':-1,'deltaE00':-1,'lpips':-1}


def _read(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).is_file() else default


def _number(value):
    if value is None:
        return None
    if value == 'Infinity':
        return float('inf')
    return float(value)


def _json_number(value):
    if value is None or np.isnan(value):
        return None
    return 'Infinity' if np.isposinf(value) else '-Infinity' if np.isneginf(value) else float(value)


def _mean(values):
    values = [_number(v) for v in values if v is not None]
    return _json_number(float(np.mean(values))) if values else None


def _csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else ['status']
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value
                             for key,value in row.items()})
    os.replace(str(temporary),str(path))


def _task_directory(output_root,task):
    for candidate in (Path(output_root)/'tasks'/task['task_id'],Path(output_root)/'runs'/task['task_id'],
                      Path(output_root)/task['task_id']):
        if candidate.is_dir():
            return candidate
    return Path(output_root)/'tasks'/task['task_id']


def _verified_predictions(run_dir,summary):
    """Ignore stale absolute paths after moving a complete project to another machine."""
    paths = {}
    for site in summary.get('sites',{}):
        path = Path(run_dir)/'evaluation/predictions'/(site+'.npy')
        sidecar = _read(path.with_suffix('.json'))
        if not sidecar or not path.is_file() or sidecar.get('sha256') != sha256(path):
            raise ValueError('Missing/corrupt prediction: '+str(path))
        paths[site] = path
    return paths


def _best_validation(run_dir):
    record = _read(Path(run_dir)/'models/best.json')
    if record is None:
        record = (_read(Path(run_dir)/'training_status.json',{}) or {}).get('best')
    return record.get('site_equal_psnr') if record else None


def _json_lines(path):
    if not Path(path).exists():
        return []
    result = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def _save_figure(fig,root,name):
    for suffix in ('png','svg','pdf'):
        fig.savefig(Path(root)/(name+'.'+suffix),dpi=220,bbox_inches='tight')


def build_report(output_root,matrix,store,frozen,lpips_fn,strict=False):
    """Full reports require every requested task, baseline and EX4 result.

    Incomplete/smoke runs still produce useful reports but never become a paper
    result by dropping failed tasks. Selections always use validation PSNR.
    """
    output_root = Path(output_root)
    root = output_root/'report'
    tables,figures = root/'tables',root/'figures'
    tables.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    loaded,missing,site_rows,cost_rows = {},[],[],[]
    for task in matrix:
        run_dir = _task_directory(output_root,task)
        summary = _read(run_dir/'evaluation/summary.json')
        reason = None
        if summary is None or not summary.get('complete'):
            reason = 'missing_or_incomplete_evaluation'
        elif set(summary.get('sites',{})) != set(task['test_sites']):
            reason = 'missing_or_unexpected_test_sites'
        elif summary.get('metric_implementation') != implementation_identity('metric'):
            reason = 'scoring_implementation_changed_rerun_evaluation_only'
        elif summary.get('lpips_identity') != getattr(lpips_fn,'identity',None):
            reason = 'perceptual_weights_changed_rerun_evaluation_only'
        if reason is None:
            try:
                predictions = _verified_predictions(run_dir,summary)
            except (ValueError,OSError) as error:
                reason = str(error)
        if reason:
            missing.append({'task_id':task['task_id'],'reason':reason})
            continue
        loaded[task['task_id']] = {'task':task,'run_dir':run_dir,'summary':summary,'predictions':predictions,
                                    'validation_psnr':_best_validation(run_dir)}
        for site,metrics in summary['sites'].items():
            site_rows.append({'task_id':task['task_id'],'experiment':task['experiment'],'model':task['model'],
                              'input_id':task['input_id'],'fold':task.get('fold'),'split_seed':task.get('split_seed'),
                              'site':site,**{metric:metrics.get(metric) for metric in METRICS},
                              'valid_pixels':metrics.get('valid_pixels'),
                              'lpips_coverage':metrics.get('lpips_details',{}).get('valid_coverage')})
        logs = _json_lines(run_dir/'train_metrics.jsonl')
        status = _read(run_dir/'training_status.json',{})
        timed=[r for r in logs if r.get('step_seconds') is not None]
        mean_step=(sum(r.get('interval_step_seconds',r['step_seconds']*r.get('interval_updates',1)) for r in timed)
                   /sum(r.get('interval_updates',1) for r in timed)) if timed else None
        cost_rows.append({'task_id':task['task_id'],'model':task['model'],'updates':status.get('updates_done'),
                          'last_logged_elapsed_seconds':logs[-1].get('elapsed_seconds') if logs else None,
                          'mean_logged_step_seconds':mean_step,
                          'best_validation_psnr':loaded[task['task_id']]['validation_psnr'],
                          'result_bytes':sum(p.stat().st_size for p in run_dir.rglob('*') if p.is_file())})
    _csv(tables/'per_site_metrics.csv',site_rows)
    _csv(tables/'missing_tasks.csv',missing)
    _csv(tables/'training_cost.csv',cost_rows)

    main_rows = []
    main_tasks = [task for task in matrix if task['experiment']=='EX2']
    for model in sorted({task['model'] for task in main_tasks}):
        for config in INPUTS:
            expected = [task for task in main_tasks if task['model']==model and task['input_id']==config['input_id']]
            if not expected:
                continue
            rows = [r for r in site_rows if r['experiment']=='EX2' and r['model']==model and r['input_id']==config['input_id']]
            expected_sites = sum(len(task['test_sites']) for task in expected)
            main_rows.append({'model':model,'input_id':config['input_id'],'n_sites':len(rows),
                              'expected_sites':expected_sites,'complete':len(rows)==expected_sites,
                              **{m:_mean([r[m] for r in rows]) for m in METRICS}})
    _csv(tables/'EX2_main_table.csv',main_rows)

    selections,gain_rows = [],[]
    for model in sorted({task['model'] for task in main_tasks}):
        for fold in sorted({task['fold'] for task in main_tasks if task.get('fold') is not None}):
            candidates = [item for item in loaded.values() if item['task']['experiment']=='EX2'
                          and item['task']['model']==model and item['task'].get('fold')==fold]
            singles = [item for item in candidates if item['task']['input_id'].startswith('S')]
            triples = [item for item in candidates if item['task']['input_id'].startswith('T')]
            if len(singles)!=5 or any(item['validation_psnr'] is None for item in singles):
                continue
            best_single = max(singles,key=lambda item:item['validation_psnr'])
            best_triple = (max(triples,key=lambda item:item['validation_psnr']) if len(triples)==10
                           and all(item['validation_psnr'] is not None for item in triples) else None)
            selections.append({'model':model,'fold':fold,'best_single':best_single['task']['input_id'],
                               'best_single_validation_psnr':best_single['validation_psnr'],
                               'best_triple':best_triple['task']['input_id'] if best_triple else None,
                               'selection_source':'validation PSNR only'})
            for triple in triples:
                members = [item for item in singles if int(item['task']['input_id'][1:]) in triple['task']['channel_map']]
                best_member = max(members,key=lambda item:item['validation_psnr'])
                for site,score in triple['summary']['sites'].items():
                    row = {'model':model,'fold':fold,'input_id':triple['task']['input_id'],'site':site,
                           'best_single':best_single['task']['input_id'],'best_member_single':best_member['task']['input_id']}
                    for metric in METRICS:
                        value = _number(score[metric])
                        row['gain_'+metric] = _json_number(DIRECTION[metric]*(value-_number(best_single['summary']['sites'][site][metric])))
                        row['member_gain_'+metric] = _json_number(DIRECTION[metric]*(value-_number(best_member['summary']['sites'][site][metric])))
                    gain_rows.append(row)
    _csv(tables/'validation_selections.csv',selections)
    _csv(tables/'EX2_three_band_per_site_gains.csv',gain_rows)
    gain_summary=[]
    for model in sorted({r['model'] for r in gain_rows}):
        for config in INPUTS[5:]:
            rows=[r for r in gain_rows if r['model']==model and r['input_id']==config['input_id']]
            gain_summary.append({'model':model,'input_id':config['input_id'],'n_sites':len(rows),
                                 'complete':len(rows)==10,
                                 **{'gain_'+m:_mean([r['gain_'+m] for r in rows]) for m in METRICS}})
    _csv(tables/'EX2_three_band_gain_summary.csv',gain_summary)

    protocol_rows=[]
    for item in loaded.values():
        task=item['task']
        if task['experiment']!='EX3':
            continue
        oof={}
        oof_tasks={}
        for main in loaded.values():
            mt=main['task']
            if mt['experiment']=='EX2' and mt['model']=='pix2pix' and mt['input_id']==task['input_id']:
                for site,path in main['predictions'].items():
                    if site in oof:
                        raise ValueError('Duplicate OOF prediction for '+site)
                    oof[site]=path
                    oof_tasks[site]=mt['task_id']
        if set(oof)!=set(task['test_sites']):
            continue
        protocol=task_protocol(task,frozen)
        cache=root/'protocol_cache'/(task['task_id']+'.json')
        identity=stable_hash({'protocol':protocol['identity'],'prediction_hashes':{s:sha256(p) for s,p in oof.items()},
                              'metric_implementation':implementation_identity('metric'),
                              'lpips':getattr(lpips_fn,'identity',None)})
        existing=_read(cache)
        if existing and existing.get('identity')==identity:
            oof_metrics=existing['sites']
        else:
            oof_metrics=score_prediction_domains(store,oof,protocol,lpips_fn)
            atomic_json(cache,{'identity':identity,'sites':oof_metrics,'oof_tasks':oof_tasks})
        for site,random_metrics in item['summary']['sites'].items():
            row={'input_id':task['input_id'],'split_seed':task['split_seed'],'site':site,
                 'random_task':task['task_id'],'oof_task':oof_tasks[site]}
            for metric in METRICS:
                row['site_isolated_'+metric]=oof_metrics[site][metric]
                row['random_'+metric]=random_metrics[metric]
                row['random_minus_isolated_'+metric]=_json_number(_number(random_metrics[metric])-_number(oof_metrics[site][metric]))
            protocol_rows.append(row)
    _csv(tables/'EX3_shared_domain_protocol_pairs.csv',protocol_rows)
    random_selections,protocol_gain_rows=[],[]
    for split_seed in (101,202,303):
        candidates=[item for item in loaded.values() if item['task']['experiment']=='EX3'
                    and item['task']['split_seed']==split_seed and item['task']['input_id'].startswith('S')]
        if len(candidates)!=5 or any(item['validation_psnr'] is None for item in candidates):
            continue
        selected=max(candidates,key=lambda item:item['validation_psnr'])
        random_selections.append({'split_seed':split_seed,'best_single':selected['task']['input_id'],
                                  'validation_psnr':selected['validation_psnr']})
        for row in protocol_rows:
            if row['split_seed']!=split_seed or not row['input_id'].startswith('T'):
                continue
            site=row['site']
            fold=next((item['task']['fold'] for item in loaded.values()
                       if item['task']['task_id']==row['oof_task']),None)
            site_selection=next((r for r in selections if r['model']=='pix2pix' and r['fold']==fold),None)
            if not site_selection:
                continue
            site_baseline=next((r for r in protocol_rows if r['input_id']==site_selection['best_single']
                               and r['site']==site and r['split_seed']==split_seed),None)
            if site_baseline is None:
                continue
            gain={'split_seed':split_seed,'site':site,'fold':fold,'input_id':row['input_id'],
                  'random_best_single':selected['task']['input_id'],
                  'site_isolated_best_single':site_selection['best_single']}
            for metric in METRICS:
                gain['random_gain_'+metric]=_json_number(DIRECTION[metric]*(
                    _number(row['random_'+metric])-_number(selected['summary']['sites'][site][metric])))
                gain['site_isolated_gain_'+metric]=_json_number(DIRECTION[metric]*(
                    _number(row['site_isolated_'+metric])-_number(site_baseline['site_isolated_'+metric])))
            protocol_gain_rows.append(gain)
    _csv(tables/'EX3_validation_single_selections.csv',random_selections)
    _csv(tables/'EX3_shared_domain_multiband_gains.csv',protocol_gain_rows)

    sensitivity=[]
    sensitivity_missing=[]
    for task in main_tasks:
        if task['model']!='pix2pix':
            continue
        report=_read(_task_directory(output_root,task)/'sensitivity/summary.json')
        if not report or not report.get('complete') or len(report.get('conditions',{}))!=17:
            sensitivity_missing.append(task['task_id'])
            continue
        for condition,sites in report['conditions'].items():
            for site,metrics in sites.items():
                sensitivity.append({'task_id':task['task_id'],'input_id':task['input_id'],'fold':task.get('fold'),
                                    'condition':condition,'site':site,**{m:metrics[m] for m in METRICS}})
    _csv(tables/'EX4_all_conditions.csv',sensitivity)
    baseline=_read(output_root/'baselines/summary.json',{})
    baseline_rows=[]
    for key in ('fits','direct'):
        for record in baseline.get(key,[]):
            for site,metrics in record['sites'].items():
                baseline_rows.append({'kind':record['kind'],'input_id':record['input_id'],'fold':record['fold'],
                                      'site':site,**{m:metrics[m] for m in METRICS}})
    _csv(tables/'EX1_EX2A_baselines.csv',baseline_rows)

    figure_status=_figures(figures,loaded,main_rows,gain_rows,protocol_rows,sensitivity,baseline_rows,
                          site_rows,cost_rows,selections,store)
    planned_figures,figure_catalog=_planned_figure_groups(
        root,loaded,main_rows,gain_rows,store,selections,cost_rows,sensitivity,output_root,figure_status)
    figure_status.extend(planned_figures)
    gallery=_gallery(root,loaded,store)
    is_full_design=len(matrix)==270 and len(main_tasks)==225
    complete=(is_full_design and not missing and not sensitivity_missing
              and baseline.get('complete') and baseline.get('linear_fit_count')==155
              and baseline.get('direct_evaluation_count')==75 and len(protocol_rows)==450
              and len(selections)==15 and len(gain_rows)==300 and len(protocol_gain_rows)==300)
    report={'status':'complete' if complete else 'partial' if is_full_design else 'smoke_or_subset',
            'expected_training_tasks':len(matrix),'verified_evaluated_tasks':len(loaded),
            'missing_tasks':missing,'missing_sensitivity_tasks':sensitivity_missing,
            'full_design':is_full_design,'figures':figure_status,'gallery_entries':gallery,
            'figure_catalog':figure_catalog,
            'limitations':['One training seed; no seed standard deviation.',
                           'LPIPS covers fixed valid rectangles; inspect per-site coverage.',
                           'Random protocol differences are not a pure causal leakage estimate.',
                           'Reconstruction and simulated shifts do not validate real registration accuracy.']}
    atomic_json(root/'report_status.json',report)
    _write_html(root,report,figure_status)
    if strict and not complete:
        raise RuntimeError('Formal report incomplete. Review '+str(root/'report_status.json'))
    return report


def _figures(root,loaded,main_rows,gain_rows,protocol_rows,sensitivity,baseline_rows,
             site_rows,cost_rows,selections,store):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white','font.size':9})
    manifest=[]
    def save(fig,name,source):
        _save_figure(fig,root,name)
        plt.close(fig)
        _csv(root/(name+'_source.csv'),source)
        manifest.append({'name':name,'status':'generated','source':name+'_source.csv'})
    # Dataset overview is descriptive only; it does not claim registration accuracy.
    from PIL import Image
    sites=list(store.sites)
    fig,axes=plt.subplots(int(np.ceil(len(sites)/5)),5,figsize=(15,3*int(np.ceil(len(sites)/5))),squeeze=False)
    source=[]
    for axis in axes.flat:axis.axis('off')
    for axis,site in zip(axes.flat,sites):
        image=Image.fromarray(store.raw(site)['RGB'])
        image.thumbnail((700,500))
        axis.imshow(image)
        axis.set_title(site)
        source.append({'site':site,'valid_pixels':int(store.masks[site].sum()),
                       'image_height':store.masks[site].shape[0],'image_width':store.masks[site].shape[1]})
    save(fig,'F0_dataset_scene_overview',source)
    if main_rows:
        models=sorted({row['model'] for row in main_rows})
        fig,axes=plt.subplots(1,len(models),figsize=(4.1*len(models),6.7),squeeze=False)
        for axis,model in zip(axes[0],models):
            values=np.full((15,1),np.nan)
            for index,config in enumerate(INPUTS):
                rows=[r for r in main_rows if r['model']==model and r['input_id']==config['input_id']]
                if rows and rows[0]['psnr'] is not None:
                    values[index,0]=_number(rows[0]['psnr'])
            image=axis.imshow(np.ma.masked_invalid(values),aspect='auto',cmap='viridis')
            axis.set_yticks(range(15),[c['input_id'] for c in INPUTS])
            axis.set_xticks([0],['PSNR (dB)'])
            axis.set_title(model)
            for index,value in enumerate(values[:,0]):
                if np.isfinite(value):axis.text(0,index,'%.2f'%value,ha='center',va='center',color='white')
            fig.colorbar(image,ax=axis,shrink=.6)
        save(fig,'F1_all_inputs_psnr',main_rows)
        fig,axes=plt.subplots(1,4,figsize=(15,5),squeeze=False)
        labels=[r['model']+'/'+r['input_id'] for r in main_rows]
        for axis,metric in zip(axes[0],METRICS[1:]):
            values=[_number(r[metric]) if r[metric] is not None else np.nan for r in main_rows]
            axis.plot(range(len(values)),values,'.-',markersize=3)
            axis.set_title(metric+(' ↑' if DIRECTION[metric]>0 else ' ↓'))
            axis.set_xticks(range(len(labels)),labels,rotation=90,fontsize=5)
        save(fig,'S1_all_metrics',main_rows)
    if gain_rows:
        models=sorted({row['model'] for row in gain_rows})
        fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),4),squeeze=False)
        for axis,model in zip(axes[0],models):
            rows=[r for r in gain_rows if r['model']==model]
            inputs=[c['input_id'] for c in INPUTS[5:]]
            for index,name in enumerate(inputs):
                values=[_number(r['gain_psnr']) for r in rows if r['input_id']==name]
                axis.scatter([index]*len(values),values,s=12,alpha=.6)
                if values:axis.scatter([index],[np.mean(values)],marker='_',s=160,color='black')
            axis.axhline(0,color='gray',lw=1)
            axis.set_xticks(range(10),inputs,rotation=60)
            axis.set_ylabel('PSNR gain vs validation-best single (dB)')
            axis.set_title(model)
        save(fig,'F2_per_site_multiband_gains',gain_rows)
    training_rows,validation_rows=[],[]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,item in loaded.items():
        train=_json_lines(item['run_dir']/'train_metrics.jsonl')
        val=_json_lines(item['run_dir']/'validation_metrics.jsonl')
        training_rows.extend(dict(row,task_id=name) for row in train)
        validation_rows.extend(dict(row,task_id=name) for row in val)
        if train:axes[0].plot([r['update'] for r in train],[r.get('g_total',np.nan) for r in train],alpha=.35,lw=.7)
        if val:axes[1].plot([r['update'] for r in val],[r.get('site_equal_psnr',np.nan) for r in val],alpha=.35,lw=.7)
    axes[0].set(xlabel='Update',ylabel='Generator total objective',title='Training histories')
    axes[1].set(xlabel='Update',ylabel='Validation PSNR (dB)',title='Validation histories')
    if training_rows or validation_rows:
        save(fig,'S2_training_and_validation_curves',training_rows+validation_rows)
    else:plt.close(fig)
    if protocol_rows:
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for axis,metric in zip(axes,('psnr','ssim','lpips')):
            a=np.array([_number(r['site_isolated_'+metric]) for r in protocol_rows])
            b=np.array([_number(r['random_'+metric]) for r in protocol_rows])
            axis.scatter(a,b,s=10,alpha=.35)
            lo,hi=min(a.min(),b.min()),max(a.max(),b.max())
            axis.plot([lo,hi],[lo,hi],'--',color='gray')
            axis.set(xlabel='Site-isolated / '+metric,ylabel='Random windows / '+metric)
        save(fig,'F3_protocol_same_region_comparison',protocol_rows)
    if sensitivity:
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for config in INPUTS:
            rows=[r for r in sensitivity if r['input_id']==config['input_id']]
            for axis,metric in zip(axes,('psnr','ssim','lpips')):
                means=[]
                for displacement in (0,.5,1,2,4):
                    selected=[r for r in rows if (r['condition']=='zero' and displacement==0)
                              or (r['condition']!='zero' and float(r['condition'].split('_')[1][:-2])==displacement)]
                    means.append(np.mean([_number(r[metric]) for r in selected]) if selected else np.nan)
                axis.plot([0,.5,1,2,4],means,'.-',label=config['input_id'],lw=.8,markersize=3)
                axis.set(xlabel='Input displacement (published pixels)',ylabel=metric)
        axes[-1].legend(bbox_to_anchor=(1.02,1),fontsize=7)
        save(fig,'F4_common_context_shift_sensitivity',sensitivity)
    if baseline_rows:
        fig,axis=plt.subplots(figsize=(8,4))
        rows=[r for r in baseline_rows if r['kind']=='linear']
        for count in range(1,6):
            vals=[_number(r['psnr']) for r in rows if len(r['input_id'])-1==count]
            if vals:axis.scatter([count]*len(vals),vals,alpha=.2,s=10)
        axis.set(xlabel='Number of physical bands (linear reference only)',ylabel='PSNR (dB)',xticks=range(1,6))
        save(fig,'S3_linear_all_band_subsets',baseline_rows)
    if site_rows:
        fig,axis=plt.subplots(figsize=(10,3.5))
        rows=[]
        for site in sites:
            values=[r['lpips_coverage'] for r in site_rows if r['site']==site and r['experiment']=='EX2'
                    and r['lpips_coverage'] is not None]
            if values:rows.append({'site':site,'coverage_percent':100*float(np.mean(values))})
        axis.bar([r['site'] for r in rows],[r['coverage_percent'] for r in rows],color='#427a91')
        axis.set(ylabel='Unique valid-pixel coverage (%)',title='LPIPS rectangle coverage; not whole-scene perceptual quality')
        save(fig,'S4_lpips_metric_coverage',rows)
        main=[r for r in site_rows if r['experiment']=='EX2']
        if main:
            labels=sorted({r['model']+'/'+r['input_id'] for r in main})
            values=np.full((len(labels),len(sites)),np.nan)
            for row in main:values[labels.index(row['model']+'/'+row['input_id']),sites.index(row['site'])]=_number(row['psnr'])
            fig,axis=plt.subplots(figsize=(9,max(4,len(labels)*.22)))
            image=axis.imshow(np.ma.masked_invalid(values),aspect='auto',cmap='viridis')
            axis.set_yticks(range(len(labels)),labels,fontsize=7)
            axis.set_xticks(range(len(sites)),sites)
            axis.set_title('Held-out site PSNR (each site appears once per input/model)')
            fig.colorbar(image,ax=axis,label='PSNR (dB)')
            save(fig,'S7_per_site_psnr',main)
    if cost_rows:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for index,model in enumerate(sorted({r['model'] for r in cost_rows})):
            rows=[r for r in cost_rows if r['model']==model]
            times=[float(r['mean_logged_step_seconds']) for r in rows if r['mean_logged_step_seconds'] is not None]
            space=[r['result_bytes']/(1024**3) for r in rows]
            if times:axes[0].scatter([model]*len(times),times,s=12,alpha=.5)
            axes[1].scatter([model]*len(space),space,s=12,alpha=.5)
        axes[0].set(ylabel='Mean logged update time (seconds)',title='Observed training cost')
        axes[1].set(ylabel='Retained task data (GiB)',title='Actual output storage')
        save(fig,'S5_compute_and_storage_cost',cost_rows)
    if selections:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for axis,key,title in zip(axes,('best_single','best_triple'),('Validation-selected single band','Validation-selected triple')):
            labels=sorted({row[key] for row in selections if row[key] is not None})
            axis.bar(labels,[sum(row[key]==label for row in selections) for label in labels],color='#427a91')
            axis.set(title=title,ylabel='Number of model/fold selections')
        save(fig,'S6_validation_band_selections',selections)
    if main_rows:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for model in sorted({r['model'] for r in main_rows}):
            rows=[r for r in main_rows if r['model']==model and all(r.get(m) is not None for m in METRICS)]
            axes[0].scatter([_number(r['psnr']) for r in rows],[_number(r['deltaE00']) for r in rows],s=24,label=model)
            axes[1].scatter([_number(r['ssim']) for r in rows],[_number(r['lpips']) for r in rows],s=24,label=model)
        axes[0].set(xlabel='PSNR (dB)',ylabel='DeltaE00',title='Color / distortion trade-off')
        axes[1].set(xlabel='SSIM',ylabel='LPIPS',title='Structure / perception trade-off')
        axes[-1].legend()
        save(fig,'S9_separate_metric_tradeoffs',main_rows)
    expected=['F1_all_inputs_psnr','F2_per_site_multiband_gains','F3_protocol_same_region_comparison',
              'F4_common_context_shift_sensitivity','S1_all_metrics','S2_training_and_validation_curves',
              'S3_linear_all_band_subsets','S4_lpips_metric_coverage','S5_compute_and_storage_cost',
              'S6_validation_band_selections','S7_per_site_psnr','S9_separate_metric_tradeoffs']
    for name in expected:
        if not any(row['name']==name for row in manifest):
            manifest.append({'name':name,'status':'unavailable_missing_required_results'})
    return manifest


def _gallery(root,loaded,store):
    assets=root/'gallery'
    assets.mkdir(exist_ok=True)
    rows=[]
    prepared=set()
    for task_id,item in loaded.items():
        task=item['task']
        for site,path in item['predictions'].items():
            truth=assets/(site+'_RGB.png')
            if site not in prepared:
                save_preview(truth,store.target(site))
                prepared.add(site)
            inp=assets/(site+'_'+task['input_id']+'_input.png')
            if not inp.exists():save_preview(inp,store.input_image(site,task['channel_map']))
            prediction=np.load(path,allow_pickle=False,mmap_mode='r')
            pred=assets/(task_id+'_'+site+'_prediction.png')
            save_preview(pred,prediction)
            error=assets/(task_id+'_'+site+'_absolute_error_x5.png')
            difference=np.mean(np.abs(np.clip(prediction,0,1)-store.target(site)),axis=2)
            difference[~store.masks[site]]=0
            save_preview(error,np.repeat(np.clip(difference*5,0,1)[...,None],3,axis=2))
            metrics=item['summary']['sites'][site]
            rows.append('<article><h3>'+html.escape(task_id+' / '+site)+'</h3><p>'+
                        html.escape(', '.join('%s=%s'%(m,metrics.get(m)) for m in METRICS))+'</p><div class="strip">'+
                        ''.join('<figure><img loading="lazy" src="'+html.escape(p.name)+'"><figcaption>'+label+'</figcaption></figure>'
                                for p,label in [(inp,'SWIR channel reference'),(truth,'RGB target'),(pred,'Prediction'),
                                                (error,'Absolute error ×5, fixed scale')])+'</div></article>')
    text='<!doctype html><meta charset="utf-8"><title>Complete prediction gallery</title><style>body{font:15px sans-serif;background:#fafafa;margin:30px}.strip{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}figure{margin:0}img{width:100%;background:#111}article{margin-bottom:35px}figcaption{padding:6px}</style><h1>All available predictions</h1><p>Fixed rendering. Preview images are downsampled; scientific scores use original floating predictions.</p>'+''.join(rows)
    (assets/'index.html').write_text(text,encoding='utf-8')
    return len(rows)


def _write_html(root,report,figures):
    links=''.join('<figure><h3>'+html.escape(row['name'])+'</h3><img src="figures/'+row['name']+'.png"><figcaption><a href="figures/'+row['name']+'.svg">SVG</a> · <a href="figures/'+row['name']+'.pdf">PDF</a> · <a href="figures/'+row['source']+'">Source CSV</a></figcaption></figure>' for row in figures if row['status']=='generated')
    tables=''.join('<li><a href="tables/'+html.escape(path.name)+'">'+html.escape(path.name)+'</a></li>' for path in sorted((root/'tables').glob('*.csv')))
    text='<!doctype html><meta charset="utf-8"><title>SWIR RGB experiment report</title><style>body{max-width:1300px;margin:40px auto;font:16px sans-serif;color:#172a3a}img{max-width:100%}figure{margin:35px 0;padding:20px;background:#f4f7f9}.status{padding:18px;background:#fff1c5}</style><h1>SWIR → RGB experiment results</h1><p class="status">Status: '+report['status']+'; verified evaluations '+str(report['verified_evaluated_tasks'])+'/'+str(report['expected_training_tasks'])+'. Missing results are not treated as zero.</p><p><a href="gallery/index.html">Open complete prediction gallery</a> · <a href="report_status.json">Completeness and limitations</a></p><h2>Tables</h2><ul>'+tables+'</ul>'+links
    (root/'index.html').write_text(text,encoding='utf-8')


def _fixed_visual_rois(store):
    """Outcome-independent, fixed 128px crop within the centre-nearest support window."""
    rows=[]
    for site in store.sites:
        height,width=store.masks[site].shape
        candidates=store.support['scenes'][site]['windows']
        x,y,w,h=min(candidates,key=lambda b:((b[0]+b[2]/2-width/2)**2+(b[1]+b[3]/2-height/2)**2,b[1],b[0]))
        size=min(128,w,h)
        rows.append({'site':site,'x':int(x+(w-size)//2),'y':int(y+(h-size)//2),'width':size,'height':size,
                     'selection_rule':'centre-nearest frozen support window; independent of predictions',
                     'display_min':0,'display_max':1,'deltaE_min':0,'deltaE_max':50,
                     'edge_error_min':0,'edge_error_max':.25,'data_identity':store.identity})
    return rows


def _training_band_statistics(store,output):
    """Train-site-only descriptive statistics, never a band selection objective."""
    import inspect
    from scipy.ndimage import binary_erosion,sobel
    from .protocols import FOLDS
    identity=stable_hash({'data':store.identity,'implementation':inspect.getsource(_training_band_statistics),
                          'folds':FOLDS,'stride':16})
    path=Path(output)/'training_band_statistics.json'
    prior=_read(path)
    if prior and prior.get('identity')==identity:return prior
    samples={}
    for site in store.sites:
        valid=binary_erosion(store.masks[site],np.ones((3,3),bool))[::16,::16]
        raw=store.raw(site)
        values=np.stack([raw['B%d'%b][::16,::16][valid] for b in range(1,6)],axis=1).astype(np.float64)/255
        if len(values)<10:raise ValueError('Too few training-statistic pixels: '+site)
        edges=[]
        for b in range(1,6):
            array=raw['B%d'%b].astype(np.float32)/255
            magnitude=np.hypot(sobel(array,0)/8,sobel(array,1)/8)
            edges.append(magnitude[::16,::16][valid])
        gray=np.sum(raw['RGB'].astype(np.float32)/255*np.array([.299,.587,.114],np.float32),axis=2)
        target_edge=np.hypot(sobel(gray,0)/8,sobel(gray,1)/8)[::16,::16][valid]
        samples[site]=(values,np.stack(edges,axis=1),target_edge)
    folds=[]
    for fold,(train,_,_) in enumerate(FOLDS,1):
        sites=['C%d'%n for n in train]
        if not set(sites).issubset(samples):continue
        values=np.concatenate([samples[s][0] for s in sites])
        edges=np.concatenate([samples[s][1] for s in sites])
        target=np.concatenate([samples[s][2] for s in sites])
        correlation=np.corrcoef(values,rowvar=False)
        entropy=[]
        for index in range(5):
            hist=np.bincount(np.round(values[:,index]*255).astype(int),minlength=256)
            probability=hist[hist>0]/hist.sum()
            entropy.append(float(-np.sum(probability*np.log2(probability))))
        combos=[]
        for config in INPUTS[5:]:
            indices=np.array(config['channel_map'])-1
            sub=correlation[np.ix_(indices,indices)]
            mean_edge=edges[:,indices].mean(axis=1)
            corr=(float(np.corrcoef(mean_edge,target)[0,1]) if np.std(mean_edge)>0 and np.std(target)>0 else None)
            combos.append({'fold':fold,'input_id':config['input_id'],'train_sites':sites,
                           'sample_pixels':len(values),'mean_band_entropy_bits':float(np.mean(np.array(entropy)[indices])),
                           'mean_abs_pairwise_correlation':_json_number(float(np.mean(np.abs(sub[np.triu_indices(3,1)])))),
                           'edge_correlation_mean_band_gradient_vs_RGB':corr,
                           'formula_scope':'separate descriptive metrics; no weighted total score'})
        folds.append({'fold':fold,'train_sites':sites,'correlation':[[ _json_number(v) for v in row] for row in correlation],
                      'band_entropy_bits':entropy,'combinations':combos})
    result={'identity':identity,'stride':16,'folds':folds,
            'edge_definition':'native-image Sobel magnitude/8; RGB luminance=.299R+.587G+.114B; full 3x3 valid support'}
    atomic_json(path,result)
    return result


def _planned_figure_groups(root,loaded,main_rows,gain_rows,store,selections,cost_rows,sensitivity,output_root,existing):
    """Map the reviewed F0-F4/S1-S12 plan to real outputs; never fabricate absent panels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from scipy.ndimage import sobel
    from skimage.color import rgb2lab,deltaE_ciede2000
    figures=Path(root)/'figures'
    manifest=[]
    rois=_fixed_visual_rois(store)
    roi_by_site={r['site']:r for r in rois}
    _csv(Path(root)/'tables/figure_crop_coordinates.csv',rois)
    atomic_json(Path(root)/'figure_selection_manifest.json',{
        'fixed_main_sites':[s for s in ('C1','C5','C8') if s in store.sites], 'rois':rois,
        'model_comparison_input':'T124','selection_input_comparison':'validation only',
        'failure_rule':'per model: minimum PSNR gain; also minimum triple PSNR, deterministic site/input tie break',
        'RGB_display_range':[0,1],'DeltaE00_display_range':[0,50],'edge_error_display_range':[0,.25]})
    def save(fig,name,rows):
        _save_figure(fig,figures,name)
        plt.close(fig)
        _csv(figures/(name+'_source.csv'),rows)
        manifest.append({'name':name,'status':'generated','source':name+'_source.csv'})
    def find(model,input_id,site):
        matches=[item for item in loaded.values() if item['task']['experiment']=='EX2'
                 and item['task']['model']==model and item['task']['input_id']==input_id and site in item['predictions']]
        return matches[0] if len(matches)==1 else None
    def meta(item,site,label):
        sidecar=_read(item['predictions'][site].with_suffix('.json'),{}) if item else {}
        return dict(roi_by_site[site],panel=label,task_id=item['task']['task_id'] if item else None,
                    model=item['task']['model'] if item else None,input_id=item['task']['input_id'] if item else None,
                    fold=item['task'].get('fold') if item else None,train_seed=item['task'].get('train_seed',11) if item else None,
                    prediction_sha256=sidecar.get('sha256') if item else None)
    def crop(array,site):
        r=roi_by_site[site]
        return np.asarray(array[r['y']:r['y']+r['height'],r['x']:r['x']+r['width']])
    def panel_figure(name,site,panels,full=False,errors=False,reason=None):
        count=len(panels)
        rows_count=2 if full or errors else 1
        fig,axes=plt.subplots(rows_count,count,figsize=(3*count,3*rows_count),squeeze=False)
        sources=[]
        truth=crop(store.target(site),site)
        for index,(label,array,item) in enumerate(panels):
            r=roi_by_site[site]
            roi=np.clip(crop(array,site),0,1)
            if full:
                thumb=Image.fromarray(np.round(np.clip(array,0,1)*255).astype(np.uint8))
                thumb.thumbnail((900,650))
                axes[0,index].imshow(thumb,extent=(0,array.shape[1],array.shape[0],0))
                axes[0,index].add_patch(Rectangle((r['x'],r['y']),r['width'],r['height'],fill=False,color='#ffcc00',lw=1))
                axes[1,index].imshow(roi,vmin=0,vmax=1)
            else:
                axes[0,index].imshow(roi,vmin=0,vmax=1)
                if errors:
                    delta=deltaE_ciede2000(rgb2lab(roi),rgb2lab(truth))
                    mask=crop(store.masks[site],site)
                    delta=np.ma.masked_where(~mask,delta)
                    image=axes[1,index].imshow(delta,vmin=0,vmax=50,cmap='magma')
            axes[0,index].set_title(label,fontsize=9)
            for axis in axes[:,index]:axis.axis('off')
            sources.append(dict(meta(item,site,label),selection_reason=reason))
        if errors:fig.colorbar(image,ax=axes[1,:].tolist(),label='DeltaE00; fixed 0–50',shrink=.75)
        fig.suptitle(site+(' — '+reason if reason else ' — fixed same-location crop'),fontsize=11)
        save(fig,name,sources)
    fixed=[site for site in ('C1','C5','C8') if site in store.sites]

    # F0 explicitly shows all five observed bands and the common input interface.
    site=store.sites[0]
    raw=store.raw(site)
    fig,axes=plt.subplots(2,6,figsize=(17,5),gridspec_kw={'height_ratios':[3,1]})
    sources=[]
    for axis,name in zip(axes[0],['RGB','B1','B2','B3','B4','B5']):
        image=Image.fromarray(raw[name]);image.thumbnail((600,450))
        axis.imshow(image,cmap='gray',vmin=0,vmax=255);axis.set_title(site+' / '+name);axis.axis('off')
        sources.append(dict(roi_by_site[site],modality=name))
    for axis in axes[1]:axis.axis('off')
    labels=['5 single bands\nrepeat each 3 times','10 triples\nascending band order','3 input channels\n256×256','pix2pix / TFD / DRUNet','3-channel RGB\nreconstruction','5 folds\n6 train / 2 val / 2 test']
    for axis,label in zip(axes[1],labels):axis.text(.5,.5,label,ha='center',va='center',bbox={'boxstyle':'round','fc':'#edf3f7','ec':'#56778b'})
    save(fig,'planned_F0_observations_and_task',sources)

    for site in fixed:
        models=[find(model,'T124',site) for model in ('pix2pix','tfd','drunet')]
        if all(models):
            panels=[('RGB target',store.target(site),None),('SWIR B124',store.input_image(site,[1,2,4]),None)]
            fold=models[0]['task'].get('fold')
            linear=Path(output_root)/'baselines/linear'/('fold%d'%fold)/'B124'/(site+'.npy') if fold else None
            if linear and linear.exists():panels.append(('Linear B124',np.load(linear,mmap_mode='r'),None))
            panels.extend((item['task']['model']+' B124',np.load(item['predictions'][site],mmap_mode='r'),item) for item in models)
            panel_figure('planned_F1_fixed_T124_'+site,site,panels,full=True,
                         reason='fixed B124 across models; linear panel requires saved baseline')
        for selection in selections:
            single=find(selection['model'],selection['best_single'],site)
            triple=find(selection['model'],selection.get('best_triple'),site)
            if not single or not triple or single['task'].get('fold')!=selection['fold']:continue
            panels=[('RGB target',store.target(site),None),('Val-selected '+selection['best_single'],np.load(single['predictions'][site],mmap_mode='r'),single),
                    ('Val-selected '+selection['best_triple'],np.load(triple['predictions'][site],mmap_mode='r'),triple)]
            panel_figure('planned_F2_validation_selected_'+selection['model']+'_'+site,site,panels,errors=True)
            fixed_triple=find('pix2pix','T124',site)
            if selection['model']=='pix2pix' and fixed_triple:
                panel_figure('planned_F2_fixed_pix2pix_single_vs_T124_'+site,site,
                             panels[:2]+[('Fixed T124',np.load(fixed_triple['predictions'][site],mmap_mode='r'),fixed_triple)],errors=True)

    # Complete same-position single/triple sheets; no input is chosen using test results.
    for site in store.sites:
        for model in ('pix2pix','tfd','drunet'):
            for group,inputs in [('S1',INPUTS[:5]),('S2',INPUTS[5:])]:
                items=[find(model,config['input_id'],site) for config in inputs]
                if not all(items):continue
                panels=[('RGB target',store.target(site),None)]
                panels.extend((item['task']['input_id'],np.load(item['predictions'][site],mmap_mode='r'),item) for item in items)
                panel_figure('planned_'+group+'_all_inputs_'+model+'_'+site,site,panels)

    if main_rows:
        models=('pix2pix','tfd','drunet')
        fig,axes=plt.subplots(1,3,figsize=(12,7))
        for axis,metric in zip(axes,('psnr','ssim','deltaE00')):
            values=np.full((15,3),np.nan)
            for row in main_rows:
                if row['model'] in models and row[metric] is not None:
                    values[[c['input_id'] for c in INPUTS].index(row['input_id']),models.index(row['model'])]=_number(row[metric])
            image=axis.imshow(np.ma.masked_invalid(values),aspect='auto',cmap='viridis' if DIRECTION[metric]>0 else 'viridis_r')
            axis.set_yticks(range(15),[c['input_id'] for c in INPUTS]);axis.set_xticks(range(3),models,rotation=35)
            axis.set_title(metric);fig.colorbar(image,ax=axis,shrink=.65)
        save(fig,'planned_F3_all_inputs_three_metrics',main_rows)

    if gain_rows:
        models=sorted({r['model'] for r in gain_rows})
        fig,axes=plt.subplots(1,len(models),figsize=(6*len(models),5),squeeze=False)
        limit=max(abs(_number(r['gain_psnr'])) for r in gain_rows if r['gain_psnr'] is not None) or 1
        for axis,model in zip(axes[0],models):
            values=np.full((len(store.sites),10),np.nan)
            for row in gain_rows:
                if row['model']==model:values[store.sites.index(row['site']),[c['input_id'] for c in INPUTS[5:]].index(row['input_id'])]=_number(row['gain_psnr'])
            image=axis.imshow(values,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
            axis.set_yticks(range(len(store.sites)),store.sites);axis.set_xticks(range(10),[c['input_id'] for c in INPUTS[5:]],rotation=60)
            axis.set_title(model);fig.colorbar(image,ax=axis,label='PSNR gain (dB)',shrink=.7)
        save(fig,'planned_S3_site_gain_heatmaps',gain_rows)
        failures=[]
        for model in models:
            negative=[r for r in gain_rows if r['model']==model and _number(r['gain_psnr'])<0]
            if negative:failures.append((min(negative,key=lambda r:(_number(r['gain_psnr']),r['site'],r['input_id'])),'minimum negative PSNR gain'))
            candidates=[item for item in loaded.values() if item['task']['experiment']=='EX2' and item['task']['model']==model and item['task']['input_id'].startswith('T')]
            scored=[(float(score['psnr']),item,site) for item in candidates for site,score in item['summary']['sites'].items() if score['psnr']!='Infinity']
            if scored:
                _,item,site=min(scored,key=lambda r:(r[0],r[2],r[1]['task']['input_id']))
                row=next((r for r in gain_rows if r['model']==model and r['site']==site and r['input_id']==item['task']['input_id']),None)
                if row:failures.append((row,'minimum triple PSNR; explicitly selected failure case'))
        for index,(row,reason) in enumerate(failures):
            site=row['site'];single=find(row['model'],row['best_single'],site);triple=find(row['model'],row['input_id'],site)
            if single and triple:panel_figure('planned_S7_failure_%02d_%s_%s'%(index,row['model'],site),site,
                [('RGB target',store.target(site),None),(row['best_single'],np.load(single['predictions'][site],mmap_mode='r'),single),
                 (row['input_id'],np.load(triple['predictions'][site],mmap_mode='r'),triple)],errors=True,reason=reason)

    stats=_training_band_statistics(store,Path(root)/'tables')
    combination_rows=[];correlation_rows=[]
    for record in stats['folds']:
        fold=record['fold'];combination_rows.extend(record['combinations'])
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        correlation=np.array([[np.nan if v is None else v for v in row] for row in record['correlation']])
        image=axes[0].imshow(correlation,vmin=-1,vmax=1,cmap='RdBu_r')
        axes[0].set_xticks(range(5),['B%d'%b for b in range(1,6)]);axes[0].set_yticks(range(5),['B%d'%b for b in range(1,6)])
        fig.colorbar(image,ax=axes[0],orientation='horizontal',shrink=.8,pad=.14,label='Pearson r')
        for row in record['combinations']:
            x=row['mean_band_entropy_bits'];y=row['mean_abs_pairwise_correlation'];z=row['edge_correlation_mean_band_gradient_vs_RGB']
            axes[1].scatter(x,y);axes[1].annotate(row['input_id'],(x,y),fontsize=7)
            if z is not None:axes[2].scatter(y,z);axes[2].annotate(row['input_id'],(y,z),fontsize=7)
        axes[1].set(xlabel='Mean band entropy (bits)',ylabel='Mean absolute pairwise r')
        axes[2].set(xlabel='Mean absolute pairwise r',ylabel='Gradient correlation with RGB')
        fig.suptitle('Fold %d training sites only: %s'%(fold,', '.join(record['train_sites'])))
        for i in range(5):
            for j in range(5):correlation_rows.append({'fold':fold,'band_i':i+1,'band_j':j+1,'correlation':record['correlation'][i][j],'train_sites':record['train_sites']})
        save(fig,'planned_S9_training_band_statistics_fold%d'%fold,record['combinations'])
    _csv(Path(root)/'tables/training_band_correlations.csv',correlation_rows)
    _csv(Path(root)/'tables/training_combination_statistics.csv',combination_rows)

    ranking=[]
    for model in ('pix2pix','tfd','drunet'):
        for fold in range(1,6):
            items=[item for item in loaded.values() if item['task']['experiment']=='EX2' and item['task']['model']==model
                   and item['task'].get('fold')==fold and item['task']['input_id'].startswith('T') and item['validation_psnr'] is not None]
            if len(items)!=10:continue
            scores=np.array([item['validation_psnr'] for item in items])
            from scipy.stats import rankdata
            ranks=rankdata(-scores,method='average')
            ranking.extend({'model':model,'fold':fold,'input_id':item['task']['input_id'],'validation_psnr':item['validation_psnr'],
                            'validation_rank':float(rank)} for item,rank in zip(items,ranks))
    if ranking:
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        models=['pix2pix','tfd','drunet']
        for config in INPUTS[5:]:
            name=config['input_id'];means=[np.mean([r['validation_rank'] for r in ranking if r['model']==m and r['input_id']==name])
                                        if any(r['model']==m and r['input_id']==name for r in ranking) else np.nan for m in models]
            axes[0].plot(range(3),means,'.-',label=name)
        axes[0].set_xticks(range(3),models);axes[0].set(ylabel='Mean validation rank across folds',ylim=(10.5,.5));axes[0].legend(fontsize=7,ncol=2)
        for index,config in enumerate(INPUTS[5:]):
            vals=[r['validation_rank'] for r in ranking if r['input_id']==config['input_id']]
            axes[1].scatter([index]*len(vals),vals,s=12,alpha=.5)
        axes[1].set_xticks(range(10),[c['input_id'] for c in INPUTS[5:]],rotation=55);axes[1].set(ylabel='Validation ranks across models/folds',ylim=(10.5,.5))
        save(fig,'planned_S11_validation_ranking_consistency',ranking)

    color_rows=[]
    for item in loaded.values():
        if item['task']['experiment']!='EX2' or item['task']['input_id']!='T124':continue
        for site,score in item['summary']['sites'].items():
            color_rows.append({'model':item['task']['model'],'site':site,**score.get('deltaE00_summary',{})})
    if color_rows:
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        for model in ('pix2pix','tfd','drunet'):
            for axis,key in zip(axes,('median','p95')):
                values=[next((r.get(key,np.nan) for r in color_rows if r['site']==site and r['model']==model),np.nan) for site in store.sites]
                axis.plot(range(len(store.sites)),values,'.-',label=model);axis.set_xticks(range(len(store.sites)),store.sites)
                axis.set_ylabel('DeltaE00 '+key)
        axes[-1].legend();save(fig,'planned_S12_color_distributions',color_rows)
        for site in fixed:
            items=[find(model,'T124',site) for model in ('pix2pix','tfd','drunet')]
            if not all(items):continue
            truth=np.clip(crop(store.target(site),site),0,1)
            gray=truth@np.array([.299,.587,.114]);reference_edge=np.hypot(sobel(gray,0)/8,sobel(gray,1)/8)
            fig,axes=plt.subplots(3,4,figsize=(12,9));sources=[]
            axes[0,0].imshow(truth);axes[0,0].set_title('RGB target')
            axes[1,0].imshow(truth);axes[1,0].set_title('Color error context')
            axes[2,0].imshow(reference_edge,cmap='gray',vmin=0,vmax=.25);axes[2,0].set_title('Target gradient')
            for column,item in enumerate(items,1):
                predicted=np.clip(crop(np.load(item['predictions'][site],mmap_mode='r'),site),0,1)
                de=deltaE_ciede2000(rgb2lab(predicted),rgb2lab(truth))
                g=predicted@np.array([.299,.587,.114]);edge=np.hypot(sobel(g,0)/8,sobel(g,1)/8)
                valid=crop(store.masks[site],site)
                axes[0,column].imshow(predicted);axes[0,column].set_title(item['task']['model']+' T124')
                im1=axes[1,column].imshow(np.ma.masked_where(~valid,de),cmap='magma',vmin=0,vmax=50)
                im2=axes[2,column].imshow(np.ma.masked_where(~valid,np.abs(edge-reference_edge)),cmap='magma',vmin=0,vmax=.25)
                sources.append(meta(item,site,'RGB / DeltaE00 / absolute Sobel magnitude difference'))
            for axis in axes.flat:axis.axis('off')
            fig.colorbar(im1,ax=axes[1,:].tolist(),label='DeltaE00 [0,50]',shrink=.7)
            fig.colorbar(im2,ax=axes[2,:].tolist(),label='Edge error [0,.25]',shrink=.7)
            save(fig,'planned_S12_same_location_color_structure_'+site,sources)

    # Keep prior filenames stable and explicitly map the reviewed figure groups.
    all_outputs=existing+manifest
    mapping={
        'F0':('Data and task overview',['planned_F0_']),
        'F1':('Fixed B124 model comparisons with same-location enlargements',['planned_F1_']),
        'F2':('Validation-selected single/triple and fixed-pix2pix T124 comparisons',['planned_F2_']),
        'F3':('All 15 inputs and three models, separate metric scales',['planned_F3_']),
        'F4':('Three-band gains relative to validation-best single',['F2_per_site_multiband_gains']),
        'S1':('All five single-band crops and complete gallery',['planned_S1_']),
        'S2':('All ten triple-band crops and complete gallery',['planned_S2_']),
        'S3':('Per-site gain heatmaps',['planned_S3_']),
        'S4':('All 31 linear subsets',['S3_linear_all_band_subsets']),
        'S5':('Same-domain protocol comparisons',['F3_protocol_same_region_comparison']),
        'S6':('Controlled displacement sensitivity',['F4_common_context_shift_sensitivity']),
        'S7':('Rule-selected negative gains and failures',['planned_S7_']),
        'S8':('Observed compute/storage cost; inference timing unavailable until logged',['S5_compute_and_storage_cost']),
        'S9':('Train-only band correlations and separate combination statistics',['planned_S9_']),
        'S10':('Training and fixed validation histories',['S2_training_and_validation_curves']),
        'S11':('Validation-only ranking consistency',['planned_S11_']),
        'S12':('Color distributions and same-location structural error',['planned_S12_'])}
    catalog=[]
    for group,(description,prefixes) in mapping.items():
        names=[r['name'] for r in all_outputs if r['status']=='generated' and any(r['name'].startswith(p) for p in prefixes)]
        catalog.append({'planned_group':group,'description':description,
                        'status':'generated_from_available_results' if names else 'unavailable_missing_required_results',
                        'output_files':[name+'.png' for name in names],
                        'source_files':[name+'_source.csv' for name in names],
                        'coordinates':'tables/figure_crop_coordinates.csv',
                        'selection_manifest':'figure_selection_manifest.json'})
    _csv(Path(root)/'figure_catalog.csv',catalog)
    atomic_json(Path(root)/'figure_catalog.json',catalog)
    return manifest,catalog
