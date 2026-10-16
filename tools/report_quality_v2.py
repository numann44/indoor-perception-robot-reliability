"""Derive quality gates, repeat uncertainty and comparison plots from measured files."""
from __future__ import annotations
import argparse,csv,datetime,json,math,os,statistics
from pathlib import Path
from indoor_perception_lab.experiments import root_path,sha,json_write
ROOT=root_path()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    protocol=json.loads((ROOT/'configs/v2/protocol.json').read_text());targets=protocol['quality_targets'];refs=[];rows=[];gates={};repeats={}
    if args.final:
        from indoor_perception_lab.quality_release import ensure_test_seal
        for task in ('semantic','detection'):
            lockpath=ROOT/f'models/v2/selection-{task}.json';lock=json.loads(lockpath.read_text());preset=lock['selected']['configuration']
            completion_path=ROOT/f'reports/v2/{task}-heldout-complete.json';completion=json.loads(completion_path.read_text())
            if completion['protocol_id']!='quality-v2' or completion['primary_seed']!=42 or completion['seed_selection_on_test'] or completion['selection_lock_sha256']!=sha(lockpath):raise ValueError('Incomplete or stale fixed-driver heldout evidence')
            if task=='semantic':
                names=[f'semantic-nyu-test-semantic-{preset}-seed{seed}.json' for seed in (42,43,44)]
                names += ['semantic-nyu-test-segmentation-trained.json','semantic-nyu-test-segmentation-initial.json',f'semantic-nyu-test-{Path(lock["additional_baselines"][0]["pack"]).name}.json',f'semantic-nyu-test-semantic-{preset}-seed42-train-majority.json',f'semantic-hypersim-test-semantic-{preset}-seed42.json','semantic-hypersim-test-segmentation-trained.json']
                expected={'reports/v2/'+name for name in names}
            else:
                names=[f'detection-{dataset}-test-detection-{preset}-seed{seed}.json' for dataset in ('jaad','mot17') for seed in (42,43,44)]
                names += ['detection-jaad-test-detection-baseline.json','detection-jaad-test-detection-trained.json',f'detection-jaad-test-{Path(lock["additional_baselines"][0]["pack"]).name}.json','detection-mot17-test-detection-baseline.json']
                expected={'reports/v2/'+name for name in names}|{'reports/v2/tracking/mot17-heldout.json'}
            entries=completion['reports']
            if len(entries)!=len(expected) or {entry['path'] for entry in entries}!=expected or any(sha(ROOT/entry['path'])!=entry['sha256'] for entry in entries):raise ValueError('Fixed comparison set is missing, changed, or duplicated')
            for entry in entries:
                path=entry['path']
                if path=='reports/v2/tracking/mot17-heldout.json':
                    value=json.loads((ROOT/path).read_text());run=json.loads((ROOT/'reports/v2/runs'/value['run_id']/'run.json').read_text())
                    ensure_test_seal(task,ROOT/lock['selected']['pack'],'mot17')
                    if value['common_detector']!=lock['selected']['pack'] or value['selected_configuration']!=json.loads((ROOT/'models/v2/selection-tracking.json').read_text())['selected']['configuration'] or run['config']['model_pack']!=lock['selected']['pack'] or run['config']['dataset']!='mot17' or run['config']['seed']!=42:raise ValueError('External tracker comparison identity differs from frozen spec')
                    continue
                dataset=Path(path).name.split('-')[1]
                tail=Path(path).stem.removeprefix(f'{task}-{dataset}-test-')
                baseline='train-majority' if tail.endswith('-train-majority') else 'model'
                pack_name=tail.removesuffix('-train-majority')
                pack=ROOT/('models/v2/' if pack_name.startswith(('semantic-','detection-D')) else 'models/')/pack_name
                expected_pack=str(pack.relative_to(ROOT));ensure_test_seal(task,pack,dataset)
                manifest=json.loads((pack/'manifest.json').read_text());value=json.loads((ROOT/path).read_text());run=json.loads((ROOT/'reports/v2/runs'/value['run_id']/'run.json').read_text());cfg=run['config']
                if value['model_pack']!=expected_pack or value.get('baseline','model')!=baseline or cfg['task']!=task or cfg['dataset']!=dataset or cfg['split']!='test' or cfg['model_pack']!=expected_pack or cfg.get('seed')!=manifest.get('seed') or cfg.get('baseline','model')!=baseline:raise ValueError('Fixed heldout report model/task/dataset/seed/baseline identity mismatch')
            refs.append({'path':str(completion_path.relative_to(ROOT)),'sha256':sha(completion_path)})
    def check_forecast(original,path):
        sidepath=ROOT/'reports/v2/forecast/full-source-provenance-replay-v2.json';side=json.loads(sidepath.read_text())
        entries=[entry for entry in side['entries'] if entry['original_report']==path]
        if not side['all_replay_metrics_equal'] or len(entries)!=1:raise ValueError('Missing exact forecast replay binding')
        entry=entries[0];replaypath=ROOT/entry['replay_report'];replay=json.loads(replaypath.read_text());directory=ROOT/entry['evaluation_run'];run=json.loads((directory/'run.json').read_text())
        if run['domain']!=entry['domain'] or run['seed']!=entry['seed'] or path!=f'reports/v2/forecast/{entry["domain"]}-test-seed{entry["seed"]}.json':raise ValueError('Forecast source replay domain/seed mismatch')
        core=lambda result:{k:v for k,v in result.items() if k!='evaluation_run_id'}
        if entry['original_report_sha256']!=sha(ROOT/path) or entry['replay_report_sha256']!=sha(replaypath) or core(original)!=core(replay) or not entry['replay_metrics_equal']:raise ValueError('Forecast original/replay metric integrity failed')
        if run['status']!='completed' or not run['source_unchanged_during_execution'] or json.loads((directory/'metrics.json').read_text())!=replay or run['run_id']!=replay['evaluation_run_id']:raise ValueError('Forecast replay is not an unchanged-source completed measurement')
        lockpath=ROOT/f'models/v2/selection-forecast-{entry["domain"]}.json';lock=json.loads(lockpath.read_text());pack=ROOT/run['pack']
        binding=[b for b in [lock['selected'],*lock['repeated_seeds']] if b['pack']==run['pack'] and b['seed']==run['seed']]
        if len(binding)!=1 or lock['test_closed'] or sha(lockpath)!=run['canonical_selection_sha256'] or sha(pack/'weights.pt')!=run['checkpoint_sha256'] or sha(pack/'manifest.json')!=run['model_manifest_sha256'] or binding[0]['checkpoint_sha256']!=original['checkpoint_sha256']:raise ValueError('Forecast canonical artifact binding changed')
        if sha(ROOT/f'data/v2/forecast/{entry["domain"]}/manifest.json')!=run['dataset_manifest_sha256'] or sha(ROOT/run['source_archive'])!=run['source_archive_sha256']:raise ValueError('Forecast source/dataset provenance changed')
        refs.extend([{'path':str(sidepath.relative_to(ROOT)),'sha256':sha(sidepath)},{'path':entry['replay_report'],'sha256':sha(replaypath)},{'path':str((directory/'run.json').relative_to(ROOT)),'sha256':sha(directory/'run.json')}])
    def read(path):
        p=ROOT/path
        if not p.exists():return None
        value=json.loads(p.read_text())
        if all(key in value for key in ['run_id','model_pack','checkpoint_sha256','split_index_sha256']):
            runpath=ROOT/'reports/v2/runs'/value['run_id'];run=json.loads((runpath/'run.json').read_text())
            if run['status']!='completed' or sha(runpath/'metrics.json')!=sha(p):raise ValueError('Unfinished/changed measurement report: '+path)
            dataset=run['config']['dataset'];index=ROOT/(f'data/v2/{dataset}/index.jsonl' if dataset in {'nyu','jaad','mot17'} else 'data/hypersim/index.jsonl' if dataset=='hypersim' else '')
            if not index.is_file() or sha(index)!=value['split_index_sha256'] or sha(ROOT/value['model_pack']/'weights.pt')!=value['checkpoint_sha256'] or run['dataset_index_sha256']!=value['split_index_sha256']:raise ValueError('Measured model/current-index identity changed: '+path)
            if run['config']['model_pack']!=value['model_pack'] or run['config'].get('baseline','model')!=value.get('baseline','model') or run['config']['checkpoint_sha256']!=value['checkpoint_sha256']:raise ValueError('Measurement config/report identity differs: '+path)
            if sha(runpath/'source-snapshot.tar.gz')!=run['source_archive_sha256']:raise ValueError('Measurement source archive changed: '+path)
        elif path.startswith('reports/v2/forecast/') and Path(path).name.startswith(('fixture-test-seed','jaad-test-seed')):
            check_forecast(value,path)
        elif all(key in value for key in ['run_id','common_detector','weights_sha256','index_sha256']):
            directory=ROOT/'reports/v2/runs'/value['run_id'];run=json.loads((directory/'run.json').read_text());pack=ROOT/value['common_detector']
            if run['status']!='completed' or sha(directory/'metrics.json')!=sha(p) or sha(pack/'weights.pt')!=value['weights_sha256'] or sha(ROOT/'data/v2/mot17/index.jsonl')!=value['index_sha256'] or sha(ROOT/'models/v2/selection-tracking.json')!=run['config']['tracking_lock_sha256'] or sha(directory/'source-snapshot.tar.gz')!=run['source_archive_sha256']:raise ValueError('Tracking measurement provenance changed')
        refs.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)});return value
    def add(dataset,model,seed,metric,value,unit,role='trained'):
        if value is not None and math.isfinite(value):rows.append({'dataset':dataset,'model':model,'seed':seed,'metric':metric,'value':value,'unit':unit,'role':role})
    def gate(name,passed,scope,values):
        gates[name]={'status':'pending' if passed is None else 'passed' if passed else 'failed','scope':scope,'values':values}
    for task,dataset,metric in [('semantic','nyu','mIoU'),('detection','jaad','AP')]:
        lock=read(f'models/v2/selection-{task}.json')
        if lock is None:continue
        preset=lock['selected']['configuration'];values=[];primary=None
        for seed in (42,43,44):
            pack=f'{task}-{preset}-seed{seed}';report=read(f'reports/v2/{task}-{dataset}-test-{pack}.json')
            if report:
                value=report[metric];add(dataset,pack,seed,metric,value,'fraction');values.append(value)
                if seed==42:primary=report
                if task=='detection':add(dataset,pack,seed,'AP50',report.get('AP50'),'fraction')
        if len(values)==3:repeats[task]={'seeds':[42,43,44],'metric':metric,'mean':statistics.mean(values),'sample_SD':statistics.stdev(values),'scope':'descriptive seed variability; not calibrated uncertainty or a confidence interval'}
        baseline_pack='segmentation-trained' if task=='semantic' else 'detection-baseline'
        baseline=read(f'reports/v2/{task}-{dataset}-test-{baseline_pack}.json')
        if baseline:
            add(dataset,baseline_pack,None,metric,baseline[metric],'fraction','historical_v1' if task=='semantic' else 'COCO')
            if task=='detection':add(dataset,baseline_pack,None,'AP50',baseline['AP50'],'fraction','COCO')
        if task=='semantic':
            improvement=primary[metric]-baseline[metric] if primary and baseline else None
            passed=primary[metric]>=targets['nyu_mIoU'] and improvement>=targets['nyu_improvement_vs_v1'] if improvement is not None else None
            gate('NYU segmentation',passed,'primary42 original-resolution valid-pixel NYU test, identical V1 rescore',{'mIoU':primary[metric] if primary else None,'target':targets['nyu_mIoU'],'absolute_gain_vs_v1':improvement,'gain_target':targets['nyu_improvement_vs_v1']})
            for base in ['segmentation-initial',Path(lock['additional_baselines'][0]['pack']).name,f'semantic-{preset}-seed42-train-majority']:
                report=read(f'reports/v2/semantic-nyu-test-{base}.json')
                if report:add('nyu',base,None,'mIoU',report['mIoU'],'fraction','initial_or_train_majority')
            report=read(f'reports/v2/semantic-hypersim-test-semantic-{preset}-seed42.json')
            if report:add('hypersim',f'semantic-{preset}-seed42',42,'mIoU',report['mIoU'],'fraction','separate_synthetic_domain')
            report=read('reports/v2/semantic-hypersim-test-segmentation-trained.json')
            if report:add('hypersim','segmentation-trained',None,'mIoU',report['mIoU'],'fraction','historical_synthetic_domain')
        else:
            improvement=primary[metric]-baseline[metric] if primary and baseline else None
            passed=primary[metric]>=targets['jaad_AP'] and improvement>=targets['jaad_improvement_vs_coco'] if improvement is not None else None
            gate('JAAD detection',passed,'primary42 fresh clip-disjoint JAAD test; original decoded pixels',{'AP':primary[metric] if primary else None,'target':targets['jaad_AP'],'absolute_gain_vs_COCO':improvement,'gain_target':targets['jaad_improvement_vs_coco']})
            for base in ['detection-trained',Path(lock['additional_baselines'][0]['pack']).name]:
                report=read(f'reports/v2/detection-jaad-test-{base}.json')
                if report:
                    for name in ('AP','AP50'):add('jaad',base,None,name,report[name],'fraction','historical_v1' if base=='detection-trained' else 'selected_pretrained_initialization')
            outside=None;outside_scores=[];outside_AP=[]
            for seed in (42,43,44):
                report=read(f'reports/v2/detection-mot17-test-detection-{preset}-seed{seed}.json')
                if report:
                    if seed==42:outside=report
                    outside_scores.append(report['AP50'])
                    outside_AP.append(report['AP'])
                    for name in ('AP','AP50'):add('mot17',f'detection-{preset}-seed{seed}',seed,name,report[name],'fraction')
            if len(outside_scores)==3:repeats['detection_mot17']={'seeds':[42,43,44],'metric':'AP50','mean':statistics.mean(outside_scores),'sample_SD':statistics.stdev(outside_scores),'scope':'one external recording; descriptive seed variability'}
            if len(outside_AP)==3:repeats['detection_mot17_AP']={'seeds':[42,43,44],'metric':'AP','mean':statistics.mean(outside_AP),'sample_SD':statistics.stdev(outside_AP),'scope':'one external recording; descriptive seed variability'}
            report=read('reports/v2/detection-mot17-test-detection-baseline.json')
            if report:
                for name in ('AP','AP50'):add('mot17','detection-baseline',None,name,report[name],'fraction','COCO')
            gate('MOT17 detection',outside['AP50']>=targets['mot17_AP50'] if outside else None,'primary42 all900frames of one external shopping-mall recording',{'AP50':outside['AP50'] if outside else None,'target':targets['mot17_AP50']})
    tracking=read('reports/v2/tracking/mot17-heldout.json')
    if tracking:
        selected=tracking['results'][tracking['selected_configuration']];old=tracking['results']['legacy-v1']
        passed=selected['IDF1']>=targets['mot17_IDF1'] and selected['IDF1']>=old['IDF1'] and selected['ID_switch']<=old['ID_switch']
        gate('MOT17 tracking',passed,'same detector outputs, all900frames, independent local identity evaluator',{'IDF1':selected['IDF1'],'target':targets['mot17_IDF1'],'legacy_IDF1':old['IDF1'],'ID_switch':selected['ID_switch'],'legacy_ID_switch':old['ID_switch']})
        for name,result in tracking['results'].items():add('mot17',name,None,'IDF1',result['IDF1'],'fraction')
    for domain in ['fixture','jaad']:
        if domain=='fixture':read('reports/v2/forecast/fixture-evaluation-provenance-replay.json')
        primary=None;ades=[];fdes=[]
        for seed in [42,43,44]:
            report=read(f'reports/v2/forecast/{domain}-test-seed{seed}.json')
            if not report:continue
            if report['split']!='test' or not report['identical_observed_histories']:raise ValueError('Forecast comparison scope changed')
            for model,result in report['metrics'].items():
                add(domain,model,seed,'ADE2',result['horizons']['2.0']['ADE'],report['units']);add(domain,model,seed,'FDE2',result['horizons']['2.0']['FDE'],report['units'])
            ades.append(report['metrics']['residual_MLP']['horizons']['2.0']['ADE']);fdes.append(report['metrics']['residual_MLP']['horizons']['2.0']['FDE'])
            if seed==42:primary=report
        if len(ades)==3:repeats['forecast_'+domain]={'seeds':[42,43,44],'ADE2_mean':statistics.mean(ades),'ADE2_sample_SD':statistics.stdev(ades),'FDE2_mean':statistics.mean(fdes),'FDE2_sample_SD':statistics.stdev(fdes),'unit':primary['units']}
        if primary:
            mlp=primary['metrics']['residual_MLP']['horizons']['2.0'];cv=primary['metrics']['constant_velocity_LS']['horizons']['2.0'];ratio=mlp['ADE']/cv['ADE'] if cv['ADE']>0 else None
            passed=ratio<=targets['forecast_ADE_ratio_vs_LS_CV'] and mlp['FDE']<=cv['FDE'] if ratio is not None else None
            coverage=primary.get('coverage');coverage_evidence=None
            if domain=='fixture' and coverage is None:
                coverage_path='reports/v2/forecast/fixture-coverage-provenance.json';record=read(coverage_path)
                frozen=json.loads((ROOT/'models/v2/selection-forecast-fixture.json').read_text())
                if record is None or record['dataset_modified'] or record['target_labels_are_features'] or record['dataset_manifest_sha256']!=frozen['dataset_manifest_sha256'] or record['dataset_manifest_sha256']!=sha(ROOT/'data/v2/forecast/fixture/manifest.json'):raise ValueError('Fixture coverage dataset identity mismatch')
                coverage=record['coverage']['test'];matched=coverage['matched_observed_history_windows'];total=coverage['all_renderer_actor_windows']
                if not total or matched!=primary['metrics']['residual_MLP']['windows'] or coverage['unavailable_observed_history_windows']!=total-matched or coverage['coverage_fraction']!=matched/total:raise ValueError('Fixture coverage denominator differs from measured forecast windows')
                coverage_evidence={'path':coverage_path,'sha256':sha(ROOT/coverage_path)}
            last2=primary['metrics']['constant_velocity_last2']['horizons']['2.0']
            gate(domain+' forecasting',passed,'primary42 shared causal observed histories; conditional error must be read with coverage; domains separate',{'ADE_ratio_vs_LS_CV':ratio,'target_maximum_ratio':targets['forecast_ADE_ratio_vs_LS_CV'],'MLP_ADE2':mlp['ADE'],'LS_CV_ADE2':cv['ADE'],'MLP_FDE2':mlp['FDE'],'LS_CV_FDE2':cv['FDE'],'last2_CV_ADE2':last2['ADE'],'last2_CV_FDE2':last2['FDE'],'ADE_ratio_vs_last2_CV':mlp['ADE']/last2['ADE'] if last2['ADE']>0 else None,'coverage':coverage,'coverage_evidence':coverage_evidence,'unit':primary['units']})
    simulation=read('reports/v2/robotics/simulation-contact-revalidation-200/summary.json')
    if simulation:
        clear=[r for r in simulation['rows'] if r['scenario']=='clear'];scenarios=sorted({r['scenario'] for r in simulation['rows']});rates={name:sum(r['goal_reached'] for r in simulation['rows'] if r['scenario']==name)/sum(r['scenario']==name for r in simulation['rows']) for name in scenarios}
        gate('Simulation clear goals',rates['clear']>=targets['simulation_clean_goal_rate'],'pre-existing clear/empty-corridor gate only; other scenario rates remain explicit',{'clear_goal_rate':rates['clear'],'target':targets['simulation_clean_goal_rate'],'all_scenario_goal_rates':rates,'unexpected_contact_episodes':simulation['collision_episodes'],'contact_sampling':'every5msphysics step','exposed_seed_revalidation':True})
        gate('Simulation contact check',simulation['collision_episodes']==0 and all(r['collision_steps']==0 for r in simulation['rows']),'60episodes, every physics step; recorded caster correction/instrumentation revalidation',{'unexpected_contacts':simulation['collision_episodes'],'target':0})
    expected=['NYU segmentation','JAAD detection','MOT17 detection','MOT17 tracking','fixture forecasting','jaad forecasting','Simulation clear goals','Simulation contact check']
    for name in expected:
        if name not in gates:gate(name,None,'awaiting executed measurement',{})
    if args.final and any(g['status']=='pending' for g in gates.values()):raise RuntimeError('Cannot finalize a pending quality measurement')
    output=ROOT/'reports/v2/quality-summary.json';output.parent.mkdir(parents=True,exist_ok=True)
    report={'protocol_id':'quality-v2','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'final' if args.final else 'in-progress','overall_status':'PARTIAL','primary_seed':42,'test_seed_selection':False,'gates':gates,'repeated_seed_statistics':repeats,'metric_rows':rows,'measurement_evidence':refs,'protocol_sha256':sha(ROOT/'configs/v2/protocol.json'),'hardware_and_quality_scopes':'physical robot and embedded target remain unavailable; passing clear corridor does not erase occlusion failures'}
    json_write(output,report)
    with (output.parent/'quality-comparisons.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset','model','seed','metric','value','unit','role']);writer.writeheader();writer.writerows(rows)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plotdir=output.parent/'plots';plotdir.mkdir(exist_ok=True)
    for dataset,metric in [('nyu','mIoU'),('jaad','AP'),('mot17','AP50'),('fixture','ADE2'),('jaad','ADE2')]:
        values=[row for row in rows if row['dataset']==dataset and row['metric']==metric and row['seed'] in {42,None}]
        if not values:continue
        fig,ax=plt.subplots(figsize=(8,4));ax.barh([row['model'] for row in values],[row['value'] for row in values],color='#237a83');ax.set_xlabel(metric+' ('+values[0]['unit']+')');ax.set_title(dataset+' held-out comparison · primary seed42');ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(plotdir/f'{dataset}-{metric}.png',dpi=170);plt.close(fig)
    print(json.dumps({'report':str(output.relative_to(ROOT)),'gates':{name:g['status'] for name,g in gates.items()},'measured_rows':len(rows)}))


if __name__=='__main__':main()
