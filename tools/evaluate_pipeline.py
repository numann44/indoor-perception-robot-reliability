"""Common-scope sensor pipeline ablations; annotations are evaluator-only."""
import json
from pathlib import Path
import numpy as np
from robot_model_sdk import fixture_frame,load_adapter
from indoor_perception_lab.experiments import Run,root_path,json_write
from indoor_perception_lab.pipeline import PerceptionPipeline
from indoor_perception_lab.datasets import read_index,sensor_from_row,jaad_targets
from indoor_perception_lab.geometry import SpatialMap,geometric_plane_mask,project_depth
from indoor_perception_lab.tracking import Tracker
from robot_reliability_bench.metrics import tracking_metrics,box_iou

root=root_path();run=Run('pipeline-ablation',{'seed':42,'common_scope':True,'tracking_frontend':'validation-selected JAAD person detector','state_thresholds':{'stationary_lt':.1,'moving_gte':.2,'approaching_range_decrease_gt':.2,'continuity_s':.5}},root/'data/jaad/index.jsonl')
rows=read_index(root,'jaad',split='test');adapter=load_adapter(root/'models/detection-trained');predictions_on={};predictions_off={};targets={};trackers={}
for row in rows:
    f=sensor_from_row(root,row);ds=[d for d in adapter.predict(f).detections if d.class_id==1 and d.score>=.3];clip=row['scene_id'];tracker=trackers.setdefault(clip,Tracker());ts=tracker.update(ds,f.timestamp_ns)
    predictions_on[f.sample_id]=[{'bbox':t.bbox,'track_id':clip+'/'+t.id,'class_id':1} for t in ts if t.observed]
    predictions_off[f.sample_id]=[{'bbox':d.bbox,'track_id':f.sample_id+'/'+str(i),'class_id':1} for i,d in enumerate(ds)]
    targets[f.sample_id]=[{'bbox':t['bbox_xyxy'],'track_id':clip+'/'+str(t['track_id']),'class_id':1,'ignore':t['ignore']} for t in jaad_targets(root,row)]
adapter.close()
tracking={'scope':'same68heldoutJAADindexedframes4clips at~2Hz; no re-identification claim; observed predictions only','threshold':.3,'on':tracking_metrics(predictions_on,targets),'off':tracking_metrics(predictions_off,targets),'off_definition':'each per-frame detection gets a new ID; deliberately association-free reference'}
json_write(run.dir/'tracking-predictions.json',{'on':predictions_on,'off':predictions_off,'targets':targets})

# Single planar object at known physical surface position, rendered into both sensors.
states={'stationary':(0.,0.),'moving':(.45,0.),'approaching':(0.,-.35),'unknown':(.15,0.)};behavior={};distance_errors=[];no_depth=0;behavior_confusion={a:{b:0 for b in states} for a in states}
for name,(vx,vz) in states.items():
    pipe=PerceptionPipeline();rgbpipe=PerceptionPipeline(use_depth=False);seen=[]
    for i in range(41):
        t=i/10;position=[-.8+vx*t,0.,4.+vz*t];objects=[{'id':'actor','position':position,'size':[.6,.8],'color':[230,50,40]}]
        f=fixture_frame(i,objects=objects);result=pipe.process(f);rgbresult=rgbpipe.process(f)
        observed=[track for track in result['tracks'] if track['observed']]
        if observed:
            estimate=observed[0]['position'];distance_errors.append(abs(estimate['surface_range_m']-np.linalg.norm(position)))
            if i>=10:
                label=observed[0]['behavior'];label=label if label in states else 'unknown';behavior_confusion[name][label]+=1;seen.append(label)
        no_depth+=sum(t['position'] is None for t in rgbresult['tracks'] if t['observed'])
    behavior[name]={'frames':41,'evaluated_after_1s':len(seen),'predicted_states':{s:seen.count(s) for s in states},'truth_velocity_xz_m_s':[vx,vz]};pipe.close();rgbpipe.close()
behavior['confusion_matrix']=behavior_confusion;behavior['accuracy']=sum(behavior_confusion[k][k] for k in states)/sum(sum(r.values()) for r in behavior_confusion.values());behavior['domain']='controlled calibrated rendered planar surfaces; observed RGB/depth histories, not annotated histories';behavior['limit']='Threshold rules rather than learned activity recognition; pixel quantization and depth spread affect decisions'
distance={'depth_assisted':{'samples':len(distance_errors),'surface_range_MAE_m':float(np.mean(distance_errors)),'surface_range_max_error_m':float(np.max(distance_errors))},'rgb_only':{'status':'unavailable','reason':'Metric distance requires depth/calibration; missing outputs counted explicitly','observed_tracks_without_distance':no_depth},'target_definition':'distance to rendered planar surface centre, not body centre of mass','common_sensors':'same164fixtureRGBframes, with/withoutdepth'}

# Same real TUM depth/pose endpoints; surface occupancy and semantic attributes differ.
segmenter=load_adapter(root/'models/segmentation-trained');maps={};agreements=[];selected=[]
tum=read_index(root,'tum')
for clip in sorted({r['sequence_id'] for r in tum}):
    cliprows=[r for r in tum if r['sequence_id']==clip and r.get('pose_world_camera') is not None]
    # Sensor availability is checked by the loader; fixed samples selected without labels.
    if not cliprows:cliprows=[r for r in tum if r['sequence_id']==clip]
    semmap=SpatialMap();geomap=SpatialMap();used=0
    for row in cliprows[::30][:20]:
        f=sensor_from_row(root,row)
        if f.T_world_camera is None:continue
        sem=segmenter.predict(f).semantic;geo,plane=geometric_plane_mask(f.depth,f.calibration);semmap.update(f,sem);geomap.update(f,geo);used+=1;selected.append(row['sample_id'])
        if plane.get('available'):
            valid=geo!=255;a=sem==1;b=geo==1;union=np.sum((a|b)&valid)
            agreements.append(float(np.sum(a&b&valid)/union) if union else None)
    maps[clip]={'frames':used,'semantic_surface_voxels':len(semmap.cells),'geometric_surface_voxels':len(geomap.cells),'same_voxel_support':set(semmap.cells)==set(geomap.cells),'semantic':semmap.to_json(),'geometric':geomap.to_json()}
segmenter.close();json_write(run.dir/'occupancy-maps.json',maps)
mapping={'scope':'TUM shared depth/pose endpoint voxels; independent maps for each sequence','samples':selected,'ground_truth':'unavailable; no TUM semantic/occupancy ground truth supplied','floor_label_agreement_mean_IoU':float(np.mean([x for x in agreements if x is not None])) if any(x is not None for x in agreements) else None,'agreement_is_accuracy':False,'geometry_assumption':'camera optical Y approximately gravity; an accepted RANSAC plane is a heuristic, not gravity calibration','representation':'occupied observed surfaces only; unobserved remains unknown','per_sequence':{k:{a:b for a,b in v.items() if a not in {'semantic','geometric'}} for k,v in maps.items()},'dynamic_exclusion_note':'This mapping-only comparison holds endpoints fixed and supplies no detector; dynamic filtering is tested separately in pipeline/core tests, so these maps are not persistent static-environment claims'}
report={'tracking':tracking,'distance':distance,'behavior':behavior,'occupancy':mapping};run.finish(report);json_write(root/'reports/pipeline-ablations.json',{'run_id':run.id,**report});print(json.dumps({'run_id':run.id,'tracking_IDF1':tracking['on'].get('IDF1'),'behavior_accuracy':behavior['accuracy'],'distance_MAE_m':distance['depth_assisted']['surface_range_MAE_m']}))
