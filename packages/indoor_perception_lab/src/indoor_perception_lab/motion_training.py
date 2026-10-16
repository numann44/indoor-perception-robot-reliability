"""Actual, resumable 2x64 MLP forecasting and matched-history baselines."""
from __future__ import annotations
import argparse, datetime, hashlib, importlib.metadata, json, os, platform, random, shlex, subprocess, sys, tarfile, time, uuid
from pathlib import Path
import numpy as np
from robot_model_sdk.models import motion_network
from .experiments import root_path, sha, json_write
from .motion_data import FUTURE_TIMES, HORIZONS, load_motion_dataset, resample_history


def seed_all(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(3)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def train_scale(histories):
    """Only training histories determine coordinate scaling; last point is origin."""
    centered=histories-histories[:,-1:, :]
    return np.maximum(np.std(centered,axis=(0,1)),1e-3).astype(np.float32)


def baseline_forecasts(history):
    h=np.asarray(history,np.float64)
    if h.ndim==2:h=h[None]
    last=h[:,-1:,:];velocity=(h[:,-1,:]-h[:,-2,:])/.1
    return {"constant_position":np.repeat(last,20,axis=1),"constant_velocity":last+FUTURE_TIMES[None,:,None]*velocity[:,None,:]}


def displacement_metrics(prediction,target,scene_ids=None):
    err=np.linalg.norm(np.asarray(prediction)-np.asarray(target),axis=-1)
    if err.ndim!=2 or err.shape[1]!=20 or not len(err):raise ValueError("Expected nonempty N x 20 displacement errors")
    result={"windows":len(err),"horizons":{str(h):{"ADE":float(err[:,:round(h*10)].mean()),"FDE":float(err[:,round(h*10)-1].mean())} for h in HORIZONS}}
    if scene_ids is not None:
        scenes=sorted(set(scene_ids));result["scene_count"]=len(scenes)
        result["scene_macro"]={str(h):{metric:float(np.mean([result_for_scene(err,np.asarray(scene_ids)==s,h,metric) for s in scenes])) for metric in ["ADE","FDE"]} for h in HORIZONS}
        # Scene cluster bootstrap keeps correlated windows together.
        rng=np.random.default_rng(20261014);ci={}
        for h in HORIZONS:
            vals=np.array([result_for_scene(err,np.asarray(scene_ids)==s,h,"ADE") for s in scenes])
            draws=np.mean(vals[rng.integers(0,len(vals),(2000,len(vals)))],axis=1)
            ci[str(h)]=np.percentile(draws,[2.5,97.5]).tolist()
        result["scene_bootstrap_ADE_95CI"]=ci
    return result


def result_for_scene(err,mask,h,metric):
    return float(err[mask,:round(h*10)].mean()) if metric=="ADE" else float(err[mask,round(h*10)-1].mean())


class ForecastAdapter:
    """Inference wrapper consumes positions only, never targets or object IDs.

    predict_history requires an already resampled11x2 history in manifest units.
    predict_timed_history additionally enforces causal history and max-gap checks.
    """
    task="forecast";backend="native"
    def __init__(self,pack,device="cpu"):
        import torch
        self.pack=Path(pack);self.metadata=json.loads((self.pack/"manifest.json").read_text());self.device=device
        weights=self.pack/self.metadata["native_weights"]
        if sha(weights)!=self.metadata["sha256"]:raise ValueError("Motion checkpoint checksum mismatch")
        self.model=motion_network(**self.metadata["network"]);self.model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True)["model"])
        self.model.to(device).eval();self.scale=np.array(self.metadata["preprocess"]["scale"],np.float32);self.last_timings_ms={}
    def predict_history(self,history):
        import torch
        h=np.asarray(history,np.float32);single=h.ndim==2
        if single:h=h[None]
        if h.ndim!=3 or h.shape[1:]!=(11,2) or not np.isfinite(h).all():raise ValueError("Forecast history must be finite N x11x2")
        start=time.perf_counter();origin=h[:,-1:,:];x=(h-origin)/self.scale
        with torch.inference_mode():y=self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        result=y*self.scale+origin;self.last_timings_ms={"total":(time.perf_counter()-start)*1000}
        return result[0] if single else result
    def predict_timed_history(self,times,positions,anchor_time=None):
        anchor_time=float(times[-1]) if anchor_time is None else float(anchor_time)
        history=resample_history(times,positions,anchor_time)
        return None if history is None else self.predict_history(history)
    def close(self):del self.model


class MotionRun:
    def __init__(self,root,dataset_path,config):
        self.root=Path(root);self.id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+Path(dataset_path).name+"-"+uuid.uuid4().hex[:6]
        self.dir=self.root/"reports/motion/runs"/self.id;self.dir.mkdir(parents=True);self.log=self.dir/"events.jsonl";self.log.touch()
        try:commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:commit="uncommitted-initial-work"
        paths=[Path(__file__),Path(__file__).with_name("motion_data.py"),Path(__file__).with_name("geometry.py"),Path(__file__).with_name("tracking.py")]
        import robot_model_sdk.models,robot_model_sdk.contracts
        paths.extend([Path(robot_model_sdk.models.__file__),Path(robot_model_sdk.contracts.__file__)])
        history_module=Path(robot_model_sdk.models.__file__).with_name("history.py")
        if history_module.exists():paths.append(history_module)
        with tarfile.open(self.dir/"source-snapshot.tar.gz","w:gz") as archive:
            for p in paths:archive.add(p,arcname=str(p.relative_to(self.root)))
            archive.add(self.root/"requirements.lock",arcname="requirements.lock")
        manifest=Path(dataset_path)/"manifest.json";snapshot=self.dir/"dataset-manifest.json";snapshot.write_bytes(manifest.read_bytes())
        self.record={"run_id":self.id,"task":"motion-train","config":config,"seed":config["seed"],"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":"running",
            "commit":commit,"source_archive_sha256":sha(self.dir/"source-snapshot.tar.gz"),"source_files":{str(p.relative_to(self.root)):sha(p) for p in paths},
            "dataset_manifest_sha256":sha(manifest),"dataset_path":str(Path(dataset_path).relative_to(self.root)),"command":shlex.join([sys.executable,*sys.argv]),
            "runtime_versions":{n:importlib.metadata.version(n) for n in ["torch","numpy","opencv-python-headless"]},"hardware":{"os":platform.platform(),"architecture":platform.machine(),"cpu_count":os.cpu_count(),"torch_threads":3},
            "checkpoint":"checkpoint.pt","resume_boundary":"epoch; no partially completed batch checkpoint","log":"events.jsonl"}
        json_write(self.dir/"run.json",self.record)
    def event(self,**fields):
        with self.log.open("a") as f:f.write(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields},allow_nan=False)+"\n")
    def finish(self,metrics):
        json_write(self.dir/"metrics.json",metrics);self.record.update(status="completed",finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());json_write(self.dir/"run.json",self.record)


def evaluate_pack(root,dataset_path,pack,split="test"):
    h,y,rows,meta=load_motion_dataset(dataset_path);ids=np.array([i for i,r in enumerate(rows) if r["split"]==split]);hist=h[ids];target=y[ids];scenes=[rows[i]["scene_id"] for i in ids]
    adapter=ForecastAdapter(pack)
    if adapter.metadata["units"]!=meta["units"] or adapter.metadata["coordinate_system"]!=meta["coordinate_system"]:
        adapter.close();raise ValueError("Forecast checkpoint/dataset coordinate units mismatch")
    predictions=baseline_forecasts(hist);predictions["MLP2x64"]=adapter.predict_history(hist);adapter.close()
    return {"split":split,"units":meta["units"],"coordinate_system":meta["coordinate_system"],"history_source":meta["history_source"],"identical_histories_for_all_methods":True,
            "evaluated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"dataset_manifest_sha256":sha(Path(dataset_path)/"manifest.json"),"evaluator_source_sha256":sha(__file__),"model_manifest_sha256":sha(Path(pack)/"manifest.json"),
            "checkpoint_sha256":sha(Path(pack)/"weights.pt"),"metrics":{name:displacement_metrics(p,target,scenes) for name,p in predictions.items()},
            "worst_MLP_windows":[{"sample":rows[ids[i]],"ADE2":float(e),"history":hist[i].tolist(),"target_evaluator_only":target[i].tolist(),"prediction":predictions["MLP2x64"][i].tolist()}
                                 for i,e in sorted(enumerate(np.linalg.norm(predictions["MLP2x64"]-target,axis=-1).mean(1)),key=lambda a:a[1],reverse=True)[:5]]}


def budget_guard(root):
    import shutil, resource
    root=Path(root);used=sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink());free=shutil.disk_usage(root).free
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
    if used>=8_000_000_000 or free<15_000_000_000:raise RuntimeError("Project8GB/free15GB disk guard exceeded")
    if rss>3_000_000_000:raise RuntimeError("Motion process3GB RSS guard exceeded")
    return {"project_bytes":used,"free_disk_bytes":free,"peak_process_rss_bytes":rss}


def train_model(root,dataset_path,lr,seed,epochs=15,batch_size=2,max_steps=600,resume=None,stop_steps=None,patience=3):
    """Budgeted primary protocol, resumable at any completed optimizer step."""
    import torch,resource
    if epochs>15 or max_steps>600 or batch_size!=2:raise ValueError("Accepted primary limits: epochs<=15,steps<=600,batch2")
    seed_all(seed);guard=budget_guard(root);h,y,rows,meta=load_motion_dataset(dataset_path)
    splits=np.array([r["split"] for r in rows]);train=np.where(splits=="train")[0];val=np.where(splits=="val")[0]
    if not len(train) or not len(val):raise ValueError("Empty train/validation split")
    scale=train_scale(h[train]);origin=h[:,-1:,:];x=torch.from_numpy((h-origin)/scale);target=torch.from_numpy((y-origin)/scale)
    model=motion_network(history=11,future=20,hidden=64);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);sampler=torch.Generator().manual_seed(seed+100000)
    config={"protocol":"accepted_budgeted_primary","lr":lr,"seed":seed,"epochs":epochs,"batch_size":2,"max_steps":max_steps,"patience":patience,"validation_interval_steps":30,"optimizer":"AdamW","weight_decay":1e-4,
            "loss":"MSE of train-history-scaled future displacement","network":{"history":11,"future":20,"hidden":64},"num_threads":3,"selection":"best validation ADE2 among checkpoints after at least1 optimizer step, not test","initial_checkpoint_eligible":False,"encoder":"none; whole small MLP trainable"}
    run=MotionRun(root,dataset_path,config);run.event(event="budget_guard_start",**guard)
    best=float("inf");best_state=None;best_step=0;step=0;epoch=0;order=None;cursor=0;bad=0;stopped="step_budget";last_val=None
    if resume:
        cp=torch.load(resume,map_location="cpu",weights_only=False)
        if cp["dataset_manifest_sha256"]!=sha(Path(dataset_path)/"manifest.json") or cp["config"]!=config:raise ValueError("Resume configuration or dataset mismatch")
        if not np.array_equal(cp["normalization_scale"],scale):raise ValueError("Resume normalization mismatch")
        model.load_state_dict(cp["model"]);opt.load_state_dict(cp["optimizer"]);sampler.set_state(cp["sampler_rng"]);torch.set_rng_state(cp["torch_rng"]);np.random.set_state(cp["numpy_rng"]);random.setstate(cp["python_rng"])
        step=cp["step"];epoch=cp["epoch"];order=cp["sampler_order"];cursor=cp["sampler_cursor"];best=cp["best_val_ADE2"];best_state=cp["best_model"];best_step=cp["best_step"];bad=cp["validation_bad_count"]
        run.event(event="resume",checkpoint_sha256=sha(resume),step=step,epoch=epoch,sampler_cursor=cursor,restored=["model","optimizer","torch_rng","numpy_rng","python_rng","sampler_rng","sampler_order","sampler_cursor","validation_bad_count"])
    def validate():
        model.eval()
        with torch.inference_mode():pred=model(x[val]).numpy()*scale+origin[val]
        return displacement_metrics(pred,y[val],[rows[i]["scene_id"] for i in val])
    def checkpoint():
        torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),"sampler_rng":sampler.get_state(),
                    "step":step,"epoch":epoch,"sampler_order":order,"sampler_cursor":cursor,"validation_bad_count":bad,"best_val_ADE2":best,"best_model":best_state,"best_step":best_step,
                    "normalization_scale":scale,"config":config,"dataset_manifest_sha256":sha(Path(dataset_path)/"manifest.json")},run.dir/"checkpoint.pt")
    initial=validate();run.event(event="initial_validation",metrics=initial,test_access=False)
    # Initial metrics are a diagnostic only. A random initial model is not a
    # trained deliverable even when its validation result happens to be better.
    started=time.time();loss_sum=0.;observations=0;end_step=min(max_steps,stop_steps) if stop_steps is not None else max_steps
    while step<end_step and epoch<epochs and bad<patience:
        if order is None:order=train[torch.randperm(len(train),generator=sampler).numpy()];cursor=0
        ids=order[cursor:cursor+2];model.train();opt.zero_grad(set_to_none=True);loss=torch.nn.functional.mse_loss(model(x[ids]),target[ids])
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite motion loss")
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10)
        if not torch.isfinite(norm):raise FloatingPointError("Nonfinite motion gradient")
        opt.step();step+=1;cursor+=len(ids);loss_sum+=float(loss.detach())*len(ids);observations+=len(ids)
        completed_epoch=cursor>=len(order)
        if completed_epoch:epoch+=1;order=None;cursor=0
        if step%30==0 or step==end_step or completed_epoch:
            vm=validate();score=vm["horizons"]["2.0"]["ADE"];last_val=vm
            if not np.isfinite(score):raise FloatingPointError("Nonfinite validation ADE")
            if score<best-1e-9:best=score;best_step=step;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};bad=0
            else:bad+=1
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)
            if rss>3_000_000_000:raise RuntimeError("Motion3GB RSS exceeded")
            run.event(event="measurement",step=step,epoch=epoch,sampler_cursor=cursor,train_loss=loss_sum/max(1,observations),validation=vm,validation_bad_count=bad,peak_rss_bytes=rss,finite_loss=True,finite_gradient=True,elapsed_s=time.time()-started)
            checkpoint();loss_sum=0.;observations=0
    if bad>=patience:stopped="early_stopping_validation_patience"
    elif epoch>=epochs:stopped="epoch_budget"
    elif stop_steps is not None and step==stop_steps and step<max_steps:stopped="intentional_resume_boundary"
    checkpoint();model.load_state_dict(best_state);model.eval()
    pack=Path(root)/"models"/f"motion-{Path(dataset_path).name}-seed{seed}-lr{lr:g}-budgeted";pack.mkdir(parents=True,exist_ok=True);torch.save({"model":model.state_dict()},pack/"weights.pt")
    manifest={"schema_version":1,"task":"forecast","architecture":"MLP2x64ReLU","network":{"history":11,"future":20,"hidden":64},"domain":meta["domain"],"units":meta["units"],"coordinate_system":meta["coordinate_system"],
            "native_weights":"weights.pt","sha256":sha(pack/"weights.pt"),"history_source":meta["history_source"],"preprocess":{"center_last":True,"scale":scale.tolist(),"normalization_fit_split":"train","sample_rate_hz":10,"history_span_s":1.,"history_points":11},
            "postprocess":{"relative_to_last":True,"unscale":True,"future_points":20,"future_times_s":FUTURE_TIMES.tolist()},"inference_adapter":"indoor_perception_lab.motion_training.ForecastAdapter",
            "predictor_input":"11x2 observed positions in declared coordinate system; no targets/identity/path inputs","output":"20x2 absolute positions in same units","seed":seed,"lr":lr,"best_step":best_step,"run_id":run.id,
            "dataset_manifest_sha256":sha(Path(dataset_path)/"manifest.json"),"training_status":"partial" if stopped=="intentional_resume_boundary" else "complete","protocol":"accepted_budgeted_primary","config":config,"redistribution":"our trained MLP weights; dataset media excluded"}
    json_write(pack/"manifest.json",manifest)
    metrics={"initial_validation":initial,"best_validation_ADE2":best,"best_step":best_step,"completed_steps":step,"completed_epochs":epoch,"stopped_reason":stopped,"train_windows":len(train),"val_windows":len(val),
             "test_access_during_selection":False,"pack":str(pack.relative_to(root)),"elapsed_s":time.time()-started,"budget_end":budget_guard(root)}
    run.finish(metrics);print(json.dumps({"event":"budgeted_motion_complete","dataset":Path(dataset_path).name,"lr":lr,"seed":seed,"steps":step,"val_ADE2":best,"run":run.id}),flush=True)
    return pack,run.dir,metrics


def _domain_dataset(domain):
    return {"fixture":"fixture-observed-world","world":"fixture-observed-world","jaad":"jaad-observed-image","image":"jaad-observed-image","oracle":"jaad-oracle-image"}.get(domain,domain)


def train_motion(domain="fixture",seed=42,lr=.001,max_steps=600,epochs=15,resume=None):
    root=root_path();dataset=root/"reports/motion/datasets"/_domain_dataset(domain)
    pack,run,metrics=train_model(root,dataset,lr,seed,epochs=epochs,max_steps=max_steps,resume=resume)
    return str(pack)


def evaluate_motion(pack,domain="fixture",split="test"):
    root=root_path();pack=Path(pack);pack=pack if pack.is_absolute() else root/pack
    return evaluate_pack(root,root/"reports/motion/datasets"/_domain_dataset(domain),pack,split)


def mark_excluded_pilots(root):
    root=Path(root);deviation={"role":"excluded exploratory protocol-deviation pilot","reason":"70epoch,batch128,LR.001/.003/.01 did not match accepted15epoch/600step,batch2 protocol",
        "heldout_exposure":"fixture test metrics were observed before correction; transparently retained; not used to select primary LR, seeds or architecture","primary_report_suffix":"-budgeted-summary.json"}
    for path in (root/"models").glob("motion-*"):
        if path.is_dir() and not path.name.endswith("-budgeted"):
            m=json.loads((path/"manifest.json").read_text());m.setdefault("protocol_status",deviation);json_write(path/"manifest.json",m)
    for path in (root/"reports/motion").glob("*-summary.json"):
        if not path.name.endswith("-budgeted-summary.json"):
            m=json.loads(path.read_text());m.setdefault("protocol_status",deviation);json_write(path,m)
    for path in (root/"reports/motion/runs").glob("*/run.json"):
        m=json.loads(path.read_text());config=m.get("config",{})
        if config.get("batch_size")!=2 or config.get("epochs",0)>15:
            m["protocol_status"]=deviation;json_write(path,m)
    json_write(root/"reports/motion/excluded-pilot-protocol-deviation.json",deviation)


def run_protocol(root,dataset_name,epochs=15,max_steps=600):
    dataset=Path(root)/"reports/motion/datasets"/dataset_name;mark_excluded_pilots(root);candidates=[]
    for lr in [.001,.0003,.0001]:
        pack,run,m=train_model(root,dataset,lr,42,epochs=epochs,max_steps=max_steps);candidates.append({"lr":lr,"pack":str(pack.relative_to(root)),"run":str(run.relative_to(root)),"validation_ADE2":m["best_validation_ADE2"],"steps":m["completed_steps"]})
    selected=min(candidates,key=lambda c:c["validation_ADE2"])
    summary={"dataset":dataset_name,"protocol":"accepted_budgeted_primary","validation_configs":candidates,"max_validation_configs":3,"selection_rule":"lowest validation2sADE seed42; no primary test selection",
        "selected_config":{"lr":selected["lr"],"epochs":epochs,"batch_size":2,"max_steps":max_steps,"patience":3,"network":{"history":11,"future":20,"hidden":64}},"selected_pack":selected["pack"],"heldout":[],
        "pilot_note":"excluded fixture pilot test was previously exposed; primary configs fixed by accepted protocol and no tuning from pilot test" if dataset_name.startswith("fixture") else "Rejected step0 checkpoint experiment automatically evaluated test before selector correction; no test metric used for corrected selection. Same3predeclaredLRs replayed; selected checkpoint must have optimizer updates."}
    summary_file=Path(root)/"reports/motion"/f"{dataset_name}-budgeted-summary.json";json_write(summary_file,summary)
    selection_file=summary_file.with_name(f"{dataset_name}-budgeted-selection-before-test.json");json_write(selection_file,summary)
    summary["selection_before_test_sha256"]=sha(selection_file)
    for seed in [42,43,44]:
        if seed==42:pack=Path(root)/selected["pack"]
        else:pack,run,m=train_model(root,dataset,selected["lr"],seed,epochs=epochs,max_steps=max_steps)
        report=evaluate_pack(root,dataset,pack);report["seed"]=seed;summary["heldout"].append(report);json_write(Path(root)/"reports/motion"/f"{dataset_name}-budgeted-heldout-seed{seed}.json",report)
    vals=np.array([[r["metrics"]["MLP2x64"]["horizons"][str(h)][metric] for h in HORIZONS for metric in ["ADE","FDE"]] for r in summary["heldout"]])
    summary["MLP_seed_aggregate"]={"seeds":[42,43,44],"horizons":{str(h):{metric:{"mean":float(vals[:,j*2+k].mean()),"std_sample":float(vals[:,j*2+k].std(ddof=1))} for k,metric in enumerate(["ADE","FDE"])} for j,h in enumerate(HORIZONS)}}
    h,y,rows,_=load_motion_dataset(dataset);adapter=ForecastAdapter(Path(root)/selected["pack"]);a=adapter.predict_history(h[:3]);y[:3]=999999;b=adapter.predict_history(h[:3]);adapter.close()
    summary["target_independence"]={"predictions_identical_after_future_target_perturbation":bool(np.array_equal(a,b)),"adapter_accepts_target_argument":False}
    json_write(summary_file,summary);return summary


def resume_audit(root,dataset_name):
    import torch
    dataset=Path(root)/"reports/motion/datasets"/dataset_name
    full,full_run,_=train_model(root,dataset,.001,137,max_steps=180);full_weights={k:v.clone() for k,v in torch.load(full/"weights.pt",weights_only=True)["model"].items()}
    cp_full=torch.load(full_run/"checkpoint.pt",weights_only=False)
    partial,partial_run,_=train_model(root,dataset,.001,137,max_steps=180,stop_steps=30)
    resumed,resumed_run,_=train_model(root,dataset,.001,137,max_steps=180,resume=partial_run/"checkpoint.pt")
    final=torch.load(resumed/"weights.pt",weights_only=True)["model"];cp_final=torch.load(resumed_run/"checkpoint.pt",weights_only=False)
    equal=all(torch.equal(full_weights[k],final[k]) for k in full_weights)
    final_equal=all(torch.equal(cp_full["model"][k],cp_final["model"][k]) for k in cp_full["model"])
    record={"dataset":dataset_name,"seed":137,"max_epochs":15,"max_steps":180,"batch_size":2,"interruption_after_step":30,"bitwise_identical_selected_model":equal,"bitwise_identical_final_training_model":final_equal,
            "uninterrupted_run":str(full_run.relative_to(root)),"partial_run":str(partial_run.relative_to(root)),"resumed_run":str(resumed_run.relative_to(root)),
            "states_restored":["model","optimizer","torch_rng","numpy_rng","python_rng","sampler_rng","sampler_order","sampler_cursor","validation_bad_count"],"role":"resume mechanics audit; no LR selection or heldout metrics"}
    json_write(Path(root)/"reports/motion"/f"{dataset_name}-budgeted-resume-audit.json",record)
    if not equal or not final_equal:raise AssertionError("Resume audit weights diverged")
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=["fixture-observed-world","jaad-observed-image","jaad-oracle-image"],required=True);parser.add_argument("--epochs",type=int,default=15);parser.add_argument("--max-steps",type=int,default=600);parser.add_argument("--resume-audit",action="store_true");args=parser.parse_args();root=root_path()
    run_protocol(root,args.dataset,args.epochs,args.max_steps)
    if args.resume_audit:resume_audit(root,args.dataset)


if __name__=="__main__":main()
