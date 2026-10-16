import hashlib,json,shutil,tempfile,unittest
from pathlib import Path
import numpy as np
from indoor_perception_lab.forecast_v2 import baselines_v2,load_forecast_data,evaluate_forecast_v2,select_forecast_v2,train_forecast_v2,build_fixture_v2,build_jaad_forecast_v2,_jaad_frontend_identity
from indoor_perception_lab.motion_data import resample_history

ROOT=Path(__file__).resolve().parents[2]


class ResidualProtocolTests(unittest.TestCase):
    def test_initial_residual_network_exact_ls_baseline(self):
        import torch
        from robot_model_sdk.quality_models import residual_motion_network
        torch.set_num_threads(3);rng=np.random.default_rng(17);history=rng.normal(size=(3,11,2)).astype(np.float32);origin=history[:,-1:,:];model=residual_motion_network()
        with torch.inference_mode():prediction=model(torch.from_numpy(history-origin)).numpy()+origin
        np.testing.assert_allclose(prediction,baselines_v2(history)["constant_velocity_LS"],rtol=1e-5,atol=2e-6)
    def test_constant_position_last2_and_ls_use_same_positions(self):
        times=np.linspace(-1,0,11);h=np.c_[1.2+2*times,3-.3*times];pred=baselines_v2(h)
        np.testing.assert_allclose(pred["constant_velocity_last2"],pred["constant_velocity_LS"],atol=1e-12)
        np.testing.assert_allclose(pred["constant_position"][0],np.tile(h[-1],(20,1)))
    def test_future_sensor_point_never_admitted(self):
        with self.assertRaisesRegex(ValueError,"Future observation"):resample_history([0,1,1.1],[[0,0],[1,1],[1.1,1.1]],1.)
    def test_fixture_observed_history_and_fresh_independent_groups(self):
        h,y,rows,meta,path=load_forecast_data(ROOT,"fixture");groups={s:{r["scene_id"] for r in rows if r["split"]==s} for s in ["train","val","test"]}
        self.assertEqual([len(groups[s]) for s in ["train","val","test"]],[72,24,0]);self.assertEqual(len(meta["split_files"]["test"]["scene_groups"]),24)
        self.assertEqual(meta["units"],"metres");self.assertIn("RGB_CC",meta["history_source"]);self.assertTrue(all(r["tracker_id"].startswith("track-") for r in rows))
        self.assertTrue(all(seed>=40000 for seeds in meta["seed_groups"].values() for seed in seeds));self.assertEqual(h.shape[1:],(11,2));self.assertEqual(y.shape[1:],(20,2))
    def test_test_predictions_require_explicit_selection_lock(self):
        with self.assertRaises(PermissionError):evaluate_forecast_v2("models/v2/nonexistent","fixture","test")
    def test_train_loader_never_opens_test_targets(self):
        from unittest.mock import patch
        original=np.load;opened=[]
        def spy(path,*args,**kwargs):opened.append(str(path));return original(path,*args,**kwargs)
        with patch("numpy.load",side_effect=spy):load_forecast_data(ROOT,"fixture")
        self.assertTrue(opened);self.assertTrue(all("-test." not in p for p in opened));self.assertTrue(all("-train." in p or "-val." in p for p in opened))
    def test_forged_diagnostic_lock_never_loads_test_targets(self):
        from unittest.mock import patch
        pack=ROOT/"models/v2/forecast-fixture-seed137-lr0.001-resume-audit";m=json.loads((pack/"manifest.json").read_text())
        forged={"domain":"fixture","checkpoint_sha256":m["sha256"]}
        with patch("indoor_perception_lab.forecast_v2.load_forecast_data") as loader:
            with self.assertRaises((PermissionError,ValueError)):evaluate_forecast_v2(pack,"fixture","test",forged)
            loader.assert_not_called()
    def test_existing_selection_is_idempotent_and_never_trains(self):
        from unittest.mock import patch
        with patch("indoor_perception_lab.forecast_v2.train_forecast_v2") as trainer:
            selection=select_forecast_v2("fixture");self.assertFalse(selection["test_closed"]);trainer.assert_not_called()
    def test_selection_reuses_all_completed_frontdoor_trials_and_repeats(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            root=Path(temporary);dataset=root/"data/v2/forecast/fixture";dataset.mkdir(parents=True);shutil.copyfile(ROOT/"data/v2/forecast/fixture/manifest.json",dataset/"manifest.json")
            for seed,lr in [(42,.001),(42,.0003),(42,.0001),(43,.001),(44,.001)]:
                relative=f"models/v2/forecast-fixture-seed{seed}-lr{lr:g}-primary";source=ROOT/relative;destination=root/relative;destination.mkdir(parents=True)
                for name in ["manifest.json","weights.pt"]:shutil.copyfile(source/name,destination/name)
                m=json.loads((source/"manifest.json").read_text());run_relative=f"reports/v2/forecast/runs/{m['run_id']}";run=root/run_relative;run.mkdir(parents=True)
                for name in ["run.json","metrics.json"]:shutil.copyfile(ROOT/run_relative/name,run/name)
            with patch("indoor_perception_lab.forecast_v2.root_path",return_value=root),patch("indoor_perception_lab.forecast_v2.train_forecast_v2") as trainer:
                selection=select_forecast_v2("fixture");self.assertEqual(len(selection["configs"]),3);self.assertEqual([r["seed"] for r in selection["repeated_seeds"]],[43,44]);trainer.assert_not_called()
                metrics=root/f"reports/v2/forecast/runs/{json.loads((root/selection['selected']['pack']/'manifest.json').read_text())['run_id']}/run.json";record=json.loads(metrics.read_text());record["status"]="running";metrics.write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError,"completed trial"):select_forecast_v2("fixture")
                trainer.assert_not_called()
    def test_opened_primary_and_dataset_cannot_be_overwritten(self):
        from unittest.mock import patch
        with patch("indoor_perception_lab.forecast_v2.load_forecast_data") as loader:
            with self.assertRaisesRegex(RuntimeError,"frozen"):train_forecast_v2("fixture")
            loader.assert_not_called()
        with self.assertRaisesRegex(RuntimeError,"frozen"):build_fixture_v2(ROOT,counts=(1,1,1))
    def test_resume_changed_scheduler_or_seed_rejected_before_new_run(self):
        manifest=json.loads((ROOT/"models/v2/forecast-fixture-seed137-lr0.001-resume-audit/manifest.json").read_text());checkpoint=ROOT/"reports/v2/forecast/runs"/manifest["run_id"]/"checkpoint.pt"
        before=set((ROOT/"reports/v2/forecast/runs").iterdir())
        with self.assertRaisesRegex(ValueError,"Strictresume"):train_forecast_v2("fixture",seed=138,lr=.001,max_steps=800,preset="resume-audit",resume=checkpoint)
        self.assertEqual(before,set((ROOT/"reports/v2/forecast/runs").iterdir()))
    def test_image_frontend_requires_selected_teacher_and_tracker_low_score(self):
        from indoor_perception_lab.tracking_v2 import TRACKING_PRESETS
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            root=Path(temporary);pack=root/"models/v2/detector";pack.mkdir(parents=True);(pack/"weights.pt").write_bytes(b"synthetic-only");(pack/"manifest.json").write_text("{}");index=root/"data/v2/jaad/index.jsonl";index.parent.mkdir(parents=True);index.write_text("synthetic-only-index")
            digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();weight=digest(pack/"weights.pt");manifest=digest(pack/"manifest.json")
            (root/"models/v2/selection-detection.json").write_text(json.dumps({"task":"detection","selection_split":"val","test_used_for_selection":False,"dataset_split_hash":digest(index),"selected":{"pack":"models/v2/detector","weights_sha256":weight,"manifest_sha256":manifest}}))
            import indoor_perception_lab.tracking_v2 as tracking_module
            tracking={"selection_split":"val","test_used_for_selection":False,"detector_weights_sha256":weight,"detector_manifest_sha256":manifest,"validation_index_sha256":digest(index),"tracker_source_sha256":digest(Path(tracking_module.__file__)),"selected":{"configuration":"T2","parameters":TRACKING_PRESETS["T2"]}}
            (root/"models/v2/selection-tracking.json").write_text(json.dumps(tracking));identity=_jaad_frontend_identity(root,pack);self.assertEqual(identity["threshold"],.1);self.assertEqual(identity["tracker_parameters"]["max_age_s"],1.);self.assertEqual(len(identity["tracker_parameters_sha256"]),64)
            with self.assertRaisesRegex(ValueError,"threshold tuning"):_jaad_frontend_identity(root,pack,.2)
            (pack/"weights.pt").write_bytes(b"changed teacher")
            with self.assertRaisesRegex(ValueError,"exact canonical"):_jaad_frontend_identity(root,pack)
    def test_coverage_counts_unobserved_people_and_targets_wait_until_sensor_cache_done(self):
        """Synthetic RGB unit fixture; no learned-model performance claim."""
        import cv2
        from unittest.mock import patch
        from robot_model_sdk.models import FixtureAdapter
        class CountingAdapter(FixtureAdapter):
            calls=0
            def predict(self,frame):self.calls+=1;return super().predict(frame)
        frames=[];annotations={}
        for i in range(41):
            x=10+i//2;rgb=np.zeros((64,96,3),np.uint8);rgb[20:40,x:x+10,0]=240;frames.append(rgb[:,:,::-1].copy());annotations[str(i)]=[{"track_id":tid,"bbox_xyxy":box,"ignore":False,"iscrowd":False,"occlusion":0} for tid,box in [("GT-visible",[x,20,x+10,40]),("GT-unobserved",[80,20,90,40])]]
        class Capture:
            cursor=0
            def get(self,key):return 10 if key==cv2.CAP_PROP_FPS else len(frames)
            def read(self):
                image=frames[self.cursor];self.cursor+=1;return True,image.copy()
            def release(self):pass
        with tempfile.TemporaryDirectory(dir=ROOT/"cache/tmp") as temporary:
            root=Path(temporary);pack=root/"models/v2/synthetic-only";pack.mkdir(parents=True);(pack/"weights.pt").write_bytes(b"unit");(pack/"manifest.json").write_text("{}");index=root/"data/v2/jaad/index.jsonl";index.parent.mkdir(parents=True);target=index.parent/"synthetic-only-target.json";target.write_text(json.dumps({"fps":10,"width":96,"height":64,"frames":annotations}));row={"sample_id":"synthetic-unit","scene_id":"synthetic-unit-clip","split":"train","video_path":"synthetic-unit-only.mp4","targets_path":str(target.relative_to(root))};index.write_text(json.dumps(row)+"\n");validation=root/"reports/v2/data/jaad-validation.json";validation.parent.mkdir(parents=True);validation.write_text('{"primary_ready":true}')
            adapter=CountingAdapter();read_text=Path.read_text
            def checked_read(path,*args,**kwargs):
                if path==target:self.assertEqual(adapter.calls,41)
                return read_text(path,*args,**kwargs)
            identity={"threshold":.1,"detector_sha256":hashlib.sha256(b"unit").hexdigest(),"tracker_configuration":"T2","tracker_parameters":{"max_age_s":1.}}
            with patch("indoor_perception_lab.forecast_v2._jaad_frontend_identity",return_value=identity),patch("robot_model_sdk.models.load_adapter",return_value=adapter),patch("cv2.VideoCapture",return_value=Capture()),patch.object(Path,"read_text",checked_read):out=build_jaad_forecast_v2(pack,root=root)
            meta=json.loads((out/"manifest.json").read_text());coverage=meta["coverage"]["train"];self.assertEqual(coverage["all_annotated_eligible_windows"],12);self.assertEqual(coverage["matched_observed_history_windows"],6);self.assertEqual(coverage["coverage_fraction"],.5)
            samples=[json.loads(line) for line in (out/"index-train.jsonl").read_text().splitlines()];self.assertTrue(all(s["tracker_id"].startswith("track-") for s in samples));self.assertTrue(all(s["target_id_evaluator_only"]=="GT-visible" for s in samples))


if __name__=="__main__":unittest.main()
