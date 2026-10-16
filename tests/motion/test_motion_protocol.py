import inspect, json, os, unittest
from pathlib import Path
import numpy as np
from indoor_perception_lab.motion_data import (FUTURE_TIMES,resample_history,interpolate_labels,fixture_scene_parameters,render_fixture_frame,sensor_world_positions,load_motion_dataset)
from indoor_perception_lab.motion_training import baseline_forecasts,displacement_metrics,train_scale,ForecastAdapter
from robot_model_sdk.models import FixtureAdapter,load_adapter


ROOT=Path(__file__).resolve().parents[2]


class CausalHistoryTests(unittest.TestCase):
    def test_actual_2997_fps_interpolation(self):
        fps=29.97002997002997;t=np.arange(0,31,3)/fps;p=np.c_[2*t,-t]
        h=resample_history(t,p,t[-1]);self.assertEqual(h.shape,(11,2))
        np.testing.assert_allclose(h[:,0],2*(t[-1]+np.linspace(-1,0,11)),atol=1e-6)
        self.assertAlmostEqual(t[-1],1.001)
    def test_future_measurements_rejected(self):
        with self.assertRaisesRegex(ValueError,"Future observation"):resample_history(np.linspace(0,1.1,12),np.zeros((12,2)),1.)
    def test_missing_history_and_occlusion_gaps_rejected(self):
        self.assertIsNone(resample_history([.1,1.],[[0,0],[1,1]],1.))
        self.assertIsNone(resample_history([0,.1,.2,.7,.8,.9,1.],np.zeros((7,2)),1.))
        self.assertIsNone(interpolate_labels([0,1,3],[[0,0],[1,1],[3,3]],np.arange(1.1,3.01,.1),.11))
    def test_invalid_timestamps_and_nonfinite_rejected(self):
        for t,p in [([0,0],[[0,0],[0,0]]),([0,1],[[0,0],[np.nan,0]])]:
            with self.assertRaises(ValueError):resample_history(t,p,1.)
    def test_only_train_histories_fit_scale(self):
        train=np.arange(44,dtype=np.float32).reshape(2,11,2)/100
        a=train_scale(train);unseen=np.full((100,11,2),9999);np.testing.assert_array_equal(a,train_scale(train.copy()))
        self.assertGreater(float(train_scale(np.concatenate([train,unseen]))[0]),0)
        self.assertEqual(a.shape,(2,))
    def test_identical_history_baselines_and_metric_horizons(self):
        history=np.c_[np.linspace(0,1,11),np.zeros(11)]
        predictions=baseline_forecasts(history);truth=np.c_[1+FUTURE_TIMES,np.zeros(20)][None]
        np.testing.assert_allclose(predictions["constant_velocity"],truth,atol=1e-6)
        cv=displacement_metrics(predictions["constant_velocity"],truth)
        self.assertLess(cv["horizons"]["2.0"]["ADE"],1e-6)
        cp=displacement_metrics(predictions["constant_position"],truth)
        self.assertAlmostEqual(cp["horizons"]["0.5"]["ADE"],.3);self.assertAlmostEqual(cp["horizons"]["2.0"]["FDE"],2.)


class ActualSensorTests(unittest.TestCase):
    def test_actual_rgb_depth_geometry_and_camera_inverse(self):
        frame,targets=render_fixture_frame(fixture_scene_parameters(10005),20);adapter=FixtureAdapter()
        measured=sensor_world_positions(frame,adapter);adapter.close()
        self.assertEqual(frame.rgb.dtype,np.uint8);self.assertEqual(frame.depth.shape,(120,160));self.assertEqual(len(measured),2)
        for detection,position in measured:
            truths=[np.array(t["position_world_m"])[[0,2]] for t in targets]
            self.assertLess(min(np.linalg.norm(position-p) for p in truths),.06)
        self.assertEqual(list(inspect.signature(sensor_world_positions).parameters),["frame","adapter"])
    def test_independent_scene_parameters(self):
        a,_=render_fixture_frame(fixture_scene_parameters(10000),20);b,_=render_fixture_frame(fixture_scene_parameters(30000),20)
        self.assertFalse(np.array_equal(a.rgb,b.rgb));self.assertFalse(np.array_equal(a.depth,b.depth))
    def test_pinned_dataset_groups_and_source_only_history(self):
        h,y,rows,meta=load_motion_dataset(ROOT/"reports/motion/datasets/fixture-observed-world")
        groups={split:{r["scene_id"] for r in rows if r["split"]==split} for split in ["train","val","test"]}
        self.assertFalse(groups["train"]&groups["val"]);self.assertFalse(groups["train"]&groups["test"]);self.assertFalse(groups["val"]&groups["test"])
        self.assertEqual(meta["units"],"metres");self.assertIn("RGB_connected_components",meta["history_source"])
        self.assertEqual(len(set(r["sample_id"] for r in rows)),len(rows));self.assertEqual(h.shape[1:],(11,2));self.assertEqual(y.shape[1:],(20,2))
    def test_checkpoint_loaded_inference_target_independence_and_units(self):
        summary=json.loads((ROOT/"reports/motion/fixture-observed-world-budgeted-summary.json").read_text());pack=ROOT/summary["selected_pack"]
        h,y,rows,meta=load_motion_dataset(ROOT/"reports/motion/datasets/fixture-observed-world");adapter=ForecastAdapter(pack)
        a=adapter.predict_history(h[:2]);y[:2]=1e9;b=adapter.predict_history(h[:2]);np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(2,20,2));self.assertEqual(adapter.metadata["units"],"metres")
        with self.assertRaises(ValueError):adapter.predict_history(np.zeros((11,3)))
        adapter.close()
    def test_actual_resume_evidence(self):
        audit=json.loads((ROOT/"reports/motion/fixture-observed-world-budgeted-resume-audit.json").read_text());self.assertTrue(audit["bitwise_identical_selected_model"])
        self.assertTrue(audit["bitwise_identical_final_training_model"])
        self.assertIn("sampler_rng",audit["states_restored"])
        self.assertIn("sampler_cursor",audit["states_restored"])
    def test_shared_sdk_forecaster_parity(self):
        summary=json.loads((ROOT/"reports/motion/fixture-observed-world-budgeted-summary.json").read_text());pack=ROOT/summary["selected_pack"]
        h,_,_,_=load_motion_dataset(ROOT/"reports/motion/datasets/fixture-observed-world")
        own=ForecastAdapter(pack);sdk=load_adapter(pack)
        np.testing.assert_allclose(own.predict_history(h[:3]),sdk.predict_history(h[:3]),rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(own.predict_timed_history(np.linspace(0,1,11),h[0]),sdk.predict_timed_history(np.linspace(0,1,11),h[0]),rtol=1e-6,atol=1e-7)
        own.close();sdk.close()
    def test_primary_limits_and_selection_before_test(self):
        summary=json.loads((ROOT/"reports/motion/fixture-observed-world-budgeted-summary.json").read_text())
        self.assertEqual(summary["protocol"],"accepted_budgeted_primary");self.assertEqual(len(summary["validation_configs"]),3)
        self.assertEqual({c["lr"] for c in summary["validation_configs"]},{.001,.0003,.0001})
        self.assertEqual(summary["selected_config"]["batch_size"],2);self.assertLessEqual(summary["selected_config"]["max_steps"],600)
        selection=json.loads((ROOT/"reports/motion/fixture-observed-world-budgeted-selection-before-test.json").read_text());self.assertEqual(selection["heldout"],[])
        for c in summary["validation_configs"]:
            m=json.loads((ROOT/c["run"]/"metrics.json").read_text());self.assertLessEqual(m["completed_steps"],600);self.assertFalse(m["test_access_during_selection"])
    def test_jaad_primary_sensor_source_actual_trained_checkpoint(self):
        dataset=ROOT/"reports/motion/datasets/jaad-observed-image";h,y,rows,meta=load_motion_dataset(dataset)
        self.assertEqual(meta["units"],"normalized_image");self.assertEqual(meta["history_source"],"detected_and_tracked_RGB")
        self.assertFalse(meta["tracker"]["GT_id_input"]);self.assertTrue(all(r["tracker_id"].startswith("track-") for r in rows))
        self.assertTrue(all(r["history_source"]=="detected_and_tracked_RGB" for r in rows))
        self.assertEqual(meta["detector_sha256"],"f32797b32c5c648fa02a25839874be793a8f58a500394286c1de739aa419a48e")
        summary=json.loads((ROOT/"reports/motion/jaad-observed-image-budgeted-summary.json").read_text())
        manifest=json.loads((ROOT/summary["selected_pack"]/"manifest.json").read_text());self.assertGreater(manifest["best_step"],0)
        adapter=ForecastAdapter(ROOT/summary["selected_pack"]);sdk=load_adapter(ROOT/summary["selected_pack"])
        np.testing.assert_allclose(adapter.predict_history(h[:3]),sdk.predict_history(h[:3]),rtol=1e-6,atol=1e-7)
        adapter.close();sdk.close()
    def test_coordinate_units_cannot_be_mixed_by_evaluator(self):
        from indoor_perception_lab.motion_training import evaluate_pack
        summary=json.loads((ROOT/"reports/motion/fixture-observed-world-budgeted-summary.json").read_text())
        with self.assertRaisesRegex(ValueError,"coordinate units mismatch"):
            evaluate_pack(ROOT,ROOT/"reports/motion/datasets/jaad-observed-image",ROOT/summary["selected_pack"])


if __name__=="__main__":unittest.main()
