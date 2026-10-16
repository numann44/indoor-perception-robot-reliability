import json
import numpy as np
import pytest
from robot_model_sdk import SensorFrameV1, ModelOutput, Detection
from robot_reliability_bench.evaluation import evaluate_corruptions, evaluate_prediction_files


def test_corruption_drop_scores_fixed_targets_as_misses():
    frames = [SensorFrameV1(str(i), i * 100_000_000, np.full((12, 16, 3), 100, np.uint8)) for i in range(100)]
    targets = {str(i): [{"bbox": [0, 0, 8, 8], "class_id": 1}] for i in range(100)}
    class Adapter:
        def predict(self, frame):
            assert not hasattr(frame, "targets")
            return ModelOutput(detections=[Detection([0, 0, 8, 8], 1, 1)])
    result = evaluate_corruptions(frames, targets, Adapter(), task="detection", conditions=["clean", "frame_drop:3"])
    clean, drop = result["conditions"]["clean"], result["conditions"]["frame_drop:3"]
    assert clean["metrics"]["AP"] == pytest.approx(1)
    assert drop["dropped_frames"] > 0
    assert drop["metrics"]["AP"] < clean["metrics"]["AP"]
    assert drop["attempted_frames"] == 100
    repeat = evaluate_corruptions(frames, targets, Adapter(), task="detection", conditions=["clean", "frame_drop:3"])
    assert repeat["conditions"]["frame_drop:3"]["dropped_frames"] == drop["dropped_frames"]
    assert repeat["conditions"]["frame_drop:3"]["metrics"]["AP"] == drop["metrics"]["AP"]


def test_prediction_file_evaluation_separate_sources_and_hashes(tmp_path):
    target = tmp_path / "targets.json"
    pred = tmp_path / "predictions.json"
    target.write_text(json.dumps({"heldout": [{"bbox": [0, 0, 8, 8], "class_id": 1}]}))
    pred.write_text(json.dumps({"heldout": [{"bbox": [0, 0, 8, 8], "class_id": 1, "score": 1}]}))
    out = tmp_path / "evaluation.json"
    result = evaluate_prediction_files(pred, target, task="detection", output_path=out)
    assert result["metrics"]["AP"] == pytest.approx(1)
    assert len(result["predictions_sha256"]) == 64
    assert result["targets_sha256"] != result["predictions_sha256"]
    assert json.loads(out.read_text())["annotation_access"].startswith("Evaluator only")
