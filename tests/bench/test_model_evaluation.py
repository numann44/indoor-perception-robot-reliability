import json
from robot_reliability_bench.model_evaluation import read_targets


def test_jaad_target_conversion_uses_processed_boxes_and_full_identity(tmp_path):
    target = tmp_path / "targets.json"
    target.write_text(json.dumps({"frames": {"15": [{"track_id": "1_2_3b", "category_id": 1,
        "bbox_xyxy": [1, 2, 3, 4], "bbox_xyxy_original": [10, 20, 30, 40], "iscrowd": True}]}}))
    manifest = tmp_path / "index.jsonl"
    rows = [{"sample_id": "clip/15", "frame_index": 15, "targets_path": "targets.json"}]
    result, hashes = read_targets(manifest, rows, "detection")
    assert result["clip/15"][0] == {"bbox": [1.0, 2.0, 3.0, 4.0], "class_id": 1, "track_id": "1_2_3b", "ignore": True}
    assert len(hashes[str(target.resolve())]) == 64
