import json
import numpy as np
import pytest
from PIL import Image
from robot_reliability_bench.inputs import load_frames
from robot_reliability_bench.runner import run_benchmark, model_identity, deployed_size, _summary


def test_inputs_never_open_annotations(tmp_path):
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(tmp_path / "rgb.png")
    manifest = tmp_path / "index.jsonl"
    manifest.write_text(json.dumps({"sample_id": "heldout", "timestamp_ns": 15,
        "split": "val",
        "rgb_path": "rgb.png", "targets_path": "does-not-exist-and-must-not-open.json"}) + "\n")
    frames, elapsed, source = load_frames(manifest)
    assert len(frames) == 1
    assert frames[0].sample_id == "heldout"
    assert frames[0].depth is None
    assert elapsed >= 0
    assert source["kind"] == "sensor_manifest"
    assert not hasattr(frames[0], "targets")
    assert len(load_frames(manifest, split="validation")[0]) == 1


def test_bad_input_is_not_silently_skipped(tmp_path):
    manifest = tmp_path / "index.json"
    manifest.write_text("[]")
    with pytest.raises(ValueError, match="no decodable"):
        load_frames(manifest)
    manifest.write_text(json.dumps([{"rgb_path": "missing.png"}]))
    with pytest.raises(FileNotFoundError):
        load_frames(manifest)


def test_model_digest_changes_with_weight(tmp_path):
    (tmp_path / "manifest.json").write_text('{"schema_version":1}')
    (tmp_path / "weights.pt").write_bytes(b"a")
    first = model_identity(tmp_path)
    (tmp_path / "weights.pt").write_bytes(b"b")
    assert first["sha256"] != model_identity(tmp_path)["sha256"]
    (tmp_path / "model.onnx.data").write_bytes(b"external weights")
    assert "model.onnx.data" in model_identity(tmp_path)["files"]


def test_runner_rejects_unqualified_protocol_before_mutation(tmp_path):
    with pytest.raises(ValueError, match="samples"):
        run_benchmark(output_dir=tmp_path / "uncreated", samples=99)
    assert not (tmp_path / "uncreated").exists()
    with pytest.raises(ValueError, match="CPU only"):
        run_benchmark(output_dir=tmp_path / "uncreated", device="mps")


def test_percentiles_have_known_values():
    summary = _summary([1.0, 2.0, 3.0])
    assert summary["p50_ms"] == 2
    assert summary["p95_ms"] == pytest.approx(2.9)
    assert _summary([])["status"] == "unavailable"


def test_project_relative_paths_and_hypersim_depth_are_explicit(tmp_path):
    root = tmp_path / "portfolio"
    data = root / "data" / "hypersim"
    data.mkdir(parents=True)
    (root / "PROJECT_ROOT").write_text(str(root))
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(data / "rgb.png")
    # Must not be opened/decoded as a registered depth image.
    (data / "raw.hdf5").write_bytes(b"not a decodable HDF5")
    manifest = data / "index.jsonl"
    manifest.write_text(json.dumps({"sample_id": "test-sample", "split": "test", "frame_index": 5,
        "rgb_path": "data/hypersim/rgb.png", "depth_path": "data/hypersim/raw.hdf5",
        "targets_path": "missing-labels.json"}) + "\n")
    frames, _, source = load_frames(manifest, split="test")
    assert frames[0].depth is None
    assert frames[0].timestamp_ns == 500_000_000
    assert source["skipped_unregistered_hdf5_depth"] == 1
    assert source["synthesized_timestamps"] == 1
    with pytest.raises(ValueError, match="no decodable"):
        load_frames(manifest, split="train")


def test_onnx_external_weight_size_includes_tensor_data(tmp_path):
    onnx = pytest.importorskip("onnx")
    tensor = onnx.helper.make_tensor("weights", onnx.TensorProto.FLOAT, [2], [1.0, 2.0])
    tensor.ClearField("float_data")
    tensor.data_location = onnx.TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="weights.data")
    tensor.external_data.add(key="offset", value="0")
    tensor.external_data.add(key="length", value="8")
    graph = onnx.helper.make_graph([], "external", [], [], [tensor])
    path = tmp_path / "model.onnx"
    onnx.save(onnx.helper.make_model(graph), str(path))
    (tmp_path / "weights.data").write_bytes(np.array([1, 2], np.float32).tobytes())
    size = deployed_size(tmp_path, path, "onnx")
    assert size["bytes"] == path.stat().st_size + 8
    assert len(size["files"]) == 2
