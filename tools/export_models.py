#!/usr/bin/env python3
"""Fixed semantic FP32 export and train-only static QDQ INT8 experiment.

Export settings and calibration selection are fixed before validation/test.
The primary trained pack is read-only. No test accuracy tunes an export variant.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import cv2
import numpy as np
import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from robot_model_sdk import load_adapter
from robot_reliability_bench.inputs import load_frames
from robot_reliability_bench.metrics import semantic_metrics
from robot_reliability_bench.evaluation import evaluate_corruptions
from robot_reliability_bench.runner import model_identity, deployed_size


def hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(frame, metadata):
    """Match SDK semantic preprocessing exactly, including cv2 linear resize."""
    ih, iw = metadata.get("input_size", [240, 320])
    x = cv2.resize(frame.rgb, (iw, ih)).astype(np.float32) / 255
    mean = np.array(metadata.get("mean", [.485, .456, .406]), np.float32)
    std = np.array(metadata.get("std", [.229, .224, .225]), np.float32)
    return ((x - mean) / std).transpose(2, 0, 1)[None].copy()


def index_rows(root, split):
    accepted = {"val", "validation"} if split in ("val", "validation") else {split}
    return [json.loads(line) for line in (root / "data/hypersim/index.jsonl").read_text().splitlines()
            if line.strip() and json.loads(line).get("split") in accepted]


def targets_for(root, rows):
    # Evaluation-only target access, never called by calibration reader.
    targets = {}
    for row in rows:
        path = root / row["mask_path"]
        target = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if target is None:
            raise ValueError(f"Unreadable target mask {path}")
        targets[row["sample_id"]] = target
    return targets


class TrainReader(CalibrationDataReader):
    def __init__(self, frames, metadata):
        self.frames, self.metadata, self.i = frames, metadata, 0
    def get_next(self):
        if self.i >= len(self.frames):
            return None
        frame = self.frames[self.i]; self.i += 1
        return {"rgb": preprocess(frame, self.metadata)}
    def rewind(self):
        self.i = 0


def clone_pack(source, destination, precision, extra):
    destination.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "manifest.json").read_text())
    shutil.copy2(source / metadata["native_weights"], destination / metadata["native_weights"])
    metadata.update({"onnx": "model.onnx", "precision": precision, "native_precision": "fp32",
                     "onnx_precision": precision, "parent_pack": str(source), "parent_native_sha256": metadata["sha256"],
                     **extra})
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def experiment(root, source, reports):
    torch.set_num_threads(4); torch.set_num_interop_threads(1); cv2.setNumThreads(4)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-semantic-export-" + uuid4().hex[:6]
    output = reports / run_id; output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def event(kind, **values):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": kind, "elapsed_s": time.perf_counter() - start, **values}
        with (output / "events.jsonl").open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    source_identity = model_identity(source)
    metadata = json.loads((source / "manifest.json").read_text())
    if metadata["task"] != "semantic":
        raise ValueError("This fixed experiment exports semantic models only")
    index = root / "data/hypersim/index.jsonl"
    selection = root / "models/selection-semantic.json"
    config = {"source_pack": str(source), "input_shape": [1, 3, 240, 320], "opset": 18, "dynamo": False,
              "external_data": False, "fp32_rtol": 1e-4, "fp32_atol": 1e-5,
              "quantization": "static QDQ S8S8 per-channel MinMax", "calibration_split": "train",
              "calibration_samples": 32, "calibration_selection": "32 uniformly spaced indices in frozen train index",
              "seed": 42, "validation_tuning": "none", "test_tuning": "none"}
    run = {"run_id": run_id, "config": config, "source_model": source_identity,
           "source_index_sha256": hash_file(index), "selection_sha256": hash_file(selection),
           "command": sys.argv, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
           "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
           "hardware": {"platform": platform.platform(), "machine": platform.machine()},
           "dependencies": {package: importlib.metadata.version(package) for package in ("torch", "torchvision", "onnx", "onnxruntime", "numpy", "opencv-python-headless")},
           "source_sha256": hash_file(Path(__file__)), "status": "running"}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    native, fp32, int8 = None, None, None
    try:
        train_frames, _, train_meta = load_frames(index, count=120, split="train")
        val_frames, _, val_meta = load_frames(index, count=120, split="validation")
        if len(train_frames) < 32 or len(val_frames) != 20:
            raise ValueError(f"Fixed protocol needs >=32 train and 20 validation RGB frames, got {len(train_frames)}/{len(val_frames)}")
        calibration_frames = [train_frames[i] for i in np.linspace(0, len(train_frames) - 1, 32, dtype=int)]
        calibration_manifest = [{"sample_id": frame.sample_id,
            "rgb_sha256": sha256(frame.rgb.tobytes()).hexdigest(), "split": "train"} for frame in calibration_frames]
        (output / "calibration-manifest.json").write_text(json.dumps(calibration_manifest, indent=2) + "\n")
        calibration_hash = hash_file(output / "calibration-manifest.json")
        native = load_adapter(source, backend="native", device="cpu")
        fp32_pack, int8_pack = root / "models/segmentation-onnx", root / "models/segmentation-int8"
        float_meta = clone_pack(source, fp32_pack, "fp32", {"export_run_id": run_id, "opset": 18})
        tensor = torch.from_numpy(preprocess(calibration_frames[0], metadata))
        event("export_started", calibration_images=32, validation_images=20)
        with torch.inference_mode():
            torch.onnx.export(native.model, (tensor,), str(fp32_pack / "model.onnx"), input_names=["rgb"],
                output_names=["logits"], opset_version=18, dynamo=False, external_data=False,
                dynamic_axes=None, do_constant_folding=True)
        onnx.checker.check_model(str(fp32_pack / "model.onnx"))
        float_meta["onnx_sha256"] = hash_file(fp32_pack / "model.onnx")
        (fp32_pack / "manifest.json").write_text(json.dumps(float_meta, indent=2) + "\n")
        fp32 = load_adapter(fp32_pack, backend="onnx", device="cpu")
        parity = []
        native_val, float_val = {}, {}
        for frame in val_frames:
            a, b = native.predict(frame), fp32.predict(frame)
            difference = np.abs(a.logits - b.logits)
            parity.append({"sample_id": frame.sample_id, "rtol": 1e-4, "atol": 1e-5,
                "logits_allclose": bool(np.allclose(a.logits, b.logits, rtol=1e-4, atol=1e-5)),
                "max_absolute_logit_difference": float(difference.max()), "mean_absolute_logit_difference": float(difference.mean()),
                "semantic_exact_agreement": bool(np.array_equal(a.semantic, b.semantic)),
                "semantic_pixel_agreement": float((a.semantic == b.semantic).mean())})
            native_val[frame.sample_id], float_val[frame.sample_id] = a.semantic, b.semantic
        parity_result = {"samples": len(parity), "all_logits_allclose": all(row["logits_allclose"] for row in parity),
                         "all_semantic_masks_exact": all(row["semantic_exact_agreement"] for row in parity),
                         "mean_semantic_pixel_agreement": float(np.mean([row["semantic_pixel_agreement"] for row in parity])), "per_image": parity}
        (output / "fp32-parity.json").write_text(json.dumps(parity_result, indent=2) + "\n")
        event("fp32_parity_complete", **{key: value for key, value in parity_result.items() if key != "per_image"})
        if not parity_result["all_logits_allclose"]:
            raise RuntimeError("FP32 export failed fixed native logits parity tolerance")
        preprocessed = output / "calibration-preprocessed.onnx"
        quant_pre_process(fp32_pack / "model.onnx", preprocessed, skip_symbolic_shape=True, save_as_external_data=False)
        quant_meta = clone_pack(source, int8_pack, "int8-QDQ", {"export_run_id": run_id,
            "calibration_manifest_sha256": calibration_hash, "calibration_split": "train", "calibration_samples": 32,
            "quantization_format": "QDQ", "activation_type": "QInt8", "weight_type": "QInt8", "per_channel": True,
            "calibration_method": "MinMax", "preprocessing": "ORT graph optimization plus ONNX shape inference"})
        event("quantization_started", calibration_manifest_sha256=calibration_hash)
        quantize_static(preprocessed, int8_pack / "model.onnx", TrainReader(calibration_frames, metadata),
            quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
            per_channel=True, calibrate_method=CalibrationMethod.MinMax, use_external_data_format=False)
        onnx.checker.check_model(str(int8_pack / "model.onnx"))
        quant_meta["onnx_sha256"] = hash_file(int8_pack / "model.onnx")
        (int8_pack / "manifest.json").write_text(json.dumps(quant_meta, indent=2) + "\n")
        int8 = load_adapter(int8_pack, backend="onnx", device="cpu")
        quant_val, quant_differences = {}, []
        for frame in val_frames:
            a, b = fp32.predict(frame), int8.predict(frame)
            quant_val[frame.sample_id] = b.semantic
            quant_differences.append({"sample_id": frame.sample_id,
                "mean_absolute_logit_difference": float(np.abs(a.logits - b.logits).mean()),
                "max_absolute_logit_difference": float(np.abs(a.logits - b.logits).max()),
                "semantic_pixel_agreement": float((a.semantic == b.semantic).mean())})
        val_rows = index_rows(root, "validation")
        val_targets = targets_for(root, val_rows)
        validation = {"native": semantic_metrics(native_val, val_targets),
                      "onnx_fp32": semantic_metrics(float_val, val_targets), "onnx_int8": semantic_metrics(quant_val, val_targets),
                      "int8_vs_fp32_output_differences": quant_differences,
                      "target_access": "Validation evaluator only; calibration uses RGB-only sensor input"}
        (output / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
        event("validation_complete", native_mIoU=validation["native"]["mIoU"], fp32_mIoU=validation["onnx_fp32"]["mIoU"], int8_mIoU=validation["onnx_int8"]["mIoU"])
        # No model/backend parameters change after this point. Test is final reporting only.
        test_frames, _, test_meta = load_frames(index, count=120, split="test")
        if len(test_frames) != 40:
            raise ValueError(f"Fixed held-out corruption protocol needs 40 test frames, got {len(test_frames)}")
        test_targets = targets_for(root, index_rows(root, "test"))
        corruption_metrics = {"protocol": "Preset clean+18 sensor corruptions; fixed40testRGBtargets; no test tuning", "seed": 42, "variants": {}}
        for name, adapter in (("native", native), ("onnx_fp32", fp32), ("onnx_int8", int8)):
            event("heldout_corruptions_started", variant=name, samples=len(test_frames))
            corruption_metrics["variants"][name] = evaluate_corruptions(test_frames, test_targets, adapter, task="semantic", seed=42)
            event("heldout_corruptions_complete", variant=name,
                  clean_mIoU=corruption_metrics["variants"][name]["conditions"]["clean"]["metrics"]["mIoU"])
        (output / "heldout-corruptions.json").write_text(json.dumps(corruption_metrics, indent=2, allow_nan=False) + "\n")
        run.update({"status": "verified", "duration_s": time.perf_counter() - start,
            "fp32_parity": parity_result, "validation_summary": {name: validation[name]["mIoU"] for name in ("native", "onnx_fp32", "onnx_int8")},
            "model_packs": {"native": str(source), "onnx_fp32": str(fp32_pack), "onnx_int8": str(int8_pack)},
            "artifact_sizes": {"native": deployed_size(source, source / metadata["native_weights"], "native"),
                               "onnx_fp32": deployed_size(fp32_pack, fp32_pack / "model.onnx", "onnx"),
                               "onnx_int8": deployed_size(int8_pack, int8_pack / "model.onnx", "onnx")},
            "sensor_sources": {"train": train_meta, "validation": val_meta, "test": test_meta},
            "calibration_manifest_sha256": calibration_hash,
            "onnx_fp32_identity": model_identity(fp32_pack), "onnx_int8_identity": model_identity(int8_pack),
            "reference": "https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html"})
        if model_identity(source)["sha256"] != source_identity["sha256"]:
            raise RuntimeError("Primary pack changed during experiment; results cannot be declared immutable")
        event("experiment_complete", status=run["status"])
    except Exception as exc:
        run.update({"status": "failed", "duration_s": time.perf_counter() - start, "error": str(exc), "traceback": traceback.format_exc()})
        event("experiment_failed", error=str(exc))
    finally:
        for adapter in (native, fp32, int8):
            if adapter is not None:
                adapter.close()
        (output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
        (reports / "latest.json").write_text(json.dumps({"run_id": run_id, "status": run["status"], "run": str(output / "run.json")}, indent=2) + "\n")
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-pack", type=Path)
    parser.add_argument("--reports", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = experiment(root, args.source_pack.resolve() if args.source_pack else root / "models/segmentation-trained",
                        args.reports.resolve() if args.reports else root / "reports/optimization")
    print(json.dumps({"status": result["status"], "run_id": result["run_id"]}), flush=True)
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
