"""Held-out model accuracy with annotation access confined to the evaluator."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from uuid import uuid4
import numpy as np
from PIL import Image

from .evaluation import evaluate_corruptions
from .inputs import load_frames, _path
from .runner import model_identity, _configure_threads, _hash_file


def read_targets(manifest, rows, task):
    """Use processed-coordinate JAAD boxes and retain complete source IDs."""
    manifest = Path(manifest).resolve()
    cache, targets, hashes = {}, {}, {}
    for row in rows:
        sample_id = row["sample_id"]
        if task == "semantic":
            path = _path(manifest, row.get("mask_path", row["targets_path"]))
            with Image.open(path) as image:
                targets[sample_id] = np.array(image)
        else:
            path = _path(manifest, row["targets_path"])
            if path not in cache:
                cache[path] = json.loads(path.read_text())
            entries = cache[path]["frames"].get(str(row["frame_index"]), [])
            targets[sample_id] = [{"bbox": list(map(float, target["bbox_xyxy"])),
                "class_id": int(target.get("category_id", 1)), "track_id": str(target["track_id"]),
                "ignore": bool(target.get("ignore", False) or target.get("iscrowd", False))} for target in entries]
        if str(path) not in hashes:
            hashes[str(path)] = _hash_file(path)
    return targets, hashes


def evaluate_model(model_spec, input_manifest, output_path, *, task="detection", backend="native",
                   device="cpu", split="test", seed=42, threads=2, conditions=None):
    """No test tuning: score a fixed adapter and preset corruptions against labels."""
    from robot_model_sdk import load_adapter
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    if task not in ("detection", "semantic"):
        raise ValueError("Model accuracy helper supports detection/semantic")
    _configure_threads(threads, native_neural=(backend == "native" and str(model_spec) != "fixture"))
    manifest = Path(input_manifest).resolve()
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    accepted = {"val", "validation"} if split in ("val", "validation") else {split}
    rows = [row for row in rows if row.get("split") in accepted]
    frames, _, source = load_frames(manifest, count=len(rows), split=split)
    targets, target_hashes = read_targets(manifest, rows, task)
    identity = model_identity(model_spec)
    adapter = load_adapter(model_spec, backend=backend, device=device)
    try:
        metrics = evaluate_corruptions(frames, targets, adapter, task=task, seed=seed, conditions=conditions)
    finally:
        adapter.close()
    if model_identity(model_spec)["sha256"] != identity["sha256"]:
        raise RuntimeError("Model pack changed while evaluating fixed held-out labels")
    output = Path(output_path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=output.parent, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=output.parent, text=True))
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    result = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-accuracy-" + uuid4().hex[:6],
              "status": "verified", "model": identity, "backend": backend, "device": device,
              "task": task, "split": split, "source": source, "images": len(frames),
              "source_index_sha256": _hash_file(manifest), "target_files_sha256": target_hashes,
              "seed": seed, "requested_framework_threads": threads, "test_tuning": "none",
              "command": sys.argv, "git_commit": commit, "git_dirty": dirty,
              "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
              "duration_s": time.perf_counter() - started,
              "hardware": {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count()},
              "config_sha256": sha256(json.dumps({"task": task, "backend": backend, "device": device, "split": split,
                                                   "seed": seed, "threads": threads, "conditions": conditions}, sort_keys=True).encode()).hexdigest(),
              "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "evaluation": metrics}
    result["dependencies"] = {}
    for package in ("robot-model-sdk", "robot-reliability-bench", "torch", "torchvision", "onnxruntime", "numpy", "pycocotools"):
        try:
            result["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["dependencies"][package] = "not-installed"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
