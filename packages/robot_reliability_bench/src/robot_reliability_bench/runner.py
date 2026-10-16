"""Actual subprocess measurement, immutable identities and JSON/CSV evidence."""

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from uuid import uuid4

import numpy as np
import psutil

from .corruptions import CONDITIONS, SEVERITIES, corrupt_frame
from .inputs import load_frames


def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(spec):
    if str(spec) == "fixture":
        import robot_model_sdk
        base = Path(robot_model_sdk.__file__).parent
        files = {name: _hash_file(base / name) for name in ("models.py", "fixture.py", "contracts.py")}
        return {"spec": "fixture", "sha256": sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "files": files,
                "kind": "synthetic_color_detector", "artifact_sizes_bytes": {}, "total_artifact_bytes": 0}
    path = Path(spec).resolve()
    manifest = path / "manifest.json" if path.is_dir() else path
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    files = {str(p.relative_to(manifest.parent)): _hash_file(p) for p in sorted(manifest.parent.rglob("*"))
             if p.is_file() and not any(part.startswith(".") for part in p.relative_to(manifest.parent).parts)}
    return {"spec": str(path), "files": files,
            "sha256": sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "artifact_sizes_bytes": {name: (manifest.parent / name).stat().st_size for name in files},
            "total_artifact_bytes": sum((manifest.parent / name).stat().st_size for name in files)}


def deployed_size(pack, artifact, backend):
    """Include referenced ONNX external tensor data, not just the graph container."""
    if not artifact.exists():
        return {"bytes": None, "files": [], "status": "unavailable"}
    files, scope = {artifact.resolve()}, "native checkpoint"
    if backend == "onnx":
        scope = "ONNX graph and referenced external tensor data"
        try:
            import onnx
        except ImportError:
            # Standard torch ONNX exporter names external tensor blobs *.data.
            # Publish an explicit conservative scope if graph inspection is absent.
            files.update(path.resolve() for path in pack.rglob("*.data") if path.is_file())
            scope = "ONNX graph plus all .data blobs in pack; conservative fallback without onnx inspection"
        else:
            model = onnx.load(str(artifact), load_external_data=False)
            def tensors(graph):
                yield from graph.initializer
                for node in graph.node:
                    for attribute in node.attribute:
                        if attribute.HasField("t"):
                            yield attribute.t
                        yield from attribute.tensors
                        if attribute.HasField("g"):
                            yield from tensors(attribute.g)
                        for child in attribute.graphs:
                            yield from tensors(child)
            for tensor in tensors(model.graph):
                for value in tensor.external_data:
                    if value.key == "location":
                        files.add((artifact.parent / value.value).resolve())
    return {"bytes": sum(path.stat().st_size for path in files),
            "files": [str(path) for path in sorted(files)], "definition": scope, "status": "available"}


def _summary(values):
    if not values:
        return {"status": "unavailable", "reason": "No measurements"}
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "mean_ms": float(values.mean()),
            "min_ms": float(values.min()), "max_ms": float(values.max())}


def _sync(device):
    if device == "mps":
        import torch
        torch.mps.synchronize()


def _configure_threads(threads, *, native_neural=False):
    # Framework thread pools inherit environment variables before their imports.
    if native_neural:
        try:
            import torch
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
        except ImportError:
            pass
    try:
        import cv2
        cv2.setNumThreads(threads)
    except ImportError:
        pass


def _serialize_output(prediction):
    if hasattr(prediction, "to_json"):
        value = prediction.to_json()
    else:
        value = {"detections": [{"bbox": list(d.bbox), "score": float(d.score), "class_id": int(d.class_id)}
                  for d in getattr(prediction, "detections", [])],
                 "semantic_shape": list(prediction.semantic.shape) if getattr(prediction, "semantic", None) is not None else None}
    if getattr(prediction, "semantic", None) is not None:
        semantic = np.ascontiguousarray(prediction.semantic)
        value["semantic_sha256"] = sha256(semantic.tobytes()).hexdigest()
        value["semantic_dtype"] = str(semantic.dtype)
    return json.dumps(value, sort_keys=True, allow_nan=False)


def measure_worker(config):
    """Executed in a fresh process once per repeat; returns raw samples.

    RAM uses a distinct prediction pass and is sampled by a monitor thread. These
    values are process RSS (not accelerator allocation or system peak memory).
    """
    from robot_model_sdk import load_adapter
    _configure_threads(config["threads"], native_neural=(config["backend"] == "native" and config["model_spec"] != "fixture"))
    np.random.seed(config["seed"])
    frames, io_ms, source = load_frames(config.get("input_manifest"), count=max(config["samples"], 120), split=config.get("split"))
    load_start = time.perf_counter_ns()
    adapter = load_adapter(config["model_spec"], backend=config["backend"], device=config["device"])
    _sync(config["device"])
    load_ms = (time.perf_counter_ns() - load_start) / 1e6
    process = psutil.Process()
    result = {"repeat": config["repeat"], "pid": os.getpid(), "source": source,
              "adapter_load_ms": load_ms, "input_decode_total_ms": io_ms,
              "input_decode_mean_ms": io_ms / len(frames), "conditions": {},
              "rss_after_load_bytes": process.memory_info().rss}
    framework_threads = {}
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        framework_threads["torch_intraop"] = torch.get_num_threads()
        framework_threads["torch_interop"] = torch.get_num_interop_threads()
    try:
        import cv2
        framework_threads["opencv"] = cv2.getNumThreads()
    except ImportError:
        pass
    if hasattr(adapter, "session"):
        options = adapter.session.get_session_options()
        framework_threads["onnx_intraop"] = options.intra_op_num_threads
        framework_threads["onnx_interop"] = options.inter_op_num_threads
    result["framework_threads"] = framework_threads
    try:
        first_start = time.perf_counter_ns()
        adapter.predict(frames[0])
        _sync(config["device"])
        result["cold_first_prediction_ms"] = (time.perf_counter_ns() - first_start) / 1e6
        result["process_birth_to_first_prediction_ms"] = (time.time() - process.create_time()) * 1000
        for i in range(config["warmup"]):
            adapter.predict(frames[i % len(frames)])
            _sync(config["device"])
        for condition in config["conditions"]:
            rng = np.random.default_rng(config["seed"])
            timings, forward, e2e, samples, digests, unique_ids = [], [], [], [], [], set()
            detection_counts, semantic_frames, depth_frames = [], 0, 0
            dropped = 0
            i = 0
            while len(timings) < config["samples"]:
                if i >= config["samples"] * 10:
                    raise RuntimeError("Corruption exceeded bounded retry budget")
                frame = frames[i % len(frames)]
                i += 1
                total_start = time.perf_counter_ns()
                altered = corrupt_frame(frame, condition, rng)
                if altered is None:
                    dropped += 1
                    continue
                _sync(config["device"])
                start = time.perf_counter_ns()
                prediction = adapter.predict(altered)
                _sync(config["device"])
                elapsed = (time.perf_counter_ns() - start) / 1e6
                timings.append(elapsed)
                detection_counts.append(len(prediction.detections))
                semantic_frames += int(prediction.semantic is not None)
                depth_frames += int(altered.depth is not None)
                last_timing = getattr(adapter, "last_timings_ms", {})
                if isinstance(last_timing, dict) and "model" in last_timing:
                    forward.append(float(last_timing["model"]))
                serialized = _serialize_output(prediction)
                e2e.append((time.perf_counter_ns() - total_start) / 1e6)
                digests.append(sha256(serialized.encode()).hexdigest())
                unique_ids.add(frame.sample_id)
                samples.append({"sample_id": frame.sample_id, "adapter_ms": elapsed,
                                "e2e_ms": e2e[-1], "model_ms": float(last_timing["model"]) if "model" in last_timing else None})
            result["conditions"][condition] = {
                "attempted_frames": i, "predicted_frames": len(timings),
                "dropped_frames": dropped, "unique_predicted_samples": len(unique_ids),
                "adapter_latency": _summary(timings), "model_forward_latency": _summary(forward),
                "e2e_latency": _summary(e2e),
                "throughput_fps": 1000 / float(np.mean(e2e)) if e2e else None,
                "output_summary": {"mean_detection_count": float(np.mean(detection_counts)) if detection_counts else None,
                                   "semantic_frames": semantic_frames, "depth_available_frames": depth_frames},
                "sample_measurements": samples, "output_digests": digests}
            print(json.dumps({"event": "condition_complete", "repeat": config["repeat"],
                              "condition": condition, "predicted_frames": len(timings), "dropped_frames": dropped,
                              "adapter_p50_ms": result["conditions"][condition]["adapter_latency"]["p50_ms"]}), flush=True)
        import threading
        rss_samples = [process.memory_info().rss]
        stop = threading.Event()
        def poll_rss():
            while not stop.wait(0.002):
                rss_samples.append(process.memory_info().rss)
        monitor = threading.Thread(target=poll_rss, daemon=True)
        monitor.start()
        try:
            for i in range(min(100, config["samples"])):
                adapter.predict(frames[i % len(frames)])
                _sync(config["device"])
                rss_samples.append(process.memory_info().rss)
        finally:
            stop.set()
            monitor.join()
        result["memory_pass"] = {"separate_from_timing": True, "samples": len(rss_samples),
                                  "rss_peak_observed_bytes": max(rss_samples),
                                  "rss_end_bytes": rss_samples[-1], "sample_interval_ms": 2,
                                  "definition": "Process resident set size; sampled peak, not allocator or device memory"}
    finally:
        adapter.close()
    return result


def run_benchmark(model_spec="fixture", output_dir="reports/bench", *, backend="native", device="cpu",
                  samples=100, warmup=10, repeats=3, threads=4, seed=42,
                  input_manifest=None, conditions=None, timeout_s=1200, split=None):
    """Run >=3 genuine fresh processes; every attempted frame is documented.

    Every condition has at least 100 timed predictions per repeat. Frame-drop
    conditions publish more frames, recording every drop and the retry count.
    """
    if samples < 100 or warmup < 10 or repeats < 3 or threads < 1:
        raise ValueError("Qualified benchmark requires samples>=100, warmup>=10, repeats>=3, threads>=1")
    if backend not in ("native", "onnx") or device not in ("cpu", "mps"):
        raise ValueError("Unsupported benchmark backend/device")
    if (backend == "onnx" or str(model_spec) == "fixture") and device != "cpu":
        raise ValueError("ONNX CPUExecutionProvider and color fixture support CPU only")
    conditions = CONDITIONS if conditions is None else conditions
    if not conditions or any(value not in CONDITIONS for value in conditions):
        raise ValueError("At least one valid corruption condition is required")
    model_spec = str(Path(model_spec).resolve()) if str(model_spec) != "fixture" else "fixture"
    input_manifest = str(Path(input_manifest).resolve()) if input_manifest else None
    config = {"model_spec": model_spec, "backend": backend, "device": device,
              "samples": samples, "warmup": warmup, "repeats": repeats, "threads": threads,
              "seed": seed, "batch_size": 1, "input_manifest": input_manifest, "conditions": conditions, "split": split}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = Path(output_dir).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    source_hash = _hash_file(input_manifest) if input_manifest else sha256(b"fixture-v1").hexdigest()
    config_hash = sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=output, text=True, stderr=subprocess.DEVNULL).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=output, text=True))
    except subprocess.CalledProcessError:
        commit, git_dirty = None, None
    versions = {}
    for package in ("robot-model-sdk", "robot-reliability-bench", "numpy", "Pillow", "psutil", "torch", "torchvision", "onnxruntime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    identity = {"run_id": run_id, "config_sha256": config_hash, "split_or_input_sha256": source_hash,
                "model": model_identity(model_spec), "git_commit": commit, "git_dirty": git_dirty,
                "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                             "logical_cpus": psutil.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total},
                "python": sys.version, "dependencies": versions,
                "corruption_severities": SEVERITIES,
                "implementation_files_sha256": {p.name: _hash_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "invocation": sys.argv,
                "definitions": {"adapter_latency": "predict(frame), including adapter preprocessing/postprocessing",
                                "model_forward_latency": "Adapter-reported synchronized network forward; detector includes TorchVision native transform/NMS",
                                "e2e_latency": "Sensor corruption, predict and result JSON serialization (semantic mask checksum included); excludes decode/render",
                                "cold_start": "Process birth to first prediction includes process/framework startup, input load and adapter load; first-prediction latency also reported separately",
                                "throughput": "1000/mean(e2e_ms), single stream batch 1",
                                "source_scope": "Fixture input is synthetic; sensor manifests establish data provenance but not target-device validation"}}
    if model_spec != "fixture":
        pack = Path(model_spec)
        if pack.is_file():
            pack = pack.parent
        model_metadata = json.loads((pack / "manifest.json").read_text())
        artifact_name = model_metadata.get("onnx", "model.onnx") if backend == "onnx" else model_metadata.get("native_weights", "weights.pt")
        artifact = pack / artifact_name
        size = deployed_size(pack, artifact, backend)
        identity["deployed_model_size_bytes"] = size["bytes"]
        identity["deployed_model_size"] = size
    else:
        identity["deployed_model_size_bytes"] = 0
    started = time.perf_counter()
    runs, failures = [], []
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": str(threads), "VECLIB_MAXIMUM_THREADS": str(threads)}
    for repeat in range(repeats):
        child_config = {**config, "repeat": repeat}
        config_path = output / f"repeat-{repeat}-config.json"
        result_path = output / f"repeat-{repeat}.json"
        config_path.write_text(json.dumps(child_config, indent=2) + "\n")
        command = [sys.executable, "-m", "robot_reliability_bench.cli", "_worker", "--config", str(config_path), "--output", str(result_path)]
        process_start = time.perf_counter()
        with (output / f"repeat-{repeat}.log").open("w") as log:
            log.write(json.dumps({"command": command, "started_utc": datetime.now(timezone.utc).isoformat()}) + "\n")
            log.flush()
            try:
                child = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s)
                status = child.returncode
            except subprocess.TimeoutExpired:
                status = "timeout"
            log.write(json.dumps({"exit_code": status, "duration_s": time.perf_counter() - process_start}) + "\n")
        if status != 0 or not result_path.exists():
            failures.append({"repeat": repeat, "exit_code": status, "log": str(output / f"repeat-{repeat}.log")})
        else:
            child_result = json.loads(result_path.read_text())
            child_result["process_total_s"] = time.perf_counter() - process_start
            runs.append(child_result)
    aggregate = {}
    for condition in conditions:
        values = [run["conditions"][condition] for run in runs]
        e2e_values = [s["e2e_ms"] for v in values for s in v["sample_measurements"]]
        aggregate[condition] = {"process_repeats": len(values),
                               "attempted_frames": sum(v["attempted_frames"] for v in values),
                               "predicted_frames": sum(v["predicted_frames"] for v in values),
                               "dropped_frames": sum(v["dropped_frames"] for v in values),
                               "adapter_latency": _summary([s["adapter_ms"] for v in values for s in v["sample_measurements"]]),
                               "model_forward_latency": _summary([s["model_ms"] for v in values for s in v["sample_measurements"] if s["model_ms"] is not None]),
                               "e2e_latency": _summary(e2e_values),
                               "throughput_fps": 1000 / float(np.mean(e2e_values)) if e2e_values else None,
                               "output_summary": {"mean_detection_count": float(np.mean([v["output_summary"]["mean_detection_count"] for v in values])) if values else None,
                                                  "semantic_frames": sum(v["output_summary"]["semantic_frames"] for v in values),
                                                  "depth_available_frames": sum(v["output_summary"]["depth_available_frames"] for v in values)},
                               "task_metrics": {"status": "unavailable", "reason": "Timing runner has no annotation access; use held-out evaluator"}}
    summary = {"status": "verified" if not failures and len(runs) == repeats else "failed",
               **identity, "config": config, "conditions": aggregate, "failures": failures,
               "repeat_evidence": [{"repeat": r["repeat"], "pid": r["pid"], "source": r["source"],
                                    "adapter_load_ms": r["adapter_load_ms"], "cold_first_prediction_ms": r["cold_first_prediction_ms"],
                                    "process_birth_to_first_prediction_ms": r["process_birth_to_first_prediction_ms"],
                                    "framework_threads": r["framework_threads"],
                                    "memory_pass": r["memory_pass"]} for r in runs],
               "duration_s": time.perf_counter() - started, "output_directory": str(output)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    import csv
    with (output / "measurements.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["run_id", "repeat", "condition", "sample_id", "adapter_ms", "model_ms", "e2e_ms"])
        writer.writeheader()
        for run in runs:
            for condition, value in run["conditions"].items():
                for sample in value["sample_measurements"]:
                    writer.writerow({"run_id": run_id, "repeat": run["repeat"], "condition": condition, **sample})
    (Path(output_dir).resolve() / "latest.json").write_text(json.dumps({"run_id": run_id, "summary": str(output / "summary.json")}, indent=2) + "\n")
    return summary
