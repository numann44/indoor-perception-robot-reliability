"""Separate prediction/target-file evaluator and sensor corruption evaluator."""

from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from PIL import Image

from .corruptions import CONDITIONS, corrupt_frame
from .metrics import detection_ap, semantic_metrics, tracking_metrics, forecasting_metrics


def _read(path):
    path = Path(path).resolve()
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {str(row["sample_id"]): row for row in rows}
    value = json.loads(path.read_text())
    if isinstance(value, dict) and "samples" in value:
        return {str(row["sample_id"]): row for row in value["samples"]}
    return value


def _masks(value, parent):
    masks = {}
    for sample_id, entry in value.items():
        if isinstance(entry, dict):
            entry = entry.get("semantic_path", entry.get("semantic"))
        if entry is None:
            masks[sample_id] = None
        elif isinstance(entry, str):
            path = parent / entry
            if path.suffix == ".npy":
                masks[sample_id] = np.load(path, allow_pickle=False)
            else:
                with Image.open(path) as image:
                    masks[sample_id] = np.array(image)
        else:
            masks[sample_id] = np.asarray(entry)
    return masks


def _detections(value):
    return {sample_id: entry.get("detections", entry.get("targets", [])) if isinstance(entry, dict) else entry
            for sample_id, entry in value.items()}


def evaluate_prediction_files(predictions_path, targets_path, *, task, output_path=None,
                              unit=None, horizons=(0.5, 1.0, 2.0), step_seconds=0.1,
                              num_classes=40, ignore_index=255):
    """Evaluate existing outputs without loading or executing an inference model.

    Detection/tracking JSON maps sample IDs to lists (or records with detections).
    Semantic JSON maps IDs to arrays/path strings/records with semantic_path.
    Forecast arrays use [windows,future_steps,2] in .npy or JSON, explicit units.
    """
    predictions_path, targets_path = Path(predictions_path).resolve(), Path(targets_path).resolve()
    predictions, targets = _read(predictions_path), _read(targets_path)
    if task == "detection":
        metrics = detection_ap(_detections(predictions), _detections(targets))
    elif task == "tracking":
        metrics = tracking_metrics(_detections(predictions), _detections(targets))
    elif task == "semantic":
        metrics = semantic_metrics(_masks(predictions, predictions_path.parent), _masks(targets, targets_path.parent),
                                   num_classes=num_classes, ignore_index=ignore_index)
    elif task == "forecast":
        if isinstance(predictions, dict):
            predictions = predictions["forecast"]
        if isinstance(targets, dict):
            targets = targets["forecast"]
        metrics = forecasting_metrics(predictions, targets, unit=unit, horizons=horizons, step_seconds=step_seconds)
    else:
        raise ValueError(f"Unknown task: {task}")
    result = {"task": task, "metrics": metrics, "predictions": str(predictions_path), "targets": str(targets_path),
              "predictions_sha256": sha256(predictions_path.read_bytes()).hexdigest(),
              "targets_sha256": sha256(targets_path.read_bytes()).hexdigest(),
              "annotation_access": "Evaluator only; no model adapter receives target files"}
    if output_path:
        path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def evaluate_corruptions(frames, targets, adapter, *, task, seed=42, conditions=None,
                         num_classes=40, ignore_index=255):
    """Hold labels fixed while corrupting sensor input and measuring task metrics.

    targets is a separate mapping consumed solely by this evaluator. It is never
    assigned to SensorFrameV1 or sent to adapter.predict. Dropped frames count as
    misses. Frame-local models cannot measure tracking or timestamp robustness;
    those require the stateful pipeline adapter and an identity evaluator.
    """
    if task not in ("detection", "semantic"):
        raise ValueError("Corruption helper supports detection/semantic frame models")
    frames = list(frames)
    if len({frame.sample_id for frame in frames}) != len(frames):
        raise ValueError("Held-out frames must have unique sample IDs")
    results = {}
    for condition in CONDITIONS if conditions is None else conditions:
        rng = np.random.default_rng(seed)
        predictions, drops = {}, 0
        for frame in frames:
            altered = corrupt_frame(frame, condition, rng)
            if altered is None:
                predictions[frame.sample_id] = [] if task == "detection" else None
                drops += 1
            else:
                output = adapter.predict(altered)
                predictions[frame.sample_id] = ([d.to_json() for d in output.detections]
                                               if task == "detection" else output.semantic)
        metric = detection_ap(predictions, targets) if task == "detection" else semantic_metrics(
            predictions, targets, num_classes=num_classes, ignore_index=ignore_index)
        results[condition] = {"attempted_frames": len(frames), "dropped_frames": drops,
                              "predicted_frames": len(frames) - drops, "metrics": metric}
    return {"seed": seed, "task": task, "conditions": results,
            "label_policy": "Fixed targets; dropout predictions are scored as misses",
            "limitation": "Frame-local model metrics cannot establish tracking/time synchronization robustness"}
