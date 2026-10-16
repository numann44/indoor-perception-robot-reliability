"""Small independently testable evaluation metrics.

No annotation ever enters an inference adapter. These functions consume already
separated predictions/targets and define the evaluation scope explicitly.
"""

import numpy as np


def box_iou(a, b):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(hi - lo, 0).prod(axis=-1)
    aa = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=-1)
    bb = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=-1)
    return intersection / np.maximum(aa[:, None] + bb[None, :] - intersection, 1e-12)


def _ignore_overlap(pred_box, ignore_boxes):
    """Crowd ignore matching uses intersection over prediction area."""
    if not ignore_boxes:
        return 0.0
    a, b = np.asarray(pred_box, float), np.asarray(ignore_boxes, float)
    intersection = np.maximum(np.minimum(a[2:], b[:, 2:]) - np.maximum(a[:2], b[:, :2]), 0).prod(axis=1)
    area = np.maximum(a[2:] - a[:2], 0).prod()
    return float((intersection / max(area, 1e-12)).max())


def _small_detection_ap(predictions, targets, thresholds=None):
    """Pooled-class AP with 101-point interpolated precision, IoU .50:.95.

    Inputs map sample_id -> lists of {bbox,class_id,score} for predictions and
    {bbox,class_id,ignore?} for targets. Ignored boxes suppress overlapping
    predictions. This small-evaluation AP has no COCO size-group/maxDets policy
    and must not be described as the official COCO evaluator.
    """
    thresholds = np.arange(0.5, 0.96, 0.05) if thresholds is None else np.asarray(thresholds, float)
    classes = sorted({int(t["class_id"]) for entries in targets.values() for t in entries if not t.get("ignore", False)})
    if not classes:
        return {"status": "unavailable", "reason": "No evaluable ground-truth detections"}
    per_class = {}
    for class_id in classes:
        gt = {sample_id: [t for t in entries if int(t["class_id"]) == class_id and not t.get("ignore", False)]
              for sample_id, entries in targets.items()}
        ignored = {sample_id: [t for t in entries if t.get("ignore", False)] for sample_id, entries in targets.items()}
        positives = sum(len(entries) for entries in gt.values())
        ordered = sorted([(float(p["score"]), sample_id, p) for sample_id, entries in predictions.items()
                          for p in entries if int(p["class_id"]) == class_id], key=lambda x: -x[0])
        aps = []
        for threshold in thresholds:
            used = {sample_id: set() for sample_id in gt}
            tp, fp = [], []
            for _, sample_id, prediction in ordered:
                boxes = [t["bbox"] for t in gt.get(sample_id, [])]
                overlap = box_iou([prediction["bbox"]], boxes)[0]
                candidates = [i for i in range(len(boxes)) if i not in used.get(sample_id, set()) and overlap[i] >= threshold]
                if candidates:
                    match = max(candidates, key=lambda i: overlap[i])
                    used[sample_id].add(match)
                    tp.append(1); fp.append(0)
                else:
                    ignore_boxes = [t["bbox"] for t in ignored.get(sample_id, [])]
                    if ignore_boxes and _ignore_overlap(prediction["bbox"], ignore_boxes) >= threshold:
                        continue
                    tp.append(0); fp.append(1)
            if not tp:
                aps.append(0.0)
                continue
            recall = np.cumsum(tp) / positives
            precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1)
            aps.append(float(np.mean([precision[recall >= r].max() if np.any(recall >= r) else 0
                                      for r in np.linspace(0, 1, 101)])))
        per_class[str(class_id)] = {"AP": float(np.mean(aps)), "AP50": aps[0], "ground_truth_count": positives,
                                    "AP_by_threshold": [float(x) for x in aps]}
    return {"status": "available", "AP": float(np.mean([v["AP"] for v in per_class.values()])),
            "AP50": float(np.mean([v["AP50"] for v in per_class.values()])), "per_class": per_class,
            "iou_thresholds": thresholds.tolist(), "definition": "101-point pooled AP with crowd-area ignore matching; no COCO maxDets/size groups"}


def detection_ap(predictions, targets, thresholds=None, prefer_coco=True):
    """Use official pycocotools COCO AP when installed; label the fallback honestly."""
    if prefer_coco:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            pass
        else:
            import contextlib
            import io
            classes = sorted({int(t["class_id"]) for entries in targets.values() for t in entries if not t.get("ignore", False)})
            if not classes:
                return {"status": "unavailable", "reason": "No evaluable ground-truth detections"}
            all_classes = sorted(set(classes) | {int(p["class_id"]) for entries in predictions.values() for p in entries})
            samples = {sample_id: i + 1 for i, sample_id in enumerate(sorted(set(targets) | set(predictions)))}
            annotations, detections = [], []
            for sample_id, entries in targets.items():
                for target in entries:
                    x1, y1, x2, y2 = map(float, target["bbox"])
                    annotations.append({"id": len(annotations) + 1, "image_id": samples[sample_id],
                        "category_id": int(target["class_id"]), "bbox": [x1, y1, max(0, x2 - x1), max(0, y2 - y1)],
                        "area": max(0, x2 - x1) * max(0, y2 - y1), "iscrowd": int(target.get("ignore", False))})
            for sample_id, entries in predictions.items():
                for prediction in entries:
                    x1, y1, x2, y2 = map(float, prediction["bbox"])
                    detections.append({"image_id": samples[sample_id], "category_id": int(prediction["class_id"]),
                                       "bbox": [x1, y1, max(0, x2 - x1), max(0, y2 - y1)], "score": float(prediction["score"])})
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                gt = COCO()
                gt.dataset = {"info": {"description": "Independent held-out evaluation"},
                    "images": [{"id": i, "file_name": sample_id} for sample_id, i in samples.items()],
                    "categories": [{"id": class_id, "name": str(class_id)} for class_id in all_classes],
                    "annotations": annotations}
                gt.createIndex()
                if detections:
                    pred = gt.loadRes(detections)
                else:
                    pred = COCO(); pred.dataset = {**gt.dataset, "annotations": []}; pred.createIndex()
                evaluator = COCOeval(gt, pred, "bbox")
                evaluator.params.imgIds = list(samples.values())
                evaluator.params.catIds = classes
                if thresholds is not None:
                    evaluator.params.iouThrs = np.asarray(thresholds, float)
                evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            precision = evaluator.eval["precision"][:, :, :, 0, -1]
            per_class = {}
            for i, class_id in enumerate(classes):
                p = precision[:, :, i]
                per_class[str(class_id)] = {"AP": float(p[p >= 0].mean()) if np.any(p >= 0) else None,
                    "AP50": float(p[0][p[0] >= 0].mean()) if np.any(p[0] >= 0) else None,
                    "ground_truth_count": sum(int(t["class_id"]) == class_id and not t.get("ignore", False)
                                              for entries in targets.values() for t in entries)}
            return {"status": "available", "AP": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]),
                    "AP75": float(evaluator.stats[2]), "per_class": per_class,
                    "iou_thresholds": evaluator.params.iouThrs.tolist(),
                    "definition": "Official pycocotools COCO bbox AP, area all, maxDets=100; ignored regions treated as crowd",
                    "evaluator": "pycocotools", "evaluator_log": log.getvalue()}
    result = _small_detection_ap(predictions, targets, thresholds)
    result["evaluator"] = "small_reference_fallback"
    return result


def semantic_metrics(predictions, targets, num_classes=40, ignore_index=255):
    """Mean IoU over classes present in GT or prediction; never average absent NaNs.

    A None prediction counts all valid GT pixels as missed rather than ignoring a
    dropped frame. Missing ground-truth masks are excluded and counted separately.
    """
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    missed = np.zeros(num_classes, dtype=np.int64)
    included, unavailable = 0, 0
    for sample_id, target in targets.items():
        if target is None:
            unavailable += 1
            continue
        target = np.asarray(target)
        valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
        prediction = predictions.get(sample_id)
        included += 1
        if prediction is None:
            missed += np.bincount(target[valid].astype(int), minlength=num_classes)
            continue
        prediction = np.asarray(prediction)
        if prediction.shape != target.shape:
            raise ValueError(f"Semantic prediction/target shape mismatch for {sample_id}")
        invalid_pred = valid & (~np.isfinite(prediction) | (prediction < 0) | (prediction >= num_classes))
        missed += np.bincount(target[invalid_pred].astype(int), minlength=num_classes)
        valid &= ~invalid_pred
        confusion += np.bincount(num_classes * target[valid].astype(int) + prediction[valid].astype(int),
                                 minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    if not included or not (confusion.sum() + missed.sum()):
        return {"status": "unavailable", "reason": "No valid target semantic pixels"}
    tp = confusion.diagonal()
    union = confusion.sum(0) + confusion.sum(1) + missed - tp
    ious = np.divide(tp, union, out=np.full(num_classes, np.nan), where=union != 0)
    return {"status": "available", "mIoU": float(np.nanmean(ious)),
            "per_class_IoU": [None if np.isnan(x) else float(x) for x in ious],
            "confusion_matrix": confusion.tolist(), "missed_ground_truth_pixels": missed.tolist(),
            "evaluated_samples": included, "unavailable_samples": unavailable,
            "pixel_accuracy": float(tp.sum() / (confusion.sum() + missed.sum()))}


def forecasting_metrics(predicted, target, *, unit, horizons=None, step_seconds=0.1):
    """ADE/FDE on windows whose first future point is t+step_seconds.

    Horizon H uses steps 1..H/dt and FDE at exactly H. Non-grid horizons are
    unavailable rather than silently rounded to a different forecast horizon.
    """
    predicted, target = np.asarray(predicted, float), np.asarray(target, float)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 2:
        raise ValueError("Forecast inputs must share shape [windows, future_steps, 2]")
    if unit not in ("m", "px", "normalized_image"):
        raise ValueError("Forecast unit must be explicit")
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("Forecast time step must be finite and positive")
    valid = np.isfinite(predicted).all(axis=(1, 2)) & np.isfinite(target).all(axis=(1, 2))
    if not np.any(valid):
        return {"status": "unavailable", "reason": "No finite matched forecast windows", "unit": unit}
    error = np.linalg.norm(predicted[valid] - target[valid], axis=-1)
    result = {"status": "available", "ADE": float(error.mean()), "FDE": float(error[:, -1].mean()),
              "unit": unit, "evaluated_windows": int(valid.sum()), "nonfinite_windows": int((~valid).sum())}
    if horizons:
        result["horizons"] = {}
        for horizon in horizons:
            n = int(round(horizon / step_seconds))
            result["horizons"][str(horizon)] = ({"ADE": float(error[:, :n].mean()), "FDE": float(error[:, n - 1].mean())}
                                                   if 1 <= n <= error.shape[1] and np.isclose(n * step_seconds, horizon)
                                                   else {"status": "unavailable", "reason": "Requested horizon not on the supported future time grid"})
    return result


def tracking_metrics(predictions, targets, *, iou_threshold=0.5):
    """Global one-to-one identity assignment over geometrically matched frames.

    Inputs map ordered sample IDs -> lists of {bbox,track_id,class_id?}. Caller
    namespaces IDs by sequence before combining videos; strings (including JAAD
    suffixes) remain intact. ID switches compare the last matched prediction for
    each target across gaps. Matching uses class-aware gated Hungarian IoU.
    """
    from scipy.optimize import linear_sum_assignment
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    if any(t.get("track_id") is None for rows in targets.values() for t in rows if not t.get("ignore", False)):
        return {"status": "unavailable", "reason": "Ground-truth identity labels are missing or null"}
    if any(p.get("track_id") is None for rows in predictions.values() for p in rows):
        return {"status": "unavailable", "reason": "Predicted track identities are missing; detection-only outputs have no identity score"}
    gt_ids = sorted({str(t["track_id"]) for rows in targets.values() for t in rows if not t.get("ignore", False)})
    pred_ids = sorted({str(p["track_id"]) for rows in predictions.values() for p in rows})
    gt_index, pred_index = {t: i for i, t in enumerate(gt_ids)}, {t: i for i, t in enumerate(pred_ids)}
    cooccurrence = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    total_gt, total_pred, false_positive, missed, switches, matched, ignored_pred = 0, 0, 0, 0, 0, 0, 0
    prior_match = {}
    frame_evidence = []
    # Dict insertion order is frame order, not lexical string order (frame-10 vs frame-2).
    sample_ids = list(targets) + [s for s in predictions if s not in targets]
    for sample_id in sample_ids:
        gt = [t for t in targets.get(sample_id, []) if not t.get("ignore", False)]
        pred = list(predictions.get(sample_id, []))
        if len({str(t["track_id"]) for t in gt}) != len(gt) or len({str(p["track_id"]) for p in pred}) != len(pred):
            raise ValueError(f"Duplicate track identity within frame {sample_id}")
        ignored_boxes = [t["bbox"] for t in targets.get(sample_id, []) if t.get("ignore", False)]
        if ignored_boxes:
            kept = []
            for p in pred:
                valid_target_overlap = any(box_iou([p["bbox"]], [t["bbox"]])[0, 0] >= iou_threshold and
                    ("class_id" not in t or "class_id" not in p or int(t["class_id"]) == int(p["class_id"])) for t in gt)
                if not valid_target_overlap and _ignore_overlap(p["bbox"], ignored_boxes) >= iou_threshold:
                    ignored_pred += 1
                else:
                    kept.append(p)
            pred = kept
        total_gt += len(gt); total_pred += len(pred)
        pairs = []
        if gt and pred:
            overlap = box_iou([t["bbox"] for t in gt], [p["bbox"] for p in pred])
            valid = overlap >= iou_threshold
            for i, t in enumerate(gt):
                for j, p in enumerate(pred):
                    if "class_id" in t and "class_id" in p and int(t["class_id"]) != int(p["class_id"]):
                        valid[i, j] = False
            row, col = linear_sum_assignment(np.where(valid, 1 - overlap, 1e6))
            pairs = [(int(i), int(j)) for i, j in zip(row, col) if valid[i, j]]
            # Identity scores use potential GT/pred identity overlap across all
            # frames, independently of each frame's best geometric assignment.
            for i, j in zip(*np.nonzero(valid)):
                cooccurrence[gt_index[str(gt[i]["track_id"])], pred_index[str(pred[j]["track_id"])]] += 1
        false_positive += len(pred) - len(pairs)
        missed += len(gt) - len(pairs)
        matched += len(pairs)
        identities = []
        for i, j in pairs:
            gt_id, pred_id = str(gt[i]["track_id"]), str(pred[j]["track_id"])
            if gt_id in prior_match and prior_match[gt_id] != pred_id:
                switches += 1
            prior_match[gt_id] = pred_id
            identities.append({"ground_truth_id": gt_id, "predicted_id": pred_id})
        frame_evidence.append({"sample_id": sample_id, "matches": identities,
                               "false_positive": len(pred) - len(pairs), "missed": len(gt) - len(pairs)})
    if total_gt == 0:
        return {"status": "unavailable", "reason": "No labeled ground-truth identities",
                "predicted_detections": total_pred, "false_positive": false_positive}
    if gt_ids and pred_ids:
        row, col = linear_sum_assignment(-cooccurrence)
        idtp = int(cooccurrence[row, col].sum())
        assignment = [{"ground_truth_id": gt_ids[i], "predicted_id": pred_ids[j], "matched_frames": int(cooccurrence[i, j])}
                      for i, j in zip(row, col) if cooccurrence[i, j] > 0]
    else:
        idtp, assignment = 0, []
    idfn, idfp = total_gt - idtp, total_pred - idtp
    return {"status": "available", "IDF1": float(2 * idtp / (2 * idtp + idfn + idfp)),
            "IDTP": idtp, "IDFN": idfn, "IDFP": idfp, "ID_switch": switches,
            "matched_detections": matched, "missed_detections": missed, "false_positive": false_positive,
            "ground_truth_detections": total_gt, "predicted_detections": total_pred,
            "ignored_predictions": ignored_pred,
            "identity_assignment": assignment, "frame_evidence": frame_evidence,
            "definition": "Class-aware IoU Hungarian matching; global identity assignment; switches compare last match across gaps"}
