#!/usr/bin/env python3
"""Decode source payloads and build sensor-only indexes plus separate supervision."""
from __future__ import annotations
import argparse
import bisect
import collections
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in rows))
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(p):
    return str(Path(p).relative_to(ROOT))


def config():
    return json.loads((ROOT / "configs/data_manifest.json").read_text())


def parse_jaad_annotation(path):
    tree = ET.parse(path)
    task = tree.find("./meta/task")
    width, height = int(task.findtext("original_size/width")), int(task.findtext("original_size/height"))
    frames = collections.defaultdict(list)
    bad = 0
    for track in tree.findall("./track"):
        label = track.get("label")
        if label not in {"pedestrian", "ped", "people"}:
            continue
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            attrs = {a.get("name"): a.text for a in box.findall("attribute")}
            bbox = [float(box.get(k)) for k in ["xtl", "ytl", "xbr", "ybr"]]
            if not all(math.isfinite(x) for x in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                bad += 1
                continue
            occlusion = {"none": 0, "part": 1, "full": 2}.get(attrs.get("occlusion"), int(box.get("occluded", 0)))
            frames[int(box.get("frame"))].append({
                "track_id": attrs["id"], "bbox_xyxy_original": bbox,
                "category_id": 1, "category_name": "person", "source_label": label,
                "occlusion": occlusion, "iscrowd": label == "people",
                "ignore": label == "people" or occlusion == 2,
                "action": attrs.get("action"), "annotation_source": "JAAD human annotation",
            })
    return {"width": width, "height": height, "frames": int(task.findtext("size")),
            "video_attributes": {a.tag: a.text for a in task.findall("video_attributes/*")},
            "invalid_boxes": bad, "targets": dict(frames)}


def fingerprints(rows, cv2, np):
    output = []
    for row in rows:
        img = cv2.imread(str(ROOT / row["rgb_path"]), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)[:8, :8].flatten()[1:]
        bits = dct > np.median(dct)
        h = sum(int(v) << i for i, v in enumerate(bits))
        output.append((row, h, cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32),
                       hashlib.sha256((ROOT / row["rgb_path"]).read_bytes()).hexdigest()))
    return output


def duplicate_audit(rows, cv2, np):
    """Image-only audit, never reads target boxes/masks or model predictions."""
    fps = fingerprints(rows, cv2, np)
    candidates = []
    for i, (a, ah, ai, ash) in enumerate(fps):
        for b, bh, bi, bsh in fps[i + 1:]:
            if a["split"] == b["split"]:
                continue
            distance = (ah ^ bh).bit_count()
            if distance <= 4 or ash == bsh:
                mae = float(np.abs(ai - bi).mean())
                if ash == bsh or mae <= 8.0:
                    candidates.append({"a": a["sample_id"], "b": b["sample_id"],
                                       "group_a": a["scene_id"], "group_b": b["scene_id"],
                                       "split_a": a["split"], "split_b": b["split"],
                                       "phash_hamming": distance, "gray_mae": mae,
                                       "exact_file_duplicate": ash == bsh})
    priority = {"train": 0, "val": 1, "test": 2}
    quarantine = set()
    for c in candidates:
        lower = "a" if priority[c["split_a"]] < priority[c["split_b"]] else "b"
        quarantine.add(c["group_" + lower])
    retained = [x for x in rows if x["scene_id"] not in quarantine]
    return retained, {"protocol": "cross-split pHash distance<=4 and gray MAE<=8; exact SHA also checked",
                      "uses_ground_truth": False, "candidates": candidates,
                      "quarantined_entire_scenes": sorted(quarantine),
                      "input_samples": len(rows), "retained_samples": len(retained),
                      "limit": "Image-level checks cannot prove original long-recording disjointness."}


def validate_jaad(conf, available_only=False):
    import cv2
    import numpy as np
    rows, reports, missing = [], [], []
    for split, videos in conf["splits"].items():
        for video in videos:
            source = ROOT / f"data/jaad/clips/{video}.mp4"
            ann = ROOT / f"data/jaad/annotations/{video}.xml"
            if not source.exists() or not ann.exists():
                if available_only:
                    missing.append(video)
                    continue
                raise RuntimeError(f"Missing JAAD payload: {video}")
            labels = parse_jaad_annotation(ann)
            cap = cv2.VideoCapture(str(source))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise RuntimeError(f"Cannot decode JAAD clip: {video}")
            scale = conf["decode_width"] / labels["width"]
            out_width, out_height = conf["decode_width"], round(labels["height"] * scale)
            targets_path = ROOT / f"data/jaad/targets/{video}.json"
            target_map = {}
            for frame, targets in labels["targets"].items():
                transformed = []
                for target in targets:
                    copy = dict(target)
                    box = np.array(copy["bbox_xyxy_original"]) * scale
                    box[[0, 2]] = np.clip(box[[0, 2]], 0, out_width)
                    box[[1, 3]] = np.clip(box[[1, 3]], 0, out_height)
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    copy["bbox_xyxy"] = box.tolist()
                    transformed.append(copy)
                target_map[str(frame)] = transformed
            dump(targets_path, {"schema_version": 1, "video_id": video, "split": split,
                                "fps": fps, "width": out_width, "height": out_height,
                                "original_width": labels["width"], "original_height": labels["height"],
                                "bbox_scale": scale, "frames": target_map})
            decoded = 0
            for frame_index in range(labels["frames"]):
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Truncated clip {video} at frame {frame_index}")
                decoded += 1
                if bgr.shape[:2] != (labels["height"], labels["width"]):
                    raise RuntimeError("Annotation/video dimension mismatch")
                if frame_index % conf["detection_stride"]:
                    continue
                bgr = cv2.resize(bgr, (out_width, out_height), interpolation=cv2.INTER_AREA)
                out = ROOT / f"data/jaad/frames/{video}/{frame_index:06d}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError("JPEG write failed")
                rows.append({"schema_version": 1, "sample_id": f"jaad/{video}/{frame_index:06d}",
                             "dataset": "jaad", "domain": "real", "split": split, "scene_id": video,
                             "sequence_id": video, "frame_index": frame_index, "timestamp": frame_index / fps,
                             "rgb_path": relative(out), "video_path": relative(source), "width": out_width,
                             "height": out_height, "original_width": labels["width"], "original_height": labels["height"],
                             "targets_path": relative(targets_path), "fps": fps,
                             "ground_truth_access": "training/evaluator only; inference loader must omit targets_path"})
            cap.release()
            reports.append({"video": video, "split": split, "decoded_frames": decoded, "fps": fps,
                            "invalid_boxes": labels["invalid_boxes"], "attributes": labels["video_attributes"]})
            print(f"JAAD decoded {split} {video}: {decoded} frames", flush=True)
    cache_bytes = sum(p.stat().st_size for p in (ROOT / "data/jaad/frames").rglob("*.jpg"))
    if cache_bytes > config()["budgets"]["decoded_jaad_cache_bytes"]:
        raise RuntimeError("JAAD decoded cache quota exceeded")
    retained, audit = duplicate_audit(rows, cv2, np)
    dump(ROOT / "reports/data/jaad_duplicate_audit.json", audit)
    split_hash = jsonl(ROOT / "data/jaad/index.jsonl", retained)
    jsonl(ROOT / f"reports/data/jaad_index_{split_hash[:12]}.jsonl", retained)
    dump(ROOT / "reports/data/jaad_validation.json", {"status": "validated_partial" if missing else "validated", "clips": reports,
         "samples": len(retained), "split_hash": split_hash, "decoded_cache_bytes": cache_bytes,
         "missing_clips": missing, "immutable_index_snapshot": f"reports/data/jaad_index_{split_hash[:12]}.jsonl",
         "duplicate_audit": "reports/data/jaad_duplicate_audit.json",
         "coordinate_space": "640x360 pixels; original boxes preserved in targets"})
    print(f"JAAD ready: {len(retained)} detection frames, split hash {split_hash}", flush=True)


def tonemap(rgb, np):
    """Official formula; RGB-only finite validity avoids privileged IDs/masks."""
    valid = np.all(np.isfinite(rgb), axis=-1)
    brightness = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    p90 = float(np.percentile(brightness[valid], 90)) if valid.any() else 0.0
    scale = 0.0 if p90 < 0.0001 else 0.8 ** 2.2 / p90
    output = np.clip(np.maximum(np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0) * scale, 0) ** (1 / 2.2), 0, 1)
    return (output * 255 + 0.5).astype(np.uint8), {"percentile": 90, "desired_brightness": 0.8,
           "gamma": 2.2, "scale": scale, "validity": "RGB finite pixels; no semantic/instance/render IDs"}


def validate_hypersim(conf):
    import cv2
    import h5py
    import numpy as np
    rows, reports = [], []
    classes = []
    with (ROOT / "data/hypersim/semantic_label_descs.csv").open() as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            source_id = int(row["semantic_id"])
            classes.append({"model_id": source_id - 1, "source_id": source_id,
                            "name": row["semantic_name"],
                            "rgb": [int(row["semantic_color_" + c]) for c in "rgb"]})
    if [x["source_id"] for x in classes] != list(range(1, 41)):
        raise ValueError("Official semantic ontology must contain NYU40 IDs1..40")
    dump(ROOT / "data/hypersim/classes.json", {"ignore_index": 255, "classes": classes,
          "source": "Apple Hypersim pinned semantic_label_descs.csv; source1..40 -> model0..39"})
    supports = {x: collections.Counter() for x in conf["splits"]}
    for split, scenes in conf["splits"].items():
        for scene in scenes:
            frames = conf["frame_exceptions"].get(scene, conf["frames_default"])
            for frame in frames:
                prefix = ROOT / "data/hypersim/raw" / scene / "images"
                color_path = prefix / f"scene_cam_00_final_hdf5/frame.{frame:04d}.color.hdf5"
                semantic_path = prefix / f"scene_cam_00_geometry_hdf5/frame.{frame:04d}.semantic.hdf5"
                depth_path = prefix / f"scene_cam_00_geometry_hdf5/frame.{frame:04d}.depth_meters.hdf5"
                with h5py.File(color_path, "r") as f:
                    color = f["dataset"][:].astype(np.float32)
                with h5py.File(semantic_path, "r") as f:
                    semantic = f["dataset"][:].astype(np.int16)
                with h5py.File(depth_path, "r") as f:
                    depth = f["dataset"][:].astype(np.float32)
                if color.ndim != 3 or color.shape[-1] != 3 or semantic.shape != color.shape[:2] or depth.shape != semantic.shape:
                    raise RuntimeError("Hypersim modality shape mismatch")
                observed = set(np.unique(semantic).tolist())
                if not observed <= {-1, *range(1, 41)}:
                    raise RuntimeError(f"Unexpected NYU40 IDs: {observed}")
                rgb, tone = tonemap(color, np)
                mask = np.where(semantic == -1, 255, semantic - 1).astype(np.uint8)
                width, height = conf["decode_width"], round(rgb.shape[0] * conf["decode_width"] / rgb.shape[1])
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                image = ROOT / f"data/hypersim/images/{scene}/frame.{frame:04d}.png"
                target = ROOT / f"data/hypersim/masks/{scene}/frame.{frame:04d}.png"
                image.parent.mkdir(parents=True, exist_ok=True)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) or not cv2.imwrite(str(target), mask):
                    raise RuntimeError("Hypersim PNG encoding failed")
                vals, counts = np.unique(mask, return_counts=True)
                supports[split].update({int(v): int(n) for v, n in zip(vals, counts)})
                rows.append({"schema_version": 1, "sample_id": f"hypersim/{scene}/cam_00/{frame:04d}",
                             "dataset": "hypersim", "domain": "synthetic", "split": split, "scene_id": scene,
                             "sequence_id": scene + "/cam_00", "camera_id": "cam_00", "frame_index": frame,
                             "timestamp": None, "rgb_path": relative(image), "targets_path": relative(target),
                             "mask_path": relative(target), "depth_path": relative(depth_path),
                             "depth_convention": conf["depth_convention"], "width": width, "height": height,
                             "classes": 40, "ignore_index": 255, "label_mapping": "NYU40 source1..40 -> model0..39",
                             "ground_truth_access": "training/evaluator only; inference loader must omit mask_path/targets_path"})
                reports.append({"sample_id": rows[-1]["sample_id"], "tone_mapping": tone,
                                "source_shape": list(color.shape), "semantic_original_ids": sorted(observed),
                                "finite_positive_depth_pixels": int((np.isfinite(depth) & (depth > 0)).sum())})
            print(f"Hypersim decoded {split} {scene}", flush=True)
    retained, audit = duplicate_audit(rows, cv2, np)
    dump(ROOT / "reports/data/hypersim_duplicate_audit.json", audit)
    split_hash = jsonl(ROOT / "data/hypersim/index.jsonl", retained)
    jsonl(ROOT / f"reports/data/hypersim_index_{split_hash[:12]}.jsonl", retained)
    dump(ROOT / "reports/data/hypersim_validation.json", {"status": "validated", "samples": len(retained),
        "split_hash": split_hash, "samples_metadata": reports,
        "pixel_support_by_split": {s: dict(c) for s, c in supports.items()},
        "training_class_ids": sorted(x for x in supports["train"] if x != 255),
        "depth_warning": "Euclidean ray range, not camera-Z; segmentation training does not fuse it with TUM",
        "asset_reuse_warning": "Official scene split still permits occasional 3D asset reuse."})
    print(f"Hypersim ready: {len(retained)} semantic images, split hash {split_hash}", flush=True)


def read_timestamp_file(path, columns=2):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            parts = line.split()
            if len(parts) < columns:
                raise ValueError(f"Malformed timestamp row: {path}")
            rows.append((float(parts[0]), parts[1:]))
    if any(b[0] <= a[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("Timestamp rows must increase strictly")
    return rows


def associate(rgb, depth, tolerance):
    """Global closest-pair greedy association with unique sensor frame use."""
    times = [x[0] for x in depth]
    candidates = []
    for i, (t, _) in enumerate(rgb):
        lo, hi = bisect.bisect_left(times, t - tolerance), bisect.bisect_right(times, t + tolerance)
        for j in range(lo, hi):
            candidates.append((abs(t - times[j]), i, j))
    used_rgb, used_depth, pairs = set(), set(), []
    for delta, i, j in sorted(candidates):
        if i not in used_rgb and j not in used_depth:
            used_rgb.add(i)
            used_depth.add(j)
            pairs.append((i, j, delta))
    return sorted(pairs)


def read_pose_file(path):
    """Discard every pose at an ambiguous duplicate timestamp; never average GT."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8 or not all(math.isfinite(float(v)) for v in parts):
            raise ValueError("Malformed/nonfinite provided camera pose")
        if sum(float(v)**2 for v in parts[4:]) < 1e-12:
            raise ValueError("Provided camera pose has zero quaternion")
        rows.append((float(parts[0]), parts[1:]))
    counts = collections.Counter(t for t, _ in rows)
    duplicated = sorted(t for t, n in counts.items() if n > 1)
    retained = sorted((t, p) for t, p in rows if counts[t] == 1)
    report = {"source_records": len(rows), "retained_records": len(retained),
              "duplicate_timestamps": duplicated,
              "discarded_ambiguous_records": len(rows)-len(retained),
              "policy": "discard all poses at repeated timestamps; sort remaining samples; interpolate only gaps<=0.1s",
              "backward_timestamp_steps": sum(b[0] < a[0] for a,b in zip(rows,rows[1:]))}
    return retained, report


def pose_at(timestamp, poses, np, max_gap=0.1):
    times = [x[0] for x in poses]
    j = bisect.bisect_left(times, timestamp)
    if j == 0 or j == len(poses):
        near = poses[0 if j == 0 else -1]
        if abs(near[0] - timestamp) > 0.02:
            return None
        return [float(x) for x in near[1]]
    a, b = poses[j - 1], poses[j]
    if b[0] - a[0] > max_gap:
        return None
    alpha = (timestamp - a[0]) / (b[0] - a[0])
    av, bv = np.asarray(a[1], dtype=np.float64), np.asarray(b[1], dtype=np.float64)
    q0, q1 = av[3:] / np.linalg.norm(av[3:]), bv[3:] / np.linalg.norm(bv[3:])
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        q /= np.linalg.norm(q)
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        q = (math.sin((1-alpha)*theta)*q0 + math.sin(alpha*theta)*q1) / math.sin(theta)
    return np.concatenate([av[:3] * (1-alpha) + bv[:3] * alpha, q]).tolist()


def validate_tum(conf, available_only=False):
    import cv2
    import numpy as np
    rows, sequences, missing = [], [], []
    calibration = {"schema_version": 1, "camera_id": "tum_freiburg3_rgb", **conf["intrinsics"],
                   "depth_scale": conf["depth_scale"], "depth_convention": conf["depth_convention"],
                   "coordinate_convention": "camera x right, y down, z forward; pose T_world_from_camera",
                   "pose_source": "dataset provided ground-truth pose; no SLAM claim",
                   "registration": "provided TUM depth registered to RGB; Freiburg3 images rectified",
                   "pose_covariance": None, "calibration_covariance": None}
    dump(ROOT / "data/tum/calibration.json", calibration)
    for sequence in conf["archives"]:
        base = ROOT / "data/tum/raw" / sequence
        marker = ROOT / f"data/tum/{sequence}.extraction_complete"
        if not marker.exists():
            if available_only:
                missing.append(sequence)
                continue
            raise RuntimeError(f"TUM archive not completely extracted: {sequence}")
        archive = ROOT / "cache/downloads" / (sequence + ".tgz")
        crc_report = ROOT / f"reports/data/tum_{sequence}_archive_crc.json"
        if not crc_report.exists():
            digest = hashlib.sha256()
            with archive.open("rb") as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != marker.read_text().strip():
                raise ValueError("TUM archive differs from extraction checksum")
            uncompressed = 0
            with gzip.open(archive, "rb") as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    uncompressed += len(block)
            dump(crc_report, {"status": "gzip_footer_crc_length_verified", "sha256": digest.hexdigest(),
                              "archive_bytes": archive.stat().st_size, "uncompressed_stream_bytes": uncompressed})
        rgb = read_timestamp_file(base / "rgb.txt")
        depth = read_timestamp_file(base / "depth.txt")
        poses, pose_audit = read_pose_file(base / "groundtruth.txt")
        dump(ROOT / f"reports/data/tum_{sequence}_pose_timestamp_audit.json", pose_audit)
        pairs = associate(rgb, depth, conf["max_timestamp_delta"])
        seqrows = []
        for i, j, delta in pairs:
            rgb_path, depth_path = base / rgb[i][1][0], base / depth[j][1][0]
            image, dep = cv2.imread(str(rgb_path)), cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if image is None or dep is None or image.shape[:2] != (480, 640) or dep.shape != (480, 640) or dep.dtype != np.uint16:
                raise RuntimeError("Invalid TUM registered RGB/depth shape or depth dtype")
            pose = pose_at(rgb[i][0], poses, np)
            row = {"schema_version": 1, "sample_id": f"tum/{sequence}/{i:06d}", "dataset": "tum", "domain": "real",
                   "split": "replay", "scene_id": sequence, "sequence_id": sequence, "frame_index": i,
                   "timestamp": rgb[i][0], "rgb_timestamp": rgb[i][0], "depth_timestamp": depth[j][0],
                   "timestamp_delta": delta, "rgb_path": relative(rgb_path), "depth_path": relative(depth_path),
                   "calibration_path": "data/tum/calibration.json", "width": 640, "height": 480,
                   "depth_scale": conf["depth_scale"], "depth_convention": conf["depth_convention"],
                   "pose_source": "provided_ground_truth", "pose_xyz_xyzw": pose,
                   "pose_available": pose is not None, "pose_covariance": None,
                   "invalid_depth_fraction": float((dep == 0).mean()),
                   "targets_path": None, "person_track_semantic_ground_truth": "unavailable"}
            seqrows.append(row)
        jsonl(ROOT / "data/tum" / sequence / "index.jsonl", seqrows)
        rows.extend(seqrows)
        sequences.append({"sequence": sequence, "rgb_frames": len(rgb), "depth_frames": len(depth),
                          "associated_frames": len(pairs), "tolerance_seconds": conf["max_timestamp_delta"],
                          "maximum_matched_delta": max((x[2] for x in pairs), default=None),
                          "pose_frames": sum(x["pose_available"] for x in seqrows),
                          "pose_missing_frames": sum(not x["pose_available"] for x in seqrows),
                          "pose_timestamp_audit": pose_audit})
        print(f"TUM decoded {sequence}: {len(pairs)} associated RGB-D frames", flush=True)
    if not rows:
        raise RuntimeError("No completely extracted TUM sequence available")
    hash_ = jsonl(ROOT / "data/tum/index.jsonl", rows)
    jsonl(ROOT / f"reports/data/tum_index_{hash_[:12]}.jsonl", rows)
    dump(ROOT / "reports/data/tum_validation.json", {"status": "validated_partial" if missing else "validated", "sequences": sequences,
                                                   "missing_sequences": missing,
                                                   "index_hash": hash_, "samples": len(rows), "calibration": calibration})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=["jaad", "hypersim", "tum"], default=["jaad", "hypersim", "tum"])
    p.add_argument("--available-only", action="store_true", help="Publish explicitly partial JAAD/TUM indexes while verified payloads arrive")
    args = p.parse_args()
    conf = config()
    allgroups = {}
    for dataset in ["jaad", "hypersim"]:
        for split, groups in conf[dataset]["splits"].items():
            for g in groups:
                if g in allgroups:
                    raise ValueError(f"Duplicate scene/clip split: {g}")
                allgroups[g] = split
    for dataset in args.datasets:
        if dataset == "jaad":
            validate_jaad(conf[dataset], available_only=args.available_only)
        elif dataset == "tum":
            validate_tum(conf[dataset], available_only=args.available_only)
        else:
            validate_hypersim(conf[dataset])


if __name__ == "__main__":
    main()
