"""Read sensors independently of annotations."""

import json
from pathlib import Path
import time
import numpy as np
from PIL import Image


def _path(manifest, value):
    """Use manifest-relative paths or discovered portfolio-root-relative data paths."""
    path = Path(value)
    if path.is_absolute():
        return path
    local = manifest.parent / path
    if local.exists():
        return local
    for parent in manifest.parents:
        if (parent / "PROJECT_ROOT").is_file():
            candidate = parent / path
            if candidate.exists() or path.parts[0] in ("data", "cache", "models"):
                return candidate
    return local


def load_frames(manifest=None, *, count=120, split=None):
    """Return sensor frames plus measured decode time and source metadata.

    Manifests may be JSON lists, {samples: [...]}, or JSONL records. Target paths
    are intentionally not opened: the inference process has no annotation access.
    """
    from robot_model_sdk import Calibration, SensorFrameV1, fixture_frames
    started = time.perf_counter_ns()
    if manifest is None:
        frames = list(fixture_frames(n=count, fps=10))
        return frames, (time.perf_counter_ns() - started) / 1e6, {"kind": "synthetic_fixture", "unique_samples": len(frames)}
    manifest = Path(manifest).resolve()
    if manifest.suffix == ".jsonl":
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    else:
        value = json.loads(manifest.read_text())
        records = value.get("samples", []) if isinstance(value, dict) else value
    if split is not None:
        accepted = {"val", "validation"} if split in ("val", "validation") else {split}
        records = [row for row in records if row.get("split") in accepted]
    frames, skipped_depth, synthesized_timestamps = [], 0, 0
    for i, row in enumerate(records[:count]):
        rgb_path = row.get("rgb_path", row.get("rgb"))
        if not rgb_path:
            raise ValueError(f"Sample {i} has no rgb_path")
        with Image.open(_path(manifest, rgb_path)) as im:
            rgb = np.array(im.convert("RGB"))
        depth_path = row.get("depth_path", row.get("depth"))
        depth = None
        calib_value = row.get("calibration")
        if not calib_value and row.get("calibration_path"):
            value = json.loads(_path(manifest, row["calibration_path"]).read_text())
            keys = ("fx", "fy", "cx", "cy", "width", "height", "depth_scale", "depth_convention")
            calib_value = {key: value[key] for key in keys if key in value}
        calibration = Calibration(**calib_value) if calib_value else None
        if depth_path:
            path = _path(manifest, depth_path)
            if path.suffix == ".npy":
                depth = np.load(path, allow_pickle=False)
            elif path.suffix in (".hdf5", ".h5"):
                # Hypersim raw radial depth is not aligned/resized/calibrated to
                # the processed RGB. Never pretend it is registered sensor-Z.
                skipped_depth += 1
            else:
                with Image.open(path) as im:
                    depth = np.array(im)
        pose = row.get("T_world_camera")
        if pose is None and row.get("pose_xyz_xyzw") is not None:
            from scipy.spatial.transform import Rotation
            values = np.asarray(row["pose_xyz_xyzw"], dtype=float)
            if values.shape != (7,):
                raise ValueError("pose_xyz_xyzw must contain translation and xyzw quaternion")
            pose = np.eye(4); pose[:3, 3] = values[:3]; pose[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
        timestamp_ns = row.get("timestamp_ns")
        if timestamp_ns is None:
            timestamp = row.get("timestamp")
            if timestamp is None:
                timestamp = row.get("frame_index", i) / 10
                synthesized_timestamps += 1
            timestamp_ns = int(round(float(timestamp) * 1e9))
        frames.append(SensorFrameV1(sample_id=str(row.get("sample_id", i)),
                                   timestamp_ns=int(timestamp_ns),
                                   rgb=rgb, depth=depth, calibration=calibration,
                                   T_world_camera=np.array(pose) if pose is not None else None,
                                   pose_source=row.get("pose_source")))
    if not frames:
        raise ValueError("Input manifest contains no decodable sensor frames")
    return frames, (time.perf_counter_ns() - started) / 1e6, {
        "kind": "sensor_manifest", "manifest": str(manifest), "unique_samples": len(frames), "split": split,
        "skipped_unregistered_hdf5_depth": skipped_depth, "synthesized_timestamps": synthesized_timestamps,
        "annotation_access": "None; RGB, depth, calibration and optional camera pose only"}
