"""Actual model-pack inference through the bounded queue, without lab imports."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .inputs import load_frames
from .replay import replay_system
from .runner import model_identity, _configure_threads


def replay_model(model_spec, output_path, *, backend="native", device="cpu", mode="realtime",
                 samples=40, fixture_fps=100, input_manifest=None, split=None,
                 worker_delay_s=0.0, max_duration_s=60.0):
    """Record actual queued model outputs and exact sensor/model identities.

    The fixture's physical renderer uses the requested sensor frame rate. An
    RGB-trained semantic model receiving colored-block fixtures is OOD; this
    verifies inference integration only, not learned task accuracy.
    """
    from robot_model_sdk import fixture_frames, load_adapter
    if samples < 1 or fixture_fps <= 0:
        raise ValueError("samples and fixture FPS must be positive")
    _configure_threads(4, native_neural=(backend == "native" and str(model_spec) != "fixture"))
    if input_manifest:
        frames, io_ms, source = load_frames(input_manifest, count=samples, split=split)
    else:
        frames = list(fixture_frames(n=samples, fps=fixture_fps))
        io_ms = None
        source = {"kind": "synthetic_fixture", "fps": fixture_fps,
                  "scope": "Colored-block fixture tests integration, not learned semantic or person accuracy"}
    identity = model_identity(model_spec)
    adapter = load_adapter(model_spec, backend=backend, device=device)
    result = replay_system(frames, adapter, mode=mode, worker_delay_s=worker_delay_s,
                           max_duration_s=max_duration_s, queue_capacity=2)
    if not result["worker_alive_at_deadline"]:
        adapter.close()
    output = Path(output_path).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=output.parent, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=output.parent, text=True))
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    source_digest = sha256()
    for frame in frames:
        source_digest.update(frame.sample_id.encode()); source_digest.update(str(frame.timestamp_ns).encode())
        source_digest.update(frame.rgb.tobytes())
        if frame.depth is not None:
            source_digest.update(frame.depth.tobytes())
    run = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-replay-" + uuid4().hex[:6],
           "status": "verified" if result["complete"] and result["counters"]["processed"] > 0 else "incomplete",
           "model": identity, "backend": backend, "device": device, "mode": mode,
           "fixture_fps": fixture_fps if not input_manifest else None, "source": source,
           "source_rgb_depth_timestamps_sha256": source_digest.hexdigest(), "input_decode_ms": io_ms,
           "command": sys.argv, "git_commit": commit, "git_dirty": dirty,
           "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
           "sensor_targets_access": "None", "result": result}
    jsonl = output.with_suffix(".jsonl")
    with jsonl.open("w") as file:
        file.write(json.dumps({"type": "run", **{key: value for key, value in run.items() if key != "result"}}) + "\n")
        for row in result["results"]:
            file.write(json.dumps({"type": "consumer_output", "run_id": run["run_id"], **row}, allow_nan=False) + "\n")
    run["consumer_jsonl"] = str(jsonl)
    output.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    return run
