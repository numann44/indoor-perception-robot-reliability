#!/usr/bin/env python3
"""Bounded sensor loading for the frozen full-heldout corruption evaluator."""
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'mot17': ('data/v2/mot17/index.jsonl', '421abf943f174834cbf8874d339ccb1778ad33e1a3ec26b280612c9a67222bd1', 'outside_holdout', 900),
    'nyu': ('data/v2/nyu/index.jsonl', '32efaf080b924a8e249a3422492ebf5079b768c3cc667b70ba0037b133a1d4ab', 'test', 654),
}
PACKS = {
    'mot17': ('detection-D3-seed42', 'fecb506928e1c42eeb81f911bcec8e2c25a1884dc134fe5b756fe01b66b2ff4a', 'detection', 'native', 'mot17'),
    'semantic-native': ('semantic-S3-seed42', '7c25472c5d56e0d855d3abc884cdff9fb5c902822e2474a7a834f6d520893b11', 'semantic', 'native', 'nyu'),
    'semantic-onnx-fp32': ('semantic-onnx-fp32', '4d3869f6fc766fd1b3f7193866acba1e5d18f4764bd7b53242897429edb690d8', 'semantic', 'onnx', 'nyu'),
    'semantic-onnx-int8': ('semantic-onnx-int8', '1610729c63726508e31263356150761ff68adbc0f21e33f77852099e7b2ec05a', 'semantic', 'onnx', 'nyu'),
}
LOCKS = {
    'models/v2/selection-detection.json': '3afc5aa89e4ed8172bce683fac0388f38f286a0abfba88b987efda226cc158e5',
    'models/v2/selection-semantic.json': 'd72b9251a51829ee082bb6139597045ce26fda65bdba2d4167ed16ab4804723b',
    'reports/v2/perception-heldout-complete.json': '8afa60178e8771ba7ff0c57d1ca1ddacc23204f4f461ceba530cddc197ba3259',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_bindings():
    for path, expected in LOCKS.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen model-selection seal changed: ' + path)
    for path, expected, split, count in SOURCES.values():
        if sha(ROOT / path) != expected:
            raise ValueError('Frozen original sensor index changed: ' + path)
    seal = json.loads((ROOT / 'reports/v2/perception-heldout-complete.json').read_text())
    if seal['status'] != 'verified' or seal['seed_selection_on_test']:
        raise ValueError('Heldout selection seal is not admitted')
    for family in ('detection', 'semantic'):
        selection = json.loads((ROOT / f'models/v2/selection-{family}.json').read_text())
        if selection['test_used_for_selection'] or selection['selection_split'] != 'val':
            raise ValueError('Heldout data used for model selection')
        expected = '44f17da2cbc5972eec95383c1113c5005635c8670d9f5c9f50dfe3ae92acbe6c' if family == 'detection' else '64c9c504c713eab994719f23d63d4adcba86e4a6c67da3a7966bf58b721ac669'
        if selection['selected']['weights_sha256'] != expected:
            raise ValueError('Primary selected weights changed')

@dataclass(frozen=True)
class PixelReference:
    path: Path
    kind: str
    checksum_kind: str | None = None
    checksum: str | None = None

@lru_cache(maxsize=2)
def pixels(ref):
    import numpy as np
    from PIL import Image
    if ref.checksum_kind == 'file' and sha(ref.path) != ref.checksum:
        raise ValueError('Sensor file bytes changed: ' + str(ref.path))
    if ref.kind == 'rgb':
        with Image.open(ref.path) as image:
            value = np.array(image.convert('RGB'))
    elif ref.path.suffix == '.npy':
        value = np.load(ref.path, allow_pickle=False)
    else:
        with Image.open(ref.path) as image:
            value = np.array(image)
    if ref.checksum_kind == 'pixels' and hashlib.sha256(value.tobytes()).hexdigest() != ref.checksum:
        raise ValueError('Decoded sensor pixels changed: ' + str(ref.path))
    return value

def frame_type():
    from robot_model_sdk import SensorFrameV1
    class LazySensorFrame(SensorFrameV1):
        # Dataclass fields and validation remain inherited. replace(frame, ...)
        # constructs this class with actual current arrays, and clean inference
        # receives the exact original pixels through the normal RGB attribute.
        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            return pixels(value) if name in {'rgb', 'depth'} and isinstance(value, PixelReference) else value
    return LazySensorFrame

def load_bounded(manifest=None, *, count=120, split=None):
    import numpy as np
    from robot_model_sdk import Calibration
    from robot_reliability_bench.inputs import _path
    started = time.perf_counter_ns()
    manifest = Path(manifest).resolve()
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if split is not None:
        accepted = {'val', 'validation'} if split in ('val', 'validation') else {split}
        records = [r for r in records if r.get('split') in accepted]
    cls = frame_type()
    frames, skipped, synthesized = [], 0, 0
    for i, row in enumerate(records[:count]):
        name = row.get('rgb_path', row.get('rgb'))
        if not name:
            raise ValueError('Missing RGB sensor path')
        checksum_kind = 'pixels' if row.get('rgb_decoded_sha256') else 'file' if row.get('rgb_sha256') else None
        rgb = PixelReference(_path(manifest, name), 'rgb', checksum_kind, row.get('rgb_decoded_sha256') or row.get('rgb_sha256'))
        calibration = row.get('calibration')
        if not calibration and row.get('calibration_path'):
            value = json.loads(_path(manifest, row['calibration_path']).read_text())
            calibration = {k: value[k] for k in ('fx', 'fy', 'cx', 'cy', 'width', 'height', 'depth_scale', 'depth_convention') if k in value}
        calibration = Calibration(**calibration) if calibration else None
        depth = None
        if row.get('depth_path', row.get('depth')):
            path = _path(manifest, row.get('depth_path', row.get('depth')))
            if path.suffix in {'.hdf5', '.h5'}:
                skipped += 1
            else:
                depth = PixelReference(path, 'depth')
        pose = row.get('T_world_camera')
        if pose is None and row.get('pose_xyz_xyzw') is not None:
            from scipy.spatial.transform import Rotation
            values = np.asarray(row['pose_xyz_xyzw'], dtype=float)
            if values.shape != (7,):
                raise ValueError('Pose translation/quaternion shape changed')
            pose = np.eye(4)
            pose[:3, 3] = values[:3]
            pose[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
        stamp = row.get('timestamp_ns')
        if stamp is None:
            value = row.get('timestamp')
            if value is None:
                value = row.get('frame_index', i) / 10
                synthesized += 1
            stamp = int(round(float(value) * 1e9))
        frames.append(cls(str(row.get('sample_id', i)), int(stamp), rgb, depth, calibration,
                          np.array(pose) if pose is not None else None, row.get('pose_source')))
    if not frames:
        raise ValueError('No admitted sensor frames')
    return frames, (time.perf_counter_ns() - started) / 1e6, {
        'kind': 'sensor_manifest', 'manifest': str(manifest), 'unique_samples': len(frames), 'split': split,
        'skipped_unregistered_hdf5_depth': skipped, 'synthesized_timestamps': synthesized,
        'annotation_access': 'None; RGB/depth/calibration/pose sensor fields only',
        'pixel_storage': 'Path references retained; global LRU maximum two decoded sensor arrays plus current corruption/adapter buffers',
        'maximum_cached_sensor_arrays': 2}

def probe(output):
    import numpy as np
    from robot_model_sdk import fixture_frames
    from robot_reliability_bench.inputs import load_frames
    from robot_reliability_bench.corruptions import CONDITIONS, corrupt_frame
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for dataset, (name, expected, split, count) in SOURCES.items():
        rows = [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r['split'] == split]
        if len(rows) != count:
            raise ValueError('Full dataset count mismatch')
        chosen = [rows[i] for i in (0, count // 3, 2 * count // 3, count - 1)]
        index = output.parent / f'stream-loader-probe-{dataset}.jsonl'
        index.write_text(''.join(json.dumps(r) + '\n' for r in chosen))
        eager = load_frames(index, count=4, split=split)[0]
        lazy = load_bounded(index, count=4, split=split)[0]
        comparisons.append(compare_frames(eager, lazy, dataset, CONDITIONS, corrupt_frame))
    # Sensor-loss replace() is exercised with genuine registered fixture depth;
    # actual NYU/MOT manifests have no depth in this frame-local evaluation.
    fixture = list(fixture_frames(n=4, fps=10))
    from PIL import Image
    lazy_fixture = []
    cls = frame_type()
    for i, frame in enumerate(fixture):
        rgb_file, depth_file = output.parent / f'loader-control-{i}.png', output.parent / f'loader-control-{i}.npy'
        Image.fromarray(frame.rgb).save(rgb_file)
        np.save(depth_file, frame.depth, allow_pickle=False)
        lazy_fixture.append(cls(frame.sample_id, frame.timestamp_ns, PixelReference(rgb_file, 'rgb'),
                                PixelReference(depth_file, 'depth'), frame.calibration, frame.T_world_camera, frame.pose_source))
    comparisons.append(compare_frames(fixture, lazy_fixture, 'registered-depth-control', CONDITIONS, corrupt_frame))
    result = {'status': 'verified', 'helper_sha256': sha(__file__), 'source_indexes_sha256': {v[0]: v[1] for v in SOURCES.values()},
              'comparisons': comparisons, 'metric_model_code_changed': False, 'conditions': CONDITIONS,
              'source_rgb_and_frame_metadata_equal': True, 'corrupted_rgb_depth_timestamp_drop_equal': True}
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result

def compare_frames(eager, lazy, dataset, conditions, corrupt_frame):
    import numpy as np
    hashes = []
    def check(a, b):
        if a is None or b is None:
            if (a is None) != (b is None):
                raise ValueError('Frame-drop parity failed')
            return
        if a.to_json() != b.to_json() or not np.array_equal(a.rgb, b.rgb):
            raise ValueError('RGB/metadata loader parity failed')
        if (a.depth is None) != (b.depth is None) or (a.depth is not None and not np.array_equal(a.depth, b.depth, equal_nan=True)):
            raise ValueError('Depth loader parity failed')
    for a, b in zip(eager, lazy, strict=True):
        check(a, b)
        hashes.append({'sample_id': a.sample_id, 'decoded_rgb_sha256': hashlib.sha256(a.rgb.tobytes()).hexdigest(), 'metadata': a.to_json()})
    for condition in conditions:
        left, right = np.random.default_rng(42), np.random.default_rng(42)
        for a, b in zip(eager, lazy, strict=True):
            check(corrupt_frame(a, condition, left), corrupt_frame(b, condition, right))
    return {'dataset': dataset, 'frames': len(eager), 'condition_count': len(conditions), 'status': 'verified', 'source_frames': hashes}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=list(PACKS))
    parser.add_argument('--output', required=True)
    parser.add_argument('--probe-result')
    parser.add_argument('--probe-only', action='store_true')
    args = parser.parse_args()
    check_bindings()
    if importlib.util.find_spec('indoor_perception_lab') is not None:
        raise ValueError('Independent Bench requires Lab absent')
    if args.probe_only:
        result = probe(ROOT / args.output)
        print(json.dumps({'status': result['status'], 'output': args.output, 'helper_sha256': sha(__file__)}))
        return 0
    proof = json.loads((ROOT / args.probe_result).read_text())
    if proof['status'] != 'verified' or proof['helper_sha256'] != sha(__file__):
        raise ValueError('Exact current helper loader/corruption parity proof required')
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'reports') or output.exists():
        raise ValueError('Require a new project report output')
    pack, manifest_sha, task, backend, source = PACKS[args.profile]
    model = ROOT / 'models/v2' / pack
    if sha(model / 'manifest.json') != manifest_sha:
        raise ValueError('Admitted model manifest changed')
    from robot_model_sdk.manifest import validate_manifest
    from robot_model_sdk import load_adapter
    from robot_reliability_bench.robotics_v2 import quality_accuracy
    from robot_reliability_bench.robotics_v2.provenance import context
    import robot_model_sdk
    import robot_reliability_bench
    validate_manifest(model, backend=backend)
    installed = {}
    for module in (robot_model_sdk, robot_reliability_bench):
        base = Path(module.__file__).parent
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json'}:
                installed[f'{module.__name__}/{path.relative_to(base)}'] = sha(path)
    name, index_sha, split, expected_count = SOURCES[source]
    def project_context(config):
        value = context(config, root=ROOT)
        expected = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if not value.get('source_sha256') or value.get('git_commit') != expected:
            raise ValueError('Require current nonempty project source provenance')
        value['installed_runtime_source_files_sha256'] = installed
        value['bounded_sensor_loading'] = {'helper_sha256': sha(__file__), 'parity_proof': args.probe_result,
            'parity_proof_sha256': sha(ROOT / args.probe_result), 'source_index_sha256': index_sha, 'split_preserved': split,
            'expected_unique_frames': expected_count, 'no_index_rewrite_or_downsampling': True, 'Lab_installed': False,
            'maximum_cached_sensor_arrays': 2, 'model_corruption_metric_implementations_unchanged': True}
        return value
    class ProgressAdapter:
        def __init__(self, adapter):
            self.adapter, self.calls, self.start = adapter, 0, time.monotonic()
        def predict(self, frame):
            value = self.adapter.predict(frame)
            self.calls += 1
            if self.calls % 100 == 0:
                import psutil
                print(json.dumps({'event': 'actual_prediction_progress', 'predictions': self.calls,
                                  'elapsed_s': time.monotonic() - self.start, 'pid': os.getpid(),
                                  'rss_bytes': psutil.Process().memory_info().rss}), flush=True)
            return value
        def close(self):
            self.adapter.close()
    quality_accuracy.context = project_context
    quality_accuracy.load_frames = load_bounded
    quality_accuracy.load_adapter = lambda *a, **k: ProgressAdapter(load_adapter(*a, **k))
    result = quality_accuracy.evaluate(model, ROOT / name, output, task=task, backend=backend,
                                       device='cpu', split=split, test_unlocked=True, conditions=None)
    if result['images'] != expected_count:
        raise RuntimeError('Full immutable heldout count not evaluated')
    print(json.dumps({'status': result['status'], 'images': result['images'], 'output': args.output}), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
