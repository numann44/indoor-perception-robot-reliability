"""Export an odometry-assisted hard-label map from saved, sensor-only replay.

No model is loaded. This preserves measured predictions rather than regenerating
them. Full online softmax sums are not available in the recorded IPC payload, so
this artifact uses class votes and explicitly makes no posterior-parity claim.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import io
import json
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
import uuid
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inside(reference, parent=ROOT):
    path = Path(reference)
    path = (path if path.is_absolute() else parent / path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Missing or out-of-project replay reference')
    return path


def recorded_mask(payload, shape):
    if set(payload) != {'shape', 'dtype', 'zlib_base64'}:
        raise ValueError('Unexpected recorded semantic fields')
    if payload['shape'] != list(shape) or payload['dtype'] != 'uint8':
        raise ValueError('Recorded mask must use original sensor coordinates')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(payload['zlib_base64'], validate=True),
                             int(np.prod(shape)) + 1)
    if len(raw) != int(np.prod(shape)) or not decoder.eof or decoder.unused_data:
        raise ValueError('Invalid or oversized recorded semantic mask')
    mask = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    if not np.isin(mask, np.r_[np.arange(40), 255]).all():
        raise ValueError('Unexpected NYU40 class in recorded prediction')
    return mask


def export(root, chapter, output):
    from indoor_perception_lab.geometry import SpatialMap
    from robot_reliability_bench.robotics_v2.openloris import frame_from_reference

    started = time.time()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-map-' + uuid.uuid4().hex[:8]
    root, chapter, output = Path(root).resolve(), Path(chapter).resolve(), Path(output).resolve()
    if root != ROOT or not chapter.is_relative_to(ROOT) or not output.is_relative_to(ROOT / 'reports/v2'):
        raise ValueError('Map export must remain in the existing portfolio')
    if output.exists():
        raise FileExistsError('Preserve prior map evidence; use a fresh output directory')
    protocol = json.loads((ROOT / 'configs/v2/protocol.json').read_text())
    project_bytes = sum(p.stat().st_size for p in ROOT.rglob('*')
                        if p.is_file() and not p.is_symlink())
    reserve_bytes = 50_000_000
    if (project_bytes + reserve_bytes >= protocol['project_limit_bytes']
            or shutil.disk_usage(ROOT).free - reserve_bytes < protocol['minimum_free_bytes']):
        raise RuntimeError('Map reconstruction cannot fit the approved disk budget')
    manifest_path = inside('chapter-manifest.json', chapter)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'verified' or not manifest.get('all_source_frames_processed'):
        raise ValueError('A completed model replay is required')
    cfg = manifest['provenance']['config']
    if cfg.get('source') != 'openloris':
        raise ValueError('The reconstruction uses only the recorded OpenLORIS stream')
    model_refs = {}
    for task, field in [('detection', 'model_pack'), ('semantic', 'semantic_pack')]:
        lock_path = inside(f'models/v2/selection-{task}.json')
        lock = json.loads(lock_path.read_text())['selected']
        pack = inside(lock['pack'] + '/manifest.json').parent
        if Path(cfg[field]).resolve() != pack:
            candidate = (ROOT / cfg[field]).resolve()
            if candidate != pack:
                raise ValueError('Replay differs from the frozen primary model')
        if digest(pack / 'manifest.json') != lock['manifest_sha256'] or digest(pack / 'weights.pt') != lock['weights_sha256']:
            raise ValueError('Selected artifact changed after the replay lock')
        model_refs[task] = {'lock_sha256': digest(lock_path), **lock}
    refs_path = inside('reports/v2/robotics/openloris-full-10hz/sensor-frames.jsonl')
    processed_path = inside('processed-frames.jsonl', chapter)
    refs = [json.loads(line) for line in refs_path.read_text().splitlines()]
    processed = [json.loads(line) for line in processed_path.read_text().splitlines()]
    if len(processed) != len(refs) or len(processed) != manifest['processed_sensor_frames']:
        raise ValueError('Full selected recorded stream must be reconstructed')
    if len({r['sample_id'] for r in refs}) != len(refs):
        raise ValueError('Duplicate recorded sample identity')
    output.mkdir(parents=True)
    source_paths = [Path(__file__).resolve(), ROOT / 'packages/indoor_perception_lab/src/indoor_perception_lab/geometry.py',
                    ROOT / 'packages/robot_model_sdk/src/robot_model_sdk/contracts.py',
                    ROOT / 'packages/robot_reliability_bench/src/robot_reliability_bench/robotics_v2/openloris.py']
    source = {str(p.relative_to(ROOT)): p.read_bytes() for p in source_paths}
    source_hashes = {p: hashlib.sha256(b).hexdigest() for p, b in source.items()}
    archive_path = output / 'source-before-reconstruction.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, data in source.items():
            entry = tarfile.TarInfo(name); entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    inputs = {str(p.relative_to(ROOT)): digest(p) for p in [manifest_path, refs_path, processed_path]}
    geometric, semantic = SpatialMap(), SpatialMap()
    updates = []
    for sensor, row in zip(refs, processed, strict=True):
        if (sensor['sample_id'], sensor['timestamp_ns']) != (row['sample_id'], row['timestamp_ns']):
            raise ValueError('Sensor/result identity or ordering mismatch')
        if (row.get('source_rgb_ref'), row.get('source_depth_ref')) != (sensor.get('rgb_path'), sensor.get('depth_path')):
            raise ValueError('Replay result references a different source image/depth pair')
        result_path = inside(row['result_ref'], chapter)
        actual = digest(result_path)
        if actual != row['result_sha256']:
            raise ValueError('Saved model result changed')
        inputs[str(result_path.relative_to(ROOT))] = actual
        payload = json.loads(result_path.read_text())
        result = payload['result']
        if (result['sample_id'], result['timestamp_ns']) != (sensor['sample_id'], sensor['timestamp_ns']):
            raise ValueError('Payload identity differs from the original sensor')
        for name in ['rgb_path', 'depth_path']:
            if sensor.get(name):
                path = inside(sensor[name]); inputs[str(path.relative_to(ROOT))] = digest(path)
        frame = frame_from_reference(sensor)
        mask = recorded_mask(payload['semantic'], frame.rgb.shape[:2])
        boxes = [d['bbox'] for d in result['detections'] if d['class_id'] == 1]
        geo_update = geometric.update(frame, dynamic_boxes=boxes)
        sem_update = semantic.update(frame, semantic=mask, dynamic_boxes=boxes)
        native = result.get('map', {})
        equal_count = (sem_update.get('observed_surface_voxels') == native.get('observed_surface_voxels')) if sem_update['updated'] else (native.get('updated') is False)
        updates.append({'sample_id':frame.sample_id, 'timestamp_ns':frame.timestamp_ns,
                        'geometric':geo_update, 'hard_label_semantic':sem_update,
                        'native_map_update':native, 'native_observed_count_matches':equal_count})
    artifacts = {}
    for label, spatial in [('geometric', geometric), ('hard-label-semantic', semantic)]:
        cells = sorted(spatial.cells.items())
        classes = [max(v['votes'], key=v['votes'].get) if v['votes'] else 255 for _, v in cells]
        votes = [np.asarray(list(value['votes'].values()), dtype=np.float64) for _, value in cells]
        vote_support = np.asarray([value.sum() for value in votes], dtype=np.float64)
        vote_available = vote_support > 0
        dominance = np.full(len(cells), np.nan)
        entropy = np.full(len(cells), np.nan)
        for i in np.flatnonzero(vote_available):
            probabilities = votes[i] / vote_support[i]
            dominance[i] = probabilities.max()
            positive = probabilities[probabilities > 0]
            entropy[i] = -np.sum(positive * np.log(positive))
        path = output / (label + '-voxels.npz')
        np.savez_compressed(path, index=np.asarray([k for k, _ in cells], np.int32).reshape(-1, 3),
                            semantic_class=np.asarray(classes, np.uint8),
                            observations=np.asarray([v['observations'] for _, v in cells], np.int64),
                            semantic_vote_support=vote_support,
                            semantic_vote_available=vote_available,
                            dominant_vote_fraction=dominance,
                            class_vote_entropy_nats=entropy,
                            last_timestamp_ns=np.asarray([v['last_timestamp_ns'] for _, v in cells], np.int64),
                            voxel_size_m=np.array(.1))
        artifacts[label] = {'path':str(path.relative_to(ROOT)), 'sha256':digest(path), 'cells':len(cells)}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, (label, spatial) in zip(axes, [('geometric', geometric), ('hard-label-semantic', semantic)], strict=True):
        cells = sorted(spatial.cells.items()); xyz = (np.asarray([k for k, _ in cells]).reshape(-1, 3) + .5) * .1
        colors = [max(v['votes'], key=v['votes'].get) if v['votes'] else 255 for _, v in cells]
        if len(xyz):
            if label == 'geometric':
                points = ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], s=2,
                                    cmap='turbo', rasterized=True)
                fig.colorbar(points, ax=ax, label='World Z (m)', shrink=.7)
            else:
                known = np.asarray(colors) != 255
                ax.scatter(xyz[~known, 0], xyz[~known, 1], c='#9ca3af', s=2, rasterized=True)
                points = ax.scatter(xyz[known, 0], xyz[known, 1], c=np.asarray(colors)[known],
                                    s=2, cmap='turbo', vmin=0, vmax=39, rasterized=True)
                fig.colorbar(points, ax=ax, label='NYU40 class index', shrink=.7)
        ax.set_title(label); ax.set_xlabel('Wheel-odometry world X (m)'); ax.set_ylabel('World Y (m)'); ax.set_aspect('equal'); ax.grid(alpha=.15)
    fig.suptitle('Observed surface projection · ground frame unverified · unobserved cells remain unknown')
    panel = output / 'observed-surface-projection.png'; fig.savefig(panel, dpi=160); plt.close(fig)
    artifacts['projection'] = {'path':str(panel.relative_to(ROOT)), 'sha256':digest(panel)}
    if any(digest(ROOT / p) != h for p, h in inputs.items()) or any(digest(ROOT / p) != h for p, h in source_hashes.items()):
        raise ValueError('Inputs/source changed during reconstruction')
    if shutil.disk_usage(ROOT).free < 15_000_000_000:
        raise RuntimeError('Minimum free disk acceptance failed')
    config = {'chapter':str(chapter.relative_to(ROOT)), 'voxel_size_m':.1, 'max_voxels':100000,
              'projection_stride':6, 'maximum_depth_z_m':8, 'semantic_dynamic_classes':[30],
              'native_probability_reconstruction':False}
    import psutil
    report = {'status':'verified', 'run_id':run_id,
              'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'started_unix':started, 'finished_unix':time.time(), 'elapsed_seconds':time.time()-started,
              'command':[sys.executable,*sys.argv], 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'hardware':{'platform':platform.platform(),'machine':platform.machine(),'cpu_logical_cores':psutil.cpu_count(),
                          'RAM_total_bytes':psutil.virtual_memory().total}, 'python':sys.version,
              'config':config, 'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
              'seed':None, 'randomness_used':False, 'numpy_version':np.__version__,
              'mapping_role':'recorded-wheel-odometry-assisted saved-prediction reconstruction; no SLAM',
              'semantic_role':'class votes from saved argmax masks; full online softmax state was not serialized',
              'uncertainty_assumptions':'Dominant-vote fraction and entropy describe repeated projected hard labels, not calibrated posteriors or confidence intervals. Pixels/frames are correlated and pose/calibration uncertainty is not propagated. Unavailable vote statistics are NaN and semantic_vote_available is false.',
              'native_probability_parity_claim':False, 'model_inference_rerun':False, 'GT_or_future_inputs':False,
              'unknown_cells':'unknown, not free', 'dynamic_filter':'same predicted person boxes in both maps; semantic person class additionally excluded; undetected dynamics can remain',
              'frame_count':len(updates), 'source_models':model_refs, 'inputs_sha256':inputs,
              'source_sha256_before_and_after':source_hashes, 'source_archive':str(archive_path.relative_to(ROOT)),
              'source_archive_sha256':digest(archive_path), 'inputs_and_source_unchanged':True,
              'native_observed_counts_match_all_frames':all(x['native_observed_count_matches'] for x in updates),
              'artifacts':artifacts, 'updates':updates,
              'accuracy_metrics':{'status':'unavailable', 'reason':'No independent occupancy map labels; class-vote maps are not calibrated posteriors'}}
    (output / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chapter', default='reports/v2/robotics/media/openloris')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export(ROOT, ROOT / args.chapter, ROOT / args.output)
    print(json.dumps({'status':result['status'], 'frames':result['frame_count'], 'artifacts':result['artifacts']}))
