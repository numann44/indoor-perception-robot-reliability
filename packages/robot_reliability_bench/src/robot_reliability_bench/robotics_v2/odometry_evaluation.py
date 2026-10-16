"""Independent OpenLORIS pose evaluator: labels never enter the frame source."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import interpolate_pose, resolve_static_transform
from .openloris import stamp_ns, message_transform, digest
from .provenance import context, finish


def first_pose_errors(observed, reference):
    """Rigid first-pose alignment; no scale or trajectory least-squares fit."""
    observed = np.asarray(observed, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if observed.shape != reference.shape or observed.ndim != 3 or observed.shape[1:] != (4, 4) or not len(observed):
        raise ValueError('Paired nonempty 4x4 pose arrays required')
    alignment = reference[0] @ np.linalg.inv(observed[0])
    aligned = alignment @ observed
    position = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    relative_rotation = np.swapaxes(reference[:, :3, :3], 1, 2) @ aligned[:, :3, :3]
    angle = Rotation.from_matrix(relative_rotation).magnitude()
    return alignment, aligned, position, np.degrees(angle)


def evaluate_odometry(bag, frame_refs, inventory, output):
    """Read /gt only here, with known mounting TF and separate sensor refs."""
    from rosbags.highlevel import AnyReader
    bag, frame_refs, inventory, output = map(Path, (bag, frame_refs, inventory, output))
    output.mkdir(parents=True, exist_ok=True)
    provenance = context({'evaluator': 'OpenLORIS recorded odometry vs separate /gt',
        'alignment': 'first paired SE3 pose; no scale fit', 'maximum_gt_interpolation_gap_ms': 100,
        'frame_refs_sha256': digest(frame_refs), 'bag_sha256': digest(bag)})
    inv = json.loads(inventory.read_text())
    edges = [(e['parent'], e['child'], np.asarray(e['T_parent_child'])) for e in inv['extrinsic_edges'] if e['static_topic']]
    mount = resolve_static_transform(edges, 'base_link', 'd400_color')
    if mount is None:
        raise ValueError('Independent evaluator requires known base_link -> d400_color mounting transform')
    stamps, transforms = [], []
    with AnyReader([bag]) as reader:
        conns = [c for c in reader.connections if c.topic == '/gt']
        if not conns:
            raise ValueError('Independent /gt labels unavailable')
        for conn, _, raw in reader.messages(connections=conns):
            message = reader.deserialize(raw, conn.msgtype)
            for tr in message.transforms:
                if tr.header.frame_id.lstrip('/') != 'gt_map' or tr.child_frame_id.lstrip('/') != 'base_link':
                    continue
                stamp = stamp_ns(tr.header)
                if not stamps or stamp > stamps[-1]:
                    stamps.append(stamp)
                    transforms.append(message_transform(tr.transform) @ mount)
    rows, observed, reference = [], [], []
    unavailable = {'missing_recorded_odometry': 0, 'missing_gt_pair': 0}
    for line in frame_refs.read_text().splitlines():
        row = json.loads(line)
        if row.get('T_world_camera') is None:
            unavailable['missing_recorded_odometry'] += 1
            continue
        ref = interpolate_pose(stamps, transforms, row['timestamp_ns'])
        if ref is None:
            unavailable['missing_gt_pair'] += 1
            continue
        rows.append({'sample_id': row['sample_id'], 'timestamp_ns': row['timestamp_ns']})
        observed.append(row['T_world_camera'])
        reference.append(ref)
    if not rows:
        raise ValueError('No paired recorded-odometry and independent-GT poses')
    alignment, aligned, error, rotation = first_pose_errors(observed, reference)
    for i, row in enumerate(rows):
        row.update({'translation_error_m': float(error[i]), 'rotation_error_deg': float(rotation[i]),
                    'recorded_odometry_aligned_camera_xyz': aligned[i, :3, 3].tolist(),
                    'independent_gt_camera_xyz': np.asarray(reference)[i, :3, 3].tolist()})
    trace = output / 'paired-pose-errors.jsonl'
    trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = {'provenance': finish(provenance), 'status': 'verified', 'pairs': len(rows),
        'unavailable': unavailable, 'gt_samples': len(stamps),
        'sensor_clock_duration_s': (rows[-1]['timestamp_ns'] - rows[0]['timestamp_ns']) / 1e9,
        'translation_rmse_m': float(np.sqrt(np.mean(error**2))), 'translation_median_m': float(np.median(error)),
        'translation_p95_m': float(np.percentile(error, 95)), 'translation_final_m': float(error[-1]),
        'rotation_rmse_deg': float(np.sqrt(np.mean(rotation**2))), 'rotation_p95_deg': float(np.percentile(rotation, 95)),
        'first_pose_alignment_T_gt_odom': alignment.tolist(), 'scale_fitted': False,
        'mount_T_base_camera': mount.tolist(), 'pose_labels': 'gt_map -> base_link from /gt, composed with static base_link -> d400_color mounting transform',
        'primary_mapping_pose': 'recorded_wheel_odometry', 'GT_in_sensor_source_or_inference': False,
        'scope': 'Independent recorded-pose comparison after first-pose rigid alignment; not SLAM or occupancy-map accuracy',
        'trace': str(trace)}
    (output / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
