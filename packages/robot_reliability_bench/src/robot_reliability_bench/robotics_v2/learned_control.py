"""Frozen, additive learned-person stop rule; no simulator truth access.

The tracker threshold is bound from the separately locked canonical tracker
configuration before a model run. It is never selected on simulation outcomes.
"""
from __future__ import annotations
import numpy as np
from robot_model_sdk import ControlCommandV1

POLICY = {
    'policy_id': 'active-person-safety-v1',
    'threshold_source': 'canonicaltracking.high_score',
    'person_class_id': 1,
    'minimum_valid_depth_pixels': 9,
    'maximum_absolute_camera_x_m': .4,
    'stop_axial_camera_z_m': .8,
    'maximum_result_age_ms': 300.,
    'position_estimator': 'median of registered positive finite depth pixels in observed track bbox; camera X median from calibrated pixel backprojection',
    'geometry_policy': 'unchanged; learned rule may only replace a command with zero linear and angular velocity',
    'OOD_scope': 'controlled primitive capsule actors are not a validated human-image domain',
}


def learned_person_observation(frame, result, high_score):
    """Return a compact sensor-only observation from currently observed tracks.

    No predicted/unobserved tracker boxes are eligible. Exact sample identity
    prevents applying a result to a different frame's depth image.
    """
    if not 0 <= high_score <= 1:
        raise ValueError('Locked tracker high_score must be in [0,1]')
    if result.get('sample_id') != frame.sample_id or result.get('timestamp_ns') != frame.timestamp_ns:
        raise ValueError('Learned result and registered depth must belong to the same sensor frame')
    output = result.get('result', result)
    rows = []
    if frame.depth is not None and frame.calibration is not None:
        c = frame.calibration
        if c.depth_convention != 'z' or c.depth_scale != 1:
            raise ValueError('Learned constraint requires registered metric camera-Z depth')
        h, w = frame.depth.shape
        for track in output.get('tracks', []):
            if not track.get('observed', False) or track.get('class_id') != 1 or track.get('score', 0.) < high_score:
                continue
            box = np.asarray(track.get('bbox', []), dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            x1, y1 = np.floor(box[:2]).astype(int)
            x2, y2 = np.ceil(box[2:]).astype(int)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            depth = frame.depth[y1:y2, x1:x2]
            valid = np.isfinite(depth) & (depth > 0)
            if int(valid.sum()) < 9:
                continue
            u = np.broadcast_to(np.arange(x1, x2)[None, :], depth.shape)
            z = float(np.median(depth[valid]))
            x = float(np.median(((u - c.cx) * depth / c.fx)[valid]))
            rows.append({'track_id': str(track.get('id', '')), 'score': float(track['score']),
                         'valid_depth_pixels': int(valid.sum()), 'camera_x_m': x, 'camera_z_m': z,
                         'hazard': bool(abs(x) <= .4 and 0 < z < .8)})
    return {'sample_id': frame.sample_id, 'timestamp_ns': frame.timestamp_ns,
            'eligible_tracks': rows, 'hazard': any(row['hazard'] for row in rows),
            'depth_available': frame.depth is not None, 'high_score': float(high_score)}


def constrain_command(geometric_command, frame, observation, now_ns):
    """Apply only an additive stop; preserve every geometric command otherwise."""
    age = None if observation is None else (now_ns - observation['timestamp_ns']) / 1e6
    reason = None
    if frame.depth is None:
        reason = 'learned_constraint_missing_depth'
    elif age is None or not np.isfinite(age) or age < 0 or age > 300:
        reason = 'learned_constraint_stale_result'
    elif observation.get('hazard', False):
        reason = 'learned_person_safety_stop'
    if reason is None:
        return geometric_command, False
    command = ControlCommandV1(timestamp_ns=int(now_ns), linear_mps=0., angular_rps=0.,
        state='stopped', reason=reason, source_sample_id=frame.sample_id,
        observed_result_age_ms=age if age is not None and np.isfinite(age) and age >= 0 else None,
        safety_stop=True)
    changed = abs(geometric_command.linear_mps) > 1e-12 or abs(geometric_command.angular_rps) > 1e-12
    return command, bool(changed)
