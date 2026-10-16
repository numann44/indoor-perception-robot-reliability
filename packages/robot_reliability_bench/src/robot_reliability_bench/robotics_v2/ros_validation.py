"""Evaluate logged recorded sensor delivery without opening labels or truth."""
from __future__ import annotations
from collections import Counter


def validate_recorded_transport(references, publisher_events, subscriber_events):
    published=[e for e in publisher_events if e.get('event')=='published']
    expected={e['timestamp_ns']:e for e in published}
    groups={name:[e for e in subscriber_events if e.get('event')==name] for name in (
        'received_rgb','received_depth','received_calibration','received_source_metadata','received_odometry')}
    offsets=[];source_errors=[]
    for i,event in enumerate(published):
        meta=event.get('source_metadata',{})
        if i>=len(references):source_errors.append('Extra published source frame');continue
        row=references[i]
        if (meta.get('source_sequence')!=i or meta.get('sample_id')!=row['sample_id'] or
            meta.get('original_timestamp_ns')!=row['timestamp_ns']):source_errors.append(f'Published source identity differs at {i}')
        offsets.append(event['timestamp_ns']-row['timestamp_ns'])
    def exact_timestamps(name,selected):
        return Counter(e['timestamp_ns'] for e in groups[name])==Counter(e['timestamp_ns'] for e in selected)
    depths=[e for e in published if e.get('depth_published')]
    poses=[e for e in published if e.get('source_metadata',{}).get('pose_expected')]
    rgb_hash_errors=[];depth_hash_errors=[];metadata_errors=[]
    for event in groups['received_rgb']:
        row=expected.get(event['timestamp_ns'])
        if row is None or row.get('RGB_payload_sha256')!=event.get('payload_sha256'):rgb_hash_errors.append(event['timestamp_ns'])
    for event in groups['received_depth']:
        row=expected.get(event['timestamp_ns'])
        if row is None or row.get('depth_payload_sha256')!=event.get('payload_sha256'):depth_hash_errors.append(event['timestamp_ns'])
    for event in groups['received_source_metadata']:
        row=expected.get(event['timestamp_ns'])
        if row is None or row.get('source_metadata')!=event.get('source_metadata'):metadata_errors.append(event['timestamp_ns'])
    checks={
        'recorded_source_identity_exact':len(published)==len(references) and not source_errors,
        'source_clock_offset_constant':bool(offsets) and len(set(offsets))==1,
        'RGB_transport_all_selected_received':exact_timestamps('received_rgb',published),
        'depth_transport_all_expected_received':exact_timestamps('received_depth',depths),
        'calibration_transport_all_selected_received':exact_timestamps('received_calibration',published),
        'source_metadata_transport_all_selected_received':exact_timestamps('received_source_metadata',published),
        'odometry_transport_all_expected_received':exact_timestamps('received_odometry',poses),
        'RGB_payload_hashes_match_published':bool(groups['received_rgb']) and not rgb_hash_errors,
        'depth_payload_hashes_match_published':bool(groups['received_depth']) and not depth_hash_errors,
        'source_metadata_received_matches_published':bool(groups['received_source_metadata']) and not metadata_errors,
    }
    return {'checks':checks,'publication_count':len(published),'receipt_counts':{k:len(v) for k,v in groups.items()},
            'clock_offset_ns':offsets[0] if offsets and len(set(offsets))==1 else None,
            'source_identity_errors':source_errors,'RGB_payload_error_timestamps':rgb_hash_errors,
            'depth_payload_error_timestamps':depth_hash_errors,'source_metadata_error_timestamps':metadata_errors,
            'scope':'Per-message publisher/subscriber sensor payload hashes and exact source identity. Worker queue drops remain separate; no inference labels or ground-truth pose accessed.'}
