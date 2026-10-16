from copy import deepcopy
from robot_reliability_bench.robotics_v2.ros_validation import validate_recorded_transport


def trace():
    refs=[{'sample_id':'person_01_suffix_9','timestamp_ns':100},{'sample_id':'other_suffix_10','timestamp_ns':200}]
    pubs=[];received=[]
    for i,row in enumerate(refs):
        ts=row['timestamp_ns']+1000
        meta={'source_sequence':i,'sample_id':row['sample_id'],'original_timestamp_ns':row['timestamp_ns'],'timestamp_ns':ts,'pose_expected':i>0}
        pubs.append({'event':'published','timestamp_ns':ts,'source_metadata':meta,'depth_published':True,'RGB_payload_sha256':str(i)+'a'*63,'depth_payload_sha256':str(i)+'b'*63})
        received.extend([{'event':'received_rgb','timestamp_ns':ts,'payload_sha256':str(i)+'a'*63},
                         {'event':'received_depth','timestamp_ns':ts,'payload_sha256':str(i)+'b'*63},
                         {'event':'received_calibration','timestamp_ns':ts},
                         {'event':'received_source_metadata','timestamp_ns':ts,'source_metadata':deepcopy(meta)}])
        if i>0:received.append({'event':'received_odometry','timestamp_ns':ts})
    return refs,pubs,received


def test_recorded_transport_requires_exact_sensor_payload_and_source_clocks():
    refs,pubs,received=trace()
    result=validate_recorded_transport(refs,pubs,received)
    assert all(result['checks'].values()) and result['clock_offset_ns']==1000
    received[0]['payload_sha256']='f'*64
    assert not validate_recorded_transport(refs,pubs,received)['checks']['RGB_payload_hashes_match_published']


def test_suffix_truncation_and_repeated_missing_message_are_not_accepted():
    refs,pubs,received=trace()
    next(e for e in received if e['event']=='received_source_metadata')['source_metadata']['sample_id']='person_01'
    assert not validate_recorded_transport(refs,pubs,received)['checks']['source_metadata_received_matches_published']
    received=[e for e in received if not(e['event']=='received_depth' and e['timestamp_ns']==1200)]
    received.append(deepcopy(next(e for e in received if e['event']=='received_depth')))
    checks=validate_recorded_transport(refs,pubs,received)['checks']
    assert not checks['depth_transport_all_expected_received']
