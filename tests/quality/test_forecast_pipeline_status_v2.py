from dataclasses import replace
import numpy as np
from robot_model_sdk import fixture_frames
from indoor_perception_lab.pipeline import PerceptionPipeline


def test_configured_forecast_distinguishes_insufficient_and_unobserved_history():
    class CausalForecast:
        metadata={'coordinate_system':'image_xy_divided_by_width_height'}
        def predict_timed_history(self,*args):raise AssertionError('A forecast needs sufficient observed history')
        def close(self):pass
    original=next(iter(fixture_frames(1)))
    frame=replace(original,sample_id='jaad-v2/oracle/0',depth=None)
    pipe=PerceptionPipeline(forecaster=CausalForecast(),use_depth=False)
    try:
        first=pipe.process(frame)
        assert first['tracks'] and all(t['learned_forecast_status']=='insufficient_causal_history' for t in first['tracks'])
        lost=pipe.process(replace(frame,sample_id='jaad-v2/oracle/1',timestamp_ns=100_000_000,rgb=np.zeros_like(frame.rgb)))
        assert lost['tracks'] and all(t['learned_forecast_status']=='unavailable_unobserved_track' for t in lost['tracks'])
    finally:pipe.close()
