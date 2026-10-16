import json
from pathlib import Path
import numpy as np
import pytest
from robot_model_sdk import load_adapter,fixture_frame
from indoor_perception_lab.pipeline import PerceptionPipeline

ROOT=Path(__file__).resolve().parents[2]
PACK=ROOT/'models/motion-fixture-observed-world-seed42-lr0.001-budgeted'

@pytest.mark.skipif(not PACK.exists(),reason='Generated experiment model absent; run primary motion training first')
def test_sdk_forecast_preprocessing_matches_training_adapter():
    from indoor_perception_lab.motion_training import ForecastAdapter
    h=np.stack([np.linspace(0,.2,11),np.linspace(4,3.9,11)],1).astype(np.float32)
    sdk=load_adapter(PACK);reference=ForecastAdapter(PACK)
    np.testing.assert_allclose(sdk.predict_history(h),reference.predict_history(h),rtol=1e-6,atol=1e-7)
    sdk.close();reference.close()

@pytest.mark.skipif(not PACK.exists(),reason='Generated experiment model absent')
def test_learned_forecast_is_available_only_with_valid_domain_history():
    p=PerceptionPipeline(forecaster=PACK)
    results=[p.process(fixture_frame(i)) for i in range(15)]
    assert any(t['learned_forecast_status']=='available' and 'MLP2x64' in t['forecasts'] for t in results[-1]['tracks'])
    f=fixture_frame(16);f.sample_id='tum/unvalidated-transfer';f.pose_source='provided_ground_truth';r=p.process(f)
    assert all(t['learned_forecast_status']=='unavailable_domain_or_coordinates' for t in r['tracks'] if t['observed'])
    p.close()
