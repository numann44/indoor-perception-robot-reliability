import numpy as np
import pytest
from robot_model_sdk import TrajectoryHistoryV1,resample_causal

def test_causal_resampling_and_gap_boundaries():
    t=np.arange(11)/10;p=np.stack([t,t*2],1)
    np.testing.assert_allclose(resample_causal(t,p),p,atol=1e-7)
    assert resample_causal(t[1:],p[1:]) is None
    with pytest.raises(ValueError,match='Future'):resample_causal(t,p,anchor_time=.9)
    keep=np.array([0,1,2,6,7,8,9,10]);assert resample_causal(t[keep],p[keep]) is None

def test_history_contract_rejects_ambiguous_units():
    with pytest.raises(ValueError):TrajectoryHistoryV1(np.array([0,1]),np.zeros((2,2)),'world','unspecified')
