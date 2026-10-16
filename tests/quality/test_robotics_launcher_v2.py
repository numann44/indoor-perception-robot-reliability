import importlib.util
from pathlib import Path
import sys
import pytest


@pytest.mark.parametrize('arguments,action,forwarded',[
    (['simulate','--active','--model-pack','selected','--tracker','T2'],'simulate-active',['--model-pack','selected','--tracker','T2']),
    (['ros2','--duration','20'],'ros2',['--duration','20']),
    (['ros2','--','--duration','20'],'ros2',['--duration','20']),
])
def test_robotics_frontdoors_forward_flags_without_remainder_rejection(monkeypatch,arguments,action,forwarded):
    spec=importlib.util.spec_from_file_location('portfolio_robotics_oracle',Path(__file__).resolve().parents[2]/'tools/portfolio_cli.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);calls=[]
    monkeypatch.setattr(sys,'argv',['portfolio',*arguments])
    monkeypatch.setattr(module.subprocess,'call',lambda args:calls.append(args) or 0)
    assert module.main()==0
    assert calls==[[sys.executable,str(module.ROOT/'tools/robotics_v2.py'),action,*forwarded]]
