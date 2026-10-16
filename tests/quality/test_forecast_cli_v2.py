"""Protect the cross-package step/epoch protocol binding without accessing targets."""
import importlib.util
from pathlib import Path
import sys
import pytest


@pytest.mark.parametrize('standalone',[False,True])
def test_forecast_frontdoors_bind_6000_steps_and_30_epochs(monkeypatch,standalone):
    from indoor_perception_lab import forecast_v2
    calls=[]
    def controlled_training(domain,seed,lr,max_steps=6000,epochs=30,**kwargs):
        calls.append((domain,seed,lr,max_steps,epochs));return Path('/controlled-model-pack')
    monkeypatch.setattr(forecast_v2,'train_forecast_v2',controlled_training)
    if standalone:
        from indoor_perception_lab.cli import main
    else:
        spec=importlib.util.spec_from_file_location('portfolio_cli_oracle',Path(__file__).resolve().parents[2]/'tools/portfolio_cli.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);main=module.main
    monkeypatch.setattr(sys,'argv',['oracle','train','--protocol','quality-v2','--task','forecast','--domain','jaad','--configuration','F2'])
    assert main()==0
    assert calls==[('jaad',42,.0003,6000,30)]


@pytest.mark.parametrize('standalone',[False,True])
@pytest.mark.parametrize('task,configuration',[('forecast','F2'),('semantic','S2')])
def test_frozen_configuration_rejects_conflicting_learning_rate(monkeypatch,standalone,task,configuration):
    if standalone:
        from indoor_perception_lab.cli import main
    else:
        spec=importlib.util.spec_from_file_location('portfolio_cli_identity_oracle',Path(__file__).resolve().parents[2]/'tools/portfolio_cli.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);main=module.main
    monkeypatch.setattr(sys,'argv',['oracle','train','--protocol','quality-v2','--task',task,'--configuration',configuration,'--lr=0.9'])
    with pytest.raises(ValueError,match='learning rate'):
        main()
