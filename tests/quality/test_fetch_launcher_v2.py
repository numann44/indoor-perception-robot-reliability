"""The public fetch verb must invoke the data tool's positional dataset parser."""
import importlib.util
from pathlib import Path
import sys


def test_fetch_dispatches_each_requested_v2_dataset(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('fetch_launcher_oracle',root/'tools/portfolio_cli.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    monkeypatch.setattr(module.subprocess,'call',lambda argv:calls.append(argv) or 0)
    monkeypatch.setattr(sys,'argv',['portfolio','fetch','--protocol','quality-v2','--datasets','nyu','openloris'])
    assert module.main()==0
    assert [call[-1] for call in calls]==['nyu','openloris']
    assert all(call[1]==str(root/'tools/data_v2.py') and len(call)==3 for call in calls)
