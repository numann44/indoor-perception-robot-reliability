import importlib.util
from pathlib import Path
import sys


def test_quality_benchmark_uses_independent_wheels_without_development_imports(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('benchmark_frontdoor',Path(__file__).resolve().parents[2]/'tools/portfolio_cli.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    independent=tmp_path/'environments/v2-bench/bin/python';independent.parent.mkdir(parents=True);independent.touch()
    module.ROOT=tmp_path;calls=[]
    monkeypatch.setenv('PYTHONPATH','/development/lab:/development/sdk')
    monkeypatch.setenv('PORTFOLIO_ROOT',str(tmp_path))
    monkeypatch.setattr(sys,'argv',['portfolio','benchmark','--protocol','quality-v2','matrix','--quiescent'])
    monkeypatch.setattr(module.subprocess,'call',lambda args,env:calls.append((args,env)) or 0)
    assert module.main()==0
    argv,environment=calls[0]
    assert argv==[str(independent),str(tmp_path/'tools/benchmark_v2.py'),'matrix','--quiescent']
    assert 'PYTHONPATH' not in environment
    assert environment['PORTFOLIO_ROOT']==str(tmp_path)
