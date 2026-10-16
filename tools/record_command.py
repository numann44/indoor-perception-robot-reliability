from pathlib import Path
import argparse,datetime,json,subprocess,os,shlex,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--log',required=True);p.add_argument('--ledger',default='reports/verified-commands.jsonl');p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args();argv=a.argv[1:] if a.argv and a.argv[0]=='--' else a.argv
log=(ROOT/a.log).resolve()
if not log.is_relative_to(ROOT):raise ValueError('Log escapes isolated root')
if log.exists():raise FileExistsError('Refusing to overwrite command evidence; choose a unique log: '+str(log))
log.parent.mkdir(parents=True,exist_ok=True);start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
with log.open('w') as f:rc=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
record={'name':a.name,'argv':argv,'command':shlex.join(argv),'cwd':str(ROOT),'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.perf_counter()-t,'exit_code':rc,'verified':rc==0,'log':str(log.relative_to(ROOT)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
ledger=(ROOT/a.ledger).resolve()
if not ledger.is_relative_to(ROOT/'reports'):raise ValueError('Command ledger must remain in project reports')
ledger.parent.mkdir(parents=True,exist_ok=True)
with ledger.open('a') as f:f.write(json.dumps(record)+'\n')
print(json.dumps(record));raise SystemExit(rc)
