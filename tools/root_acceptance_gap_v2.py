"""Run root acceptance only inside the declared no-NN barrier, then release it.

Browser visual QA remains explicitly pending; it runs after the neural episodes.
This local scheduler makes progress without relying on an agent's waiting session.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reusable_completed_outputs(prior_path):
    """Reuse successful immutable map/demo outputs after a software-only repair."""
    prior_path = (ROOT / prior_path).resolve()
    if not prior_path.is_relative_to(ROOT / 'reports/v2/root-acceptance') or not prior_path.is_file():
        raise ValueError('Prior root acceptance must be a saved local report')
    prior = json.loads(prior_path.read_text())
    if prior['status'] != 'failed':
        raise ValueError('Only a retained failed acceptance may be resumed')
    exits = {check['name']: check['exit_code'] for check in prior['checks']}
    if exits.get('saved-output-map') != 0 or exits.get('source-clock-demo') != 0 or exits.get('software-all-suite') == 0:
        raise ValueError('Reuse requires completed map/demo and a failed software suite')
    def checked(reference, digest, parent=ROOT):
        path = (parent / reference).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != digest:
            raise ValueError('Previously completed evidence changed: ' + str(reference))
        return path
    mapping_path = checked(prior['map']['path'], prior['map']['sha256'])
    mapping = json.loads(mapping_path.read_text())
    if mapping['status'] != 'verified' or mapping['model_inference_rerun'] or not mapping['inputs_and_source_unchanged']:
        raise ValueError('Prior map was not accepted')
    for identities in [mapping['inputs_sha256'], mapping['source_sha256_before_and_after']]:
        for name, digest in identities.items(): checked(name, digest)
    for artifact in mapping['artifacts'].values(): checked(artifact['path'], artifact['sha256'])
    demo_path = checked(prior['demo']['path'], prior['demo']['sha256'])
    demo = json.loads(demo_path.read_text()); base = ROOT / demo['output']
    if demo['status'] != 'verified' or demo['chapters'] != 4 or not 60 <= demo['source_duration_s'] <= 90:
        raise ValueError('Prior replay was not accepted')
    checked('index.html', demo['viewer_sha256'], base)
    replay = json.loads(checked('manifest.json', demo['manifest_sha256'], base).read_text())
    for chapter in replay['chapters']:
        checked(chapter['evidence_ref'], chapter['evidence_sha256'], base)
        for frame in chapter['frames']:
            checked(frame['image'], frame['image_sha256'], base)
            checked(frame['result_ref'], frame['result_sha256'], base)
    return prior, {'path':str(prior_path.relative_to(ROOT)), 'sha256':sha(prior_path),
                   'policy':'Rechecked every map input/source/artifact and displayed replay image/result hash; no repeated inference or reconstruction'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-prior', help='Retained failed root report with verified immutable map/demo outputs')
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol = json.loads((ROOT / 'configs/v2/protocol.json').read_text())
    deadline = datetime.datetime.fromisoformat(protocol['deadline_utc'].replace('Z', '+00:00')).timestamp()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-root-' + uuid.uuid4().hex[:6]
    output = ROOT / 'reports/v2/root-acceptance' / run_id
    output.mkdir(parents=True)
    state = {'run_id':run_id, 'status':'waiting-for-gap', 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source_tool_sha256':sha(__file__), 'checks':[], 'browser_visual_QA':'pending after neural wave',
             'policy':'No model/config/runtime changes; actual exits required; no release on failure'}
    def save():
        (output / 'run.json').write_text(json.dumps(state, indent=2) + '\n')
    def stage(name, args):
        log = output / (name + '.log')
        argv = [str(ROOT / 'environments/main/bin/python'), str(ROOT / 'tools/record_command.py'),
                '--name', name + '-' + run_id, '--log', str(log.relative_to(ROOT)),
                '--ledger', 'reports/v2/verified-commands.jsonl', '--', *args]
        started = time.time()
        code = subprocess.call(argv, cwd=ROOT)
        state['checks'].append({'name':name, 'argv':argv, 'exit_code':code,
                                'seconds':time.time()-started, 'log':str(log.relative_to(ROOT)), 'log_sha256':sha(log)})
        save()
        if code:
            raise RuntimeError(name + ' failed; retained its log and blocked simulation release')
    save()
    gap_path = ROOT / 'reports/v2/robotics/final-root-acceptance-gap.json'
    release_path = ROOT / 'reports/v2/robotics/final-active-wave-release.json'
    try:
        if release_path.exists():
            raise FileExistsError('Preserve previous release; do not run duplicate root acceptance')
        while not gap_path.exists():
            if time.time() >= deadline - 3600:
                raise TimeoutError('No root acceptance gap before the final reserved hour')
            time.sleep(20)
        gap = json.loads(gap_path.read_text())
        state['gap'] = {'path':str(gap_path.relative_to(ROOT)), 'sha256':sha(gap_path), 'record':gap}
        state['status'] = 'running'; save()
        if args.reuse_prior:
            prior, binding = reusable_completed_outputs(args.reuse_prior)
            if prior['gap']['sha256'] != state['gap']['sha256']:
                raise ValueError('Cannot reuse outputs from another dispatch barrier')
            state.update(map=prior['map'], demo=prior['demo'], reused_completed_evidence=binding); save()
        else:
            map_output = 'reports/v2/robotics/maps/' + run_id
            stage('saved-output-map', ['environments/main/bin/python','tools/reconstruct_map_v2.py',
                                      '--chapter','reports/v2/robotics/media/openloris','--output',map_output])
            map_path = ROOT / map_output / 'run.json'
            mapping = json.loads(map_path.read_text())
            if mapping['status'] != 'verified' or mapping['model_inference_rerun'] or not mapping['inputs_and_source_unchanged']:
                raise ValueError('Saved-output mapping acceptance failed')
            state['map'] = {'path':str(map_path.relative_to(ROOT)), 'sha256':sha(map_path)}; save()
            stage('source-clock-demo', ['./portfolio','demo','--protocol','quality-v2'])
            demo_path = ROOT / 'reports/v2/demo-build.json'
            demo = json.loads(demo_path.read_text())
            if demo['status'] != 'verified' or demo['chapters'] != 4 or not 60 <= demo['source_duration_s'] <= 90:
                raise ValueError('Actual four-chapter replay acceptance failed')
            state['demo'] = {'path':str(demo_path.relative_to(ROOT)), 'sha256':sha(demo_path)}; save()
        previous = set((ROOT / 'reports/v2/verification').glob('*/run.json'))
        stage('software-all-suite', ['./portfolio','verify','--protocol','quality-v2','--suite','all'])
        created = set((ROOT / 'reports/v2/verification').glob('*/run.json')) - previous
        if len(created) != 1:
            raise ValueError('Final software invocation must have one unambiguous report')
        verification_path = created.pop()
        verification = json.loads(verification_path.read_text())
        if verification['status'] != 'passed' or not verification['source_unchanged_during_execution']:
            raise ValueError('Current-source software acceptance failed')
        state['verification'] = {'path':str(verification_path.relative_to(ROOT)), 'sha256':sha(verification_path)}
        import psutil
        root_prefix = str(ROOT)
        blocked = []
        for process in psutil.process_iter(['pid','cmdline']):
            try:
                argv = process.info['cmdline'] or []
                if process.pid == os.getpid():continue
                own = root_prefix in ' '.join(argv) or process.cwd() == root_prefix
                known_runtime = any((arg.endswith('/tools/benchmark_v2.py') or arg.endswith('/tools/robotics_v2.py')
                                     or arg.endswith('/tools/verify_v2.py') or 'inference_worker' in arg
                                     or 'ml_worker' in arg or 'quality_training' in arg or 'render-failure-pairs.py' in arg)
                                    for arg in argv)
                if own and known_runtime:blocked.append({'pid':process.pid, 'argv':argv})
            except (psutil.Error, FileNotFoundError):pass
        if blocked:
            state['blocked_processes'] = blocked
            raise RuntimeError('Owned runtime process still active; release blocked')
        if sha(gap_path) != state['gap']['sha256']:
            raise ValueError('Declared gap evidence changed during root acceptance')
        state['status'] = 'passed'; state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
        release = {'authorized':True, 'other_NN_stopped':True,
                   'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'root_acceptance':str((output/'run.json').relative_to(ROOT)), 'root_acceptance_sha256':sha(output/'run.json'),
                   'gap_sha256':sha(gap_path), 'browser_visual_QA':'pending after 120 episodes; no acceptance claim',
                   'qualification_policy':'Benchmark orchestrator independently rechecks frozen SDK/Bench/Lab and three qualified tools'}
        with release_path.open('x') as stream:stream.write(json.dumps(release, indent=2) + '\n')
        print(json.dumps({'status':'passed','root_acceptance':str((output/'run.json').relative_to(ROOT)),'release':str(release_path.relative_to(ROOT))}), flush=True)
    except Exception as exc:
        state['status'] = 'failed'; state['error'] = repr(exc); state['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(); save()
        print(json.dumps({'status':'failed','root_acceptance':str((output/'run.json').relative_to(ROOT)),'error':state['error']}), flush=True)
        return 1
    return 0


if __name__ == '__main__':raise SystemExit(main())
