"""Audit completed local delivery metadata without inference or target access."""
from __future__ import annotations

import collections
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(reference):
    path = (ROOT / reference).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError('Missing or out-of-project evidence: ' + str(reference))
    return path


def software_sources():
    paths = []
    for directory in ('packages', 'tools', 'tests', 'configs'):
        for path in (ROOT / directory).rglob('*'):
            relative = path.relative_to(ROOT / directory)
            if (path.is_file() and not path.is_symlink()
                    and path.suffix in {'.py', '.json', '.toml', '.lock'}
                    and not any(part in {'build', '__pycache__'} or part.endswith('.egg-info') for part in relative.parts)):
                paths.append(path)
    paths.extend([ROOT / 'portfolio', ROOT / 'requirements.lock'])
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}


def main():
    os.chdir(ROOT)
    stamp = datetime.datetime.now(datetime.timezone.utc)
    output = ROOT / 'reports/v2/final-delivery' / (stamp.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    record = {'scope': 'Final local metadata audit; no inference, labels or benchmark rerun',
              'started_utc': stamp.isoformat(), 'command': [sys.executable, *sys.argv],
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'checks': {}, 'model_inference_executed': False}
    documents = sorted(path for path in (ROOT / 'docs/v2').iterdir() if path.suffix in {'.md', '.py'})
    documents.extend([ROOT / name for name in ['README.md', 'FINAL_REPORT_V2.md', 'CHECKPOINT.md', 'RESUME.md']])
    try:
        matrix_path = inside('docs/v2/requirements.json')
        paths = [Path(__file__).resolve(), matrix_path, *documents]
        snapshot = {str(path.relative_to(ROOT)): path.read_bytes() for path in paths}
        before = {name: hashlib.sha256(data).hexdigest() for name, data in snapshot.items()}
        archive_path = output / 'source-and-documents-before-audit.tar.gz'
        with tarfile.open(archive_path, 'w:gz') as archive:
            for name, data in snapshot.items():
                entry = tarfile.TarInfo(name); entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
        record['source_and_documents_before_execution'] = before
        record['archive'] = {'path': str(archive_path.relative_to(ROOT)), 'sha256': sha(archive_path)}

        matrix = json.loads(matrix_path.read_text())
        rows = matrix['rows']; allowed = {'implemented-tested', 'implemented-unverified', 'deferred', 'unavailable'}
        if len({row['id'] for row in rows}) != len(rows) or any(row['status'] not in allowed for row in rows):
            raise ValueError('Duplicate requirement IDs or invalid status')
        counts = dict(collections.Counter(row['status'] for row in rows))
        actual = {'Complete': counts.get('implemented-tested', 0),
                  'Partial': counts.get('implemented-unverified', 0) + counts.get('deferred', 0),
                  'Blocked': counts.get('unavailable', 0)}
        if counts != {name: value for name, value in matrix['counts'].items() if value} or actual != matrix['complete_partial_blocked']:
            raise ValueError('Requirement counts differ from actual rows')
        if matrix['overall_status'] != 'PARTIAL' or matrix['phase'] != 'final':
            raise ValueError('Final matrix must retain explicit PARTIAL overall status')
        evidence_files = 0
        for row in rows:
            if row['status'] == 'implemented-tested' and row.get('acceptance_predicate_passed') is not True:
                raise ValueError('Tested row lacks a passing scoped predicate: ' + row['id'])
            for binding in row['evidence']:
                if binding.get('exists') is not True or sha(inside(binding['path'])) != binding['sha256']:
                    raise ValueError('Matrix evidence binding changed: ' + row['id'])
                evidence_files += 1
        record['checks']['requirements'] = {'passed': True, 'rows': len(rows), **actual,
                                             'evidence_references_checked': evidence_files}

        acceptance = json.loads(inside('reports/v2/robotics/final-active-wave-release.json').read_text())
        root_path = inside(acceptance['root_acceptance'])
        if sha(root_path) != acceptance['root_acceptance_sha256']:
            raise ValueError('Root release proof changed')
        root = json.loads(root_path.read_text())
        verification_path = inside(root['verification']['path'])
        verification = json.loads(verification_path.read_text())
        if (root['status'] != 'passed' or sha(verification_path) != root['verification']['sha256']
                or verification['status'] != 'passed' or verification['source_unchanged_during_execution'] is not True
                ):
            raise ValueError('Original root acceptance proof is invalid')
        # Packaging-only repairs after the completed simulation release require
        # a fresh all-suite proof. Preserve the original release instead of
        # rewriting it or rerunning completed neural experiments.
        supplement_path = ROOT / 'reports/v2/final-delivery-software-requalification.json'
        if supplement_path.exists():
            supplement = json.loads(supplement_path.read_text())
            if (supplement['original_root_acceptance']['path'] != str(root_path.relative_to(ROOT))
                    or supplement['original_root_acceptance']['sha256'] != sha(root_path)):
                raise ValueError('Supplemental acceptance references another original root proof')
            verification_path = inside(supplement['verification']['path'])
            verification = json.loads(verification_path.read_text())
            if (sha(verification_path) != supplement['verification']['sha256']
                    or verification['status'] != 'passed'
                    or verification['source_unchanged_during_execution'] is not True):
                raise ValueError('Supplemental all-suite proof is invalid')
            record['supplemental_current_software_acceptance'] = {
                'path': str(supplement_path.relative_to(ROOT)), 'sha256': sha(supplement_path)}
        if software_sources() != verification['source_after_execution']:
            raise ValueError('Current exact software tree differs from successful all-suite proof')
        record['checks']['current_software'] = {'passed': True, 'path': str(verification_path.relative_to(ROOT)),
                                               'source_files': len(verification['source_after_execution'])}

        pipeline = json.loads(inside('reports/v2/bench/final-posttiming-streamed-execution.json').read_text())
        if pipeline['status'] not in {'verified', 'executed-with-failures'}:
            raise ValueError('Final execution pipeline has not finished')
        neural = json.loads(inside('reports/v2/robotics/final-active-person-matched/summary.json').read_text())
        episode_rows = neural['rows']
        keys = {(row['condition'], row['scenario'], row['seed']) for row in episode_rows}
        frozen = json.loads(inside('configs/v2/robotics-active-person.json').read_text())
        expected = {(condition, scenario, seed) for condition in frozen['conditions']
                    for scenario in frozen['scenarios']
                    for seed in range(frozen['seed_start'], frozen['seed_start'] + frozen['seed_count'])}
        if neural['episodes'] != 120 or len(episode_rows) != 120 or keys != expected:
            raise ValueError('Matched simulation episode coverage is incomplete')
        record['checks']['execution_coverage'] = {'passed': True, 'pipeline_status': pipeline['status'],
                                                 'matched_episodes': len(episode_rows),
                                                 'note': 'Completed execution does not assert successful quality or navigation gates'}

        import psutil
        active = []
        for process in psutil.process_iter(['pid', 'name']):
            try:
                if process.pid == os.getpid() or not Path(process.cwd()).is_relative_to(ROOT):
                    continue
                argv = process.cmdline()
                if any(any(token in arg for token in ('quality_training', 'inference_worker', 'ml_worker',
                                                      'benchmark_v2.py', 'benchmark_streamed_accuracy_v2.py',
                                                      'robotics_v2.py', 'render-failure-pairs.py')) for arg in argv):
                    active.append({'pid': process.pid, 'name': process.info['name'], 'argv': argv})
            except (psutil.Error, FileNotFoundError):
                pass
        if active:
            record['active_owned_runtimes'] = active
            raise ValueError('An owned inference/benchmark/robotics runtime remains active')
        record['checks']['runtime_quiescence'] = {'passed': True}

        demo = json.loads(inside('reports/v2/demo-build.json').read_text())
        visual_path = inside('reports/v2/demo-browser-qa.json'); visual = json.loads(visual_path.read_text())
        if (demo['status'] != 'verified' or demo['chapters'] != 4 or not 60 <= demo['source_duration_s'] <= 90
                or visual['status'] != 'passed' or visual['actual_browser_visual_inspection'] is not True):
            raise ValueError('Built replay or actual browser visual acceptance is absent')
        for name, field in [('demo/v2/index.html', 'viewer_sha256'), ('demo/v2/manifest.json', 'manifest_sha256')]:
            if sha(inside(name)) != demo[field] or visual[field] != demo[field]:
                raise ValueError('Browser inspected a different replay build')
        record['checks']['demo'] = {'passed': True, 'seconds': demo['source_duration_s'],
                                    'visual_QA_path': str(visual_path.relative_to(ROOT)), 'visual_QA_sha256': sha(visual_path)}

        commands_path = inside('reports/v2/verified-commands.jsonl')
        commands = [json.loads(line) for line in commands_path.read_text().splitlines() if line.strip()]
        for command in commands:
            if sha(inside(command['log'])) != command['log_sha256'] or bool(command['verified']) != (command['exit_code'] == 0):
                raise ValueError('Command exit/log proof changed: ' + command['name'])
        record['checks']['commands'] = {'passed': True, 'completed_records': len(commands),
                                        'successful': sum(command['verified'] for command in commands),
                                        'failed_attempts_retained': sum(not command['verified'] for command in commands)}

        preservation = json.loads(inside('reports/v2/v1-preservation.json').read_text())
        changed = [row['path'] for row in preservation['files'] if sha(inside(row['path'])) != row['sha256']]
        if changed: raise ValueError('V1 artifacts changed: ' + repr(changed))
        record['checks']['v1_preserved'] = {'passed': True, 'files': len(preservation['files'])}

        protocol = json.loads(inside('configs/v2/protocol.json').read_text())
        ledger = [json.loads(line) for line in inside('reports/v2/data/download_ledger.jsonl').read_text().splitlines() if line.strip()]
        project_bytes = sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file() and not path.is_symlink())
        free = shutil.disk_usage(ROOT).free; downloaded = ledger[-1]['cumulative_bytes'] if ledger else 0
        if project_bytes >= protocol['project_limit_bytes'] or downloaded >= protocol['download_limit_bytes'] or free < protocol['minimum_free_bytes']:
            raise ValueError('Approved resource ceiling failed')
        record['checks']['budget'] = {'passed': True, 'project_bytes': project_bytes,
                                      'recorded_cumulative_download_bytes': downloaded, 'free_bytes': free}
        changed = [name for name, digest in before.items() if sha(ROOT / name) != digest]
        if changed: raise ValueError('Source/documents changed during final audit: ' + repr(changed))
        record['source_and_documents_unchanged_during_execution'] = True
        record['status'] = 'passed'
    except Exception as exc:
        record.update(status='failed', error=repr(exc))
    record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report_path = output / 'run.json'
    report_path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'path': str(report_path.relative_to(ROOT)),
                      'error': record.get('error')}))
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
