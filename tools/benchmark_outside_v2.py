#!/usr/bin/env python3
"""Admit the immutable MOT17 external holdout to the unchanged Bench evaluator."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = {
    'data/v2/mot17/index.jsonl': '421abf943f174834cbf8874d339ccb1778ad33e1a3ec26b280612c9a67222bd1',
    'models/v2/selection-detection.json': '3afc5aa89e4ed8172bce683fac0388f38f286a0abfba88b987efda226cc158e5',
    'reports/v2/perception-heldout-complete.json': '8afa60178e8771ba7ff0c57d1ca1ddacc23204f4f461ceba530cddc197ba3259',
    'models/v2/detection-D3-seed42/manifest.json': 'fecb506928e1c42eeb81f911bcec8e2c25a1884dc134fe5b756fe01b66b2ff4a',
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / 'reports') or output.exists():
        raise ValueError('Require a new report path within the project')
    for path, expected in BINDINGS.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen external-test admission binding changed: ' + path)
    rows = [json.loads(line) for line in (ROOT / 'data/v2/mot17/index.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != 900 or len({r['sample_id'] for r in rows}) != 900:
        raise ValueError('Require all 900 unique external-test frames')
    if any(r.get('split') != 'outside_holdout' or r.get('dataset') != 'mot17' or r.get('sequence_id') != 'MOT17-11-SDP' for r in rows):
        raise ValueError('External-test dataset/split/sequence contract mismatch')
    selection = json.loads((ROOT / 'models/v2/selection-detection.json').read_text())
    seal = json.loads((ROOT / 'reports/v2/perception-heldout-complete.json').read_text())
    selected = selection['selected']
    if selection['test_used_for_selection'] or selected['pack'] != 'models/v2/detection-D3-seed42' or selected['weights_sha256'] != '44f17da2cbc5972eec95383c1113c5005635c8670d9f5c9f50dfe3ae92acbe6c':
        raise ValueError('Selected detector or selection protocol changed')
    if seal['status'] != 'verified' or seal['seed_selection_on_test'] or seal['family_bindings']['detection']['selection_lock_sha256'] != BINDINGS['models/v2/selection-detection.json']:
        raise ValueError('Held-out model-selection seal is not admitted')
    if importlib.util.find_spec('indoor_perception_lab') is not None:
        raise ValueError('This external evaluation must use the independent Bench environment without Lab')
    from robot_model_sdk.manifest import validate_manifest
    from robot_reliability_bench.robotics_v2 import quality_accuracy
    from robot_reliability_bench.robotics_v2.provenance import context
    import robot_model_sdk
    import robot_reliability_bench
    validate_manifest(ROOT / selected['pack'])
    installed = {}
    for module in (robot_model_sdk, robot_reliability_bench):
        base = Path(module.__file__).parent
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json'}:
                installed[f'{module.__name__}/{path.relative_to(base)}'] = digest(path)
    admission = {'bindings_sha256': BINDINGS, 'all_unique_external_frames': 900,
                 'split_preserved': 'outside_holdout', 'model_selection_uses_test': False,
                 'Lab_installed': False, 'helper_sha256': digest(__file__),
                 'scope': 'Admission wrapper only; unchanged installed evaluator/model/metrics'}
    def project_context(config):
        proof = context(config, root=ROOT)
        expected = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if not proof.get('source_sha256') or proof.get('git_commit') != expected:
            raise ValueError('Require nonempty project source and current project Git identity')
        proof['installed_runtime_source_files_sha256'] = installed
        proof['external_holdout_admission'] = admission
        return proof
    quality_accuracy.context = project_context
    result = quality_accuracy.evaluate(ROOT / selected['pack'], ROOT / 'data/v2/mot17/index.jsonl', output,
                                       task='detection', backend='native', device='cpu', split='outside_holdout',
                                       test_unlocked=True, conditions=None)
    if result['images'] != 900:
        raise RuntimeError('External evaluator did not consume all 900 frames')
    print(json.dumps({'status': result['status'], 'images': result['images'], 'output': str(output),
                      'admission': admission}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
