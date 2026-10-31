#!/usr/bin/env python3
"""Freeze a production FA3 quality pilot; no remote writes or model execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from controller.durable import digest, secure_directory, write_new_json
from controller.policy import PRODUCTION_SOURCES, QUALITY_SOURCES, XQA_SOURCES, XQA_OPERATORS
from controller.protocols import TASKS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-native-quality-20261026')
XQA_SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-xqa-20261026')


def prepare(base_package: Path, old_quality: Path, run_id: str, bundle: Path, out: Path,
            completion: Path | None = None, parent_audit: Path | None = None, profile: str = 'fa3',
            recovery_audit: Path | None = None) -> dict:
    if profile not in ('fa3', 'xqa'):
        raise ValueError('Unsupported production quality profile')
    candidate_source = XQA_SOURCE if profile == 'xqa' else SOURCE
    original = json.loads(base_package.read_text())
    historical = json.loads(old_quality.read_text())
    request = original['request']
    remote = request['approved_workspace'] + '/agent_work/runs/' + run_id
    fragment = completion is not None
    pilot = completion.absolute() if fragment else ROOT / 'results/prepared/native-quality-pilot-20261026T075846Z'
    manifest_name = 'completion_manifest.json' if fragment else 'pilot_manifest.json'
    data_relative = '/input/cohort' if fragment else '/input/pilot'
    inputs = ROOT / 'results/mirrors/quality-protocols-20261026T075846Z/files/support/inputs'
    scorer = inputs / 'longbench-scorer'
    shots = Path(historical['assets']['input/five_train_examples.json']['local_path'])
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=candidate_source, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=candidate_source, text=True):
        raise ValueError('Commit the candidate quality source before freezing')
    sys.path.insert(0, str(candidate_source))
    from tools import run_qwen25_joint_kv_quality as runner
    args = runner._parser().parse_args([
        '--model', str(ROOT / 'results/mirrors/quality-protocols-20261026T075846Z/files/model_metadata'),
        '--gsm8k', str(pilot / 'gsm8k.jsonl'), '--gsm8k-shots', str(shots), '--longbench-data', str(pilot),
        '--longbench-repo', str(scorer), '--attention-backend', 'native-flashinfer', '--native-planner', 'fa3',
        '--fp8-q-scale', '1.0', '--fp8-k-scale', '0.125', '--fp8-v-scale', '0.0625', '--graph-mode', 'cudagraphs',
        '--arms', 'auto', 'fp8_e4m3', '--max-seqs-per-batch', '128', '--concurrency', '8',
        '--cohort-scope', 'fragment' if fragment else 'pilot', '--binary-source-commit', request['server_source_commit'], '--output', str(out / 'unused')])
    runner._validate_args(args)
    audit = runner._data_audit(args, {task: pilot / (task + '.jsonl') for task in TASKS})
    for task, item in audit['datasets'].items():
        item['path'] = remote + data_relative + '/' + task + '.jsonl'
    if out.exists():
        raise FileExistsError('Quality package must be a new immutable directory')
    secure_directory(out)
    write_new_json(out / 'approved_data_manifest.json', {'data_audit': audit})
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    files = {name: Path(entry['local_path']) for name, entry in original['assets'].items() if name.startswith('runtime/')}
    files['input/source.bundle'] = bundle.absolute()
    files['input/approved_data_manifest.json'] = out / 'approved_data_manifest.json'
    for name, entry in historical['assets'].items():
        if name.startswith('input/') and not name.startswith('input/pilot/') and name != 'input/approved_data_manifest.json':
            files[name] = Path(entry['local_path'])
    for task in TASKS:
        files[data_relative.lstrip('/') + '/' + task + '.jsonl'] = pilot / (task + '.jsonl')
    files[data_relative.lstrip('/') + '/' + manifest_name] = pilot / manifest_name
    request.update(run_id=run_id, remote_output=remote, concurrency=8, arms=['bf16', 'fp8_e4m3'],
                   python_source_repo=remote + '/source_repo', python_source_commit=commit,
                   backend_source_sha256={name: sha(candidate_source / name) for name in
                                          (XQA_SOURCES if profile == 'xqa' else PRODUCTION_SOURCES) + QUALITY_SOURCES},
                   dataset_path=remote + data_relative + '/' + manifest_name, dataset_sha256=sha(pilot / manifest_name),
                   max_output_gib=5, max_wall_seconds=7200,
                   workload={'kind': 'native_quality_fa3', 'cohort': 'completion_fragment' if fragment else 'pilot',
                             'samples_per_arm': 1639 if fragment else 80, 'ignore_eos': False,
                             'source_content_sha256': sha(pilot / manifest_name),
                             'scorer_dependencies': request['approved_workspace'] + '/experiments/joint-kv-14b/longbench-deps-py312',
                             'longbench_repo': remote + '/input/scorer', 'gsm8k_shots': remote + '/input/five_train_examples.json',
                             'approved_data_manifest': remote + '/input/approved_data_manifest.json'},
                   research_reason='Check the production FA3 FP8-Q/FP8-KV candidate on frozen GSM8K64 and eight LongBench tasks x2; natural EOS, same scales, prefix and scheduler; pilot is not full quality certification.')
    if profile == 'xqa':
        request['workload']['kind'] = 'native_quality_xqa'
        request['operators'] = XQA_OPERATORS
        request['arms'] = ['fp8_e4m3', 'bf16']
        request['research_reason'] = ('Evaluate a deployable quality/performance candidate after full FA3 static-Q quality loss: '
            'BF16-Q FA2 paged prefill plus SM90 XQA decode on resident FP8 KV versus the existing FA3 BF16 baseline; '
            'declare internal FP8 query conversion; operator/partial-page/padded-Graph reference gate before real pilot80, '
            'natural EOS and unchanged production scheduler. No full quality certification from the pilot.')
    if fragment:
        if parent_audit is None:
            raise ValueError('Completion requires an independently verified parent audit')
        parent = json.loads(parent_audit.read_text())
        if parent['classification'] != 'confirmed_pilot_prediction_integrity_and_rescoring' or parent['samples_per_arm'] != 80:
            raise ValueError('Parent quality predictions have not been independently verified')
        if parent['production_configuration']['operators'] != request['operators']:
            raise ValueError('Completion cannot reuse a different operator candidate')
        request['workload'].update(reuse_run_id=parent['run_id'], parent_result_sha256=parent['terminal_result_sha256'],
                                   parent_audit_sha256=sha(parent_audit))
        files['input/parent_quality_audit.json'] = parent_audit.absolute()
        request['research_reason'] = ('Complete the declared production '+profile.upper()+' candidate on only the remaining '
            '1,639 original records; reuse the independently verified80 from this exact operator/engine configuration, '
            'then merge/rescore1719. No repeated pilot predictions or old-implementation quality reuse.')
    if recovery_audit is not None:
        recovered = json.loads(recovery_audit.read_text())
        if (profile != 'xqa' or not fragment or recovered.get('classification') !=
                'confirmed_recovered_fragment_arm_prediction_integrity_and_rescoring'
                or recovered.get('audited_arms') != ['fp8_e4m3'] or recovered.get('samples_per_arm') != 1639
                or recovered['production_configuration']['operators'] != request['operators']):
            raise ValueError('Only an independently recovered XQA FP8 fragment can unlock its missing BF16')
        request['arms'] = ['bf16']
        request['workload']['recovery'] = {'prior_run_id': recovered['run_id'], 'verified_arm': 'fp8_e4m3',
            'prior_terminal_result_sha256': recovered['terminal_result_sha256'],
            'arm_measurement_sha256': recovered['arm_measurement_sha256']['fp8_e4m3'],
            'recovered_audit_sha256': sha(recovery_audit)}
        files['input/recovered_fragment_audit.json'] = recovery_audit.absolute()
        request['research_reason'] = ('Generate only the missing BF16 reference1639 after independently recovering the complete '
            'FP8 fragment from its late-teardown CONFLICT. Preserve the original failure, never repeat FP8 predictions; '
            'same engine/input/operators and pilot80, then perform an explicit three-run1719 merge.')
    scientific = {key: request[key] for key in (
        'python_source_commit', 'binary_sha256', 'dataset_sha256', 'backend_source_sha256', 'image_id',
        'graph_mode', 'prefix_cache', 'fp8_scales', 'operators', 'service', 'concurrency')}
    scientific['workload'] = {key: request['workload'][key] for key in
                              ('kind', 'cohort', 'samples_per_arm', 'ignore_eos', 'source_content_sha256')}
    scientific['scorer_sha256'] = sha(scorer / 'LongBench/metrics.py')
    scientific['shots_sha256'] = sha(shots)
    if recovery_audit is not None:
        scientific['missing_arm_recovery'] = request['workload']['recovery']
        scientific['arms'] = request['arms']
    request['experiment_key'] = digest(scientific)
    package = {'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal', 'request': request,
               'gpu_job_launched': False,
               'assets': {name: {'local_path': str(path), 'sha256': sha(path)} for name, path in files.items()}}
    write_new_json(out / 'package.json', package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-package', 'old-quality', 'bundle', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--completion', type=Path)
    parser.add_argument('--parent-audit', type=Path)
    parser.add_argument('--profile', choices=('fa3', 'xqa'), default='fa3')
    parser.add_argument('--recovery-audit', type=Path)
    args = parser.parse_args()
    result = prepare(args.base_package, args.old_quality, args.run_id, args.bundle, args.out.absolute(), args.completion, args.parent_audit, args.profile, args.recovery_audit)
    print(json.dumps({'package': str(args.out / 'package.json'), 'experiment_key': result['request']['experiment_key'],
                      'gpu_job_launched': False, 'samples_per_arm': result['request']['workload']['samples_per_arm']}))


if __name__ == '__main__':
    main()
