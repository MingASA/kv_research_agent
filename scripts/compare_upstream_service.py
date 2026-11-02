"""Pair the three original real service cases after independent raw recomputation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.audit_upstream_service import audit
from scripts.controller.durable import write_new_json
from scripts.controller.upstream_qualification import BASELINE_COMMIT, OFFICIAL_COMMIT
from scripts.controller.upstream_service import CASES


def bound(audit_path, manifest_path):
    recorded = json.loads(audit_path.read_bytes())
    recomputed = audit(manifest_path)
    if recorded != recomputed:
        raise ValueError('Service audit changed or does not recompute from its original complete raw evidence')
    manifest = json.loads(manifest_path.read_bytes())
    request = json.loads((Path(manifest['mirror_directory'])/'files/ticket.json').read_bytes())['request']
    return recorded, request


def compare(reference_path, reference_manifest, candidate_path, candidate_manifest):
    left, a = bound(reference_path, reference_manifest)
    right, c = bound(candidate_path, candidate_manifest)
    for name in ('model_id', 'model_path', 'model_config_sha256', 'image_id', 'gpu_uuids',
                 'service', 'graph_mode', 'prefix_cache', 'upstream_commit', 'benchmark_cases'):
        if a[name] != c[name]:
            raise ValueError('Service hardware/model/mechanism differs: '+name)
    if (a['arms'] != ['bf16'] or c['arms'] not in (['fp8_e4m3'], ['nvfp4'])
            or a['server_source_commit'] != BASELINE_COMMIT or c.get('native_base_commit') != BASELINE_COMMIT
            or a['upstream_commit'] != OFFICIAL_COMMIT or a.get('native_configuration') is not None
            or left['input_files_sha256'] != right['input_files_sha256']):
        raise ValueError('Require frozen official A and native C with identical original inputs')
    comparisons = {}
    rows = []
    for name, _, _, _, concurrency in CASES:
        av, cv = left['cases'][name], right['cases'][name]
        ap, cp = av['profile'], cv['profile']
        for field in ('profiling_records', 'profile_session_ids_sha256', 'profile_prompt_counts_sha256'):
            if ap[field] != cp[field]:
                raise ValueError('Actual offered request/token cohort differs: '+name+'/'+field)
        row = {'case': name, 'candidate': right['arm'], 'client_concurrency': concurrency,
               'A_RPS': ap['request_rps'], 'C_RPS': cp['request_rps'],
               'RPS_speedup': cp['request_rps']/ap['request_rps'],
               'A_output_TPS': ap['output_tps'], 'C_output_TPS': cp['output_tps'],
               'output_TPS_speedup': cp['output_tps']/ap['output_tps'],
               'A_actual_output_tokens': ap['actual_output_tokens'], 'C_actual_output_tokens': cp['actual_output_tokens'],
               'A_mean_output_tokens': ap['osl_mean'], 'C_mean_output_tokens': cp['osl_mean'],
               'A_window_seconds': ap['observation_seconds'], 'C_window_seconds': cp['observation_seconds'],
               'A_empty_answers': av['Goodput']['empty_or_whitespace_answers'],
               'C_empty_answers': cv['Goodput']['empty_or_whitespace_answers']}
        for field in ('ttft_p50_ms', 'ttft_p95_ms', 'ttft_p99_ms', 'itl_p50_ms', 'itl_p95_ms',
                      'itl_p99_ms', 'e2e_p50_ms', 'e2e_p95_ms', 'e2e_p99_ms', 'gpu_memory_used_mib_max'):
            row['A_'+field] = ap[field];row['C_'+field] = cp[field]
        for key in ('num_running_requests', 'num_waiting_requests', 'num_used_blocks'):
            row['A_'+key+'_sampled_max'] = av['application']['ranges'][key]['max']
            row['C_'+key+'_sampled_max'] = cv['application']['ranges'][key]['max']
        curves = []
        for al, cl in zip(av['Goodput']['points'], cv['Goodput']['points']):
            if any(al[k] != cl[k] for k in ('TTFT_limit_ms', 'request_mean_ITL_limit_ms')):
                raise ValueError('Predeclared Goodput thresholds differ')
            curves.append({'TTFT_limit_ms': al['TTFT_limit_ms'], 'request_mean_ITL_limit_ms': al['request_mean_ITL_limit_ms'],
                           'A_goodput_RPS': al['goodput_rps'], 'C_goodput_RPS': cl['goodput_rps']})
        comparisons[name] = {'rates_and_tails': row, 'Goodput_curve': curves,
                             'A_application': av['application'], 'C_application': cv['application']}
        rows.append(row)
    result = {'schema': 'latest-upstream-real-service-pair-v1', 'reference_run': a['run_id'],
        'candidate_run': c['run_id'], 'candidate_arm': right['arm'], 'same_frozen_official_base': OFFICIAL_COMMIT,
        'same_physical_GPU': a['gpu_uuids'], 'same_weights_model_path': a['model_path'], 'same_image': a['image_id'],
        'service': a['service'], 'native_configuration': c['native_configuration'], 'cases': comparisons,
        'source_audit_SHA256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (reference_path, candidate_path)},
        'all_original_requests_and_complete_measured_windows_used': True,
        'ITL_tail_is_request_mean_not_individual_token_interval': True,
        'quality_or_business_SLO_accepted': False, 'maximum_supported_capacity_claim': False,
        'cross_version_D_speedup_claim': False}
    return result, rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'reference-manifest', 'candidate', 'candidate-manifest', 'out', 'csv'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result, rows = compare(args.reference, args.reference_manifest, args.candidate, args.candidate_manifest)
    write_new_json(args.out, result)
    with args.csv.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({'candidate': result['candidate_arm'], 'cases': len(rows)}))
