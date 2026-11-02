"""Recompute frozen upstream service rates, tails, residency gauges and Goodput."""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from scripts.audit_upstream_run import audit as audit_lifecycle
from scripts.audit_upstream_model import native_runtime
from scripts.controller.durable import write_new_json
from scripts.controller.nextgen_performance import aiperf_command
from scripts.controller.production import audit_requests
from scripts.controller.upstream_service import KIND, CASES
from scripts.controller.upstream_qualification import server_command


def read(path):
    return json.loads(path.read_bytes())


def jsonl(path):
    return [json.loads(line) for line in path.read_bytes().split(b'\n') if line]


def goodput(directory, profile):
    metrics = {r['metadata']['x_request_id']: r for r in jsonl(directory/'artifacts/profile_export.jsonl')}
    content = {}
    for row in jsonl(directory/'artifacts/profile_export_raw.jsonl'):
        pieces = []
        for response in row['responses']:
            for packet in response.get('packets', []):
                if packet.get('name') != 'data' or packet.get('value') == '[DONE]':
                    continue
                for choice in json.loads(packet['value']).get('choices', []):
                    text = choice.get('delta', {}).get('content', choice.get('text', ''))
                    if text is not None:
                        if not isinstance(text, str):
                            raise ValueError('Unexpected streamed answer type')
                        pieces.append(text)
        content[row['metadata']['x_request_id']] = ''.join(pieces)
    if set(content) != set(metrics):
        raise ValueError('Goodput raw answer inventory differs from audited requests')
    curve = []
    for ttft in (1000, 5000, 10000, 30000):
        for itl in (25, 50, 100, 200):
            accepted = 0
            for identity, row in metrics.items():
                value = row['metrics']
                tokens = value['output_token_count']['value']
                if (content[identity].strip() and value['time_to_first_token']['value'] <= ttft
                        and (tokens == 1 or value['inter_token_latency']['value'] <= itl)):
                    accepted += 1
            curve.append({'TTFT_limit_ms': ttft, 'request_mean_ITL_limit_ms': itl,
                          'useful_completed_requests': accepted,
                          'goodput_rps': accepted/profile['observation_seconds']})
    return {'predeclared_sensitivity_curve_not_business_SLO': True, 'points': curve,
            'empty_or_whitespace_answers': sum(not v.strip() for v in content.values()),
            'all_intended_requests_remain_denominator': len(metrics),
            'ITL_statistic': 'per-request mean, one-token interval vacuous'}


def gauges(directory, request, pages, profile):
    records = jsonl(directory/'application_metrics.jsonl')
    header, rows = records[0], records[1:]
    if (header['run_id'] != request['run_id'] or header['physical_pages'] != pages
            or header['accounting_source'] != 'upstream261_effective_usage_excludes_padding'
            or not rows or any(a['monotonic_ns'] >= b['monotonic_ns'] for a, b in zip(rows, rows[1:]))):
        raise ValueError('Application observations changed source, capacity or clock')
    start = profile['worker_observation_started_monotonic_ns']
    end = profile['worker_observation_ended_monotonic_ns']
    requests = jsonl(directory/'artifacts/profile_export.jsonl')
    request_start = min(r['metadata']['request_start_ns'] for r in requests)/1e9
    request_end = max(r['metadata']['request_end_ns'] for r in requests)/1e9
    if not request_start < request_end:
        raise ValueError('Invalid actual request observation window')
    during = [r for r in rows if start <= r['monotonic_ns'] <= end
              and request_start <= datetime.datetime.fromisoformat(r['utc']).timestamp() <= request_end]
    ranges = {}
    for name in header['units']:
        values = [r['values'][name] for r in during if name in r['values']]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Nonfinite or negative application gauge')
        ranges[name] = {'min': min(values), 'max': max(values), 'mean': statistics.mean(values)} if values else None
    window = read(directory/'prefix_counter_window.json')
    hits = window['after']['prefix_cache_hit_tokens_total']-window['before']['prefix_cache_hit_tokens_total']
    prompts = window['after']['prefix_cache_prompt_tokens_total']-window['before']['prefix_cache_prompt_tokens_total']
    expected = int(round(profile['isl_mean']*profile['profiling_records']))
    if hits != window['hit_tokens_delta'] or prompts != window['prompt_tokens_delta'] or hits < 0 or prompts <= 0:
        raise ValueError('Prefix admission counter window differs')
    return {'samples': len(rows), 'samples_during_client_window': len(during),
            'sample_window': 'first actual AIPerf request start through last request end; CLI preparation excluded',
            'error_samples': sum(bool(r['errors']) for r in rows), 'ranges': ranges,
            'real_prefix_hit_tokens': hits, 'offered_prefix_admission_tokens': prompts,
            'actual_successful_prompt_tokens': expected, 'counter_span_exact_prompt_match': prompts == expected,
            'cross_request_hit_fraction': hits/prompts,
            'physical_Full_KV_pages': pages, 'Full_KV_token_slots': pages*request['service']['block_size'],
            'API_cached_tokens_used': False, 'raw_HTML_independently_reparsed': False,
            'limitations': ['One-second sequential samples can miss peaks.',
                           'Prefix blocks can overlap active usage and include GDN checkpoints; do not sum these gauges.',
                           'Gauge values are bound to the frozen collector and SHA; original HTML is not retained.']}


def audit(manifest_path: Path):
    lifecycle = audit_lifecycle(manifest_path)
    manifest = read(manifest_path)
    root = Path(manifest['mirror_directory'])/'files'
    ticket = read(root/'ticket.json')
    request = ticket['request']
    arm = request['arms'][0]
    if request['workload']['kind'] != KIND or request.get('graph_observer') is not None:
        raise ValueError('Require the observer-free frozen upstream production workload')
    output = root/'output'/arm
    results = {}
    for name, dataset, measured, warmup, concurrency in CASES:
        directory = output/name
        command = read(directory/'server_command.json')
        if command != server_command(request, Path('/results')/name):
            raise ValueError('Actual formal command differs from the qualified normal/native path')
        log = (directory/'server.log').read_text()
        runtime = native_runtime(log, request.get('native_configuration'))
        capacities = re.findall(r'kv cache capacity: [^\n]*?blocks: (\d+), slot_size:', log)
        if len(capacities) != 1:
            raise ValueError('No unique actual Full-KV capacity')
        pages = int(capacities[0])
        phases = {}
        windows = {}
        for phase, source_name, count in [('warmup', warmup, 16 if dataset == 'sharegpt' else 4),
                                          ('measured', measured, 512 if dataset == 'sharegpt' else 64)]:
            location = directory/phase
            source = root/'input/production'/dataset/(source_name+'.json')
            expected_command = aiperf_command(request['model_id'],
                Path(request['remote_output'])/'input/production'/dataset/(source_name+'.json'),
                Path('/results')/name/phase/'artifacts', requests=count, concurrency=min(count, concurrency),
                sessions=count, latest_upstream=True)
            cli = read(location/'aiperf_cli_CPU_acceptance.json')
            if (read(location/'command.json') != expected_command
                    or cli['request_count'] != count or cli['conversation_num'] != count
                    or cli['concurrency'] != min(count, concurrency) or cli['warmup_request_count'] is not None
                    or cli['cli_config_sha256'] != request['service_benchmark']['aiperf_cli_config_sha256']):
                raise ValueError('Actual command/session/warmup semantics differ')
            recomputed = audit_requests(location/'artifacts', source, count, 0, allow_multi_turn=True)
            cached = read(location/'independent_worker_audit.json')
            recomputed.update(all_official_tokenizer_and_server_prompt_counts_match=True,
                worker_observation_started_monotonic_ns=cached['worker_observation_started_monotonic_ns'],
                worker_observation_ended_monotonic_ns=cached['worker_observation_ended_monotonic_ns'])
            if recomputed != cached:
                raise ValueError('Worker headline differs from independently recomputed raw API evidence')
            if dataset == 'sharegpt':
                counts = {r['session_id']: r['prompt_tokens'] for r in read(
                    root/'input/production/sharegpt/retokenized_Qwen38_CPU-a1.json')['requests'] if r['phase'] == source_name}
            else:
                counts = {r['session_id']: r['prompt_tokens_cpu'] for r in read(
                    root/'input/production/govreport/manifest.json')['files'][source_name]['lineage']}
            observed = {r['metadata']['conversation_id']: int(r['metrics']['usage_prompt_tokens']['value'])
                        for r in jsonl(location/'artifacts/profile_export.jsonl')}
            if observed != counts:
                raise ValueError('Server prompt tokens differ from fixed official27B tokenizer')
            summary = read(location/'artifacts/profile_export_aiperf.json')
            windows[phase] = [datetime.datetime.fromisoformat(summary[k]) for k in ('start_time', 'end_time')]
            phases[phase] = recomputed
        if windows['warmup'][1] > windows['measured'][0]:
            raise ValueError('Warmup overlaps actual measurement')
        results[name] = {'profile': phases['measured'], 'external_warmup': phases['warmup'],
                         'actual_native_runtime': runtime,
                         'Goodput': goodput(directory/'measured', phases['measured']),
                         'application': gauges(directory/'measured', request, pages, phases['measured'])}
    return {'schema': 'upstream-real-service-independent-audit-v1', 'run_id': request['run_id'], 'arm': arm,
            'lifecycle': lifecycle, 'native_configuration': request.get('native_configuration'),
            'source_identity': {k: request[k] for k in ('model_id', 'model_config_sha256', 'image_id',
                'server_source_commit', 'binary_sha256', 'gpu_uuids', 'service', 'operators', 'upstream_commit')},
            'input_files_sha256': request['service_benchmark']['input_files_sha256'], 'cases': results,
            'SHA_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'quality_or_business_SLO_acceptance': False, 'full_formal_Graph_coverage_claim': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.sha_manifest)
    write_new_json(args.out, result)
    print(json.dumps({'run_id': result['run_id'], 'cases': list(result['cases']), 'raw_audit_complete': True}))
