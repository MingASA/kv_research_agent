#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Verify original heldout AIPerf/Graph and recompute bounded KV distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest,write_new_json
from scripts.controller.production import audit_requests,verify_backend,aiperf_service_command,_jsonl_rows,_verify_observation_rates
from scripts.controller.evidence import verify_allocation
from scripts.controller.range_probe import audit_export, CHAT_SCOPE, LONG_QA_SCOPE, validate_longcontext_inputs


def _sha(path: Path) -> str:
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):value.update(chunk)
    return value.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:raise ValueError(message)


def audit(mirror: Path,run_id: str,source: Path,prepared: Path,
          recover_postprocessing_failure: bool = False) -> dict:
    snapshot=json.loads((mirror/'SNAPSHOT.json').read_text())
    for row in snapshot['files']:
        path=mirror/row['relative_path']
        _require(path.resolve().is_relative_to(mirror.resolve()) and not path.is_symlink()
                 and path.stat().st_size==row['size'] and _sha(path)==row['sha256'],
                 'Mirrored original evidence differs')
    root=mirror/'support/agent_work/runs'/run_id
    ticket=json.loads((root/'ticket.json').read_text());request=ticket['request']
    terminal=json.loads((root/'result.json').read_text())
    recovered=terminal['status']=='FAILED' and recover_postprocessing_failure
    _require(request['run_id']==run_id and digest(request)==ticket['request_sha256']
        and terminal['request_sha256']==ticket['request_sha256'] and (terminal['status']=='COMPLETE' or recovered)
        and terminal['gpu_release_verified'] is True and terminal['live_owned_containers']==[]
        and request['arms']==['bf16'] and request['workload']['kind']=='heldout_kv_range_probe',
        'Missing exact operator/diagnostic scope or actual clean terminal seal')
    if recovered:
        _require(not (root/'COMPLETE').exists() and not (root/'output/bf16/measurement.json').exists()
            and 'Raw actual-token throughput mismatch' in (root/'output/bf16/container.log').read_text(),
            'Recovery is only the preserved rate-clock postprocessing failure with independently complete raw data')
    else:
        _require(json.loads((root/'COMPLETE').read_text())['result_sha256']==_sha(root/'result.json'),
            'COMPLETE seal differs')
    for name,expected in ticket['assets_sha256'].items():
        _require(_sha(root/name)==expected,'Frozen runtime/input asset differs: '+name)
    for name,expected in request['backend_source_sha256'].items():
        _require(_sha(source/name)==expected,'Independent scientific source differs: '+name)
    _require(_sha(root/'input/canonical_inputs.json')==request['dataset_sha256']
        and _sha(prepared/'canonical_inputs.json')==request['dataset_sha256']
        and _sha(root/'input/production_manifest.json')==_sha(prepared/'manifest.json'),
        'Original heldout prompt/selection bytes differ')
    manifest=json.loads((prepared/'manifest.json').read_text())
    content_scope = request['workload']['content_scope']
    if content_scope == LONG_QA_SCOPE:
        validate_longcontext_inputs(json.loads((prepared/'canonical_inputs.json').read_text()), manifest)
    else:
        _require(content_scope == CHAT_SCOPE and manifest['quality_labels_loaded'] is False
            and manifest['service_excluded_count']==528
            and len({r['session_id'] for r in manifest['selected_lineage']})==128,
            'Heldout input provenance/answer separation differs')
    output=root/'output/bf16';measurement=None
    if not recovered:
        measurement=json.loads((output/'measurement.json').read_text())
        _require(measurement['measurement_success'] is True and measurement['arm']=='bf16'
            and measurement['request_sha256']==ticket['request_sha256']
            and measurement['official_aiperf_complete'] is True and measurement['model_quality_validated'] is False
            and measurement['scope']=='production_KV_distribution_diagnostic_not_headline_performance'
            and terminal['arm_results']==[{'arm':'bf16','result_sha256':_sha(output/'measurement.json')}],
            'Missing AIPerf/diagnostic worker seal or inappropriate quality/performance certification')
        for name,expected in measurement['evidence_sha256'].items():
            path=output/name
            _require(path.resolve().is_relative_to(output.resolve()) and _sha(path)==expected,'Worker evidence differs')
    cli=json.loads((output/'aiperf_cli_CPU_acceptance.json').read_text())
    _require(cli.get('actual_Cyclopts_CLIConfig_accepted') is True
        and cli.get('cli_config_sha256')==request['workload']['aiperf_cli_config_sha256']
        and cli.get('aiperf_version')=='0.13.0' and cli.get('warmup_request_count') is None
        and cli.get('request_count')==128 and cli.get('concurrency')==128
        and cli.get('benchmark_started') is False and cli.get('GPU_used') is False
        and json.loads((output/'aiperf_command.json').read_text())==aiperf_service_command(request,Path('/results')),
        'Actual own-environment full CLI/zero-warmup configuration or benchmark argv differs')
    native=json.loads((output/'operator_acceptance.json').read_text())
    probe=json.loads((output/'range_operator/acceptance.json').read_text())
    _require(native['real_capture_and_replay'] is True and native['padded_graph_bucket_checked'] is True
        and probe['model_loaded'] is False and probe['service_or_model_quality_certified'] is False
        and all(probe.get(k) is True for k in (
            'exact_integer_counts_match_Torch_QDQ','FP32_moments_match_Torch_QDQ',
            'nonpositive_padding_NaN_Inf_excluded','valid_nonfinite_values_counted',
            'actual_capture_and_changed_input_replay','reset_excludes_warmup_and_preserves_Graph_buffers',
            'actual_quiescent_reset_export_thread_handshake','layers_and_phases_separate')),
        'Missing actual small numerical/Graph/reset/export admission')
    backend=verify_backend((output/'server.log').read_text(),'bf16',request['operators']['bf16'])
    allocation=verify_allocation(json.loads((output/'allocation.json').read_text()),'bf16')
    metrics=audit_requests(output/'artifacts',root/'input/canonical_inputs.json',128,0)
    if measurement is not None:
        _require(metrics==measurement['metrics'],'Independent actual AIPerf raw/usage outcomes differ')
    control=output/'range_probe'
    reset=json.loads((control/'RESET_DONE.json').read_text())
    exported=json.loads((control/'EXPORT_DONE.json').read_text())
    _require(exported.get('scope')=='BF16_valid_cache_writes_QDQ_proxy_not_FP8_generation_or_quality',
        'Actual activation observation scope differs')
    for name,response in (('RESET',reset),('EXPORT',exported)):
        command=json.loads((control/(name+'.json')).read_text())
        _require(command=={'command':name,'run_id':run_id,'request_sha256':ticket['request_sha256']}
            and response['command_sha256']==_sha(control/(name+'.json'))
            and all(response[k]==v for k,v in command.items()),'Actual quiescent command epoch/SHA differs')
    distribution=audit_export(exported,ticket['request_sha256'],run_id,reset['reset_monotonic_ns'],content_scope)
    if measurement is not None:
        _require(distribution==measurement['KV_distribution_diagnostic'],'Worker distribution differs')
    _require(distribution==json.loads((control/'audit.json').read_text()),'Raw counter-derived distribution differs')
    starts=[]
    for row in _jsonl_rows(output/'artifacts/profile_export_raw.jsonl'):
        starts.append(row['start_perf_ns'])
    _require(starts and 0<=min(starts)-reset['reset_monotonic_ns']<1800*10**9,
        'Actual requests did not begin after the acknowledged reset in the common Linux monotonic clock')
    summary=json.loads((output/'artifacts/profile_export_aiperf.json').read_text())
    clock=_verify_observation_rates(summary,metrics['actual_output_tokens'],128)
    return {'classification':'confirmed_raw_heldout_BF16_KV_distribution_and_AIPerf_integrity_not_FP8_quality',
        'original_root_status':terminal['status'],'postprocessing_failure_recovered_independently':recovered,
        'worker_measurement_seal_present':measurement is not None,'original_raw_or_terminal_modified':False,
        'run_id':run_id,'source_commit':request['python_source_commit'],
        'dataset_sha256':request['dataset_sha256'],'binary_sha256':request['binary_sha256'],
        'model_config_sha256':request['model_config_sha256'],
        'files_SHA_verified':len(snapshot['files']),'bytes_SHA_verified':sum(r['size'] for r in snapshot['files']),
        'terminal_result_sha256':_sha(root/'result.json'),'backend':backend,'allocation':allocation,
        'actual_requests_start_after_reset_verified':True,'actual_original_requests':128,
        'actual_CLI_config_source_sha256':cli['cli_config_sha256'],
        'rate_clock_precision_verification':clock,
        'actual_output_tokens':metrics['actual_output_tokens'],'finish_reason_counts':metrics['finish_reason_counts'],
        **({'content_scope':LONG_QA_SCOPE, 'input_prompt_tokens_cpu':manifest['prompt_tokens_cpu'],
            'exact_previous_evaluation_overlap':manifest['overlap_with_previous_evaluation'],
            'official_generation_caps_preserved':True} if content_scope == LONG_QA_SCOPE else {}),
        'distribution':distribution,'rows':exported['rows'],'model_quality_certified':False,
        'new_scales_derived':False,'headline_performance_result':False,
        'limits':['BF16 cache-write activations; FP8 generation feedback and internal-Q quantization not measured',
            ('Heldout real long-document QA only; no code or semantic-independence coverage'
             if content_scope == LONG_QA_SCOPE else
             'Heldout short chat32-2980 CPU estimated prompt tokens, not32K or code coverage'),
            'Counts describe actual executed valid writes, not every logical prefix token or unique text',
            'Prefill phase includes chunked/mixed batches; decode is the pure decode path',
            'Diagnostic kernels add overhead; AIPerf timing is not a formal new speed result']}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('mirror','source','prepared','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--recover-postprocessing-failure',action='store_true')
    args=parser.parse_args()
    result=audit(args.mirror,args.run_id,args.source,args.prepared,args.recover_postprocessing_failure);write_new_json(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k in ('classification','run_id','files_SHA_verified','actual_output_tokens','distribution')}))


if __name__=='__main__':main()
