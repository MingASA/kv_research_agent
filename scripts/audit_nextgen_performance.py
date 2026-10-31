"""Independently recompute real-text AIPerf results from a complete run mirror.

The service audit proves lifecycle/Graph/resident storage. This additional audit
requires both actual external-warmup and measured raw exports; optimistic worker
summaries alone cannot become a performance result. Failed runs remain failed.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

from scripts.audit_nextgen_validation import _check, _mirror_paths, audit as audit_service
from scripts.controller.durable import write_new_json
from scripts.controller.nextgen_performance import PERF_KIND, aiperf_command, verify_frozen_case, validate_cli_protocol_proof
from scripts.controller.production import audit_requests


def _read(path: Path):
    return json.loads(path.read_text())


def audit_phase(directory: Path, source: Path, *, expected_requests: int, phase: str,
                command: list[str], cli_config_sha256: str) -> dict:
    """Recompute payloads, wire completion, usage, clocks, rates, tails and GPU samples."""
    _check(_read(directory/'command.json')==command,'Actual AIPerf argv differs from its reviewed production case')
    cli=_read(directory/'aiperf_cli_CPU_acceptance.json')
    _check(cli.get('actual_Cyclopts_CLIConfig_accepted') is True and cli.get('aiperf_version')=='0.13.0'
           and cli.get('cli_config_sha256')==cli_config_sha256 and cli.get('request_count')==expected_requests
           and cli.get('concurrency')==int(command[command.index('--concurrency')+1])
           and cli.get('conversation_num')==(int(command[command.index('--num-sessions')+1]) if '--num-sessions' in command else None)
           and cli.get('warmup_request_count') is None and cli.get('benchmark_started') is False
           and cli.get('GPU_used') is False,'Actual CLI parser/config/warmup admission differs')
    recomputed=audit_requests(directory/'artifacts',source,expected_requests,0,allow_multi_turn=True)
    recorded=_read(directory/'independent_worker_audit.json')
    start=recorded.get('worker_observation_started_monotonic_ns')
    end=recorded.get('worker_observation_ended_monotonic_ns')
    _check(type(start) is int and type(end) is int and 0<start<end,'Invalid bounded worker observation clock')
    recomputed.update(phase=phase,worker_observation_started_monotonic_ns=start,
                      worker_observation_ended_monotonic_ns=end)
    _check(recomputed==recorded,'Cached worker performance differs from independently recomputed actual raw data')
    return recomputed


def audit(manifest_path: Path, *, source_repo: Path | None = None) -> dict:
    lifecycle=audit_service(manifest_path)
    manifest=_read(manifest_path);mirror,root=_mirror_paths(manifest)
    ticket=_read(root/'ticket.json');request=ticket['request'];arm=request['arms'][0]
    _check(request['workload']['kind']==PERF_KIND,'A correctness/quality run is not an AIPerf performance run')
    # Every frozen input/runtime file must be present in the SHA-verified mirror,
    # rather than silently taking an unlisted file from a previous local mirror.
    verified_paths={(mirror/row['base_rel']).resolve(strict=True) for row in manifest['files']}
    for name,sha in ticket['assets_sha256'].items():
        path=root/name
        _check(path.resolve(strict=True) in verified_paths and hashlib.sha256(path.read_bytes()).hexdigest()==sha,
               'Frozen input/runtime asset missing from independent mirror identity: '+name)
    output=root/'output'/arm;measurement=_read(output/'measurement.json')
    _check(measurement.get('measurement_success') is True and measurement.get('service_performance_measured') is True
           and measurement.get('official_aiperf_complete') is True,'Worker did not complete the actual production benchmark')
    config=request['performance_evaluation'];cohort=root/'input/production'
    case=verify_frozen_case(cohort,config['case'],model_id=request['model_id'])
    _check(case['input_manifest_sha256']==config['input_manifest_sha256']
           and case['concurrency']==config['concurrency'],'Original real-text case/demand identity differs')
    quality_path=root/'input/performance_quality_scores.json';quality=_read(quality_path)
    _check(hashlib.sha256(quality_path.read_bytes()).hexdigest()==config['quality_scores_sha256']
           and quality.get('prediction_count')==1720
           and quality.get('all_raw_API_and_usage_contracts_independently_verified') is True,
           'Full independent quality snapshot missing or changed')
    for key in ('model_id','binary_sha256','server_source_commit','operators','service'):
        _check(quality.get(key)==request[key],'Quality and performance deployment differ: '+key)
    _check(quality.get('arm')==arm and quality.get('kv_scales')==request.get('kv_scales'),
           'Quality snapshot uses another KV format/scale')
    if quality.get('python_source_commit')!=request['python_source_commit']:
        from scripts.controller.model_source_equivalence import verify_model_source_equivalence
        _check(source_repo is not None,'Independently inspect model source commits before reusing older quality evidence')
        equivalence=verify_model_source_equivalence(source_repo,quality['python_source_commit'],request['python_source_commit'])
        path=root/'quality_model_source_equivalence.json'
        _check(path.resolve(strict=True) in verified_paths and _read(path)==equivalence,
               'Actual quality source-equivalence proof differs or was omitted from mirror')
    proof_path=root/'input/full_CLI_protocol_CPU_audit.json';proof=_read(proof_path)
    _check(hashlib.sha256(proof_path.read_bytes()).hexdigest()==config['full_CLI_protocol_CPU_audit_sha256']
           and proof.get('all_checks_pass') is True and proof.get('AIPerf_version')=='0.13.0'
           and proof.get('observed_wire_request_total')==194,'Full CLI/session protocol prerequisite changed')
    session_limited=proof.get('schema')=='nextgen-full-cli-session-stop-v2'
    validate_cli_protocol_proof(proof,require_session_limit=session_limited)
    phases={}
    for phase in ('warmup','measured'):
        expected=case[phase];directory=output/phase
        for name in ('command.json','aiperf_cli_CPU_acceptance.json','independent_worker_audit.json','gpu_telemetry.csv',
                     'artifacts/profile_export.jsonl','artifacts/profile_export_raw.jsonl','artifacts/profile_export_aiperf.json'):
            _check((directory/name).resolve(strict=True) in verified_paths,'Actual AIPerf evidence omitted from mirror: '+phase+'/'+name)
        concurrency=case['warmup_concurrency'] if phase=='warmup' else case['concurrency']
        command=aiperf_command(request['model_id'],Path(request['remote_output'])/'input/production'/expected['filename'],
                               Path('/results')/phase/'artifacts',requests=expected['requests'],concurrency=concurrency,
                               sessions=expected['sessions'] if session_limited else None)
        phases[phase]=audit_phase(directory,cohort/expected['filename'],expected_requests=expected['requests'],phase=phase,
                                 command=command,cli_config_sha256=config['aiperf_cli_config_sha256'])
    _check(phases['warmup']['worker_observation_ended_monotonic_ns']<=phases['measured']['worker_observation_started_monotonic_ns'],
           'Measured CLI overlapped external warmup')
    _check(phases['measured']['worker_observation_ended_monotonic_ns']-phases['warmup']['worker_observation_started_monotonic_ns']
           <=config['max_seconds']*10**9,'Actual production measurement exceeded its bounded window')
    phase_times={}
    for phase in phases:
        summary=_read(output/phase/'artifacts/profile_export_aiperf.json')
        phase_times[phase]=[datetime.datetime.fromisoformat(summary[key].replace('Z','+00:00')) for key in ('start_time','end_time')]
    _check(phase_times['warmup'][1]<=phase_times['measured'][0],'Actual AIPerf measurement clock overlaps warmup')
    performance=measurement['performance']
    _check(performance.get('case')==config['case'] and performance.get('source_manifest_sha256')==case['input_manifest_sha256']
           and performance.get('warmup_is_external_and_unmeasured') is True
           and performance.get('all_actual_payload_and_SSE_and_usage_and_counts_verified') is True
           and performance.get('official_AIPerf_complete') is True
           and performance.get('profile')==phases['measured'] and performance.get('external_warmup')==phases['warmup'],
           'Cached headline/warmup result differs from original complete raw observations')
    return {'schema':'kv-nextgen-independent-production-performance-audit-v1','run_id':manifest['run_id'],'arm':arm,
            'source_SHA_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'classification':'confirmed_actual_model_AIPerf_real_text_production_performance',
            'request_identity':{key:request[key] for key in ('model_id','model_config_sha256','image_id','binary_sha256','server_source_commit',
                               'python_source_commit','gpu_uuids','service','operators','graph_mode','prefix_cache')},
            'kv_scales':request.get('kv_scales'),
            'case':config['case'],'concurrency':config['concurrency'],'input_manifest_sha256':case['input_manifest_sha256'],
            'lifecycle':lifecycle,'profile':phases['measured'],'external_warmup':phases['warmup'],
            'actual_AIPerf_phase_times':{key:[value.isoformat() for value in times] for key,times in phase_times.items()},
            'offered_requests':case['measured']['requests'],'successful_requests':phases['measured']['profiling_records'],
            'all_offered_requests_completed':True,'actual_model_and_Graph_and_raw_service_performance_verified':True,
            'quality_snapshot_reference_sha256':config['quality_scores_sha256'],'production_quality_accepted':False,
            'SLO_goodput_measured':False,'GPU_memory_utilization_is_bandwidth_GBps':False,
            'max_successful_admission_capacity_measured':False,
            'limitations':['One declared closed-loop real-text workload, not a production arrival trace.',
                           'Natural output lengths are reported; quality acceptance and SLO thresholds are not assumed.',
                           'GPU memory utilization is an activity percentage; no HBM bandwidth counter is measured.']}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-repo',type=Path)
    args=parser.parse_args();result=audit(args.sha_manifest,source_repo=args.source_repo)
    write_new_json(args.out,result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
