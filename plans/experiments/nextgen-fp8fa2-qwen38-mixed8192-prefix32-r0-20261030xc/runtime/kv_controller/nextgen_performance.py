"""Production text workload contracts and exact AIPerf command construction.

Only the existing frozen real-text cohorts are admitted. A request cap is a
safety limit, never an instruction to ignore EOS or generate a fixed length.
Warmup is a separate dataset/CLI invocation, outside measured artifacts.
"""

from __future__ import annotations

import hashlib
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import time

from .durable import digest,write_new_json


PERF_KIND='native_hybrid_production_aiperf'
CASES={'chat-c4':('chat128.json','warmup_chat16.json',4),
       'chat-c16':('chat128.json','warmup_chat16.json',16),
       'long-unique-c8':('long_unique16.json','warmup_long2.json',8),
       'long-unique-c16':('long_unique16.json','warmup_long2.json',16),
       'long-prefix-c16':('long_prefix16x2.json','warmup_long2.json',16)}
EXPECTED_COUNTS={'chat128.json':(128,128),'warmup_chat16.json':(16,16),
                 'long_unique16.json':(16,16),'long_prefix16x2.json':(16,32),'warmup_long2.json':(2,2)}
SESSION_REPAIR='aiperf_frozen_session_limit_v2'


def validate_session_recovery(prior: dict, result: dict, current: dict, failure: dict,
                              result_sha256: str) -> None:
    recovery=current.get('session_protocol_recovery',{})
    ignored={'run_id','remote_output','python_source_repo','dataset_path','replicate',
             'experiment_key','research_reason','performance_evaluation','session_protocol_recovery'}
    old_config={k:v for k,v in prior.get('performance_evaluation',{}).items()
                if k!='full_CLI_protocol_CPU_audit_sha256'}
    new_config={k:v for k,v in current.get('performance_evaluation',{}).items()
                if k!='full_CLI_protocol_CPU_audit_sha256'}
    if (prior.get('replicate')!=0 or current.get('replicate')!=1
            or prior.get('workload',{}).get('kind')!=PERF_KIND
            or old_config.get('case')!='long-prefix-c16' or old_config!=new_config
            or {k:v for k,v in prior.items() if k not in ignored}!={k:v for k,v in current.items() if k not in ignored}
            or result.get('status')!='FAILED' or result.get('gpu_release_verified') is not True
            or result.get('live_owned_containers')!=[] or result.get('arm_results')!=[]
            or result.get('run_id')!=prior['run_id'] or result.get('request_sha256')!=digest(prior)
            or recovery.get('prior_run_id')!=prior['run_id']
            or recovery.get('prior_request_sha256')!=digest(prior)
            or recovery.get('prior_result_sha256')!=result_sha256
            or recovery.get('repair_kind')!=SESSION_REPAIR
            or recovery.get('CPU_repair_evidence_sha256')!=current['performance_evaluation']['full_CLI_protocol_CPU_audit_sha256']
            or failure.get('run_id')!=prior['run_id'] or failure.get('terminal')!='FAILED'
            or failure.get('GPU_release_verified') is not True
            or failure.get('valid_intended_performance_measurement') is not False
            or failure.get('actual_rows')!=32 or failure.get('expected_unique_session_turns')!=32
            or not 0<failure.get('observed_unique_session_turns',0)<32
            or not failure.get('duplicate_session_turns') or not failure.get('missing_session_turns')):
        raise ValueError('Session protocol recovery requires the exact released failed cohort and unchanged deployment')


def verify_session_recovery(request: dict, root: Path, uid: int) -> None:
    recovery=request.get('session_protocol_recovery')
    if recovery is None:return
    def read(path: Path) -> bytes:
        if (path.resolve(strict=True)!=path or path.stat().st_uid!=uid
                or not path.is_file() or path.stat().st_size>64*2**20):
            raise ValueError('Unsafe session protocol recovery evidence')
        return path.read_bytes()
    old=Path(request['approved_workspace'])/'agent_work/runs'/recovery['prior_run_id']
    ticket=json.loads(read(old/'ticket.json'));prior=ticket['request']
    result_raw=read(old/'result.json');result=json.loads(result_raw)
    failure_raw=read(root/'input/performance_session_failure_audit.json');failure=json.loads(failure_raw)
    if (hashlib.sha256(failure_raw).hexdigest()!=recovery['failure_audit_sha256']
            or ticket['request_sha256']!=digest(prior) or prior['remote_output']!=str(old)
            or (old/'COMPLETE').exists()):
        raise ValueError('Original session protocol failure identity changed')
    validate_session_recovery(prior,result,request,failure,hashlib.sha256(result_raw).hexdigest())
    output=old/'output'/prior['arms'][0]
    raw=read(output/'measured/artifacts/profile_export.jsonl')
    source=read(old/'input/production/long_prefix16x2.json')
    rows=[json.loads(line) for line in raw.split(b'\n') if line]
    expected={(s['session_id'],i) for s in json.loads(source)['data'] for i in range(len(s['payloads']))}
    counts=Counter((r['metadata']['conversation_id'],r['metadata']['turn_index']) for r in rows)
    duplicates=[{'session':s,'turn':i,'count':n} for (s,i),n in sorted(counts.items()) if n>1]
    missing=[{'session':s,'turn':i} for s,i in sorted(expected-counts.keys())]
    validation=json.loads(read(output/'validation_result.json'))
    if (hashlib.sha256(raw).hexdigest()!=failure['profile_export_sha256']
            or hashlib.sha256(source).hexdigest()!=failure['source_sha256']
            or len(rows)!=32 or len(expected)!=32 or len(counts)!=failure['observed_unique_session_turns']
            or set(counts)-expected or duplicates!=failure['duplicate_session_turns']
            or missing!=failure['missing_session_turns']
            or len({r['metadata']['x_request_id'] for r in rows})!=32
            or any(r.get('error') is not None or r['metadata']['benchmark_phase']!='profiling'
                   or r['metadata']['was_cancelled'] is not False for r in rows)
            or any(validation.get(name) is not True for name in ('model_loaded','actual_model_Graph_replay_logged','A_B_A_repeated_answers_match'))
            or validation.get('known_answers_checked')!=20
            or b'Duplicated/missing actual request/session/phase' not in read(output/'container.log')):
        raise ValueError('Original raw cohort does not reproduce the bound session-protocol failure')


def validate_cli_protocol_proof(proof: dict, *, require_session_limit: bool=False) -> None:
    if (proof.get('all_checks_pass') is not True or proof.get('AIPerf_version')!='0.13.0'
            or proof.get('observed_wire_request_total')!=194):
        raise ValueError('Full installed CLI and frozen session protocol lack verified194-turn evidence')
    if require_session_limit and (
            proof.get('schema')!='nextgen-full-cli-session-stop-v2'
            or proof.get('all_cases_session_limit_verified') is not True
            or proof.get('heterogeneous_latency_verified') is not True
            or proof.get('multi_turn_concurrency')!=16
            or proof.get('request_count_only_negative_control_reproduced') is not True):
        raise ValueError('New dispatch requires heterogeneous-latency full-session CLI qualification')


def validate_performance_contract(request: dict) -> None:
    config=request.get('performance_evaluation',{})
    fields={'case','concurrency','input_manifest_sha256','full_CLI_protocol_CPU_audit_sha256',
            'aiperf_cli_config_sha256','quality_scores_sha256','max_seconds'}
    if (set(config)!=fields or config.get('case') not in CASES
            or config.get('concurrency')!=CASES[config['case']][2] or config.get('max_seconds')!=3600
            or any(re.fullmatch('[0-9a-f]{64}',config.get(name,'')) is None for name in fields
                   if name.endswith('sha256'))):
        raise ValueError('Unreviewed production case/demand or missing frozen protocol/quality identities')


def verify_performance_prerequisites(request: dict, root: Path, uid: int) -> None:
    from .deployment_quality import verify_quality_reference
    config=request['performance_evaluation']
    verify_quality_reference(request,root,uid,'input/performance_quality_scores.json',config['quality_scores_sha256'])
    # Quality need not be accepted to measure its tradeoff. No quality acceptance
    # or SLO tolerance is invented by this performance prerequisite.
    verify_frozen_case(root/'input/production',config['case'],model_id=request['model_id'])
    proof_path=root/'input/full_CLI_protocol_CPU_audit.json'
    if proof_path.resolve(strict=True)!=proof_path or proof_path.stat().st_uid!=uid:
        raise ValueError('Unowned full CLI protocol evidence')
    raw=proof_path.read_bytes();proof=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=config['full_CLI_protocol_CPU_audit_sha256']:
        raise ValueError('The full CLI/session protocol lacks its independently verified frozen proof')
    validate_cli_protocol_proof(proof,require_session_limit=True)
    verify_session_recovery(request,root,uid)
    cli=Path(request['approved_workspace'])/'.venv-aiperf/lib/python3.12/site-packages/aiperf/config/flags/cli_config.py'
    if cli.resolve(strict=True)!=cli or cli.stat().st_uid!=uid or hashlib.sha256(cli.read_bytes()).hexdigest()!=config['aiperf_cli_config_sha256']:
        raise ValueError('Existing installed AIPerf CLI changed; no automatic dependency replacement')


def validate_payloads(raw: bytes, *, model_id: str) -> dict:
    document=json.loads(raw);sessions=document.get('data',[])
    if not isinstance(sessions,list) or not sessions or len(sessions)>128:
        raise ValueError('Production text cohort must contain1-128 actual sessions')
    identifiers=set();requests=0;hashes=[]
    for session in sessions:
        identifier=session.get('session_id');payloads=session.get('payloads')
        if not isinstance(identifier,str) or not identifier or identifier in identifiers:
            raise ValueError('Duplicated/missing production session identity')
        identifiers.add(identifier)
        if not isinstance(payloads,list) or not 1<=len(payloads)<=2:
            raise ValueError('Only declared original single/two-question document sessions are admitted')
        for payload in payloads:
            if (payload.get('model')!=model_id or payload.get('stream') is not True
                    or payload.get('stream_options')!={'include_usage':True}
                    or payload.get('ignore_eos',False) or payload.get('stop') is not None
                    or payload.get('temperature')!=0 or payload.get('tools')
                    or payload.get('chat_template_kwargs')!={'enable_thinking':False}
                    or type(payload.get('max_tokens')) is not int or not 1<=payload['max_tokens']<=1024
                    or not isinstance(payload.get('messages'),list) or not payload['messages']):
                raise ValueError('Production payload changed model/natural stopping/template or generation limit')
            if any(not isinstance(row,dict) or row.get('role') not in ('system','user','assistant')
                   or not isinstance(row.get('content'),str) for row in payload['messages']):
                raise ValueError('Malformed original text conversation')
            requests+=1;hashes.append(digest(payload))
    return {'sessions':len(sessions),'requests':requests,'source_sha256':hashlib.sha256(raw).hexdigest(),
            'payload_hashes':hashes,'original_session_order_preserved':True,'arrival_trace_claim':False}


def verify_frozen_case(directory: Path, case: str, *, model_id: str) -> dict:
    if case not in CASES:raise ValueError('Unknown core production pressure point')
    manifest=json.loads((directory/'manifest.json').read_text())
    if (manifest.get('natural_stopping') is not True or manifest.get('all_documents_untruncated') is not True
            or manifest.get('original_dataset_generation_caps_preserved') is not True):
        raise ValueError('Production input provenance/cap policy differs')
    measured,warmup,concurrency=CASES[case];result={'case':case,'concurrency':concurrency}
    for label,name in (('measured',measured),('warmup',warmup)):
        raw=(directory/name).read_bytes();observed=validate_payloads(raw,model_id=model_id)
        expected=manifest['files'][name]
        if (observed['source_sha256']!=expected['sha256'] or observed['sessions']!=expected['sessions']
                or observed['requests']!=expected['requests']
                or (observed['sessions'],observed['requests'])!=EXPECTED_COUNTS[name]):
            raise ValueError('Frozen production cohort bytes/counts changed')
        result[label]={'filename':name,**observed}
    result['warmup_concurrency']=min(concurrency,result['warmup']['sessions'])
    result['input_manifest_sha256']=hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()
    return result


def aiperf_command(model_id: str, source: Path, artifacts: Path, *, requests: int, concurrency: int,
                   timeout_seconds: int = 600, sessions: int | None=None) -> list[str]:
    if (type(requests) is not int or not 1<=requests<=128
            or type(concurrency) is not int or not 1<=concurrency<=16
            or timeout_seconds!=600 or not source.is_absolute() or not artifacts.is_absolute()):
        raise ValueError('Unreviewed/unbounded AIPerf production demand/path')
    if sessions is not None and (type(sessions) is not int or not 1<=sessions<=requests<=sessions*2):
        raise ValueError('Invalid frozen session/turn cardinality')
    command=['/workspace/.venv-aiperf/bin/aiperf','profile','--model',model_id,
        '--tokenizer','/models/'+model_id,'--url','http://127.0.0.1:19001',
        '--endpoint-type','chat','--streaming','--use-server-token-count','--use-legacy-max-tokens',
        '--input-file',str(source),'--custom-dataset-type','inputs_json',
        '--dataset-sampling-strategy','sequential','--request-count',str(requests),
        '--concurrency',str(concurrency),'--workers-max','1','--no-gpu-telemetry',
        '--random-seed','20261027','--request-timeout-seconds',str(timeout_seconds),
        '--ui','none','--export-level','raw','--output-artifact-dir',str(artifacts)]
    # Keep legacy argv reproducible for historical independent audits. Every
    # newly executed phase supplies its exact frozen session count explicitly.
    if sessions is not None:command.extend(['--num-sessions',str(sessions)])
    return command


def run_aiperf_measurement(request: dict, root: Path, output: Path,
                           server: subprocess.Popen, env: dict) -> dict:
    """Run separate external warmup and measured CLI, then audit every raw turn."""
    from .aiperf_cli_admission import validate_service_cli
    from .production import audit_requests
    from .worker import monitor,stop_child

    config=request['performance_evaluation'];case=config['case']
    cohort=root/'input/production';verified=verify_frozen_case(cohort,case,model_id=request['model_id'])
    if (verified['input_manifest_sha256']!=config['input_manifest_sha256']
            or verified['concurrency']!=config['concurrency']):
        raise ValueError('Actual production cohort/config differs from the immutable request')
    cpu_proof=root/'input/full_CLI_protocol_CPU_audit.json'
    if hashlib.sha256(cpu_proof.read_bytes()).hexdigest()!=config['full_CLI_protocol_CPU_audit_sha256']:
        raise ValueError('Actual full CLI protocol proof changed')
    proof=json.loads(cpu_proof.read_text())
    validate_cli_protocol_proof(proof,require_session_limit=True)
    client_env=dict(env,PYTHONPATH=str(root/'runtime_support/tokenizer_compat'),
                    CJYU_AIPERF_LOCAL_TOKENIZER_SHIM='1',CUDA_VISIBLE_DEVICES='',
                    AIPERF_DATASET_MMAP_CACHE_ENABLED='true',
                    AIPERF_DATASET_MMAP_CACHE_DIR='/run_cache/aiperf_mmap')
    results={}
    deadline=time.monotonic_ns()+config['max_seconds']*1_000_000_000
    for phase in ('warmup','measured'):
        directory=output/phase;directory.mkdir(mode=0o700)
        source=cohort/verified[phase]['filename']
        concurrency=verified['warmup_concurrency'] if phase=='warmup' else verified['concurrency']
        command=aiperf_command(request['model_id'],source,directory/'artifacts',
            requests=verified[phase]['requests'],concurrency=concurrency,sessions=verified[phase]['sessions'])
        write_new_json(directory/'command.json',command)
        validate_service_cli(command,root,directory,client_env,config['aiperf_cli_config_sha256'])
        started=time.monotonic_ns()
        with (directory/'aiperf.log').open('x') as log:
            process=subprocess.Popen(command,cwd=root,env=client_env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            try:
                monitor(process,server,directory/'gpu_telemetry.csv',deadline_ns=deadline)
                if process.wait()!=0:raise ValueError('Actual AIPerf CLI failed; retain all actual records')
            finally:stop_child(process)
        measured=audit_requests(directory/'artifacts',source,verified[phase]['requests'],0,allow_multi_turn=True)
        measured.update(phase=phase,worker_observation_started_monotonic_ns=started,
                        worker_observation_ended_monotonic_ns=time.monotonic_ns())
        write_new_json(directory/'independent_worker_audit.json',measured);results[phase]=measured
    return {'protocol':'native_hybrid_real_text_AIPerf_v1','case':case,
        'source_manifest_sha256':verified['input_manifest_sha256'],'warmup_is_external_and_unmeasured':True,
        'profile':results['measured'],'external_warmup':results['warmup'],
        'all_actual_payload_and_SSE_and_usage_and_counts_verified':True,'official_AIPerf_complete':True,
        'production_quality_accepted':False,'quality_snapshot_reference':config['quality_scores_sha256'],
        'memory_utilization_metric_is_memory_bandwidth_GBps':False,
        'scope':'Same real source/caps/natural stopping and declared closed-loop demand; actual successful request/token rates and tails. '
                'No arrival-trace claim; warmup and model cold-start excluded from steady-state statistics.'}
