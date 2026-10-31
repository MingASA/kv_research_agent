"""Freeze new-binary native model gates/calibration; no SSH or GPU dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.controller.approval import TRUSTED_APPROVAL, load_approval_snapshot
from scripts.controller.durable import digest, write_new_bytes, write_new_json
from scripts.controller.nextgen import GDN_OVERLAY, MODEL, QUALITY_KIND, SCHEMA, SERVICE_KIND
from scripts.controller.nextgen_service import SERVICE_SOURCE_FILES,MIXED_PREFILL_PROFILE
from scripts.controller.policy import validate_request
from scripts.controller.nextgen_performance import CASES,PERF_KIND,verify_frozen_case
from scripts.controller.nextgen_tools_service import TOOLS_KIND,TOOLS_SCOPE


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_performance_imports(runtime: Path) -> dict:
    """Import the actual deployed audit outside the control checkout/site paths."""
    source = '''import json,sys
from pathlib import Path
runtime=Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0,str(runtime))
import kv_controller.production as production
import kv_controller.evidence as evidence
import kv_controller.nextgen_performance as performance
assert all(Path(module.__file__).resolve().is_relative_to(runtime)
           for module in (production,evidence,performance))
assert callable(production.audit_requests) and callable(evidence.read_records)
assert not any(name == 'scripts' or name.startswith('scripts.') for name in sys.modules)
assert not any(name.split('.')[0] in ('torch','cuda','flashinfer','triton') for name in sys.modules)
print(json.dumps({'frozen_runtime_imports_verified':True,
                  'control_checkout_or_site_packages_required':False,'GPU_used':False}))
'''
    with tempfile.TemporaryDirectory(prefix='kv-frozen-import-') as directory:
        result = subprocess.run([sys.executable,'-B','-I','-S','-c',source,str(runtime.resolve())],
                                cwd=directory,text=True,capture_output=True,timeout=30)
    if result.returncode:
        raise ValueError('Frozen performance runtime import failed before GPU dispatch: '+result.stderr[-2048:])
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--replicate', type=int, default=0,
                        help='Explicit ledger replicate for a reviewed retry; never implies an independent sample')
    parser.add_argument('--session-recovery-ticket',type=Path)
    parser.add_argument('--session-recovery-result',type=Path)
    parser.add_argument('--session-recovery-audit',type=Path)
    parser.add_argument('--packaging-recovery-ticket',type=Path)
    parser.add_argument('--packaging-recovery-result',type=Path)
    parser.add_argument('--packaging-repair-proof',type=Path)
    parser.add_argument('--resource-recovery-ticket', type=Path)
    parser.add_argument('--resource-recovery-result', type=Path)
    parser.add_argument('--transport-recovery-ticket', type=Path)
    parser.add_argument('--transport-recovery-certificate', type=Path)
    parser.add_argument('--reuse-assets-from-run')
    parser.add_argument('--orchestration-recovery-ticket',type=Path)
    parser.add_argument('--orchestration-recovery-result',type=Path)
    parser.add_argument('--orchestration-repair-proof',type=Path)
    parser.add_argument('--idle-wait-recovery-ticket',type=Path)
    parser.add_argument('--idle-wait-recovery-result',type=Path)
    parser.add_argument('--arm', choices=('bf16', 'fp8_e4m3', 'nvfp4'), required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--decode', choices=('flashinfer', 'xqa'), default='flashinfer')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--mixed-prefill',action='store_true',
                        help='Explicit8192 batch/2048 chunk profile; fresh matched quality required for formal performance')
    parser.add_argument('--mixed-prefill-pilot-proof',type=Path)
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--quality-cohort',type=Path)
    parser.add_argument('--quality-completion-certificate',type=Path)
    parser.add_argument('--scales-audit', type=Path)
    parser.add_argument('--performance-case',choices=tuple(CASES))
    parser.add_argument('--production-inputs',type=Path)
    parser.add_argument('--performance-quality-scores',type=Path)
    parser.add_argument('--full-cli-cpu-proof',type=Path)
    parser.add_argument('--aiperf-cli-config-sha256')
    parser.add_argument('--native-tools',action='store_true')
    parser.add_argument('--tools-inputs',type=Path)
    parser.add_argument('--tools-source',type=Path)
    parser.add_argument('--tools-quality-scores',type=Path)
    parser.add_argument('--tools-cpu-protocol-proof',type=Path)
    parser.add_argument('--tools-cpu-image-proof',type=Path)
    args = parser.parse_args()
    if args.quality and args.calibrate:
        raise ValueError('Formal core quality must disable the diagnostic range observer')
    if args.performance_case and (args.quality or args.calibrate):
        raise ValueError('Performance has a separate explicit workload and disables the observer')
    if args.native_tools and (args.quality or args.calibrate or args.performance_case):
        raise ValueError('Native tool task generation is a separate observer-free workload')
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'state/NEXTGEN_CONFIG.local.json').read_text())
    if not config.get('new_binary_CPU_tests_verified') or not config.get('remote_source_repo'):
        raise ValueError('New-binary source/test/ldd/SHA mapping must be verified before preparing this model gate')
    source = Path(config['local_repo'])
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Freeze only a clean source commit')
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    changed = subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only',
                                       config['source_commit'], head], text=True).splitlines()
    if any(name.endswith(('.cpp', '.h', '.hpp', '.cc', '.cu', '.cuh')) for name in changed):
        raise ValueError('Current Python source has C++ differences from the verified new binary')
    approval, approval_sha = load_approval_snapshot(TRUSTED_APPROVAL)
    remote = approval['approved_workspace'] + '/agent_work/runs/' + args.run_id
    calibration = root / 'results/prepared/nextgen-calibration-chat128-long8-20261027b'
    if (args.calibrate and args.arm != 'bf16') or (args.arm != 'bf16' and not args.scales_audit):
        raise ValueError('Calibrate only BF16; quantized model gates require independently audited new-model scales')
    validation = root / 'plans/experiments/nextgen-bf16-qwen38-validation-r2-20261027e/input/validation.json'
    if not validation.exists():
        validation = root / 'plans/experiments/nextgen-bf16-qwen38-validation-r2-20261027e/validation.json'
    proof = root / 'reports/operations/nextgen-native-kv-20261027a/GDN_private_overlay_CPU_import_a1.json'
    dependency = json.loads(proof.read_text())
    request = {key: approval[key] for key in ('ssh_target', 'approved_host_fingerprint', 'approved_workspace')}
    request.update(schema=SCHEMA, run_id=args.run_id, model_id=MODEL, model_path=approval['read_only_model_path'],
                   image_id=approval['approved_image_id'], gpu_uuids=[args.gpu_uuid],
                   remote_output=remote, max_wall_seconds=7200 if args.quality or args.performance_case or args.native_tools else 3600,
                   startup_seconds=900, max_output_gib=4,
                   arms=[args.arm], replicate=args.replicate, source_repo=config['remote_source_repo'],
                   server_binary=config['remote_new_binary_path'], server_source_commit=config['source_commit'],
                   binary_sha256=config['binary_sha256'], server_binary_execution='required',
                   python_source_repo=remote + '/source_repo', python_source_commit=head,
                   dataset_path=remote + '/input/validation.json', dataset_sha256=_sha(validation),
                   model_config_sha256=_sha(root / '.deps/nextgen-official-models-20261027-a1'/MODEL/'config.json'),
                   backend_source_sha256={name: _sha(source/name) for name in SERVICE_SOURCE_FILES},
                   graph_mode='cudagraphs', prefix_cache=True,
                   max_context_scope='validation_input_budget_not_server_context_cap',
                   service={'max_batch': 8, 'max_tokens_per_batch': 2048, 'linear_state_slots': 16,
                            'max_context': 131072, 'max_memory_utilization': .85, 'page_size': 64},
                   operators={args.arm: {'planner': 'fa3' if args.arm == 'bf16' else 'fa2', 'decode': args.decode}},
                   workload=({'kind':QUALITY_KIND,'scope':'natural_stop_core_quality_generations_not_service_performance'}
                             if args.quality else {'kind':SERVICE_KIND,
                             'scope':'format_correctness_lifecycle_and_optional_untimed_calibration'}),
                   runtime_dependency_overlay={'overlay_name': GDN_OVERLAY,
                       'overlay_manifest_sha256': dependency['manifest_sha256'],
                       'CPU_import_evidence_sha256': _sha(proof)},
                   range_observation=args.calibrate,
                   research_reason='New C++ allocator/typed native backend, actual model Graph/cache lifecycle; '
                       'optional independent real-text BF16 range calibration before low-precision quality. Not throughput.')
    if args.calibrate:
        request.update(calibration_sha256=_sha(calibration/'calibration.json'), calibration_count=136,
                       calibration_concurrency=4)
    if args.scales_audit:
        audit = json.loads(args.scales_audit.read_text())
        if (audit.get('state') != 'BF16_CALIBRATION_INDEPENDENTLY_VERIFIED'
                or audit.get('model_id') != MODEL or audit.get('actual_observer_GPU_Graph_passed') is not True):
            raise ValueError('Scales lack independently verified new-model BF16 observations')
        request.update(kv_scales=audit['candidate_scales'][args.arm],
                       scale_calibration_audit_sha256=_sha(args.scales_audit))
    quality = (args.quality_cohort or root/'results/prepared/nextgen-quality-real-text-native-json-v2-20261028a').resolve(strict=True)
    if args.quality:
        manifest = json.loads((quality/'manifest.json').read_text())
        if _sha(quality/'requests.jsonl')!=manifest['request_payload_sha256']:
            raise ValueError('Frozen real quality requests changed')
        request['service'].update(max_batch=16,linear_state_slots=32,max_memory_utilization=.90)
        count=1720
        if args.quality_completion_certificate:
            completion=json.loads(args.quality_completion_certificate.read_text())
            if (completion.get('state')!='INDEPENDENT_PARTIAL_RAW_QUALITY_VALIDATED_NOT_FULL_COHORT'
                    or manifest.get('partial_raw_certificate_sha256')!=_sha(args.quality_completion_certificate)
                    or manifest.get('completion_requests')!=completion['missing_count']
                    or manifest['request_payload_sha256']!=completion['completion_requests_sha256']):
                raise ValueError('Quality completion subset does not match its independent raw audit')
            count=completion['missing_count']
            request['quality_completion']={'anchor_run_id':completion['anchor_run_id'],
                'partial_raw_certificate_sha256':_sha(args.quality_completion_certificate)}
        request['quality_evaluation']={'requests':count,'requests_sha256':manifest['request_payload_sha256'],
             'source_manifest_sha256':_sha(quality/'manifest.json'),'client_concurrency':16,
             'client_token_budget':160000,'request_timeout_seconds':600,'max_seconds':4800}
        request['research_reason']='New-model1720 real-text core quality generations with natural stopping and '
        request['research_reason']+='matched typed native service configuration; labels/scoring stay independent, observer disabled. Not AIPerf.'
        if args.quality_completion_certificate:
            request['research_reason']=f'Logical1720 quality cohort: generate only{count} missing durable request IDs. '
            request['research_reason']+='Retain all independently verified prior responses; exact source/operator/config/scales, '
            request['research_reason']+='no answer-based selection, no single uninterrupted quality/performance claim.'
    performance_case=None
    if args.performance_case:
        if not all((args.production_inputs,args.performance_quality_scores,args.full_cli_cpu_proof,args.aiperf_cli_config_sha256)):
            raise ValueError('Performance requires frozen real inputs, evaluated quality, full CLI proof and exact installed CLI hash')
        from scripts.controller.nextgen_performance import validate_cli_protocol_proof
        validate_cli_protocol_proof(json.loads(args.full_cli_cpu_proof.read_text()),require_session_limit=True)
        performance_case=verify_frozen_case(args.production_inputs,args.performance_case,model_id=MODEL)
        request['service'].update(max_batch=16,linear_state_slots=32,max_memory_utilization=.90)
        request['workload']={'kind':PERF_KIND,'scope':'natural_stop_real_text_AIPerf_end_to_end'}
        request['performance_evaluation']={'case':args.performance_case,'concurrency':performance_case['concurrency'],
            'input_manifest_sha256':performance_case['input_manifest_sha256'],
            'full_CLI_protocol_CPU_audit_sha256':_sha(args.full_cli_cpu_proof),
            'aiperf_cli_config_sha256':args.aiperf_cli_config_sha256,
            'quality_scores_sha256':_sha(args.performance_quality_scores),'max_seconds':3600}
        request['research_reason']='Real-text AIPerf end-to-end production case with separate disjoint external warmup, '
        request['research_reason']+='natural stopping, raw request/SSE/usage/count audit and matched evaluated-quality deployment. '
        request['research_reason']+='One workload per fresh model instance; no synthetic padding or implicit warmup cursor reuse.'
        request['research_reason']+=' Explicit frozen session count prevents fast-session recycling; request cap remains unchanged.'
    if args.native_tools:
        from scripts.controller.nextgen_tools_service import validate_cpu_protocol
        if not all((args.tools_inputs,args.tools_source,args.tools_quality_scores,
                    args.tools_cpu_protocol_proof,args.tools_cpu_image_proof)):
            raise ValueError('Native tools require real inputs/source,scored deployment and actual CPU image/protocol proof')
        tools_manifest=json.loads((args.tools_inputs/'manifest.json').read_text())
        image_proof=json.loads(args.tools_cpu_image_proof.read_text())
        protocol_proof=json.loads(args.tools_cpu_protocol_proof.read_text())
        validate_cpu_protocol(protocol_proof)
        if (tools_manifest.get('tasks')!=32 or tools_manifest.get('user_turns')!=112
                or image_proof.get('state')!='COMPLETE_CPU_TOOL_IMAGE_VALIDATION'
                or image_proof.get('image_id')!=request['image_id'] or image_proof.get('GPU_requested') is not False
                or image_proof.get('native_CPP_reference_sha256')!=protocol_proof['native_CPP_reference_sha256']):
            raise ValueError('Real32-task inventory or actual isolated H800 CPU image qualification is missing')
        for name in ('official_tokenization.py','nextgen_multiturn.py','nextgen_tool_child.py','nextgen_tools_env.py','native_tools.py'):
            if image_proof['runtime_sources_sha256'].get('pkg/scripts/controller/'+name)!=_sha(root/'scripts/controller'/name):
                raise ValueError('Actual H800 CPU qualification predates current native tool runtime: '+name)
        request['service'].update(max_batch=16,linear_state_slots=32,max_memory_utilization=.90)
        request['range_observation']=False;request['workload']={'kind':TOOLS_KIND,'scope':TOOLS_SCOPE}
        request['tool_evaluation']={'tasks':32,'user_turns':112,'client_concurrency':4,'max_steps_per_turn':10,
            'max_new_tokens':1024,'max_seconds':3600,'tool_parser':'qwen3_coder',
            'manifest_sha256':_sha(args.tools_inputs/'manifest.json'),'tasks_sha256':_sha(args.tools_inputs/'tasks.jsonl'),
            'tool_source_manifest_sha256':_sha(args.tools_source/'source_manifest.json'),
            'CPU_protocol_rollup_sha256':_sha(args.tools_cpu_protocol_proof),
            'CPU_image_evidence_sha256':_sha(args.tools_cpu_image_proof),
            'quality_scores_sha256':_sha(args.tools_quality_scores)}
        request['research_reason']='Original32-task BFCL native-tool conversations with bounded in-memory APIs; '
        request['research_reason']+='no reference answers on serving path,separate independent state/response scoring,not AIPerf.'
    if args.mixed_prefill:
        if not (args.calibrate or args.quality or args.performance_case or args.native_tools):
            raise ValueError('Mixed-prefill pilot must use bounded real BF16 calibration before formal matched evaluation')
        request['scheduling_profile']=MIXED_PREFILL_PROFILE
        request['service'].update(max_batch=16,linear_state_slots=32,max_memory_utilization=.90,
            max_tokens_per_batch=8192,max_tokens_per_chunk_for_prefill=2048)
        if args.calibrate:
            if args.mixed_prefill_pilot_proof:raise ValueError('A pilot cannot qualify itself')
        else:
            if not args.mixed_prefill_pilot_proof:raise ValueError('Formal mixed profile requires the completed actual GPU pilot proof')
            request['mixed_prefill_pilot_audit_sha256']=_sha(args.mixed_prefill_pilot_proof)
        request['research_reason']+=' Explicit mixed-prefill headroom profile; prior2048 quality cannot qualify8192 formal performance.'
    elif args.mixed_prefill_pilot_proof:
        raise ValueError('Pilot proof belongs only to the explicit mixed-prefill profile')
    experiment={name: request[name] for name in
        ('model_id', 'server_source_commit', 'python_source_commit', 'dataset_sha256', 'operators', 'service',
         'range_observation','workload')} | {'kv_scales':request.get('kv_scales'),
                                          'quality_evaluation':request.get('quality_evaluation')}
    if performance_case is not None:experiment['performance_evaluation']=request['performance_evaluation']
    if args.native_tools:experiment['tool_evaluation']=request['tool_evaluation']
    request['experiment_key'] = digest(experiment)
    if any((args.session_recovery_ticket,args.session_recovery_result,args.session_recovery_audit)):
        from scripts.controller.nextgen_performance import SESSION_REPAIR,validate_session_recovery
        if not all((args.session_recovery_ticket,args.session_recovery_result,args.session_recovery_audit,args.full_cli_cpu_proof)):
            raise ValueError('Original failed ticket/result/cohort audit and corrected full CLI proof are required')
        prior=json.loads(args.session_recovery_ticket.read_text())['request']
        result=json.loads(args.session_recovery_result.read_text());failure=json.loads(args.session_recovery_audit.read_text())
        request['session_protocol_recovery']={'prior_run_id':prior['run_id'],'prior_request_sha256':digest(prior),
            'prior_result_sha256':_sha(args.session_recovery_result),'repair_kind':SESSION_REPAIR,
            'CPU_repair_evidence_sha256':_sha(args.full_cli_cpu_proof),'failure_audit_sha256':_sha(args.session_recovery_audit)}
        validate_session_recovery(prior,result,request,failure,_sha(args.session_recovery_result))
        from scripts.controller.quota import QuotaLedger
        with QuotaLedger(root/'state/controller') as ledger:old=ledger.get(prior['run_id'])
        if old['state']!='SETTLED' or old['request_sha256']!=digest(prior):
            raise ValueError('Original failed session-protocol run must be reconciled and settled first')
    if any((args.packaging_recovery_ticket,args.packaging_recovery_result,args.packaging_repair_proof)):
        if not all((args.packaging_recovery_ticket,args.packaging_recovery_result,args.packaging_repair_proof)):
            raise ValueError('Packaging recovery needs the original ticket/result and isolated CPU repair proof')
        prior=json.loads(args.packaging_recovery_ticket.read_text())['request']
        result=json.loads(args.packaging_recovery_result.read_text())
        packaging_proof=json.loads(args.packaging_repair_proof.read_text())
        ignored={'run_id','remote_output','python_source_repo','dataset_path','replicate','runtime_packaging_recovery'}
        if (result.get('status')!='FAILED' or result.get('gpu_release_verified') is not True
                or result.get('live_owned_containers')!=[] or prior.get('replicate')!=0
                or args.replicate!=(2 if args.transport_recovery_ticket else 1)
                or result.get('request_sha256')!=digest(prior)
                or {k:v for k,v in prior.items() if k not in ignored}!={k:v for k,v in request.items() if k not in ignored}
                or packaging_proof.get('state')!='FROZEN_PERFORMANCE_AUDIT_IMPORT_REPRODUCED_FIXED_CPU'
                or packaging_proof.get('original_run_id')!=prior['run_id']
                or packaging_proof.get('original_terminal_sha256')!=_sha(args.packaging_recovery_result)
                or packaging_proof.get('standalone_auditor_sha256')!=_sha(root/'scripts/audit_native_results.py')):
            raise ValueError('Packaging recovery changed the deployment or lacks its bound CPU repair')
        request['runtime_packaging_recovery']={'prior_run_id':prior['run_id'],
            'prior_result_sha256':_sha(args.packaging_recovery_result),'repair_kind':'standalone_performance_audit_import',
            'audit_module_sha256':_sha(root/'scripts/audit_native_results.py'),
            'CPU_repair_evidence_sha256':_sha(args.packaging_repair_proof)}
    if args.resource_recovery_ticket or args.resource_recovery_result:
        if not args.resource_recovery_ticket or not args.resource_recovery_result:
            raise ValueError('Resource recovery requires both original ticket and released terminal result')
        prior = json.loads(args.resource_recovery_ticket.read_text())['request']
        result = json.loads(args.resource_recovery_result.read_text())
        if (args.replicate != 1 or prior['replicate'] != 0 or result['status'] != 'CONFLICT'
                or result.get('gpu_release_verified') is not True or result.get('live_owned_containers') != []
                or result['run_id'] != prior['run_id'] or result.get('arm_results') != []
                or request['experiment_key'] != prior['experiment_key']):
            raise ValueError('Only a reviewed exact-configuration released pre-model conflict may be retried')
        request['resource_recovery'] = {'reason':'foreign_GPU_process_before_model_start',
            'prior_run_id':prior['run_id'], 'prior_result_sha256':_sha(args.resource_recovery_result)}
    transport_prior=None
    if args.transport_recovery_ticket or args.transport_recovery_certificate:
        from scripts.controller.absent_recovery import verify_absence_certificate
        if not args.transport_recovery_ticket or not args.transport_recovery_certificate:
            raise ValueError('Both prior ticket and reviewed absence certificate are required')
        transport_prior=json.loads(args.transport_recovery_ticket.read_text())['request']
        certificate=json.loads(args.transport_recovery_certificate.read_text())
        verify_absence_certificate(transport_prior,certificate)
        if args.replicate!=transport_prior['replicate']+1:raise ValueError('Explicit next transport recovery replicate required')
        ignored={'run_id','remote_output','python_source_repo','dataset_path','replicate','transport_recovery'}
        if {k:v for k,v in transport_prior.items() if k not in ignored}!={k:v for k,v in request.items() if k not in ignored}:
            raise ValueError('Transport recovery changed the actual deployment/input/operator')
        request['transport_recovery']={'prior_run_id':transport_prior['run_id'],
            'prior_request_sha256':digest(transport_prior),'absence_certificate_sha256':_sha(args.transport_recovery_certificate)}
    if args.orchestration_recovery_ticket or args.orchestration_recovery_result or args.orchestration_repair_proof:
        if not all((args.orchestration_recovery_ticket,args.orchestration_recovery_result,args.orchestration_repair_proof)):
            raise ValueError('Original ticket/result and CPU repair proof are all required')
        prior=json.loads(args.orchestration_recovery_ticket.read_text())['request']
        result=json.loads(args.orchestration_recovery_result.read_text());repair_proof=json.loads(args.orchestration_repair_proof.read_text())
        ignored={'run_id','remote_output','python_source_repo','dataset_path','replicate',
                 'transport_recovery','resource_recovery','orchestration_recovery'}
        if (result['status']!='FAILED' or result['gpu_release_verified'] is not True or result['live_owned_containers']!=[]
                or args.replicate!=prior['replicate']+1
                or {k:v for k,v in prior.items() if k not in ignored}!={k:v for k,v in request.items() if k not in ignored}):
            raise ValueError('Not an exact released deployment eligible for the verified accounting repair')
        request['orchestration_recovery']={'prior_run_id':prior['run_id'],
            'prior_result_sha256':_sha(args.orchestration_recovery_result),'repair_kind':'bounded_output_scan_atomic_progress_race',
            'supervisor_source_sha256':_sha(root/'scripts/controller/supervisor.py'),
            'CPU_repair_evidence_sha256':_sha(args.orchestration_repair_proof)}
        if repair_proof.get('supervisor_source_sha256')!=request['orchestration_recovery']['supervisor_source_sha256']:
            raise ValueError('CPU proof is not from the exact repaired supervisor source')
    if args.idle_wait_recovery_ticket or args.idle_wait_recovery_result:
        from scripts.controller.idle_wait_recovery import REASON,verify_same_deployment,verify_zero_allocation_wait
        if not args.idle_wait_recovery_ticket or not args.idle_wait_recovery_result:
            raise ValueError('The original zero-allocation ticket and exact terminal result are both required')
        prior=json.loads(args.idle_wait_recovery_ticket.read_text())['request']
        result=json.loads(args.idle_wait_recovery_result.read_text())
        verify_zero_allocation_wait(prior,result);verify_same_deployment(prior,request)
        from scripts.controller.quota import QuotaLedger
        with QuotaLedger(root/'state/controller') as ledger:old=ledger.get(prior['run_id'])
        if old['state']!='SETTLED' or old['actual_gpu_us']!=0 or old['request_sha256']!=digest(prior):
            raise ValueError('Original zero-GPU wait must be independently reconciled/settled before preparation')
        request['idle_wait_recovery']={'prior_run_id':prior['run_id'],'prior_request_sha256':digest(prior),
            'prior_result_sha256':_sha(args.idle_wait_recovery_result),'reason':REASON}
    validate_request(request, approval)
    output = root / 'plans/experiments' / args.run_id
    output.mkdir(mode=0o700, exist_ok=False)
    bundle = output/'source.bundle'
    subprocess.run(['git', '-C', str(source), 'bundle', 'create', str(bundle), 'HEAD',
                    '^ceff00bbd38054d2b6b4aa44513d773f2bd174b2'], check=True)
    assets = {}

    def add(name: str, data: bytes) -> None:
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_new_bytes(path, data)
        assets[name] = {'local_path': str(path), 'sha256': _sha(path)}

    assets['input/source.bundle'] = {'local_path': str(bundle), 'sha256': _sha(bundle)}
    add('input/validation.json', validation.read_bytes())
    tokenization = root / 'plans/experiments/nextgen-bf16-qwen38-validation-r2-20261027e/input/validation_tokenization_CPU.json'
    add('input/validation_tokenization_CPU.json', tokenization.read_bytes())
    add('input/gdn_dependency_CPU_import.json', proof.read_bytes())
    if args.mixed_prefill_pilot_proof:
        add('input/mixed_prefill_pilot_audit.json',args.mixed_prefill_pilot_proof.read_bytes())
    if args.calibrate:
        add('input/calibration.json', (calibration/'calibration.json').read_bytes())
        add('input/calibration_manifest.json', (calibration/'manifest.json').read_bytes())
    if args.scales_audit:
        add('input/scale_calibration_audit.json', args.scales_audit.read_bytes())
    if transport_prior is not None:
        from scripts.controller.durable import canonical_bytes
        add('input/transport_prior_request.json',canonical_bytes(transport_prior))
        add('input/transport_absence_certificate.json',args.transport_recovery_certificate.read_bytes())
    if args.orchestration_repair_proof:
        add('input/orchestration_repair_CPU.json',args.orchestration_repair_proof.read_bytes())
    if args.packaging_repair_proof:
        add('input/packaging_repair_CPU.json',args.packaging_repair_proof.read_bytes())
    if args.session_recovery_audit:
        add('input/performance_session_failure_audit.json',args.session_recovery_audit.read_bytes())
    if args.quality:
        # Reuse the one immutable CPU source instead of duplicating 23MB per
        # arm in local Git; remote input files still have their own exact SHA.
        for filename,source_file in [('quality_requests.jsonl','requests.jsonl'),('quality_manifest.json','manifest.json')]:
            path=quality/source_file
            assets['input/'+filename]={'local_path':str(path),'sha256':_sha(path)}
        add('input/scoring_protocol.json',(quality/'scoring_protocol.json').read_bytes())
        if args.quality_completion_certificate:
            add('input/partial_quality_certificate.json',args.quality_completion_certificate.read_bytes())
    if performance_case is not None:
        for name in {'manifest.json',performance_case['measured']['filename'],performance_case['warmup']['filename']}:
            path=args.production_inputs/name
            assets['input/production/'+name]={'local_path':str(path.resolve()),'sha256':_sha(path)}
        add('input/performance_quality_scores.json',args.performance_quality_scores.read_bytes())
        add('input/full_CLI_protocol_CPU_audit.json',args.full_cli_cpu_proof.read_bytes())
        shim=root/'results/mirrors/takeover-20261026T040426Z-attempt02/files/support/inputs/aiperf/aiperf_v3_tokenizer_compat/sitecustomize.py'
        add('runtime_support/tokenizer_compat/sitecustomize.py',shim.read_bytes())
    if args.native_tools:
        for name in ('manifest.json','tasks.jsonl','pre_gold_freeze_receipt.json'):
            path=args.tools_inputs/name;assets['input/tools32/'+name]={'local_path':str(path.resolve()),'sha256':_sha(path)}
        add('input/tool_source/source_manifest.json',(args.tools_source/'source_manifest.json').read_bytes())
        tools_source_manifest=json.loads((args.tools_source/'source_manifest.json').read_text())
        for row in tools_source_manifest['files']:
            path=args.tools_source/'files'/row['path']
            assets['input/tool_source/files/'+row['path']]={'local_path':str(path.resolve()),'sha256':_sha(path)}
        add('input/tool_quality_scores.json',args.tools_quality_scores.read_bytes())
        add('input/tool_CPU_protocol_rollup.json',args.tools_cpu_protocol_proof.read_bytes())
        add('input/tool_CPU_image_evidence.json',args.tools_cpu_image_proof.read_bytes())
    for path in sorted((root/'scripts/controller').glob('*.py')):
        # The local evidence shim imports the control checkout. Deploy the same
        # standalone auditor already used by the established production runner.
        source_file=root/'scripts/audit_native_results.py' if path.name=='evidence.py' else path
        add('runtime/kv_controller/' + path.name, source_file.read_bytes())
    if performance_case is not None:
        write_new_json(output/'frozen_runtime_import_CPU.json',
                       validate_frozen_performance_imports(output/'runtime'))
    write_new_json(output/'request.json', request)
    package={'schema': 'kv-frozen-local-package-v1', 'classification': 'proposal',
        'request': request, 'approval_sha256_at_preparation': approval_sha, 'assets': assets}
    if args.reuse_assets_from_run:
        import re
        from scripts.controller.quota import QuotaLedger
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',args.reuse_assets_from_run) is None:
            raise ValueError('Invalid completed reuse parent')
        prior_ticket=json.loads((root/'state/controller/tickets'/(args.reuse_assets_from_run+'.json')).read_text())
        with QuotaLedger(root/'state/controller') as ledger:parent=ledger.get(args.reuse_assets_from_run)
        result=json.loads(parent['evidence_json'])
        if (parent['state']!='SETTLED' or result['status']!='COMPLETE' or result['gpu_release_verified'] is not True
                or result['live_owned_containers']!=[] or prior_ticket['request']['model_id']!=MODEL):
            raise ValueError('Only a bound completed owned input source can reduce transfer')
        package['remote_asset_reuse']={name:{'sha256':entry['sha256'],'reuse_from_run_id':args.reuse_assets_from_run,
            'reuse_parent_result_sha256':digest(result)} for name,entry in assets.items()
            if name.startswith(('input/','runtime/')) and prior_ticket['assets_sha256'].get(name)==entry['sha256']}
        if args.quality_completion_certificate:
            completion=json.loads(args.quality_completion_certificate.read_text());name='input/quality_requests.jsonl'
            if prior_ticket['assets_sha256'].get(name)!=completion['logical_full_requests_sha256']:
                raise ValueError('Completed input-reuse parent does not contain the logical full cohort')
            package['remote_asset_reuse'][name]={'sha256':assets[name]['sha256'],
                'derive_requests_from_run_id':args.reuse_assets_from_run,
                'source_requests_sha256':completion['logical_full_requests_sha256'],
                'source_parent_result_sha256':digest(result),'selected_request_ids':completion['missing_ids']}
    write_new_json(output/'package.json',package)


if __name__ == '__main__':
    main()
