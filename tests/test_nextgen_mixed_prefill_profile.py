import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.nextgen import QUALITY_KIND
from scripts.controller.nextgen_service import MIXED_PREFILL_PROFILE,native_server_command
from scripts.controller.nextgen_service import verify_mixed_prefill_prerequisites
from scripts.controller.nextgen_service import resolved_scheduling_options
from scripts.controller.durable import digest
from scripts.controller.policy import validate_request
from scripts.controller.deployment_quality import verify_quality_reference
from test_nextgen_native_service import _service


def mixed(request):
    request=copy.deepcopy(request);request['scheduling_profile']=MIXED_PREFILL_PROFILE
    request['service'].update(max_batch=16,linear_state_slots=32,max_memory_utilization=.9,
        max_tokens_per_batch=8192,max_tokens_per_chunk_for_prefill=2048)
    return request


class MixedPrefillProfileTests(unittest.TestCase):
    def options_log(self):
        return 'Master init options: max_tokens_per_batch: 8192, max_tokens_per_chunk_for_prefill: 2048, '+\
            'max_seqs_per_batch: 16, max_linear_state_cache_slots: 32, max_memory_utilization: 0.9, '+\
            'cp_size: 1, num_speculative_tokens: 0'

    def test_resolved_runtime_must_match_candidate_not_just_request(self):
        _,request=_service();request=mixed(request);log=self.options_log()
        self.assertEqual(resolved_scheduling_options(log,request['service'])['resolved']['max_tokens_per_batch'],8192)
        for wrong in [log.replace('batch: 8192','batch: 2048'),log+'\n'+log,
                      log.replace('num_speculative_tokens: 0','num_speculative_tokens: 1')]:
            with self.subTest(log=wrong),self.assertRaises(ApprovalError):resolved_scheduling_options(wrong,request['service'])

    def test_formal_profile_rechecks_original_pilot_terminal_and_real_allocation(self):
        _,pilot=_service();pilot=mixed(pilot);pilot['run_id']='pilot'
        formal=copy.deepcopy(pilot);formal['run_id']='formal';formal['range_observation']=False
        formal['workload']={'kind':QUALITY_KIND}
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);prior=base/'agent_work/runs/pilot';root=base/'agent_work/runs/formal'
            (prior/'output/bf16').mkdir(parents=True);(root/'input').mkdir(parents=True)
            (prior/'ticket.json').write_text(json.dumps({'request':pilot,'request_sha256':digest(pilot)}))
            validation={'known_answers_checked':20,'calibration_requests':136,'A_B_A_repeated_answers_match':True,
                'actual_model_Graph_replay_logged':True,'native_allocation':{'token_capacity':6400}}
            vp=prior/'output/bf16/validation_result.json';vp.write_text(json.dumps(validation))
            log=prior/'output/bf16/server.log';log.write_text(self.options_log())
            terminal={'status':'COMPLETE','gpu_release_verified':True,'live_owned_containers':[],
                'request_sha256':digest(pilot)};tp=prior/'result.json';tp.write_text(json.dumps(terminal))
            proof={key:formal[key] for key in ('model_id','model_config_sha256','image_id','binary_sha256',
                'server_source_commit','python_source_commit','service','gpu_uuids')}
            proof.update(state='MIXED_PREFILL_BF16_PILOT_INDEPENDENTLY_VERIFIED',scheduling_profile=MIXED_PREFILL_PROFILE,
                run_id='pilot',request_sha256=digest(pilot),terminal_result_sha256=hashlib.sha256(tp.read_bytes()).hexdigest(),
                validation_result_sha256=hashlib.sha256(vp.read_bytes()).hexdigest(),known_answer_checks_verified=20,
                calibration_requests_verified=136,actual_observer_GPU_Graph_passed=True,
                actual_runtime_options_verified=True,server_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                resolved_runtime_options=resolved_scheduling_options(log.read_text(),pilot['service']),
                observed_running_during_long_requests=2,observed_active_FullKV_pages_during_long_requests=90,
                largest_single_request_page_upper_bound=80)
            formal['remote_output']=str(root)
            def check(value):
                path=root/'input/mixed_prefill_pilot_audit.json';path.write_text(json.dumps(value))
                formal['mixed_prefill_pilot_audit_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                verify_mixed_prefill_prerequisites(formal,base,os.getuid())
            check(proof)
            for changes in [{'observed_running_during_long_requests':1},
                    {'observed_active_FullKV_pages_during_long_requests':101},
                    {'observed_active_FullKV_pages_during_long_requests':float('inf')},
                    {'observed_active_FullKV_pages_during_long_requests':80}]:
                with self.subTest(changes=changes),self.assertRaises(ApprovalError):check({**proof,**changes})
            terminal['status']='FAILED';tp.write_text(json.dumps(terminal))
            changed={**proof,'terminal_result_sha256':hashlib.sha256(tp.read_bytes()).hexdigest()}
            with self.assertRaises(ApprovalError):check(changed)

    def test_old_quality_snapshot_cannot_qualify_new_scheduling_geometry(self):
        _,old=_service();old['service'].update(max_batch=16,linear_state_slots=32,max_memory_utilization=.9)
        quality={key:old[key] for key in ('model_id','binary_sha256','server_source_commit',
            'python_source_commit','operators','service')}
        quality.update(arm='bf16',kv_scales=None,prediction_count=1720,
            all_raw_API_and_usage_contracts_independently_verified=True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'quality.json';raw=json.dumps(quality).encode();path.write_bytes(raw)
            sha=hashlib.sha256(raw).hexdigest()
            verify_quality_reference(old,root,os.getuid(),'quality.json',sha)
            with self.assertRaisesRegex(ValueError,'deployment differs'):
                verify_quality_reference(mixed(old),root,os.getuid(),'quality.json',sha)

    def test_explicit_bounded_BF16_pilot_sets_separate_chunk_and_batch(self):
        approval,request=_service();request=mixed(request);validate_request(request,approval)
        command=native_server_command(request,'bf16')
        self.assertIn('--max_tokens_per_batch=8192',command)
        self.assertIn('--max_tokens_per_chunk_for_prefill=2048',command)
        self.assertIn('--enable_graph=true',command);self.assertIn('--enable_prefix_cache=true',command)
        self.assertEqual(request['calibration_count'],136)

    def test_old_capsules_remain_exact_and_do_not_acquire_implicit_tuning(self):
        approval,request=_service();validate_request(request,approval)
        command=native_server_command(request,'bf16')
        self.assertIn('--max_tokens_per_batch=2048',command)
        self.assertFalse(any(arg.startswith('--max_tokens_per_chunk_for_prefill=') for arg in command))
        request['service']['max_tokens_per_batch']=8192
        with self.assertRaises(ApprovalError):validate_request(request,approval)

    def test_unknown_or_unmatched_profile_is_rejected(self):
        approval,request=_service();request=mixed(request)
        for change in [{'scheduling_profile':'auto'},{'range_observation':False},
                       {'service':{**request['service'],'max_tokens_per_batch':2048}},
                       {'service':{**request['service'],'max_tokens_per_chunk_for_prefill':8192}},
                       {'service':{**request['service'],'max_memory_utilization':.95}},
                       {'gpu_uuids':[]}]:
            with self.subTest(change=change),self.assertRaises(ApprovalError):
                validate_request({**request,**change},approval)

    def test_new_profile_keeps_full_quality_and_observer_exclusion(self):
        approval,request=_service();request=mixed(request)
        request.update(range_observation=False,max_wall_seconds=7200,
            workload={'kind':QUALITY_KIND,'scope':'natural_stop_core_quality_generations_not_service_performance'},
            mixed_prefill_pilot_audit_sha256='f'*64,
            quality_evaluation={'requests':1720,'client_concurrency':16,'client_token_budget':160000,
                'request_timeout_seconds':600,'max_seconds':4800,'requests_sha256':'a'*64,'source_manifest_sha256':'b'*64})
        validate_request(request,approval)
        for change in [{'range_observation':True},{'mixed_prefill_pilot_audit_sha256':None},
                       {'quality_evaluation':{**request['quality_evaluation'],'requests':80}}]:
            with self.subTest(change=change),self.assertRaises(ApprovalError):
                validate_request({**request,**change},approval)
