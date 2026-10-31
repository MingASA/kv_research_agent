"""CPU checks for new-model admission without relaxing shared-server guards."""

import copy
import json
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.approval import ApprovalError, validate_scope
from scripts.controller.nextgen import GDN_OVERLAY, KIND, MODEL, PAGES_KIND, PAGES_SOURCE_FILES, SCHEMA, SOURCE_FILES, _stream_case, native_python_environment, server_command, verify_startup_recovery
from scripts.controller.policy import docker_command, validate_request
from scripts.controller.quota import budget_key
from test_execution import fixture


def _fixture():
    approval, request, now = fixture()
    approval['read_only_model_path'] = '/fixture/models/' + MODEL
    request.update(schema=SCHEMA, model_id=MODEL, model_path=approval['read_only_model_path'], arms=['bf16'],
                   python_source_repo=request['remote_output']+'/source_repo', python_source_commit='b'*40,
                   dataset_path=request['remote_output']+'/input/validation.json', prefix_cache=True,
                   backend_source_sha256={name:'a'*64 for name in SOURCE_FILES},
                   operators={'bf16':{'planner':'fa3','decode':'flashinfer'}},
                   workload={'kind':KIND,'scope':'correctness_cache_lifecycle_not_performance'},
                   service={'max_batch':8,'max_tokens_per_batch':2048,'linear_state_slots':16,
                            'max_context':16384,'max_memory_utilization':.85},
                   max_context_scope='validation_input_budget_not_server_context_cap')
    return approval, request, now


class NextgenContractTests(unittest.TestCase):
    def test_NVFP4_pages_gate_is_a_distinct_operator_job_and_requires_complete_native_sources(self):
        approval,request,_=_fixture()
        request.update(arms=['nvfp4'],max_wall_seconds=1800,
                       server_binary_execution='none',
                       workload={'kind':PAGES_KIND,'scope':'SM90_native_compressed_pages_no_model_or_performance'},
                       backend_source_sha256={name:'a'*64 for name in PAGES_SOURCE_FILES},
                       operators={'nvfp4':{'planner':'fa2','decode':'flashinfer'}})
        validate_request(request,approval)
        for field,value in [('arms',['bf16']),('backend_source_sha256',{name:'a'*64 for name in SOURCE_FILES}),
                            ('operators',{'nvfp4':{'planner':'fa3','decode':'xqa'}}),
                            ('workload',{'kind':PAGES_KIND,'scope':'production_model_performance'})]:
            with self.subTest(field=field),self.assertRaises(ApprovalError):
                validate_request({**request,field:value},approval)
        with self.assertRaises(ApprovalError):validate_request({**request,'server_binary_execution':'required'},approval)
        approval,server,_=_fixture()
        with self.assertRaises(ApprovalError):validate_request({**server,'server_binary_execution':'none'},approval)

    def test_dependency_repair_requires_a_distinct_verified_private_overlay(self):
        approval, request, _ = _fixture()
        request.update(startup_seconds=1500,replicate=2,
                       startup_recovery={'prior_run_id':'initial-failed','prior_result_sha256':'a'*64,
                                         'reason':'weight_load_exceeded_initial900s_limit'},
                       environment_recovery={'prior_run_id':'dependency-failed','prior_result_sha256':'b'*64,
                                             'overlay_name':GDN_OVERLAY,'overlay_manifest_sha256':'c'*64,
                                             'CPU_import_evidence_sha256':'d'*64,
                                             'reason':'missing_DSL_core_and_embedded_site_path'})
        validate_request(request,approval)
        for field,value in [('overlay_name','other-package'),('CPU_import_evidence_sha256',''),
                            ('reason','install_global_dependencies'),('prior_run_id',request['run_id'])]:
            changed=copy.deepcopy(request);changed['environment_recovery'][field]=value
            with self.subTest(field=field),self.assertRaises(ApprovalError):validate_request(changed,approval)

    def test_embedded_python_receives_only_the_fixed_overlay_and_keeps_existing_libraries(self):
        _,request,_=_fixture();original={'PYTHONPATH':'/existing/runtime','LD_LIBRARY_PATH':'/existing/cuda'}
        self.assertEqual(native_python_environment(request,original),original)
        request['environment_recovery']={'overlay_name':GDN_OVERLAY}
        fixed=native_python_environment(request,original)
        base=request['approved_workspace']+'/agent_work/envs/'+GDN_OVERLAY+'/site-packages'
        self.assertEqual(fixed['PYTHONPATH'],base+'/nvidia_cutlass_dsl/dsl_packages:'+base+':/existing/runtime')
        self.assertEqual(fixed['LD_LIBRARY_PATH'],original['LD_LIBRARY_PATH'])
        self.assertEqual(original['PYTHONPATH'],'/existing/runtime')

    def test_longer_startup_is_one_audited_retry_and_preserves_runtime(self):
        approval, request, _ = _fixture()
        request.update(startup_seconds=1500,replicate=1,
                       startup_recovery={'prior_run_id':'prior-failed','prior_result_sha256':'a'*64,
                                         'reason':'weight_load_exceeded_initial900s_limit'})
        validate_request(request,approval)
        for field,value in [('replicate',2),('startup_seconds',2000),('startup_recovery',None)]:
            with self.subTest(field=field),self.assertRaises(ApprovalError):
                validate_request({**request,field:value},approval)
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);prior=base/'agent_work/runs/prior-failed';(prior/'output/bf16').mkdir(parents=True)
            raw=json.dumps({'status':'FAILED','gpu_release_verified':True,'live_owned_containers':[], 'arm_results':[]}).encode()
            (prior/'result.json').write_bytes(raw)
            request['startup_recovery']['prior_result_sha256']=hashlib.sha256(raw).hexdigest()
            prior_request=copy.deepcopy(request);prior_request['service']['max_context']=8192
            request['startup_recovery']['context_budget_correction']={
                'prior_nominal_bound':8192,'new_input_budget':16384,
                'validated_input_max_prompt_plus_output':8890,'tokenization_evidence_sha256':'a'*64}
            (prior/'ticket.json').write_text(json.dumps({'request':prior_request}))
            (prior/'output/bf16/container.log').write_text('TimeoutError: Native hybrid startup deadline exceeded')
            verify_startup_recovery(request,base,os.getuid())
            changed=copy.deepcopy(request);changed['python_source_commit']='f'*40
            with self.assertRaises(ApprovalError):verify_startup_recovery(changed,base,os.getuid())

    def test_new_model_native_BF16_contract_is_admitted(self):
        approval, request, _ = _fixture()
        validate_request(request, approval)
        command = server_command(request)
        self.assertIn('--backend=llm', command)
        self.assertIn('--max_linear_state_cache_slots=16', command)
        self.assertIn('--enable_chunked_prefill=true', command)
        self.assertIn('--python_graph_backend=cudagraphs', command)

    def test_initial_contract_rejects_unverified_formats_and_configuration(self):
        for field, value in [('arms',['fp8_e4m3']),('model_id','Qwen2.5-14B-Instruct'),
                             ('model_path','/foreign/model'),('python_source_repo','/fixture/project/other'),
                             ('dataset_path','/fixture/project/other.json'),('graph_mode','off'),
                             ('operators',{'bf16':{'planner':'auto','decode':'flashinfer'}}),
                             ('service',{'max_batch':128}),('backend_source_sha256',{})]:
            with self.subTest(field=field):
                approval, request, _ = _fixture()
                request[field] = value
                with self.assertRaises(ApprovalError): validate_request(request, approval)

    def test_model_scope_update_preserves_cumulative_budget_identity(self):
        approval, _, _ = fixture()
        changed = copy.deepcopy(approval)
        changed['read_only_model_path'] = '/fixture/models/' + MODEL
        changed['container_limits']['memory_gib'] = 128
        self.assertEqual(budget_key(approval),budget_key(changed))

    def test_existing_quota_unknown_and_busy_gpu_gates_apply_to_new_model(self):
        approval, request, now = _fixture()
        usage={'known_run_ids':[],'reconciled':True,'uncertain_runs':[],'active_gpu_uuids':[],
               'gpu_hours_consumed':0,'gpu_hours_reserved':0,'disk_gib_consumed':0,'disk_gib_reserved':0}
        validate_scope(approval,request,usage,now)
        for change in [{'uncertain_runs':['prior']},{'active_gpu_uuids':request['gpu_uuids']},
                       {'gpu_hours_consumed':3},{'known_run_ids':[request['run_id']]}]:
            with self.subTest(change=change), self.assertRaises(ApprovalError):
                validate_scope(approval,request,{**usage,**change},now)

    def test_container_has_only_verified_readonly_model_and_existing_isolation(self):
        approval, request, _ = _fixture()
        ticket={'request':request,'approval':approval,'reservation_id':'a'*32,'request_sha256':'b'*64}
        command=docker_command(ticket,'bf16',0)
        self.assertIn('type=bind,src='+request['model_path']+',dst=/models/'+MODEL+',readonly',command)
        for flag in ['--read-only','--cap-drop','--pids-limit','--network','--ipc','--user']:
            self.assertIn(flag,command)

    def test_legacy_model_mount_cannot_be_changed_by_extra_model_id(self):
        approval, request, _ = fixture()
        request['model_id']='../../untrusted'
        ticket={'request':request,'approval':approval,'reservation_id':'a'*32,'request_sha256':'b'*64}
        self.assertNotIn('untrusted',' '.join(docker_command(ticket,'bf16',0)))

    def test_invalid_SSE_is_preserved_and_not_accepted(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self,*_args): return False
            def __iter__(self): return iter([b'data: not-json\n'])
        with patch('urllib.request.urlopen',return_value=Response()):
            row=_stream_case({'id':'bad','messages':[],'expected':'answer'},MODEL)
        self.assertFalse(row['contract_valid'])
        self.assertEqual(row['raw_SSE'],['data: not-json'])
        self.assertIn('JSONDecodeError',row['error'])


if __name__=='__main__': unittest.main()
