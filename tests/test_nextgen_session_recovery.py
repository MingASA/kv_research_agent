"""A session-stop repair cannot hide a changed deployment or a completed run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from collections import Counter

from scripts.controller.durable import digest,canonical_bytes
from scripts.controller.nextgen_performance import SESSION_REPAIR,validate_session_recovery,verify_session_recovery
from scripts.controller.policy import validate_request
from scripts.controller.approval import ApprovalError
import test_nextgen_performance_contract as performance_fixtures


class SessionProtocolRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.approval,self.prior=performance_fixtures.NativePerformanceContractTests().fixture()
        self.approval['approved_workspace']=str(self.base)
        self.old=self.base/'agent_work/runs/prior';self.root=self.base/'agent_work/runs/current'
        self.output=self.old/'output/nvfp4';(self.output/'measured/artifacts').mkdir(parents=True)
        (self.old/'input/production').mkdir(parents=True);(self.root/'input').mkdir(parents=True)
        self.prior.update(run_id='prior',approved_workspace=str(self.base),remote_output=str(self.old),
            python_source_repo=str(self.old/'source_repo'),dataset_path=str(self.old/'input/validation.json'),
            source_repo=str(self.base/'source'),server_binary=str(self.base/'xllm'),replicate=0)
        self.prior['performance_evaluation']['case']='long-prefix-c16'
        self.request=copy.deepcopy(self.prior);self.request.update(run_id='current',remote_output=str(self.root),
            python_source_repo=str(self.root/'source_repo'),dataset_path=str(self.root/'input/validation.json'),replicate=1)
        self.request['performance_evaluation']['full_CLI_protocol_CPU_audit_sha256']='9'*64
        self.result={'run_id':'prior','request_sha256':digest(self.prior),'status':'FAILED',
                     'gpu_release_verified':True,'live_owned_containers':[],'arm_results':[]}
        sessions=[{'session_id':f'doc-{i}','payloads':[{'question':str(i),'max_tokens':128}]*2} for i in range(16)]
        self.source=canonical_bytes({'data':sessions});expected=[(row['session_id'],t) for row in sessions for t in (0,1)]
        self.rows=[{'metadata':{'conversation_id':s,'turn_index':t,'x_request_id':str(n),
            'benchmark_phase':'profiling','was_cancelled':False}} for n,(s,t) in enumerate(expected[:27]+expected[:5])]
        self.raw=b'\n'.join(canonical_bytes(row) for row in self.rows)+b'\n'
        counts=Counter((r['metadata']['conversation_id'],r['metadata']['turn_index']) for r in self.rows)
        self.failure={'run_id':'prior','terminal':'FAILED','GPU_release_verified':True,
            'valid_intended_performance_measurement':False,'actual_rows':32,'expected_unique_session_turns':32,
            'observed_unique_session_turns':27,'profile_export_sha256':hashlib.sha256(self.raw).hexdigest(),
            'source_sha256':hashlib.sha256(self.source).hexdigest(),
            'duplicate_session_turns':[{'session':s,'turn':t,'count':n} for (s,t),n in sorted(counts.items()) if n>1],
            'missing_session_turns':[{'session':s,'turn':t} for s,t in sorted(set(expected)-counts.keys())]}
        (self.old/'ticket.json').write_bytes(canonical_bytes({'request':self.prior,'request_sha256':digest(self.prior)}))
        result_raw=canonical_bytes(self.result);(self.old/'result.json').write_bytes(result_raw)
        (self.output/'measured/artifacts/profile_export.jsonl').write_bytes(self.raw)
        (self.old/'input/production/long_prefix16x2.json').write_bytes(self.source)
        (self.output/'validation_result.json').write_bytes(canonical_bytes({'model_loaded':True,
            'actual_model_Graph_replay_logged':True,'A_B_A_repeated_answers_match':True,'known_answers_checked':20}))
        (self.output/'container.log').write_text('ValueError: Duplicated/missing actual request/session/phase')
        failure_raw=canonical_bytes(self.failure);(self.root/'input/performance_session_failure_audit.json').write_bytes(failure_raw)
        self.request['session_protocol_recovery']={'prior_run_id':'prior','prior_request_sha256':digest(self.prior),
            'prior_result_sha256':hashlib.sha256(result_raw).hexdigest(),'repair_kind':SESSION_REPAIR,
            'CPU_repair_evidence_sha256':'9'*64,'failure_audit_sha256':hashlib.sha256(failure_raw).hexdigest()}

    def verify(self):verify_session_recovery(self.request,self.root,os.getuid())

    def test_exact_released_failed_original_raw_cohort_admits_one_explicit_repair(self):
        validate_request(self.request,self.approval);self.verify()

    def test_model_configuration_or_workload_change_is_not_a_protocol_repair(self):
        changes=[('binary_sha256','f'*64),('kv_scales',{'key':1,'value':1}),('service',{}),
                 ('operators',{}),('arms',['bf16']),('gpu_uuids',['another'])]
        for key,value in changes:
            old=self.request[key];self.request[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.verify()
            self.request[key]=old
        self.request['performance_evaluation']['concurrency']=8
        with self.assertRaises(ValueError):self.verify()

    def test_completed_or_unreleased_predecessor_and_second_retry_are_rejected(self):
        sha=self.request['session_protocol_recovery']['prior_result_sha256']
        for field,value in [('status','COMPLETE'),('gpu_release_verified',False),('live_owned_containers',['live']),
                            ('arm_results',[{'already':'measured'}])]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_session_recovery(self.prior,self.result|{field:value},self.request,self.failure,sha)
        self.request['replicate']=2
        with self.assertRaises(ApprovalError):validate_request(self.request,self.approval)

    def test_recovery_cannot_be_combined_with_another_reason_or_plain_service(self):
        self.request['resource_recovery']={'reason':'foreign_GPU_process_before_model_start'}
        with self.assertRaises(ApprovalError):validate_request(self.request,self.approval)
        del self.request['resource_recovery'];self.request['workload']={'kind':'native_hybrid_format_service_validation'}
        with self.assertRaises(ApprovalError):validate_request(self.request,self.approval)

    def test_changed_original_raw_or_fake_failure_manifest_fails_closed(self):
        (self.output/'measured/artifacts/profile_export.jsonl').write_bytes(self.raw+b'\n')
        with self.assertRaises(ValueError):self.verify()
        (self.output/'measured/artifacts/profile_export.jsonl').write_bytes(self.raw)
        self.failure['observed_unique_session_turns']=32
        raw=canonical_bytes(self.failure);(self.root/'input/performance_session_failure_audit.json').write_bytes(raw)
        self.request['session_protocol_recovery']['failure_audit_sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(ValueError):self.verify()


if __name__=='__main__':unittest.main()
