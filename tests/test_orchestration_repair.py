"""Only the exact released scan race may use its independently tested repair."""

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.durable import canonical_bytes,digest
from scripts.controller.nextgen_service import verify_orchestration_recovery
from scripts.controller.policy import validate_request
from test_nextgen_native_service import _service


class OrchestrationRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.approval,self.prior=_service('fp8_e4m3');self.prior.update(run_id='prior',replicate=1)
        self.old=self.base/'agent_work/runs/prior';self.old.mkdir(parents=True)
        self.request=copy.deepcopy(self.prior);self.new=self.base/'agent_work/runs/new'
        self.request.update(run_id='new',replicate=2,remote_output=str(self.new))
        (self.new/'input').mkdir(parents=True);(self.new/'runtime/kv_controller').mkdir(parents=True)
        self.source=self.new/'runtime/kv_controller/supervisor.py';self.source.write_bytes(b'fixed-source')
        self.result={'status':'FAILED','gpu_release_verified':True,'live_owned_containers':[],'arm_results':[],
            'run_id':'prior','request_sha256':digest(self.prior),'error':"FileNotFoundError: [Errno 2] No such file or directory: '"+str(self.old/'output/fp8_e4m3/.quality_progress.json.unique')+"'"}
        (self.old/'ticket.json').write_bytes(canonical_bytes({'request':self.prior,'request_sha256':digest(self.prior)}))
        self.proof={'state':'BOUNDED_TREE_SCAN_RACE_REPRODUCED_AND_FIXED_CPU_ONLY','GPU_used':False,
            'old_code_result':{'FileNotFound':True},'repaired_code_result':{'FileNotFound':False,'bytes':8},
            'supervisor_source_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),
            'quota_limits_or_checks_removed':False,'complete_scan_retries_bounded':3}
        self.freeze()

    def freeze(self):
        raw=canonical_bytes(self.result);(self.old/'result.json').write_bytes(raw)
        proof=canonical_bytes(self.proof);(self.new/'input/orchestration_repair_CPU.json').write_bytes(proof)
        self.request['orchestration_recovery']={'prior_run_id':'prior','prior_result_sha256':hashlib.sha256(raw).hexdigest(),
            'repair_kind':'bounded_output_scan_atomic_progress_race','supervisor_source_sha256':self.proof['supervisor_source_sha256'],
            'CPU_repair_evidence_sha256':hashlib.sha256(proof).hexdigest()}

    def verify(self):verify_orchestration_recovery(self.request,self.base,os.getuid())

    def test_exact_released_scan_failure_and_exact_tested_source_are_admitted(self):self.verify()

    def test_unrelated_failure_unreleased_run_and_existing_completion_are_rejected(self):
        original=copy.deepcopy(self.result)
        for name,value in [('error','CUDA illegal access'),('status','UNKNOWN'),('gpu_release_verified',False),
                           ('arm_results',[{'already':'completed'}])]:
            self.result=original|{name:value};self.freeze()
            with self.subTest(name=name),self.assertRaises(ApprovalError):self.verify()
        self.result=original;self.freeze();(self.old/'COMPLETE').write_bytes(b'')
        with self.assertRaises(ApprovalError):self.verify()

    def test_changed_model_input_scale_or_missing_cpu_source_proof_is_rejected(self):
        for name,value in [('binary_sha256','0'*64),('dataset_sha256','0'*64),('kv_scales',{'key':1.,'value':1.})]:
            old=self.request[name];self.request[name]=value
            with self.subTest(name=name),self.assertRaises(ApprovalError):self.verify()
            self.request[name]=old
        self.source.write_bytes(b'untested-source')
        with self.assertRaises(ApprovalError):self.verify()

    def test_contract_repair_is_explicit_bounded_and_does_not_mix_recoveries(self):
        approval,request=_service('fp8_e4m3');request.update(replicate=2,
            orchestration_recovery=self.request['orchestration_recovery'])
        validate_request(request,approval)
        for change in ({'replicate':3},{'orchestration_recovery':{}},{'transport_recovery':{}},
                       {'resource_recovery':{'hidden':True}}):
            with self.subTest(change=change),self.assertRaises(ApprovalError):validate_request(request|change,approval)


if __name__=='__main__':unittest.main()
