"""A reviewed resource retry must preserve provenance and prove no model ran."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.durable import canonical_bytes, digest
from scripts.controller.nextgen_service import verify_resource_recovery
from scripts.controller.policy import validate_request
from test_nextgen_native_service import _service


class ResourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name);self.approval,self.prior=_service('fp8_e4m3')
        self.prior.update(run_id='prior',replicate=0,experiment_key='a'*64)
        self.root=self.base/'agent_work/runs/prior';(self.root/'output/fp8_e4m3').mkdir(parents=True)
        self.request=copy.deepcopy(self.prior);self.request.update(run_id='new',replicate=1)
        self.result={'run_id':'prior','request_sha256':digest(self.prior),'status':'CONFLICT',
            'gpu_release_verified':True,'live_owned_containers':[],'arm_results':[],
            'error':'UnsafeState: Foreign GPU activity/reservation; stop this run only'}
        (self.root/'events.jsonl').write_bytes(canonical_bytes({'action':'GPU_CONFLICT_OBSERVATION',
            'reason':'live_unowned_gpu_process','process':{'uid':os.getuid()+1}}))
        self.freeze()

    def freeze(self):
        (self.root/'ticket.json').write_bytes(canonical_bytes({'request':self.prior,'request_sha256':digest(self.prior)}))
        (self.root/'result.json').write_bytes(canonical_bytes(self.result))
        self.request['resource_recovery']={'reason':'foreign_GPU_process_before_model_start','prior_run_id':'prior',
            'prior_result_sha256':hashlib.sha256((self.root/'result.json').read_bytes()).hexdigest()}

    def verify(self):verify_resource_recovery(self.request,self.base,os.getuid())

    def test_exact_released_pre_model_conflict_is_admitted(self):self.verify()

    def test_no_model_results_or_second_hidden_retry_is_admitted(self):
        for field,value in [('replicate',2),('kv_scales',{'key':1.,'value':1.}),
                            ('binary_sha256','f'*64),('gpu_uuids',['another-GPU'])]:
            with self.subTest(field=field):
                old=self.request[field];self.request[field]=value
                with self.assertRaises(ApprovalError):self.verify()
                self.request[field]=old

    def test_unreleased_running_or_completed_prior_is_rejected(self):
        original=copy.deepcopy(self.result)
        for field,value in [('status','COMPLETE'),('gpu_release_verified',False),
                            ('live_owned_containers',['still-live']),('arm_results',[{'result':'already-ran'}])]:
            with self.subTest(field=field):
                self.result=original|{field:value};self.freeze()
                with self.assertRaises(ApprovalError):self.verify()

    def test_any_model_start_or_output_artifact_prevents_retry(self):
        for name in ('server.log','server_command.json','quality_predictions.jsonl','measurement.json'):
            path=self.root/'output/fp8_e4m3'/name;path.write_bytes(b'')
            with self.subTest(name=name),self.assertRaises(ApprovalError):self.verify()
            path.unlink()

    def test_foreign_process_proof_and_terminal_hash_are_required(self):
        (self.root/'events.jsonl').write_bytes(canonical_bytes({'action':'GPU_CONFLICT_OBSERVATION',
            'reason':'live_unowned_gpu_process','process':{'uid':os.getuid()}}))
        with self.assertRaises(ApprovalError):self.verify()
        (self.root/'result.json').write_bytes(b'{}')
        with self.assertRaises(ApprovalError):self.verify()

    def test_contract_requires_explicit_recovery_and_first_retry_only(self):
        approval,request=_service('fp8_e4m3');request.update(replicate=1)
        with self.assertRaises(ApprovalError):validate_request(request,approval)
        request['resource_recovery']={'reason':'foreign_GPU_process_before_model_start',
            'prior_run_id':'prior','prior_result_sha256':'e'*64}
        validate_request(request,approval)
        for field,value in [('replicate',0),('replicate',2),('resource_recovery',{})]:
            with self.subTest(field=field),self.assertRaises(ApprovalError):
                validate_request(request|{field:value},approval)


if __name__=='__main__':unittest.main()
