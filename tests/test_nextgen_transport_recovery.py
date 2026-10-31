"""A zero-GPU transport recovery cannot hide a real or uncertain prior run."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.absent_recovery import verify_absence_certificate
from scripts.controller.approval import ApprovalError
from scripts.controller.durable import canonical_bytes,digest
from scripts.controller.nextgen_service import verify_transport_recovery
from scripts.controller.policy import validate_request
from scripts.reconcile_absent_dispatch import terminal_from_witnesses
from test_nextgen_native_service import _service


class TransportRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.approval,self.prior=_service('fp8_e4m3');self.prior.update(run_id='prior',replicate=0)
        self.ticket={'request':self.prior,'reservation_id':'prior-nonce','request_sha256':digest(self.prior)}
        witness={'schema':'kvra-absent-dispatch-readonly-witness-v1','ok':True,'remote_writes':0,
            'run_id':'prior','reservation_id':'prior-nonce','request_sha256':digest(self.prior),
            'root_absent':True,'boot_id':'boot','monotonic_ns':1,'remote_rpc_processes':[],
            'run_processes':[],'all_run_containers':[],'selected_compute_apps':[]}
        self.certificate=terminal_from_witnesses(self.ticket,[witness,dict(witness,monotonic_ns=30_000_000_001)])
        self.request=copy.deepcopy(self.prior);self.request.update(run_id='new',replicate=1,
            remote_output=str(self.base/'agent_work/runs/new'),python_source_repo='new/source_repo',dataset_path='new/input/validation.json')
        self.input=Path(self.request['remote_output'])/'input';self.input.mkdir(parents=True)
        self.freeze()

    def freeze(self):
        (self.input/'transport_prior_request.json').write_bytes(canonical_bytes(self.prior))
        raw=canonical_bytes(self.certificate);(self.input/'transport_absence_certificate.json').write_bytes(raw)
        self.request['transport_recovery']={'prior_run_id':'prior','prior_request_sha256':digest(self.prior),
            'absence_certificate_sha256':hashlib.sha256(raw).hexdigest()}

    def verify(self):
        with patch('scripts.controller.nextgen_service.subprocess.check_output',return_value=''):
            verify_transport_recovery(self.request,self.base,os.getuid())

    def test_exact_non_launch_certificate_and_same_configuration_is_admitted(self):self.verify()

    def test_witness_activity_or_changed_boot_cannot_authorize_recovery(self):
        for name,value in [('root_absent',False),('remote_rpc_processes',[123]),('run_processes',[123]),
                           ('all_run_containers',['exists']),('boot_id','changed'),('monotonic_ns',30_000_000_000)]:
            certificate=copy.deepcopy(self.certificate);certificate['readonly_witnesses'][1][name]=value
            with self.subTest(name=name),self.assertRaises(ApprovalError):verify_absence_certificate(self.prior,certificate)

    def test_gpu_use_or_success_or_missing_bound_proof_is_rejected(self):
        for name,value in [('status','COMPLETE'),('gpu_release_verified',False),('allocation_started_monotonic_ns',1),
                           ('remote_result_file_created',True),('new_GPU_admission_authorized_by_this_proof',True),
                           ('readonly_witnesses',[])]:
            with self.subTest(name=name),self.assertRaises(ApprovalError):
                verify_absence_certificate(self.prior,self.certificate|{name:value})

    def test_changed_model_input_operator_or_prior_root_blocks_new_attempt(self):
        for name,value in [('binary_sha256','f'*64),('operators',{'fp8_e4m3':{'planner':'fa2','decode':'other'}}),
                           ('dataset_sha256','0'*64),('replicate',2)]:
            original=self.request[name];self.request[name]=value
            with self.subTest(name=name),self.assertRaises(ApprovalError):self.verify()
            self.request[name]=original
        (self.base/'agent_work/runs/prior').mkdir()
        with self.assertRaises(ApprovalError):self.verify()

    def test_contract_bounded_explicit_transport_retry_only(self):
        approval,request=_service('fp8_e4m3');request.update(replicate=1)
        with self.assertRaises(ApprovalError):validate_request(request,approval)
        request['transport_recovery']={'prior_run_id':'prior','prior_request_sha256':'a'*64,'absence_certificate_sha256':'b'*64}
        validate_request(request,approval)
        for change in ({'replicate':0},{'replicate':3},{'resource_recovery':{'hidden':'second recovery'}},
                       {'transport_recovery':{'prior_run_id':'../escape'}}):
            with self.subTest(change=change),self.assertRaises(ApprovalError):validate_request(request|change,approval)


if __name__=='__main__':unittest.main()
