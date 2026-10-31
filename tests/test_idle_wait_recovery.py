"""A fresh idle-wait attempt cannot hide old GPU use, output or changed settings."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.durable import canonical_bytes,digest
from scripts.controller.idle_wait_recovery import ERROR,REASON,verify_idle_wait_recovery,verify_zero_allocation_wait
from scripts.controller.policy import validate_request
from test_nextgen_native_service import _service


class IdleWaitRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        _,self.prior=_service('fp8_e4m3');self.root=self.base/'agent_work/runs/prior'
        self.root.mkdir(parents=True)
        self.prior.update(run_id='prior',remote_output=str(self.root),python_source_repo=str(self.root/'source_repo'),
                          dataset_path=str(self.root/'input/validation.json'))
        self.ticket={'request':self.prior,'request_sha256':digest(self.prior),'reservation_id':'original-reservation'}
        (self.root/'ticket.json').write_bytes(canonical_bytes(self.ticket))
        self.result={'run_id':'prior','request_sha256':digest(self.prior),'reservation_id':'original-reservation',
            'status':'NOT_LAUNCHED','error':ERROR,'gpu_release_verified':True,'live_owned_containers':[],
            'allocation_started_monotonic_ns':None,'allocation_ended_monotonic_ns':None,
            'allocation_intervals_ns':[],'arm_results':[]}
        self.request=copy.deepcopy(self.prior);new=self.base/'agent_work/runs/new'
        self.request.update(run_id='new',replicate=1,remote_output=str(new),python_source_repo=str(new/'source_repo'),
                            dataset_path=str(new/'input/validation.json'))
        self.identity={'pid':999999999,'uid':os.getuid(),'start_ticks':1,'cgroup':'owned-fixture',
                       'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        (self.root/'SUPERVISOR_STARTED.json').write_bytes(canonical_bytes({'identity':self.identity}))
        (self.root/'events.jsonl').write_bytes(canonical_bytes({'action':'WAIT_GPU'}))
        self.freeze()

    def freeze(self):
        raw=canonical_bytes(self.result);(self.root/'result.json').write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
        (self.root/'WATCHDOG_FINISHED.json').write_bytes(canonical_bytes({'request_sha256':digest(self.prior),
            'gpu_release_verified':True,'result_sha256':sha}))
        self.request['idle_wait_recovery']={'prior_run_id':'prior','prior_request_sha256':digest(self.prior),
                                          'prior_result_sha256':sha,'reason':REASON}

    def verify(self):
        with patch('scripts.controller.idle_wait_recovery.process_identity',side_effect=FileNotFoundError):
            verify_idle_wait_recovery(self.request,self.base,os.getuid())

    def test_exact_terminal_unallocated_wait_can_receive_one_explicit_attempt(self):self.verify()

    def test_any_GPU_use_uncertainty_or_other_failure_is_rejected(self):
        for name,value in (('status','FAILED'),('error','TimeoutError: worker deadline'),('gpu_release_verified',False),
                           ('allocation_started_monotonic_ns',1),('allocation_ended_monotonic_ns',2),
                           ('allocation_intervals_ns',[{'start_ns':1,'end_ns':2}]),('arm_results',[{}])):
            with self.subTest(name=name),self.assertRaises(ValueError):
                verify_zero_allocation_wait(self.prior,self.result|{name:value})

    def test_changed_settings_or_hidden_replicate_is_not_a_wait_recovery(self):
        for name,value in (('binary_sha256','f'*64),('dataset_sha256','0'*64),('replicate',0),('replicate',2),
                           ('gpu_uuids',['different']),('kv_scales',{'key':.25,'value':.5})):
            original=self.request[name];self.request[name]=value
            with self.subTest(name=name),self.assertRaises(ValueError):self.verify()
            self.request[name]=original

    def test_container_or_worker_output_or_creation_event_blocks_zero_allocation_claim(self):
        files=[('owned_containers.json',canonical_bytes({'containers':[{'id':'exists'}]})),
               ('output/quality_predictions.jsonl',b'{}\n'),
               ('events.jsonl',canonical_bytes({'action':'WAIT_GPU'})+canonical_bytes({'action':'OWNED_CONTAINER_CREATED'}))]
        for name,raw in files:
            path=self.root/name;path.parent.mkdir(exist_ok=True);old=path.read_bytes() if path.exists() else None
            path.write_bytes(raw)
            with self.subTest(name=name),self.assertRaises(ValueError):self.verify()
            if old is None:path.unlink()
            else:path.write_bytes(old)

    def test_live_original_supervisor_or_unbound_watchdog_cannot_recover(self):
        with patch('scripts.controller.idle_wait_recovery.process_identity',return_value=self.identity),\
                patch('scripts.controller.idle_wait_recovery.process_lifecycle',return_value={'state':'S'}):
            with self.assertRaisesRegex(ValueError,'still alive'):
                verify_idle_wait_recovery(self.request,self.base,os.getuid())
        (self.root/'WATCHDOG_FINISHED.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'watchdog evidence'):self.verify()

    def test_contract_requires_explicit_bounded_nonoverlapping_recovery_metadata(self):
        approval,request=_service('fp8_e4m3');request.update(replicate=1,idle_wait_recovery={
            'prior_run_id':'prior','prior_request_sha256':'a'*64,'prior_result_sha256':'b'*64,'reason':REASON})
        validate_request(request,approval)
        for change in ({'replicate':0},{'replicate':3},{'resource_recovery':{}},
                       {'idle_wait_recovery':{**request['idle_wait_recovery'],'reason':'unknown_timeout'}}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_request(request|change,approval)

    def test_missing_subset_can_continue_only_with_the_explicit_zero_GPU_wait_exception(self):
        from scripts.controller.nextgen import QUALITY_KIND
        approval,request=_service('fp8_e4m3');request.update(replicate=1,max_wall_seconds=7200,
            workload={'kind':QUALITY_KIND,'scope':'natural_stop_core_quality_generations_not_service_performance'},
            service={**request['service'],'max_batch':16,'linear_state_slots':32,'max_memory_utilization':.9},
            quality_evaluation={'requests':802,'client_concurrency':16,'client_token_budget':160000,
                'request_timeout_seconds':600,'max_seconds':4800,'requests_sha256':'a'*64,'source_manifest_sha256':'b'*64},
            quality_completion={'anchor_run_id':'original-t','partial_raw_certificate_sha256':'c'*64},
            idle_wait_recovery={'prior_run_id':'prior-y','prior_request_sha256':'d'*64,
                'prior_result_sha256':'e'*64,'reason':REASON})
        validate_request(request,approval)
        request.pop('idle_wait_recovery')
        with self.assertRaises(ValueError):validate_request(request,approval)


if __name__=='__main__':unittest.main()
