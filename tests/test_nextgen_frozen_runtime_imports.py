"""Frozen performance packages must work without the control checkout."""

from pathlib import Path
import hashlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_nextgen_service import validate_frozen_performance_imports
from scripts.controller.durable import digest,write_new_json
from scripts.controller.nextgen_service import verify_performance_packaging_recovery,verify_transport_recovery
from scripts.controller.nextgen_performance import PERF_KIND
from scripts.controller.policy import validate_request
import test_nextgen_performance_contract as performance_contract


class FrozenRuntimeImportsTests(unittest.TestCase):
    def test_original_alias_fails_and_standalone_auditor_closes_dependencies(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            runtime=Path(temporary)/'runtime'
            package=runtime/'kv_controller'
            shutil.copytree(root/'scripts/controller',package,
                            ignore=shutil.ignore_patterns('__pycache__'))
            with self.assertRaisesRegex(ValueError,"No module named 'scripts'"):
                validate_frozen_performance_imports(runtime)
            shutil.copyfile(root/'scripts/audit_native_results.py',package/'evidence.py')
            proof=validate_frozen_performance_imports(runtime)
            self.assertTrue(proof['frozen_runtime_imports_verified'])
            self.assertFalse(proof['control_checkout_or_site_packages_required'])
            self.assertFalse(proof['GPU_used'])

    def test_packaging_retry_is_explicit_bounded_and_exclusive(self):
        approval,request=performance_contract.NativePerformanceContractTests().fixture()
        recovery={'prior_run_id':'original-run','prior_result_sha256':'a'*64,
                  'repair_kind':'standalone_performance_audit_import','audit_module_sha256':'b'*64,
                  'CPU_repair_evidence_sha256':'c'*64}
        request.update(replicate=1,runtime_packaging_recovery=recovery)
        validate_request(request,approval)
        transport={'prior_run_id':'absent-dispatch','prior_request_sha256':'d'*64,
                   'absence_certificate_sha256':'e'*64}
        validate_request(request|{'replicate':2,'transport_recovery':transport},approval)
        for changes in ({'replicate':0},{'replicate':2},{'resource_recovery':{}},
                        {'replicate':3,'transport_recovery':transport},
                        {'replicate':2,'transport_recovery':transport|{'absence_certificate_sha256':'invalid'}},
                        {'runtime_packaging_recovery':recovery|{'repair_kind':'arbitrary_retry'}},
                        {'runtime_packaging_recovery':recovery|{'prior_run_id':request['run_id']}}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_request(request|changes,approval)

    def test_actual_predecessor_and_exact_repair_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);old=base/'agent_work/runs/original-run';new=base/'agent_work/runs/new-run'
            output=old/'output/bf16';output.mkdir(parents=True)
            (new/'input').mkdir(parents=True);(new/'runtime/kv_controller').mkdir(parents=True)
            prior={'run_id':'original-run','remote_output':str(old),'replicate':0,'arms':['bf16'],
                   'workload':{'kind':PERF_KIND},'binary_sha256':'d'*64}
            current=prior|{'run_id':'new-run','remote_output':str(new),'replicate':1}
            result={'status':'FAILED','gpu_release_verified':True,'live_owned_containers':[],
                    'arm_results':[],'run_id':prior['run_id'],'request_sha256':digest(prior)}
            write_new_json(old/'ticket.json',{'request':prior,'request_sha256':digest(prior)})
            write_new_json(old/'result.json',result)
            write_new_json(output/'validation_result.json',{'known_answers_checked':20,
                'A_B_A_repeated_answers_match':True,'actual_model_Graph_replay_logged':True})
            (output/'container.log').write_text("ModuleNotFoundError: No module named 'scripts.audit_native_results'\n")
            auditor=new/'runtime/kv_controller/evidence.py';auditor.write_bytes(b'bound auditor fixture')
            proof={'state':'FROZEN_PERFORMANCE_AUDIT_IMPORT_REPRODUCED_FIXED_CPU','original_run_id':prior['run_id'],
                   'original_terminal_sha256':hashlib.sha256((old/'result.json').read_bytes()).hexdigest(),
                   'standalone_auditor_sha256':hashlib.sha256(auditor.read_bytes()).hexdigest(),
                   'original_alias_failed_isolated_import':True,'standalone_auditor_passed_isolated_import':True,
                   'model_source_or_binary_or_tolerance_changed':False}
            write_new_json(new/'input/packaging_repair_CPU.json',proof)
            recovery={'prior_run_id':prior['run_id'],'prior_result_sha256':proof['original_terminal_sha256'],
                      'repair_kind':'standalone_performance_audit_import',
                      'audit_module_sha256':proof['standalone_auditor_sha256'],
                      'CPU_repair_evidence_sha256':hashlib.sha256((new/'input/packaging_repair_CPU.json').read_bytes()).hexdigest()}
            current['runtime_packaging_recovery']=recovery
            verify_performance_packaging_recovery(current,base,os.getuid())
            absent=current|{'run_id':'never-prepared-run',
                            'remote_output':str(base/'agent_work/runs/never-prepared-run')}
            witnesses=[{'schema':'kvra-absent-dispatch-readonly-witness-v1','ok':True,'remote_writes':0,
                        'root_absent':True,'run_id':absent['run_id'],'request_sha256':digest(absent),
                        'reservation_id':'reservation','remote_rpc_processes':[],'run_processes':[],
                        'all_run_containers':[],'boot_id':'same-boot','monotonic_ns':value}
                       for value in (100,30_000_000_100)]
            certificate={'status':'NOT_LAUNCHED','gpu_release_verified':True,'live_owned_containers':[],
                         'new_output_bytes':0,'allocation_started_monotonic_ns':None,
                         'allocation_ended_monotonic_ns':None,'remote_result_file_created':False,
                         'new_GPU_admission_authorized_by_this_proof':False,'run_id':absent['run_id'],
                         'request_sha256':digest(absent),'reservation_id':'reservation',
                         'readonly_witnesses':witnesses}
            write_new_json(new/'input/transport_prior_request.json',absent)
            write_new_json(new/'input/transport_absence_certificate.json',certificate)
            chained=current|{'replicate':2,'transport_recovery':{'prior_run_id':absent['run_id'],
                'prior_request_sha256':digest(absent),'absence_certificate_sha256':
                hashlib.sha256((new/'input/transport_absence_certificate.json').read_bytes()).hexdigest()}}
            verify_performance_packaging_recovery(chained,base,os.getuid())
            with patch('scripts.controller.nextgen_service.subprocess.check_output',return_value=''):
                verify_transport_recovery(chained,base,os.getuid())
                with self.assertRaises(ValueError):
                    verify_transport_recovery(chained|{'binary_sha256':'e'*64},base,os.getuid())
            with self.assertRaises(ValueError):
                verify_performance_packaging_recovery(current|{'binary_sha256':'e'*64},base,os.getuid())
            (output/'warmup').mkdir()
            with self.assertRaises(ValueError):
                verify_performance_packaging_recovery(current,base,os.getuid())
            (output/'warmup').rmdir()
            auditor.write_bytes(b'changed after proof')
            with self.assertRaises(ValueError):
                verify_performance_packaging_recovery(current,base,os.getuid())


if __name__=='__main__':
    unittest.main()
