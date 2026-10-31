"""Real E2E cases are distinct from quality and retain the production profile."""

import copy
import unittest

from scripts.controller.policy import validate_request
from scripts.controller.nextgen_performance import PERF_KIND
from test_nextgen_native_service import _service


class NativePerformanceContractTests(unittest.TestCase):
    def fixture(self):
        approval,request=_service('nvfp4')
        request.update(max_wall_seconds=7200,range_observation=False,
            service={**request['service'],'max_batch':16,'linear_state_slots':32,'max_memory_utilization':.90},
            workload={'kind':PERF_KIND,'scope':'natural_stop_real_text_AIPerf_end_to_end'},
            performance_evaluation={'case':'chat-c16','concurrency':16,'input_manifest_sha256':'a'*64,
                'full_CLI_protocol_CPU_audit_sha256':'b'*64,'aiperf_cli_config_sha256':'c'*64,
                'quality_scores_sha256':'d'*64,'max_seconds':3600})
        return approval,request
    def test_production_case_is_observer_free_and_matched_to_quality_service(self):
        approval,request=self.fixture();validate_request(request,approval)
    def test_different_shape_or_synthetic_unknown_load_is_rejected(self):
        approval,request=self.fixture()
        for changes in ({'range_observation':True},{'service':{**request['service'],'max_batch':128}},
                        {'performance_evaluation':{**request['performance_evaluation'],'case':'fixed-padding-128k'}},
                        {'performance_evaluation':{**request['performance_evaluation'],'concurrency':128}},
                        {'performance_evaluation':{**request['performance_evaluation'],'max_seconds':100000}}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_request(request|changes,approval)
    def test_benchmark_cannot_hide_inside_plain_service_or_quality_kind(self):
        approval,request=_service('bf16');request['performance_evaluation']={'case':'chat-c16'}
        with self.assertRaises(ValueError):validate_request(request,approval)


if __name__=='__main__':unittest.main()
