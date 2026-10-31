"""Disjoint quality completion is declared; it never passes as a full new run."""

import unittest

from scripts.controller.nextgen import QUALITY_KIND
from scripts.controller.policy import validate_request
from test_nextgen_native_service import _service


class QualityCompletionContractTests(unittest.TestCase):
    def fixture(self):
        approval,request=_service('fp8_e4m3')
        request.update(range_observation=False,max_wall_seconds=7200,
            workload={'kind':QUALITY_KIND,'scope':'natural_stop_core_quality_generations_not_service_performance'},
            service={**request['service'],'max_batch':16,'linear_state_slots':32,'max_memory_utilization':.9},
            quality_evaluation={'requests':802,'client_concurrency':16,'client_token_budget':160000,
                'request_timeout_seconds':600,'max_seconds':4800,'requests_sha256':'a'*64,'source_manifest_sha256':'b'*64},
            quality_completion={'anchor_run_id':'previous','partial_raw_certificate_sha256':'c'*64})
        return approval,request
    def test_explicit_missing_id_completion_with_same_production_profile_is_admitted(self):
        a,r=self.fixture();validate_request(r,a)
    def test_missing_certificate_or_impossible_count_or_hidden_replicate_is_rejected(self):
        a,r=self.fixture()
        for changes in ({'quality_completion':None},{'replicate':1},
                        {'quality_evaluation':{**r['quality_evaluation'],'requests':1720}},
                        {'quality_completion':{'anchor_run_id':'../escape','partial_raw_certificate_sha256':'c'*64}}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_request(r|changes,a)


if __name__=='__main__':unittest.main()
