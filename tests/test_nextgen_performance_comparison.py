"""Natural output volume, capacity and latency must not become interchangeable gains."""

import copy
import unittest

from scripts.compare_nextgen_performance import compare


class ActualPerformancePairTests(unittest.TestCase):
    def fixture(self):
        reference={'schema':'kv-nextgen-independent-production-performance-audit-v1','arm':'bf16','run_id':'reference',
            'actual_model_and_Graph_and_raw_service_performance_verified':True,'all_offered_requests_completed':True,
            'offered_requests':128,'successful_requests':128,'case':'chat-c16','concurrency':16,'input_manifest_sha256':'a'*64,
            'request_identity':{key:'same' for key in ('model_id','model_config_sha256','image_id','binary_sha256',
                'server_source_commit','python_source_commit','gpu_uuids','service','graph_mode','prefix_cache')},
            'quality_snapshot_reference_sha256':'b'*64,'actual_AIPerf_phase_times':{'measured':['first','last']},
            'profile':{'request_rps':10,'output_tps':100,'osl_mean':10,'profile_session_ids_sha256':'c'*64,
                'profile_prompt_counts_sha256':'d'*64,**{key:20 for key in
                ('ttft_p50_ms','ttft_p95_ms','itl_p50_ms','itl_p95_ms','e2e_p50_ms','e2e_p95_ms')}},
            'lifecycle':{'Full_KV_physical_token_slots':1000,'Full_KV_resident_bytes':65536000,
                'GDN_state_resident_bytes':512000000,'total_cache_resident_bytes':577536000}}
        reference['request_identity']['operators']={'bf16':{'planner':'fa3','decode':'xqa'}}
        candidate=copy.deepcopy(reference);candidate.update(arm='nvfp4',run_id='candidate')
        candidate['request_identity']['operators']={'nvfp4':{'planner':'fa2','decode':'flashinfer'}}
        candidate['profile'].update(request_rps=15,output_tps=165,osl_mean=11,ttft_p95_ms=22,itl_p50_ms=None)
        candidate['lifecycle'].update(Full_KV_physical_token_slots=3555,Full_KV_resident_bytes=65525760,
                                      total_cache_resident_bytes=577525760)
        return reference,candidate

    def test_real_rates_output_lengths_and_slower_tail_are_all_retained(self):
        a,b=self.fixture();result=compare(a,b);delta=result['actual_rates_and_tails']
        self.assertAlmostEqual(delta['request_rps']['change_percent'],50)
        self.assertAlmostEqual(delta['output_tps']['change_percent'],65)
        self.assertAlmostEqual(delta['osl_mean']['change_percent'],10)
        self.assertAlmostEqual(delta['ttft_p95_ms']['latency_reduction_percent'],-10)
        self.assertIsNone(delta['itl_p50_ms']['latency_reduction_percent'])
        self.assertFalse(result['quality_accepted_or_SLO_gain_inferred'])

    def test_more_physical_slots_do_not_imply_a_smaller_total_resident_pool(self):
        a,b=self.fixture();result=compare(a,b);storage=result['resident_storage_and_physical_pool_capacity']
        self.assertAlmostEqual(storage['Full_KV_physical_token_slots']['candidate_over_BF16'],3.555)
        self.assertAlmostEqual(storage['total_cache_resident_bytes']['candidate_over_BF16'],1,places=4)
        self.assertFalse(result['maximum_successful_admission_capacity_inferred_from_physical_slots'])

    def test_changed_GPU_demand_prompt_or_incomplete_side_cannot_be_a_pair(self):
        for name in ('GPU','demand','prompt','incomplete','GDN'):
            a,b=self.fixture()
            if name=='GPU':b['request_identity']['gpu_uuids']='another'
            elif name=='demand':b['concurrency']=8
            elif name=='prompt':b['profile']['profile_prompt_counts_sha256']='changed'
            elif name=='incomplete':b['all_offered_requests_completed']=False
            else:b['lifecycle']['GDN_state_resident_bytes']+=1
            with self.subTest(name=name),self.assertRaises(ValueError):compare(a,b)


if __name__=='__main__':unittest.main()
