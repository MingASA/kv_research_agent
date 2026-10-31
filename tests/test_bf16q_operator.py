# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Fail closed on changed precision, missing Graph evidence and unsafe scope."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.bf16q_operator import run_operator
from scripts.controller.policy import BF16Q_DECODE_OPERATORS,BF16Q_DECODE_SOURCES,validate_request
from test_execution import fixture


def _request() -> tuple:
    approval,request,_=fixture()
    request.update(schema='kv-production-run-v1',prefix_cache=True,concurrency=128,
        max_wall_seconds=1800,max_output_gib=3,arms=['fp8_e4m3'],
        python_source_repo=request['remote_output']+'/source_repo',python_source_commit='c'*40,
        backend_source_sha256={n:'a'*64 for n in BF16Q_DECODE_SOURCES},
        operators=copy.deepcopy(BF16Q_DECODE_OPERATORS),service={'max_seqs_per_batch':128,
            'max_tokens_per_batch':10240,'max_concurrent_requests':200,'memory_utilization':.8},
        workload={'kind':'native_bf16q_decode_operator','ignore_eos':False,'source_content_sha256':'a'*64,
            'method':'resident_FP8_BF16Q_production_Graph128_gate_v1', 'relative_l2_limit':.01,
            'geometry':{'batch':128,'heads':40,'kv_heads':8,'head_dim':128,'page_size':128},
            'primary_sources_sha256':{n:'a'*64 for n in ('include/flashinfer/attention/prefill.cuh',
                'include/flashinfer/mma.cuh','flashinfer/decode.py')},'model_loaded':False,'AIPerf_ran':False})
    return approval,request


class BF16QOperatorTests(unittest.TestCase):
    def test_single_candidate_is_bounded_and_cannot_become_a_serving_job(self) -> None:
        approval,request=_request();validate_request(request,approval)
        for key,value in [('max_wall_seconds',1801),('concurrency',16),('arms',['bf16']),('max_output_gib',4)]:
            changed=copy.deepcopy(request);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(changed,approval)
        for key,value in [('model_loaded',True),('AIPerf_ran',True),('relative_l2_limit',.05),('geometry',{'batch':16}),
                          ('kind','sharegpt_natural_xqa_bf16_reference')]:
            changed=copy.deepcopy(request);changed['workload'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(changed,approval)

    def test_failed_operator_never_launches_model_or_AIPerf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,request=_request();source=root/'operator_manifest.json'
            source.write_text(json.dumps(request['workload']));request['dataset_path']=str(source)
            with patch('scripts.controller.bf16q_operator.subprocess.run',side_effect=subprocess.CalledProcessError(1,['probe'])):
                with patch('scripts.controller.bf16q_operator.subprocess.Popen') as spawned:
                    with self.assertRaises(subprocess.CalledProcessError):run_operator({'request':request},root,'fp8_e4m3',{})
                    spawned.assert_not_called()
            self.assertFalse((root/'measurement.json').exists())

    def test_wrong_Q_precision_or_missing_actual_Graph_cannot_be_sealed(self) -> None:
        for corruption in ('none','Q','graph','primary','error','native_graph'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);_,request=_request();source=root/'input.json'
                source.write_text(json.dumps(request['workload']));request['dataset_path']=str(source)
                graph={k:True for k in ('real_capture_and_replay','writer_in_same_graph','noncontiguous_QKV',
                    'reserved_page0_unchanged','resident_storage_stable','rebind_preserves_live_cache')}
                graph.update(production_max_batch=128,reference_Q='BF16_without_FP8_QDQ',relative_l2_limit=.01,
                    dynamic_live_rows_and_context=[{'live_rows':n,'context':c,'relative_l2':.002}
                        for n,c in ((128,257),(126,130),(1,130),(127,257))],backend={
                        'profile_scope':'production_bf16q_decode','planner_backend':'fa2','decode_impl':'flashinfer',
                        'q_data_type':'torch.bfloat16','decode_internal_q_data_type':'torch.bfloat16',
                        'kv_data_type':'torch.float8_e4m3fn','decode_tensor_cores':True,'q_scale':None,
                        'fp8_views_alias_resident_pages':True,'resident_initialization_policy':'finite_full_initialization',
                        'full_BF16_KV_shadow':False,'k_scale':.125,'v_scale':.0625})
                response={'classification':'confirmed_BF16Q_FP8KV_operator_Graph128_only','graph128':graph,
                    'model_loaded':False,'AIPerf_ran':False,'model_quality_validated':False,'service_performance_measured':False,
                    'installed_sources_sha256':request['workload']['primary_sources_sha256'],
                    'native_operator':{k:True for k in ('real_capture_and_replay','partial_page_and_replan',
                        'padded_graph_bucket_checked','noncontiguous_q_checked')}}
                response['native_operator']['reference_comparisons']={k:{'relative_l2':.002} for k in
                    ('prefill','capture_replay','changed_context','padded_graph')}
                if corruption=='Q':graph['backend']['decode_internal_q_data_type']='torch.float8_e4m3fn'
                elif corruption=='graph':graph['writer_in_same_graph']=False
                elif corruption=='primary':response['installed_sources_sha256']={}
                elif corruption=='error':graph['dynamic_live_rows_and_context'][0]['relative_l2']=float('nan')
                elif corruption=='native_graph':response['native_operator']['real_capture_and_replay']=False
                def finish(*args,**kwargs):
                    (root/'operator_acceptance.json').write_text(json.dumps(response))
                with patch('scripts.controller.bf16q_operator.subprocess.run',side_effect=finish):
                    if corruption=='none':self.assertTrue(run_operator({'request':request},root,'fp8_e4m3',{})['operator_only'])
                    else:
                        with self.assertRaises(ValueError):run_operator({'request':request},root,'fp8_e4m3',{})


if __name__=='__main__':unittest.main()
