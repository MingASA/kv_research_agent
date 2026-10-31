# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU corruption tests; synthetic activity metadata is not GPU evidence."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.component_profile import CONFIG, ComponentProfileCoordinator, validate_windows
from scripts.controller.policy import COMPONENT_SOURCES, BF16_XQA_OPERATORS, validate_request
from test_execution import fixture


def _request() -> tuple[dict, dict]:
    approval, request, _ = fixture()
    remote = request['remote_output']
    request.update(schema='kv-production-run-v1', prefix_cache=True, concurrency=1024,
        max_wall_seconds=5400, max_output_gib=3, operators=copy.deepcopy(BF16_XQA_OPERATORS),
        python_source_repo=remote+'/source_repo', python_source_commit='2'*40,
        backend_source_sha256={key: '3'*64 for key in COMPONENT_SOURCES},
        dataset_path=remote+'/input/payload_trace.jsonl',
        dataset_archive={'relative_path':'input/payload_trace.jsonl.xz','codec':'xz','uncompressed_size_bytes':1000},
        workload={'kind':'anonymous_trace_component_diagnostic','family':'toolagent','requests':637,'warmup':0,
            'ignore_eos':True,'arrival_mode':'fixed_schedule_continuous_warm_cache','arrival_scaling':.125,
            'content_scope':'anonymous_arrival_length_prefix_emulation','trace_manifest':remote+'/input/trace_manifest.json',
            'source_content_sha256':'4'*64},
        service={'max_seqs_per_batch':128,'max_tokens_per_batch':10240,'max_concurrent_requests':200,'memory_utilization':.8},
        diagnostic={'method':'owned_process_torch_profiler','max_windows':3,'max_forwards_per_window':16,
                    'max_seconds_per_window':6,'max_trace_mib':32,'window_source_seconds':90,
                    'semantic_quality_claim':False,'headline_performance_claim':False})
    return approval, request


class ComponentContractTests(unittest.TestCase):
    def test_explicit_bounded_diagnostic_and_required_probe_source(self):
        approval, request = _request(); validate_request(request, approval)
        request['backend_source_sha256'].pop('tools/check_component_profiler.py')
        with self.assertRaises(ApprovalError): validate_request(request, approval)

    def test_observer_limits_and_scope_cannot_be_increased(self):
        for key, value in [('max_windows',4),('max_forwards_per_window',100),('max_trace_mib',128),
                           ('headline_performance_claim',True),('method','system_attach')]:
            approval, request = _request(); request['diagnostic'][key]=value
            with self.subTest(key=key), self.assertRaises(ApprovalError): validate_request(request,approval)
        approval, request = _request(); request['max_output_gib']=True
        with self.assertRaises(ApprovalError): validate_request(request,approval)

    def test_no_observer_injection_into_ordinary_run(self):
        approval, request = _request(); request['workload']['kind']='anonymous_trace_xqa_bf16_reference'
        with self.assertRaises(ApprovalError): validate_request(request,approval)

    def test_window_anchor_waits_for_real_post_client_counter_increase(self):
        with tempfile.TemporaryDirectory() as td:
            ticket={'request':{'run_id':'CPU-fixture'},'request_sha256':'a'*64}
            c=ComponentProfileCoordinator(Path(td),ticket,'bf16',{'window_begin_ms':0,'first_retained_timestamp_ms':0,'arrival_scaling':.125})
            sample={'utc':'CPU fixture','values':{'prefix_cache_prompt_tokens_total':100},'errors':{}}
            c.on_sample(sample); self.assertFalse((c.root/'window_plan.json').exists())
            c.client_started();c.on_sample(sample);c.on_sample(sample)
            self.assertFalse((c.root/'window_plan.json').exists())
            sample['values']['prefix_cache_prompt_tokens_total']=200;c.on_sample(sample)
            plan=json.loads((c.root/'window_plan.json').read_text())
            self.assertEqual(len(plan['windows']),3);self.assertFalse(plan['anchor_is_exact_offered_origin'])
            first=(c.root/'window_plan.json').read_bytes();c.on_sample(sample)
            self.assertEqual(first,(c.root/'window_plan.json').read_bytes())

    def test_activity_validation_rejects_corrupt_or_wrong_phase_evidence(self):
        for mutation in ('none','hash','no_kernels','wrong_device','warmup','no_graph','GPU_outside_phase'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                root=Path(td); cfg={**CONFIG,'run_id':'CPU-fixture','request_sha256':'a'*64,'arm':'bf16'}
                (root/'PROCESS.json').write_text(json.dumps({'config':cfg,'pid':123,'scope':'owned_process_only_no_system_attach'}))
                windows=[]
                for i in range(3):
                    start=(10+i*100)*10**9
                    windows.append({'index':i,'not_before_epoch_ns':start-1,'start_deadline_epoch_ns':start+10**9})
                    events=[{'cat':'kernel','name':'CPU_fixture_not_real_kernel','ts':start/1000,
                             'dur':100000,'args':{'device':0}}]
                    events += [{'cat':'user_annotation','name':'xllm.component.bf16.'+name}
                               for name in ('kv_materialize','kv_write','prefill_attention')]
                    events.append({'name':'cudaGraphLaunch'})
                    if mutation=='no_kernels':events=[row for row in events if row.get('cat')!='kernel']
                    if mutation=='wrong_device':events[0]['args']['device']=1
                    if mutation=='GPU_outside_phase':events[0]['ts']=0
                    path=root/f'window-{i:02d}.trace.json';path.write_text(json.dumps({
                        'baseTimeNanoseconds':0,'deviceProperties':[{'CPU_fixture':True}],'traceEvents':events}))
                    result={'status':'CAPTURED_REQUIRES_INDEPENDENT_TRACE_VALIDATION','pid':123,
                        **{key:cfg[key] for key in ('run_id','request_sha256','arm')},
                        'start_epoch_ns':start,'end_epoch_ns':start+10**8,
                        'steps':[{'completed':True,'prefill':True,'chunked_prefill':False,
                                  'sequences':1,'input_tokens':512,'start_epoch_ns':start,'end_epoch_ns':start+10**8,
                                  'components':{'graph_replay':0 if mutation=='no_graph' else 1}}],
                        'trace_file':path.name,'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                    if mutation=='hash':result['trace_sha256']='b'*64
                    (root/f'window-{i:02d}.json').write_text(json.dumps(result))
                (root/'window_plan.json').write_text(json.dumps({'request_sha256':cfg['request_sha256'],'windows':windows}))
                audit={'warm_cache_end_epoch_ns':(20 if mutation=='warmup' else 1)*10**9,
                       'phase_start_ns':0,'offered_window_seconds':300}
                if mutation=='none':self.assertEqual(len(validate_windows(root,cfg,audit)['windows']),3)
                else:
                    with self.assertRaises(ValueError):validate_windows(root,cfg,audit)


if __name__=='__main__':
    unittest.main()
