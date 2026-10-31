# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Fault injection for diagnostic scope, epoch, complete counters and acknowledgements."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.controller.policy import BF16_XQA_OPERATORS, KV_RANGE_SOURCES, validate_request
from scripts.controller.durable import digest
from scripts.controller.range_probe import (METHOD, RangeProbeCoordinator, audit_export,
    LONG_QA_SCOPE, LONG_QA_TASK_CAPS, validate_longcontext_inputs)
from test_prefill_operator import _request


def _range_request() -> tuple:
    approval,request=_request()
    request.update(arms=['bf16'],concurrency=128,
        backend_source_sha256={name:'a'*64 for name in KV_RANGE_SOURCES},
        operators={'bf16':BF16_XQA_OPERATORS['bf16']},
        workload={'kind':'heldout_kv_range_probe','ignore_eos':False,'requests':128,'warmup':0,
            'method':METHOD,'quality_labels_loaded':False,'content_scope':'heldout_original_ShareGPT_natural_chat',
            'aiperf_zero_warmup_policy':'omit_explicit_option',
            'jsonl_framing':'physical_LF_UTF8','export_before_postprocessing':True,
            'aiperf_cli_config_sha256':'a'*64,
            'input_manifest':request['remote_output']+'/input/production_manifest.json','source_content_sha256':'a'*64})
    return approval,request


def _export() -> dict:
    rows=[]
    for layer in range(48):
        for phase in ('prefill','decode'):
            for kind in ('key','value'):
                for head in range(8):
                    rows.append(dict(layer=layer,phase=phase,kind=kind,head=head,finite=128,
                        nonfinite=0,clipped=0,underflow_nonzero_to_zero=0,normal=128,subnormal=0,zero=0,
                        valid_values=128,absmax=1.,squared_energy=128.,squared_error=.125,relative_l2=.03125))
    return dict(method=METHOD,request_sha256='a'*64,run_id='test',command='EXPORT',reset_monotonic_ns=123,
        layers=48,heads=8,dimension=128,key_scale=.125,value_scale=.0625,host_counter_reads_in_forward=False,
        activation_tensors_retained=False,reserved_bytes=105216,
        padding_policy='exclude_all_nonpositive_slots_including_reserved_page0',
        device_writer_calls_by_layer_phase=[[1,1] for _ in range(48)],rows=rows)


def _long_inputs() -> tuple:
    rows, lineage = [], []
    for task, cap in LONG_QA_TASK_CAPS.items():
        for index in range(32):
            identity = task + '-' + str(index)
            prompt = 'CPU fixture only: ' + identity
            payload = {'model':'Qwen2.5-14B-Instruct','messages':[{'role':'user','content':prompt}],
                'max_tokens':cap,'seed':17,'temperature':0,'top_p':1,'stream':True,
                'stream_options':{'include_usage':True}}
            rows.append({'session_id':identity,'payloads':[payload]})
            lineage.append({'session_id':identity,'task':task,'record_id':identity,
                'context_sha256':digest(identity),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'payload_sha256':digest(payload),'max_tokens':cap,'prompt_tokens_cpu':4096})
    data = {'data':rows}
    manifest = {'schema':'kv-real-heldout-longcontext-QA-input-v1','content_scope':LONG_QA_SCOPE,
        'requests':128,'canonical_sha256':digest(data),'quality_labels_used_in_selection':False,
        'answers_exported_to_probe':False,'previous_predictions_used_in_selection':False,
        'official_generation_caps_preserved':True,'messages_and_natural_stop_controls_unchanged':True,
        'evaluation_excluded_count':400,'overlap_with_previous_evaluation':{
            'record_ids':0,'exact_records':0,'exact_contexts':0},'selected_lineage':lineage,
        'source_content_sha256':'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'}
    return data, manifest


class RangeProbeTests(unittest.TestCase):
    def test_long_document_scope_keeps_original_caps_and_discloses_distinct_coverage(self) -> None:
        data, manifest = _long_inputs()
        validate_longcontext_inputs(data, manifest)
        approval, request = _range_request()
        workload = request['workload']
        workload.pop('quality_labels_loaded')
        workload.update(content_scope=LONG_QA_SCOPE,quality_labels_used_in_selection=False,
            answers_exported_to_probe=False,official_generation_caps_preserved=True,
            evaluation_excluded_count=400)
        validate_request(request, approval)
        actual = audit_export(_export(),'a'*64,'test',123,LONG_QA_SCOPE)
        self.assertFalse(actual['heldout_chat_only_not_long_document_or_code_coverage'])
        self.assertTrue(actual['long_document_QA_only_not_code_coverage'])

    def test_long_document_policy_cannot_relabel_unknown_data_or_relax_answer_separation(self) -> None:
        approval, request = _range_request()
        request['workload'].update(content_scope=LONG_QA_SCOPE,quality_labels_used_in_selection=False,
            answers_exported_to_probe=False,official_generation_caps_preserved=True,evaluation_excluded_count=400)
        for key, value in [('content_scope','arbitrary_long_text'),('answers_exported_to_probe',True),
                           ('quality_labels_used_in_selection',True),('official_generation_caps_preserved',False),
                           ('evaluation_excluded_count',0)]:
            changed = copy.deepcopy(request); changed['workload'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):validate_request(changed, approval)

    def test_long_document_original_prompt_cap_context_bounds_and_labels_are_fail_closed(self) -> None:
        for corruption in ('cap','prompt','answers','eos','duplicate_context','label_ranking','old_context','overflow','task'):
            data, manifest = _long_inputs()
            payload = data['data'][0]['payloads'][0]
            if corruption == 'cap':payload['max_tokens'] = 1024
            elif corruption == 'prompt':payload['messages'][0]['content'] += ' unrelated padding'
            elif corruption == 'answers':payload['answers'] = ['fixture label']
            elif corruption == 'eos':payload['ignore_eos'] = True
            elif corruption == 'duplicate_context':manifest['selected_lineage'][0]['context_sha256'] = manifest['selected_lineage'][1]['context_sha256']
            elif corruption == 'label_ranking':manifest['previous_predictions_used_in_selection'] = True
            elif corruption == 'old_context':manifest['overlap_with_previous_evaluation']['exact_contexts'] = 1
            elif corruption == 'overflow':manifest['selected_lineage'][0]['prompt_tokens_cpu'] = 32768
            else:manifest['selected_lineage'][0]['task'] = 'lcc'
            # Even a caller updating the outer manifest SHA cannot bypass payload lineage.
            manifest['canonical_sha256'] = digest(data)
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate_longcontext_inputs(data, manifest)

    def test_single_BF16_diagnostic_cannot_escape_into_production_or_larger_budget(self) -> None:
        approval,request=_range_request();validate_request(request,approval)
        for key,value in [('arms',['fp8_e4m3']),('concurrency',8),('max_wall_seconds',1801),('max_output_gib',4)]:
            changed=copy.deepcopy(request);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(changed,approval)
        for key,value in [('kind','sharegpt_natural_xqa_bf16_reference'),('quality_labels_loaded',True),('warmup',16)]:
            changed=copy.deepcopy(request);changed['workload'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(changed,approval)

    def test_complete_grid_aggregate_is_recomputed_from_raw_counts_and_moments(self) -> None:
        result=audit_export(_export(),'a'*64,'test',123)
        self.assertEqual(result['valid_tokens_written_per_layer_by_phase'],{'prefill':1,'decode':1})
        self.assertEqual(result['aggregate']['prefill/key']['finite'],48*8*128)
        self.assertEqual(result['aggregate']['prefill/key']['relative_l2'],.03125)

    def test_stale_epoch_missing_layer_nonfinite_padding_corruption_cannot_be_certified(self) -> None:
        for corruption in ('epoch','row','calls','nan','partition','l2','token_count','duplicate'):
            value=_export()
            if corruption=='epoch':value['reset_monotonic_ns']=122
            elif corruption=='row':value['rows'].pop()
            elif corruption=='calls':value['device_writer_calls_by_layer_phase'][2][1]=0
            elif corruption=='nan':value['rows'][0]['nonfinite']=1
            elif corruption=='partition':value['rows'][0]['subnormal']=1
            elif corruption=='l2':value['rows'][0]['relative_l2']=.3
            elif corruption=='token_count':value['rows'][0].update(finite=256,valid_values=256,normal=256)
            else:value['rows'][1]=value['rows'][0]
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):audit_export(value,'a'*64,'test',123)

    def test_export_before_reset_and_foreign_acknowledgement_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            coordinator=RangeProbeCoordinator(Path(directory),{'request':{'run_id':'test'},'request_sha256':'a'*64})
            with self.assertRaises(ValueError):coordinator.export_after_requests()
            (coordinator.root/'RESET_DONE.json').write_text(json.dumps({'run_id':'other','command':'RESET'}))
            with self.assertRaises(ValueError):coordinator.reset_after_bootstrap()


if __name__=='__main__':unittest.main()
