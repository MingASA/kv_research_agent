# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU fault injection for the explicitly bounded production prefill gate."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.policy import PREFILL_WORKSPACE_OPERATORS, PREFILL_WORKSPACE_SOURCES, validate_request
from scripts.controller.prefill_operator import run_operator
from test_execution import fixture


def _request() -> tuple:
    approval, request, _ = fixture()
    request.update(schema='kv-production-run-v1', prefix_cache=True, concurrency=1,
        max_wall_seconds=1800, max_output_gib=3, arms=['fp8_e4m3'],
        python_source_repo=request['remote_output']+'/source_repo', python_source_commit='c'*40,
        backend_source_sha256={name: 'a'*64 for name in PREFILL_WORKSPACE_SOURCES},
        operators=copy.deepcopy(PREFILL_WORKSPACE_OPERATORS), service={'max_seqs_per_batch':128,
            'max_tokens_per_batch':10240,'max_concurrent_requests':200,'memory_utilization':.8},
        workload={'kind':'native_prefill_workspace_operator','ignore_eos':False,
            'method':'recorded_production_prefill_repair_operator_gate','source_content_sha256':'a'*64,
            'model_loaded':False,'AIPerf_ran':False,
            'geometry':{'sequences':1,'query_tokens':510,'kv_tokens':1022,'layers':48}})
    return approval, request


class PrefillOperatorTests(unittest.TestCase):
    def test_only_explicit_single_arm_recorded_bounded_operator_scope_is_accepted(self) -> None:
        approval, request = _request();validate_request(request,approval)
        for key,value in [('max_wall_seconds',1801),('concurrency',128),('arms',['bf16']),
                          ('max_output_gib',4)]:
            changed=copy.deepcopy(request);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(changed,approval)
        for key,value in [('AIPerf_ran',True),('model_loaded',True),('geometry',{'sequences':2})]:
            changed=copy.deepcopy(request);changed['workload'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(changed,approval)
        changed=copy.deepcopy(request);changed['operators']['fp8_e4m3']['prefill_workspace_mib']=3072
        with self.assertRaises(ValueError):validate_request(changed,approval)

    def test_single_FP8_arm_cannot_escape_into_an_ordinary_serving_workload(self) -> None:
        approval,request=_request();request['workload']['kind']='sharegpt_natural_xqa_bf16_reference'
        with self.assertRaises(ValueError):validate_request(request,approval)

    def test_operator_failure_never_loads_a_model_or_launches_AIPerf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,request=_request();source=root/'geometry.json'
            source.write_text(json.dumps({'geometry':request['workload']['geometry'],
                'recorded_source_sha256':'a'*64,'method':request['workload']['method']}))
            request['dataset_path']=str(source);request['python_source_repo']=str(root)
            with patch('scripts.controller.prefill_operator.subprocess.run',side_effect=subprocess.CalledProcessError(1,['probe'])) as called:
                with patch('scripts.controller.prefill_operator.subprocess.Popen') as spawned:
                    with self.assertRaises(subprocess.CalledProcessError):run_operator({'request':request},root,'fp8_e4m3',{})
                    spawned.assert_not_called()
            argv=called.call_args.args[0]
            self.assertTrue(argv[2].endswith('/tools/check_fp8_prefill_workspace.py'))
            self.assertFalse((root/'measurement.json').exists())

    def test_missing_real_page_or_footprint_evidence_cannot_be_sealed(self) -> None:
        valid={'classification':'confirmed_operator_correctness_and_recorded_geometry_screen_only',
            'model_loaded':False,'AIPerf_ran':False,'native_operator':{'real_capture_and_replay':True},
            'workspace_page_checks':{name:True for name in ('shared_page_deduplicated','noncontiguous_FP8_strides',
                'inactive_pages_not_materialized','multiple_layers_reuse_same_storage','next_prefill_replaces_active_set')},
            'screen':{'prepare_and_active_page_cost_included':True,'candidate_backend':{
                'prefill_kv_data_type':'torch.bfloat16','kv_data_type':'torch.float8_e4m3fn',
                'fp8_views_alias_resident_pages':True,'profile_scope':'production_xqa_bf16_prefill_workspace',
                'prefill_workspace_reserved_bytes':1539*2**20+4,'prefill_workspace_view_bytes':9*2**19}}}
        for corruption in ('none','pages','alias','reserved','graph'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);_,request=_request();source=root/'geometry.json'
                source.write_text(json.dumps({'geometry':request['workload']['geometry'],
                    'recorded_source_sha256':'a'*64,'method':request['workload']['method']}))
                request['dataset_path']=str(source);request['python_source_repo']=str(root)
                response=copy.deepcopy(valid)
                if corruption=='pages':response['workspace_page_checks']['shared_page_deduplicated']=False
                elif corruption=='alias':response['screen']['candidate_backend']['fp8_views_alias_resident_pages']=False
                elif corruption=='reserved':response['screen']['candidate_backend']['prefill_workspace_reserved_bytes']=1
                elif corruption=='graph':response['native_operator']['real_capture_and_replay']=False
                def finish(*args,**kwargs):
                    (root/'operator_acceptance.json').write_text(json.dumps(response))
                with patch('scripts.controller.prefill_operator.subprocess.run',side_effect=finish):
                    if corruption=='none':
                        self.assertTrue(run_operator({'request':request},root,'fp8_e4m3',{})['operator_only'])
                    else:
                        with self.assertRaises(ValueError):run_operator({'request':request},root,'fp8_e4m3',{})


if __name__=='__main__':unittest.main()
