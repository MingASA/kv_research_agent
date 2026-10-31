# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Coverage evidence cannot authorize an arbitrary scale, model path or GPU matrix."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.durable import digest, write_new_json
from scripts.controller.policy import BF16_XQA_OPERATORS, XQA_SOURCES, QUALITY_SOURCES, validate_request
from scripts.controller.production import candidate_quality_command, run_candidate_quality, verify_backend
from scripts.controller.scale_coverage import METHOD, RUN, SCALES, BASE_SCALES, TOOLS_CHANGED, validate_prerequisites
import test_production_serving as production_tests


def _request() -> tuple:
    approval, request = production_tests.ProductionPolicyTests().request()
    baseline = {name:'a'*64 for name in XQA_SOURCES + QUALITY_SOURCES}
    request.update(arms=['fp8_e4m3'], max_wall_seconds=3600, max_output_gib=5,
        operators=copy.deepcopy(BF16_XQA_OPERATORS), fp8_scales=dict(SCALES),
        backend_source_sha256={name:('b'*64 if name in TOOLS_CHANGED else value) for name,value in baseline.items()})
    remote = request['remote_output']
    request['workload'] = {'kind':'native_quality_xqa_production_batch', 'cohort':'full_production_batch',
        'samples_per_arm':1719, 'ignore_eos':False, 'source_content_sha256':'a'*64,
        'scorer_dependencies':approval['approved_workspace']+'/experiments/joint-kv-14b/longbench-deps-py312',
        'longbench_repo':remote+'/input/scorer', 'gsm8k_shots':remote+'/input/five_train_examples.json',
        'approved_data_manifest':remote+'/input/approved_data_manifest.json',
        'scale_calibration':{'method':METHOD, 'calibration_run_id':RUN,
            'calibration_audit_path':remote+'/input/longcontext_range_independent_audit.json',
            'proposal_path':remote+'/input/scale_coverage_candidate.json',
            'reference_audit_path':remote+'/input/existing_modern_reference_audit.json',
            'calibration_audit_sha256':'a'*64, 'proposal_sha256':'a'*64,
            'calibration_terminal_sha256':'a'*64, 'reference_audit_sha256':'a'*64,
            'reference_run_id':'retained_reference', 'baseline_backend_source_sha256':baseline}}
    return approval, request


def _prerequisites(directory: Path, request: dict) -> None:
    item = request['workload']['scale_calibration']
    audit = {'classification':'confirmed_raw_heldout_BF16_KV_distribution_and_AIPerf_integrity_not_FP8_quality',
        'run_id':RUN,'original_root_status':'COMPLETE','content_scope':'heldout_original_LongBench_long_document_QA',
        'terminal_result_sha256':item['calibration_terminal_sha256'], 'actual_original_requests':128,
        'binary_sha256':request['binary_sha256'], 'model_config_sha256':request['model_config_sha256'],
        'rows':[{'kind':'key','clipped':0,'absmax':45.25},{'kind':'value','clipped':68,'absmax':37.}]}
    reference = {'classification':'confirmed_full_production_batch_prediction_integrity_and_rescoring',
        'run_id':'retained_reference','audited_arms':['bf16','fp8_e4m3'],'samples_per_arm':1719,
        'root_complete':True,'gpu_release_verified':True,
        'production_configuration':{k:copy.deepcopy(request[k]) for k in ('binary_sha256','model_config_sha256',
            'image_id','operators','service','graph_mode','concurrency')},
        'audited_source_files_sha256':item['baseline_backend_source_sha256']}
    reference['production_configuration']['fp8_scales'] = dict(BASE_SCALES)
    for path_key, sha_key, data in [('calibration_audit_path','calibration_audit_sha256',audit),
                                  ('reference_audit_path','reference_audit_sha256',reference)]:
        write_new_json(directory/Path(item[path_key]).name,data)
        item[sha_key] = digest(data)
    proposal = {'calibration_audit_sha256':item['calibration_audit_sha256'],
        'calibration_terminal_sha256':item['calibration_terminal_sha256'],'old_scales':dict(BASE_SCALES),
        'candidate_scales':dict(SCALES),'actual_clipped_count':{'key':0,'value':68},
        'actual_absmax':{'key':45.25,'value':37.},'quality_labels_used':False,
        'calibration_exact_overlap_with_old_LongBench400':0,'candidate_quality_accepted':False}
    write_new_json(directory/'scale_coverage_candidate.json',proposal)
    item['proposal_sha256'] = digest(proposal)


class ScaleCoverageTests(unittest.TestCase):
    def test_unique_coverage_request_changes_only_scales_and_selected_quality_arm(self) -> None:
        approval, request = _request();validate_request(request,approval)
        command = candidate_quality_command({'request':request},Path('/CPU'),'fp8_e4m3')
        self.assertEqual(command[command.index('--fp8-v-scale')+1], '0.125')
        self.assertEqual(command[command.index('--arms')+1], 'fp8_e4m3')
        self.assertNotIn('--fp8-bf16q-decode',command)

    def test_coverage_cannot_escape_to_other_scales_workload_budget_or_runtime_changes(self) -> None:
        approval, request = _request()
        for key,value in [('arms',['bf16','fp8_e4m3']),('fp8_scales',{'key':.125,'value':.25}),
                          ('concurrency',8),('max_wall_seconds',3601),('max_output_gib',6)]:
            bad=copy.deepcopy(request);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(bad,approval)
        for key,value in [('method','answer_selected_scale'),('proposal_path','/tmp/unfrozen'),
                          ('calibration_run_id','unknown')]:
            bad=copy.deepcopy(request);bad['workload']['scale_calibration'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(bad,approval)
        bad=copy.deepcopy(request);bad['backend_source_sha256']['xllm/python/attention/flashinfer.py']='c'*64
        with self.assertRaises(ValueError):validate_request(bad,approval)
        bad=copy.deepcopy(request);bad['workload'].pop('scale_calibration')
        with self.assertRaises(ValueError):validate_request(bad,approval)

    def test_mutated_or_missing_evidence_fails_before_model_start(self) -> None:
        _,request=_request()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_prerequisites(root,request)
            result=validate_prerequisites(request,input_root=root)
            self.assertFalse(result['model_quality_validated'])
            path=root/'scale_coverage_candidate.json'; path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):validate_prerequisites(request,input_root=root)

    def test_actual_operator_failure_prevents_new_model_and_receives_candidate_scales(self) -> None:
        _,request=_request()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch('scripts.controller.scale_coverage.validate_prerequisites') as evidence:
                with patch('scripts.controller.production.subprocess.run',side_effect=[
                        subprocess.CompletedProcess([],0),subprocess.CalledProcessError(1,['GPU_gate'])]) as gate:
                    with patch('scripts.controller.production.subprocess.Popen') as model:
                        with self.assertRaises(subprocess.CalledProcessError):
                            run_candidate_quality({'request':request},root,root,'fp8_e4m3',{'PYTHONPATH':''})
                        model.assert_not_called()
                    argv=gate.call_args.args[0]
                    self.assertEqual(argv[argv.index('--value-scale')+1],'0.125')
                evidence.assert_called_once()

    def test_actual_backend_new_scale_needs_explicit_matching_contract(self) -> None:
        row={'class':'FlashInferFP8Backend','attention_impl':'flashinfer','planner_backend':'fa2',
            'paged_prefill':True,'graph_enabled':True,'profile_scope':'production_xqa',
            'quality_transform':False,'kv_cache_dtype':'fp8_e4m3','q_data_type':'torch.bfloat16',
            'kv_data_type':'torch.float8_e4m3fn','decode_impl':'xqa',
            'decode_internal_q_data_type':'torch.float8_e4m3fn','xqa_enable_pdl':True,
            'xqa_unused_kv_policy':'finite_full_initialization','xqa_max_context':32768,
            'xqa_graph_padding':'reserved_page0_one_token_discard','q_scale':None,
            'k_scale':.125,'v_scale':.125,'fp8_views_alias_resident_pages':True,'graph_captured_batches':[1]}
        log='NATIVE_KV_BACKEND '+json.dumps(row)+'\nNATIVE_KV_GRAPH_REPLAY '+json.dumps({
            'class':row['class'],'captured_batch_size':1})
        operator=BF16_XQA_OPERATORS['fp8_e4m3']
        with self.assertRaises(ValueError):verify_backend(log,'fp8_e4m3',operator)
        self.assertEqual(verify_backend(log,'fp8_e4m3',operator,SCALES)['v_scale'],.125)
        with self.assertRaises(ValueError):verify_backend(log,'fp8_e4m3',operator,{'key':.125,'value':.25})


if __name__ == '__main__':unittest.main()
