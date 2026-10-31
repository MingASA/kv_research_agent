# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU score fixtures must not permit joining different model/task records."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_bf16q_quality import compare
from scripts.controller.policy import BF16Q_DECODE_OPERATORS,BF16_XQA_OPERATORS
from scripts.controller.protocols import TASKS


def _fixture() -> tuple:
    shared={k:{'CPU_fixture':True} for k in ('model_weights_sha256','model_metadata_sha256','versions','protocol_files_sha256')}
    shared.update(samples_per_arm=1719,root_complete=True,gpu_release_verified=True,full_1719_evaluation=True,
        audited_source_files_sha256={'xllm/python/models/qwen2.py':'same'},reuse_mapping={},paired_items={},
        task_scores={},length_finish_counts={})
    b=copy.deepcopy(shared);c=copy.deepcopy(shared)
    b.update(classification='confirmed_full_production_batch_prediction_integrity_and_rescoring',run_id='CPU-reference',
        audited_arms=['bf16','fp8_e4m3'],evaluator_sha256='old',
        production_configuration={'concurrency':128,'operators':copy.deepcopy(BF16_XQA_OPERATORS)})
    c.update(classification='confirmed_full_BF16Q_candidate_prediction_integrity_and_rescoring',run_id='CPU-candidate',
        audited_arms=['fp8_e4m3'],evaluator_sha256='new',
        production_configuration={'concurrency':128,'operators':copy.deepcopy(BF16Q_DECODE_OPERATORS)})
    for task in TASKS:
        n=1319 if task=='gsm8k' else 50
        b['reuse_mapping'][task]=c['reuse_mapping'][task]=list(range(n))
        b['paired_items'][task]=[];c['paired_items'][task]=[]
        for i in range(n):
            row={'index':i,'original_source_index':i,'id':str(i),'scores':{'bf16':1.0,'fp8_e4m3':1.0}}
            after={'index':i,'original_source_index':i,'id':str(i),'scores':{'fp8_e4m3':.5 if task=='lcc' and i==0 else 1.0}}
            if task=='gsm8k':
                row['flexible_scores']={'bf16':1.0,'fp8_e4m3':1.0};after['flexible_scores']={'fp8_e4m3':1.0}
            b['paired_items'][task].append(row);c['paired_items'][task].append(after)
        b['task_scores'][task]=({'bf16_strict_em':1.0,'bf16_flexible_em':1.0,
            'fp8_e4m3_strict_em':1.0,'fp8_e4m3_flexible_em':1.0} if task=='gsm8k' else {'bf16':1.0,'fp8_e4m3':1.0})
        c['task_scores'][task]=({'fp8_e4m3_strict_em':1.0,'fp8_e4m3_flexible_em':1.0}
            if task=='gsm8k' else {'fp8_e4m3':.99 if task=='lcc' else 1.0})
    return b,c,{'only_new_strict_profile_literal_diff':True,'baseline_evaluator_sha256':'old','candidate_evaluator_sha256':'new'}


@unittest.skipUnless(importlib.util.find_spec('numpy') is not None, 'Scientific comparison runs in the pinned analysis venv')
class BF16QComparisonTests(unittest.TestCase):
    def test_exact_global_record_pairing_recomputes_the_macro_without_new_users(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/(str(i)+'.json') for i in range(3)]
            for p,d in zip(paths,_fixture()):p.write_text(json.dumps(d))
            r=compare(*paths)
            self.assertAlmostEqual(r['LongBench_macro']['FP8_BF16Q'],.99875)
            self.assertAlmostEqual(r['task_deltas']['lcc']['official_score']['versus_BF16_pp'],-1.0)
            self.assertEqual(r['paired_score_changes']['lcc_official_score_versus_BF16'],{'down':1,'tie':49})
            self.assertEqual(r['independent_new_users'],0);self.assertFalse(r['quality_accepted'])

    def test_changed_ID_runtime_or_evaluator_prevents_a_quality_claim(self) -> None:
        for corruption in ('id','runtime','evaluator','score'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as directory:
                b,c,logic=_fixture()
                if corruption=='id':c['paired_items']['lcc'][0]['id']='different'
                elif corruption=='runtime':c['audited_source_files_sha256']['xllm/python/models/qwen2.py']='different'
                elif corruption=='evaluator':c['evaluator_sha256']='different'
                else:c['task_scores']['lcc']['fp8_e4m3']=1.0
                paths=[Path(directory)/(str(i)+'.json') for i in range(3)]
                for p,d in zip(paths,(b,c,logic)):p.write_text(json.dumps(d))
                with self.assertRaises(ValueError):compare(*paths)


if __name__=='__main__':unittest.main()
