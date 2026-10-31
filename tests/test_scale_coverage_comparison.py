# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""A scale recipe cannot inherit old scores or join changed source prompts."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.compare_scale_coverage_quality import compare, LABEL
from scripts.controller.scale_coverage import SCALES, BASE_SCALES
from test_bf16q_comparison import _fixture


def _sealed_rows(mirror: Path, audit: dict, arm: str, task: str) -> list:
    return [{'_record_sha256':str(i),'_prompt_sha256':str(i),
        'api_response':{'usage':{'prompt_tokens':100}}} for i in range(1319 if task == 'gsm8k' else 50)]


def _write(directory: Path, corruption: str | None = None) -> list:
    b,c,_ = _fixture()
    c['classification'] = 'confirmed_full_scale_coverage_candidate_prediction_integrity_and_rescoring'
    b['evaluator_sha256'] = c['evaluator_sha256'] = 'same'
    c['production_configuration'] = copy.deepcopy(b['production_configuration'])
    b['production_configuration']['fp8_scales'] = dict(BASE_SCALES)
    c['production_configuration']['fp8_scales'] = dict(SCALES)
    proof = {'evaluator_and_scoring_source_bytes_identical':True,'all_inference_source_bytes_identical':True,
        'baseline_evaluator_sha256':'same','candidate_evaluator_sha256':'same'}
    if corruption == 'id':c['paired_items']['lcc'][0]['id'] = 'other'
    elif corruption == 'source':c['audited_source_files_sha256']['xllm/python/models/qwen2.py'] = 'other'
    elif corruption == 'scale':c['production_configuration']['fp8_scales']['value'] = .25
    elif corruption == 'score':c['task_scores']['lcc']['fp8_e4m3'] = 1.
    paths = [directory/(name+'.json') for name in ('reference','candidate','proof')]
    paths[0].write_text(json.dumps(b))
    proof['reference_audit_sha256'] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    paths[2].write_text(json.dumps(proof))
    c['scale_quality_provenance_sha256'] = hashlib.sha256(paths[2].read_bytes()).hexdigest()
    if corruption == 'proof':paths[2].write_text(paths[2].read_text()+' ')
    paths[1].write_text(json.dumps(c))
    return paths


@unittest.skipUnless(importlib.util.find_spec('numpy') is not None, 'Scientific analysis uses the pinned local venv')
class CoverageComparisonTests(unittest.TestCase):
    def test_exact_records_recompute_deltas_and_remain_unaccepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write(Path(directory))
            with patch('scripts.compare_scale_coverage_quality._rows',side_effect=_sealed_rows):
                result = compare(*paths,Path('reference-mirror'),Path('candidate-mirror'))
            self.assertEqual(result['records'],1719)
            self.assertAlmostEqual(result['LongBench_macro'][LABEL],.99875)
            self.assertAlmostEqual(result['task_deltas']['lcc']['official_score']['versus_BF16_pp'],-1.)
            self.assertFalse(result['quality_accepted'])
            self.assertEqual(result['independent_new_users'],0)

    def test_changed_identity_inference_scales_scores_or_seal_is_refused(self) -> None:
        for corruption in ('id','source','scale','score','proof'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                paths = _write(Path(directory),corruption)
                with patch('scripts.compare_scale_coverage_quality._rows',side_effect=_sealed_rows):
                    with self.assertRaises(ValueError):compare(*paths,Path('reference-mirror'),Path('candidate-mirror'))

    def test_identical_ID_cannot_hide_a_different_original_prompt(self) -> None:
        def altered(mirror: Path, audit: dict, arm: str, task: str) -> list:
            rows = _sealed_rows(mirror,audit,arm,task)
            if mirror == Path('candidate-mirror'):rows[0]['_prompt_sha256'] = 'other'
            return rows
        with tempfile.TemporaryDirectory() as directory:
            paths = _write(Path(directory))
            with patch('scripts.compare_scale_coverage_quality._rows',side_effect=altered):
                with self.assertRaisesRegex(ValueError,'prompt'):compare(*paths,Path('reference-mirror'),Path('candidate-mirror'))


if __name__ == '__main__':unittest.main()
