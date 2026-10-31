# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU-only quality comparison fixtures; never model predictions or GPU results."""

import copy
from pathlib import Path
import tempfile
import unittest

from scripts.compare_quality_reference import _combine_verified, _evaluation_logic
from scripts.controller.policy import XQA_OPERATORS, BF16_XQA_OPERATORS
from scripts.controller.protocols import TASKS


def fixture() -> tuple[dict, dict]:
    provenance = {key: {'fixture': 'fixed'} for key in (
        'model_weights_sha256', 'model_metadata_sha256', 'versions', 'protocol_files_sha256')}
    config = {'binary_sha256': 'same', 'concurrency': 8, 'operators': copy.deepcopy(XQA_OPERATORS)}
    provenance['production_configuration'] = config
    candidate = {'classification': 'confirmed_full_frozen_cohort_independent_rescoring', 'records_per_arm': 1719,
                 'provenance': provenance, 'paired_items': {}, 'task_scores': {}}
    reference = {**copy.deepcopy(provenance), 'classification': 'confirmed_full_bf16_reference_prediction_integrity_and_rescoring',
                 'samples_per_arm': 1719, 'audited_arms': ['bf16'], 'full_1719_evaluation': True,
                 'paired_items': {}, 'reuse_mapping': {}}
    reference['production_configuration']['operators'] = copy.deepcopy(BF16_XQA_OPERATORS)
    for task in TASKS:
        count = 1319 if task == 'gsm8k' else 50
        reference['reuse_mapping'][task] = list(range(count))
        reference['paired_items'][task], candidate['paired_items'][task] = [], []
        for index in range(count):
            new = {'index': index, 'original_source_index': index, 'id': str(index), 'scores': {'bf16': 1.0}}
            old = {'index': index, 'id': str(index), 'scores': {'bf16': 1.0, 'fp8_e4m3': 0.0 if task == 'lcc' and index == 0 else 1.0}}
            if task == 'gsm8k':
                new['flexible_scores'] = {'bf16': 1.0}
                old['flexible_scores'] = {'bf16': 1.0, 'fp8_e4m3': 0.0 if index == 0 else 1.0}
            reference['paired_items'][task].append(new); candidate['paired_items'][task].append(old)
        candidate['task_scores'][task] = ({'fp8_e4m3_strict_em': 1.0, 'fp8_e4m3_flexible_em': 1318/1319}
                                         if task == 'gsm8k' else {'fp8_e4m3': .98 if task == 'lcc' else 1.0})
    return reference, candidate


class ReferenceComparisonTests(unittest.TestCase):
    def test_scores_and_flips_preserve_retained_candidate(self) -> None:
        reference, candidate = fixture()
        result = _combine_verified(reference, candidate)
        self.assertEqual(result['task_scores']['gsm8k']['fp8_e4m3_flexible_em'], 1318/1319)
        self.assertEqual(result['paired_flips']['gsm8k'], {'down': 1, 'tie': 1318})
        self.assertAlmostEqual(result['longbench_8_task_macro']['fp8_e4m3'], .9975)

    def test_missing_or_mismatched_global_records_are_rejected(self) -> None:
        reference, candidate = fixture()
        candidate['paired_items']['lcc'][3]['id'] = 'other-record'
        with self.assertRaises(ValueError): _combine_verified(reference, candidate)
        reference, candidate = fixture()
        reference['paired_items']['gsm8k'].pop()
        with self.assertRaises(ValueError): _combine_verified(reference, candidate)

    def test_changed_provenance_candidate_score_or_operator_is_rejected(self) -> None:
        for group, key, value in [('versions', 'fixture', 'different'),
                                  ('production_configuration', 'concurrency', 128)]:
            reference, candidate = fixture(); reference[group][key] = value
            with self.subTest(group=group), self.assertRaises(ValueError): _combine_verified(reference, candidate)
        reference, candidate = fixture(); candidate['task_scores']['lcc']['fp8_e4m3'] = 1.0
        with self.assertRaises(ValueError): _combine_verified(reference, candidate)

    def test_only_new_strict_profile_recognition_can_differ_in_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory)/'old.py', Path(directory)/'new.py'
            old.write_text("def _load_server_manifest():\n return ('old-profile',)\ndef score():\n return 1\n")
            new.write_text("def _load_server_manifest():\n return ('old-profile','native-flashinfer-bf16-xqa-reference-v1')\ndef score():\n return 1\n")
            self.assertEqual(_evaluation_logic(old), _evaluation_logic(new))
            new.write_text(new.read_text().replace('return 1', 'return 0'))
            self.assertNotEqual(_evaluation_logic(old), _evaluation_logic(new))


if __name__ == '__main__':
    unittest.main()
