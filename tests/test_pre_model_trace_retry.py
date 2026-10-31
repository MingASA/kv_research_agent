"""A safety retry cannot silently become a duplicate service measurement."""

import copy
import unittest

from scripts.controller.durable import digest
from scripts.prepare_trace_package import validate_pre_model_retry


class PreModelTraceRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prior = {'run_id': 'prior', 'python_source_commit': 'a' * 40,
                      'dataset_sha256': 'b' * 64, 'replicate': 0, 'max_wall_seconds': 3600,
                      'workload': {'kind': 'anonymous_trace_xqa_bf16_reference',
                                   'family': 'conversation', 'arrival_scaling': 1.0}}
        self.proof = {'classification': 'confirmed_original_pre_model_CONFLICT_no_service_observations',
                      'run_id': 'prior', 'request_sha256': digest(self.prior),
                      'original_status': 'CONFLICT', 'GPU_release_verified': True,
                      'model_loaded': False, 'AIPerf_ran': False, 'arm_measurement_present': False,
                      'successful_model_predictions': 0, 'terminal_result_sha256': 'c' * 64,
                      'source_commit': 'a' * 40, 'dataset_sha256': 'b' * 64}
        self.options = dict(replicate=1, family='conversation', arrival_scaling=1.0,
                            profile='xqa-bf16-reference', source_commit='a' * 40,
                            dataset_sha256='b' * 64, max_wall_seconds=3600)

    def test_released_pre_model_failure_accepts_only_next_attempt(self) -> None:
        validate_pre_model_retry(self.proof, self.prior, **self.options)

    def test_missing_or_service_started_or_unreleased_evidence_rejected(self) -> None:
        for field, value in (('GPU_release_verified', False), ('model_loaded', True),
                             ('AIPerf_ran', True), ('arm_measurement_present', True),
                             ('successful_model_predictions', 1), ('request_sha256', 'd' * 64),
                             ('original_status', 'COMPLETE'), ('terminal_result_sha256', None)):
            with self.subTest(field=field):
                proof = copy.deepcopy(self.proof)
                proof[field] = value
                with self.assertRaises((ValueError, TypeError)):
                    validate_pre_model_retry(proof, self.prior, **self.options)
        proof = copy.deepcopy(self.proof)
        proof.pop('AIPerf_ran')
        with self.assertRaises(ValueError):
            validate_pre_model_retry(proof, self.prior, **self.options)

    def test_changed_workload_source_or_budget_is_not_orchestration_retry(self) -> None:
        for field, value in (('replicate', 0), ('family', 'synthetic'), ('arrival_scaling', .5),
                             ('source_commit', 'd' * 40), ('dataset_sha256', 'e' * 64),
                             ('max_wall_seconds', 7200), ('profile', 'xqa')):
            with self.subTest(field=field):
                options = {**self.options, field: value}
                with self.assertRaises(ValueError):
                    validate_pre_model_retry(self.proof, self.prior, **options)


if __name__ == '__main__':
    unittest.main()
