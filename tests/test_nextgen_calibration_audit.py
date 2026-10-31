"""Candidate scales require complete raw range evidence rather than cached flags."""

import copy
import math
import unittest

from scripts.audit_nextgen_calibration import _snapshot_scales
from scripts.controller.nextgen_service import OBSERVATION_METHOD


def _snapshot():
    rows = [{'layer_id': layer, 'phase': phase, 'kind': kind, 'head': head,
             'finite_values': 512, 'nonfinite_values': 0,
             'absmax': (layer + head + 1) * (1 if kind == 'key' else 2)}
            for layer in range(3, 64, 4) for phase in ('prefill', 'decode')
            for kind in ('key', 'value') for head in range(4)]
    maxima = {kind: max(row['absmax'] for row in rows if row['kind'] == kind) for kind in ('key', 'value')}
    def scales(denominator):
        return {kind: 2.0**math.ceil(math.log2(value * 1.1 / denominator)) for kind, value in maxima.items()}
    return {'method': OBSERVATION_METHOD, 'quality_or_performance_certified': False,
            'layer_ids': list(range(3, 64, 4)), 'heads': 4, 'dimension': 256, 'rows': rows,
            'phase_calls_by_layer': [[1, 1] for _ in range(16)], 'observed_absmax': maxima,
            'counter_storage_bytes': 5376, 'FP8_E4M3_candidate_scales': scales(448),
            'NVFP4_global_candidate_scales': scales(2688)}


class CalibrationRawEvidenceTests(unittest.TestCase):
    def test_complete_ranges_produce_independently_recomputed_format_candidates(self):
        candidates = _snapshot_scales(_snapshot())
        self.assertEqual(candidates['fp8_e4m3'], {'key': .25, 'value': .5})
        self.assertEqual(candidates['nvfp4'], {'key': .03125, 'value': .0625})

    def test_missing_nonfinite_or_optimistic_observations_are_rejected(self):
        for mutation in ('empty', 'nonfinite', 'missing', 'duplicate', 'scale', 'maximum', 'phase', 'bytes'):
            value = copy.deepcopy(_snapshot())
            if mutation == 'empty':value['rows'][0]['finite_values'] = 0
            elif mutation == 'nonfinite':value['rows'][0]['nonfinite_values'] = 1
            elif mutation == 'missing':value['rows'].pop()
            elif mutation == 'duplicate':value['rows'][0] = value['rows'][1]
            elif mutation == 'scale':value['FP8_E4M3_candidate_scales']['key'] *= .5
            elif mutation == 'maximum':value['observed_absmax']['value'] += 1
            elif mutation == 'phase':value['phase_calls_by_layer'][0][1] = 0
            else:value['counter_storage_bytes'] -= 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _snapshot_scales(value)


if __name__ == '__main__':
    unittest.main()
