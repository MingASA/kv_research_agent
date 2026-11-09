"""The combined allocation executes main before model and retains both receipts."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.next_stage_deployment import (
    P1_EXTENSION, P0_REPAIR_SOURCE, P0_REPAIR_BACKEND, run, operator_acceptance_passed,
)
from tests.test_kv_p1_operator import P1AcceptanceTest


class InlineMainTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prior = self.root/'prior'
        self.output = self.root/'output'
        self.output.mkdir()
        self.pilot = P1AcceptanceTest().acceptance(mx=True)
        self.pilot['phase'] = 'pilot'
        directory = self.prior/'output/mxfp4/operators'
        directory.mkdir(parents=True)
        (directory/'acceptance.json').write_text(json.dumps(self.pilot))
        self.old = dict(python_source_commit='1'*40, backend_source_sha256={'tool.py': '2'*64},
                        model_id='model', model_config_sha256='4'*64, image_id='image',
                        arms=['mxfp4'], gpu_uuids=['gpu'], p1_extension=P1_EXTENSION,
                        probe_driver_sha256='2'*64, p1_phase='pilot')
        self.request = dict(self.old, run_id='combined', remote_output='/new',
                            python_source_repo='/new/source_repo', dataset_path='/new/input/validation.json',
                            max_wall_seconds=3600, native_configuration={'value_scale': 1.0},
                            p1_inline_main_operator=True,
                            operator_qualification_reference={'run_id':'actual-pilot','result_sha256':'5'*64})

    def invoke(self, returned=None, error=None):
        events = []
        acceptance = returned if returned is not None else P1AcceptanceTest().acceptance(mx=True)

        def operator(ticket, root, output, arm, env):
            events.append('main')
            request = ticket['request']
            self.assertEqual(request['p1_phase'], 'main')
            self.assertEqual(request['p1_pilot_reference'], self.request['operator_qualification_reference'])
            self.assertEqual(request['python_source_repo'], '/new/source_repo')
            if error:
                raise error
            return {'acceptance': acceptance}

        def model(*args):
            events.append('model')
            return {'actual_model_called': True}

        with patch('scripts.controller.next_stage_deployment._completed_reference', return_value=(self.prior,self.old)), \
                patch('scripts.controller.kv_p1_operator.run', side_effect=operator), \
                patch('scripts.controller.upstream_qualification.run', side_effect=model):
            try:
                result = run({'request':self.request}, self.root, self.output, 'mxfp4', {})
            except ValueError:
                self.assertNotIn('model',events)
                raise
        return result, events

    def test_actual_main_precedes_model_and_phase_is_not_relabelled(self):
        self.assertFalse(operator_acceptance_passed(self.pilot,'mxfp4'))
        result, events = self.invoke()
        self.assertEqual(events,['main','model'])
        self.assertTrue(result['actual_model_called'])
        receipt = json.loads((self.output/'inline_operator_main.json').read_text())
        self.assertEqual(receipt['acceptance']['phase'],'main')
        self.assertFalse(receipt['separate_GPU_allocation'])
        self.assertEqual(self.pilot['phase'],'pilot')

    def test_failed_main_or_changed_identity_never_loads_model(self):
        for key, value in [('pass',False), ('phase','pilot'), ('runtime_source_commit','9'*40), ('tool_sha256','9'*64)]:
            acceptance = P1AcceptanceTest().acceptance(mx=True)
            acceptance[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.invoke(returned=acceptance)
            receipt = self.output/'inline_operator_main.json'
            if receipt.exists():
                receipt.unlink()

    def test_model_not_called_after_main_exception(self):
        with self.assertRaises(ValueError):
            self.invoke(error=ValueError('actual numerical failure'))

    def test_pilot_gpu_mismatch_or_missing_inline_request_is_rejected(self):
        self.request['gpu_uuids']=['other-gpu']
        with self.assertRaises(ValueError):self.invoke()
        self.request['gpu_uuids']=['gpu']
        self.request.pop('p1_inline_main_operator')
        with self.assertRaises(ValueError):self.invoke()

    def repair_request(self):
        self.request.pop('p1_inline_main_operator')
        self.request['p0_repair_only']=True
        for request in (self.request,self.old):
            request['python_source_commit']=P0_REPAIR_SOURCE
            request['backend_source_sha256']={'xllm/python/attention/flashinfer_native_kv.py':P0_REPAIR_BACKEND}
        self.pilot['runtime_source_commit']=P0_REPAIR_SOURCE
        (self.prior/'output/mxfp4/operators/acceptance.json').write_text(json.dumps(self.pilot))

    def test_exact_repair_source_reuses_actual_pilot_without_repeating_main_timings(self):
        self.repair_request()
        result,events=self.invoke()
        self.assertEqual(events,['model'])
        self.assertTrue(result['actual_model_called'])
        self.assertEqual(self.pilot['phase'],'pilot')

    def test_repair_flag_never_accepts_other_source_or_failed_numerics(self):
        self.repair_request()
        self.request['python_source_commit']='9'*40
        with self.assertRaises(ValueError):self.invoke()
        self.request['python_source_commit']=P0_REPAIR_SOURCE
        self.pilot['pass']=False
        (self.prior/'output/mxfp4/operators/acceptance.json').write_text(json.dumps(self.pilot))
        with self.assertRaises(ValueError):self.invoke()


if __name__ == '__main__':
    unittest.main()
