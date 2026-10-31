# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Missing-arm recovery requires released, independently recovered exact input."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.controller.durable import digest
from scripts.controller.supervisor import UnsafeState, verify_quality_recovery


class QualityRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name).resolve()
        self.old = self.base/'agent_work/runs/prior'; self.new = self.base/'agent_work/runs/missing'
        (self.old/'output/fp8_e4m3').mkdir(parents=True); (self.new/'input').mkdir(parents=True)
        self.request = {key:'fixture' for key in ('binary_sha256','model_config_sha256','operators','service','graph_mode',
            'fp8_scales','concurrency','python_source_commit','backend_source_sha256','dataset_sha256','model_path','image_id')}
        prior = copy.deepcopy(self.request)
        self._write(self.old/'ticket.json', {'request': prior})
        result_sha = self._write(self.old/'result.json', {'status':'CONFLICT','gpu_release_verified':True,
            'live_owned_containers':[], 'request_sha256':digest(prior)})
        measurement_sha = self._write(self.old/'output/fp8_e4m3/measurement.json', {'measurement_success':True,
            'request_sha256':digest(prior), 'evidence_sha256':{}})
        self.audit = {'classification':'confirmed_recovered_fragment_arm_prediction_integrity_and_rescoring',
            'run_id':'prior','audited_arms':['fp8_e4m3'],'samples_per_arm':1639,'root_complete':False,
            'terminal_result_sha256':result_sha,'conflict_seconds_after_last_prediction':4.6,
            'arm_measurement_sha256':{'fp8_e4m3':measurement_sha}}
        audit_sha = self._write(self.new/'input/recovered_fragment_audit.json', self.audit)
        self.request.update(arms=['bf16'],workload={'recovery':{'prior_run_id':'prior','verified_arm':'fp8_e4m3',
            'prior_terminal_result_sha256':result_sha,'arm_measurement_sha256':measurement_sha,'recovered_audit_sha256':audit_sha}})

    def _write(self, path: Path, value: dict) -> str:
        data = (json.dumps(value)+'\n').encode();path.write_bytes(data);return hashlib.sha256(data).hexdigest()

    def _check(self) -> None:
        verify_quality_recovery(self.request,self.base,self.new,os.getuid())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_recovered_fp8_allows_only_missing_bf16(self) -> None:
        self._check()
        self.request['arms']=['fp8_e4m3']
        with self.assertRaises(UnsafeState): self._check()

    def test_changed_input_or_existing_bf16_blocks_duplicate_work(self) -> None:
        original = self.request['dataset_sha256'];self.request['dataset_sha256']='changed'
        with self.assertRaises(UnsafeState): self._check()
        self.request['dataset_sha256']=original
        (self.old/'output/bf16/quality').mkdir(parents=True)
        with self.assertRaises(UnsafeState): self._check()

    def test_overlap_or_tampered_audit_does_not_unlock_gpu(self) -> None:
        self.audit['conflict_seconds_after_last_prediction']=-1
        path=self.new/'input/recovered_fragment_audit.json'
        self.request['workload']['recovery']['recovered_audit_sha256']=self._write(path,self.audit)
        with self.assertRaises(UnsafeState): self._check()


if __name__=='__main__':
    unittest.main()
