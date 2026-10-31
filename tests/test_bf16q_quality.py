# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""A production quality run must bind the exact previously admitted operator."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.durable import digest
from scripts.controller.policy import BF16Q_DECODE_SOURCES,QUALITY_SOURCES,validate_request
from scripts.controller.production import candidate_quality_command,run_candidate_quality
from test_bf16q_operator import _request as operator_request


def _request() -> tuple:
    approval,r=operator_request();r.update(max_wall_seconds=5400,max_output_gib=5,
        backend_source_sha256={n:'a'*64 for n in BF16Q_DECODE_SOURCES+QUALITY_SOURCES})
    r['workload'].update(kind='native_quality_bf16q_production_batch',cohort='full_production_batch',samples_per_arm=1719,
        scorer_dependencies=approval['approved_workspace']+'/experiments/joint-kv-14b/longbench-deps-py312',
        longbench_repo=r['remote_output']+'/input/scorer',gsm8k_shots=r['remote_output']+'/input/five_train_examples.json',
        approved_data_manifest=r['remote_output']+'/input/approved_data_manifest.json',operator_validation={
            'run_id':'prod-bf16q-decode-op-r0-20261027a','source_commit':'78004505e2db44e88994899e8ea3e25c109978a7',
            'terminal_result_sha256':'b'*64,'audit_sha256':'d'*64,
            'audit_path':r['remote_output']+'/input/bf16q_operator_audit.json',
            'source_sha256':{n:'a'*64 for n in BF16Q_DECODE_SOURCES}})
    return approval,r


class BF16QQualityTests(unittest.TestCase):
    def test_full_candidate_requires_exact_admitted_source_and_real_production_scope(self) -> None:
        a,r=_request();validate_request(r,a)
        for location,key,value in [('request','concurrency',8),('request','arms',['bf16']),
                ('request','max_wall_seconds',5401),('workload','samples_per_arm',80),('workload','cohort','pilot')]:
            changed=copy.deepcopy(r);(changed if location=='request' else changed['workload'])[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_request(changed,a)
        changed=copy.deepcopy(r);changed['backend_source_sha256']['xllm/python/attention/flashinfer.py']='c'*64
        with self.assertRaises(ValueError):validate_request(changed,a)
        command=candidate_quality_command({'request':r},Path('/output'),'fp8_e4m3')
        self.assertIn('--fp8-bf16q-decode',command)
        self.assertEqual(command[command.index('--native-decode')+1],'flashinfer')
        self.assertEqual(command[command.index('--cohort-scope')+1],'full')
        self.assertEqual(command[command.index('--concurrency')+1],'128')

    def test_false_prerequisite_cannot_start_scorer_probe_model_or_client(self) -> None:
        _,r=_request();prior=r['workload']['operator_validation']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'audit.json';prior['audit_path']=str(path)
            audit={'classification':'confirmed_operator_evidence_integrity_and_BF16Q_Graph128_contract_only',
                **{k:prior[k] for k in ('run_id','source_commit','terminal_result_sha256')},
                'actual_numerical_Graph_reported_and_sealed':True,'model_loaded':False,
                'graph128':{'production_max_batch':128},'backend_source_sha256':prior['source_sha256'],
                'binary_sha256':'wrong','image_id':r['image_id'],'operators':r['operators']}
            path.write_text(json.dumps(audit));prior['audit_sha256']=digest(audit)
            import hashlib
            prior['audit_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch('scripts.controller.production.subprocess.run') as probe:
                with patch('scripts.controller.production.subprocess.Popen') as model:
                    with self.assertRaises(ValueError):run_candidate_quality({'request':r},root,root,'fp8_e4m3',{})
                    probe.assert_not_called();model.assert_not_called()


if __name__=='__main__':unittest.main()
