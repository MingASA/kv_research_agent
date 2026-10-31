"""Missing-only recovery keeps every prior response and rejects any overlap."""

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from scripts.controller.durable import canonical_bytes,digest
from scripts.controller.nextgen_service import verify_quality_completion
from test_nextgen_native_service import _service


class QualityCompletionPrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        _,self.prior=_service('fp8_e4m3');self.prior['run_id']='old'
        self.request=copy.deepcopy(self.prior);self.old=self.base/'agent_work/runs/old';self.new=self.base/'agent_work/runs/new'
        self.request.update(run_id='new',remote_output=str(self.new),quality_evaluation={'requests':802})
        (self.old/'input').mkdir(parents=True);(self.old/'output/fp8_e4m3').mkdir(parents=True);(self.new/'input').mkdir(parents=True)
        self.full=[{'id':f'id-{i}','request':{'original':i}} for i in range(1720)]
        self.pred=[{'id':r['id']} for r in self.full[:918]]
        oldraw=b''.join(canonical_bytes(r) for r in self.full);predraw=b''.join(canonical_bytes(r) for r in self.pred)
        (self.old/'input/quality_requests.jsonl').write_bytes(oldraw);(self.old/'output/fp8_e4m3/quality_predictions.jsonl').write_bytes(predraw)
        (self.old/'ticket.json').write_bytes(canonical_bytes({'request':self.prior,'request_sha256':digest(self.prior)}))
        result=canonical_bytes({'run_id':'old','request_sha256':digest(self.prior),'status':'FAILED','gpu_release_verified':True,'live_owned_containers':[]})
        (self.old/'result.json').write_bytes(result);self.missing=self.full[918:]
        current=b''.join(canonical_bytes(r) for r in self.missing);(self.new/'input/quality_requests.jsonl').write_bytes(current)
        self.cert={'state':'INDEPENDENT_PARTIAL_RAW_QUALITY_VALIDATED_NOT_FULL_COHORT',
            'all_existing_raw_HTTP_usage_payloads_verified':True,'invalid_responses_filtered':False,
            'selection_uses_answers_labels_or_scores':False,'logical_full_input_count':1720,'source_request':self.prior,
            'source_request_sha256':digest(self.prior),'original_result_sha256':hashlib.sha256(result).hexdigest(),
            'parent_prediction_sha256':hashlib.sha256(predraw).hexdigest(),'durable_count':918,
            'durable_ids':sorted(r['id'] for r in self.pred),'logical_full_requests_sha256':hashlib.sha256(oldraw).hexdigest(),
            'missing_count':802,'completion_requests_sha256':hashlib.sha256(current).hexdigest()}
        self.freeze()
    def freeze(self):
        raw=canonical_bytes(self.cert);(self.new/'input/partial_quality_certificate.json').write_bytes(raw)
        self.request['quality_completion']={'anchor_run_id':'old','partial_raw_certificate_sha256':hashlib.sha256(raw).hexdigest()}
    def verify(self):verify_quality_completion(self.request,self.base,os.getuid())
    def test_exact_original_complement_and_unchanged_deployment_are_admitted(self):self.verify()
    def test_overlap_changed_payload_or_reordered_missing_rows_is_rejected(self):
        for rows in ([self.full[0]]+self.missing[1:],list(reversed(self.missing)),self.missing[:-1]):
            (self.new/'input/quality_requests.jsonl').write_bytes(b''.join(canonical_bytes(r) for r in rows))
            with self.subTest(first=rows[0]['id']),self.assertRaises(ValueError):self.verify()
    def test_unreleased_parent_or_changed_model_cannot_be_resumed(self):
        self.request['binary_sha256']='0'*64
        with self.assertRaises(ValueError):self.verify()
    def test_score_selected_or_filtered_previous_responses_cannot_be_a_completion_source(self):
        for name,value in [('selection_uses_answers_labels_or_scores',True),('invalid_responses_filtered',True)]:
            original=self.cert[name];self.cert[name]=value;self.freeze()
            with self.subTest(name=name),self.assertRaises(ValueError):self.verify()
            self.cert[name]=original


if __name__=='__main__':unittest.main()
