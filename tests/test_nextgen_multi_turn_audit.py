"""Reuse complete raw/metric auditing for exact original document sessions."""

import json
import unittest

from scripts.controller.production import audit_requests
from test_production_serving import ProductionRawAuditTests


class MultiTurnAuditTests(ProductionRawAuditTests):
    def setUp(self):
        super().setUp()
        data=json.loads(self.source.read_text())
        data['data'][1]['payloads'].append(data['data'][2]['payloads'][0]);data['data'].pop()
        self.source.write_text(json.dumps(data))
        self.rows[2]['metadata']['conversation_id']='source-1';self.rows[2]['metadata']['turn_index']=1

    def run_audit(self):
        (self.artifacts/'profile_export.jsonl').write_text('\n'.join(json.dumps(row) for row in self.rows))
        (self.artifacts/'profile_export_raw.jsonl').write_text('\n'.join(json.dumps(row) for row in self.raws))
        return audit_requests(self.artifacts,self.source,2,1,allow_multi_turn=True)

    def test_complete_two_turn_session_preserves_counts_and_one_token_semantics(self):
        result=self.run_audit();self.assertEqual(result['profiling_records'],2)
        self.assertEqual(result['warmup_records'],1);self.assertEqual(result['actual_output_tokens'],3)

    def test_overlap_between_session_turns_is_not_a_valid_replay(self):
        self.rows[2]['metadata']['request_start_ns']=self.rows[1]['metadata']['request_start_ns']
        with self.assertRaises(ValueError):self.run_audit()


if __name__=='__main__':unittest.main()
