"""A cached prediction cannot bypass actual HTTP/body/token audit."""

import base64
import copy
import hashlib
import json
import unittest

from scripts.controller.durable import canonical_bytes
from scripts.score_nextgen_core_quality import _verify_prediction


class RawQualityScoringTests(unittest.TestCase):
    def fixture(self):
        expected={'id':'example','endpoint':'/v1/chat/completions','prompt_tokens_cpu':5,
                  'request':{'model':'Qwen3.8-27B','messages':[{'role':'user','content':'fixture'}],'max_tokens':32}}
        response={'choices':[{'message':{'content':'42'},'finish_reason':'stop'}],
                  'usage':{'prompt_tokens':5,'completion_tokens':1,'total_tokens':6}}
        row={'id':'example','endpoint':expected['endpoint'],'request':expected['request'],
             'request_body_sha256':hashlib.sha256(canonical_bytes(expected['request'])).hexdigest(),
             'raw_HTTP_body_base64':base64.b64encode(json.dumps(response).encode()).decode(),
             'HTTP_status':200,'response':response,'answer':'42','usage':response['usage'],
             'finish_reason':'stop','request_started_monotonic_ns':10,'request_ended_monotonic_ns':20,'error':None}
        return expected,row

    def test_raw_complete_prediction_verifies(self):
        expected,row=self.fixture()
        self.assertEqual(_verify_prediction(row,expected),'42')

    def test_empty_first_stop_output_can_be_retained_as_quality_zero(self):
        expected,row=self.fixture()
        response=row['response'];response['choices'][0]['message']['content']=''
        row['answer']='';row['raw_HTTP_body_base64']=base64.b64encode(json.dumps(response).encode()).decode()
        with self.assertRaises(ValueError):_verify_prediction(row,expected)
        self.assertEqual(_verify_prediction(row,expected,allow_empty_stop=True),'')
        response['choices'][0]['finish_reason']='length';row['finish_reason']='length'
        row['raw_HTTP_body_base64']=base64.b64encode(json.dumps(response).encode()).decode()
        with self.assertRaises(ValueError):_verify_prediction(row,expected,allow_empty_stop=True)

    def test_cached_answer_wrong_raw_usage_or_failure_cannot_be_scored_as_valid(self):
        for mutation in ('answer','raw_answer','usage','error','notice','request','clock'):
            expected,row=self.fixture();row=copy.deepcopy(row)
            if mutation=='answer':row['answer']='43'
            elif mutation=='raw_answer':
                response=copy.deepcopy(row['response']);response['choices'][0]['message']['content']='43'
                row['raw_HTTP_body_base64']=base64.b64encode(json.dumps(response).encode()).decode()
            elif mutation=='usage':row['usage']['prompt_tokens']=2
            elif mutation=='error':row['error']='request timeout'
            elif mutation=='notice':row['HTTP_status']=429
            elif mutation=='request':row['request']['max_tokens']=16
            else:row['request_ended_monotonic_ns']=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                _verify_prediction(row,expected)


if __name__=='__main__':unittest.main()
