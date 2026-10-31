"""Actual response contracts and bounded mixed-length quality demand."""

import base64
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.controller.nextgen_quality import _request, _validate_inputs, run_core_generations


def _row(index=0,prompt=5):
    return {'id':'test-'+str(index),'task':'fixture','group':'core','source_index':index,
            'endpoint':'/v1/chat/completions','prompt_tokens_cpu':prompt,
            'request':{'model':'test-model','messages':[{'role':'user','content':'fixture'}],
                       'temperature':0,'max_tokens':1,'chat_template_kwargs':{'enable_thinking':False}}}


class QualityClientTests(unittest.TestCase):
    def test_actual_chat_and_completion_bytes_usage_are_preserved(self):
        response={'usage':{'prompt_tokens':5,'completion_tokens':1,'total_tokens':6},
                  'choices':[{'message':{'content':'42'},'finish_reason':'stop'}]}
        class Response:
            status=200
            def __enter__(self):return self
            def __exit__(self,*_args):return False
            def read(self,_maximum):return json.dumps(response).encode()
        with patch('urllib.request.urlopen',return_value=Response()):
            result=_request(_row(),3)
            self.assertTrue(result['contract_valid'])
            self.assertEqual(json.loads(base64.b64decode(result['raw_HTTP_body_base64'])),response)
            row=_row();row['endpoint']='/v1/completions'
            row['request']={'model':'test-model','prompt':'code prefix','temperature':0,'max_tokens':1}
            response['choices']=[{'text':' next','finish_reason':'length'}]
            self.assertTrue(_request(row,3)['contract_valid'])

    def test_HTTP200_notice_wrong_usage_and_nonobject_body_are_not_quality_success(self):
        bodies=[{'error':{'message':'capacity limit'}},
                {'choices':[{'message':{'content':'42'},'finish_reason':'stop'}],
                 'usage':{'prompt_tokens':2,'completion_tokens':1,'total_tokens':3}}, []]
        for body in bodies:
            with self.subTest(body=body):
                class Response:
                    status=200
                    def __enter__(self):return self
                    def __exit__(self,*_args):return False
                    def read(self,_maximum):return json.dumps(body).encode()
                with patch('urllib.request.urlopen',return_value=Response()):result=_request(_row(),3)
                self.assertFalse(result['contract_valid'])
                self.assertEqual(json.loads(base64.b64decode(result['raw_HTTP_body_base64'])),body)

    def test_input_validation_refuses_duplicates_forced_EOS_or_unbounded_context(self):
        row=_row();_validate_inputs([row],1,'test-model',12)
        for rows,budget in [([row,row],12),([{**row,'request':{**row['request'],'ignore_eos':True}}],12),([row],5)]:
            with self.subTest(rows=rows,budget=budget),self.assertRaises(ValueError):
                _validate_inputs(rows,1,'test-model',budget)

    def test_long_requests_cannot_exceed_the_common_client_budget_and_no_predictions_drop(self):
        rows=[_row(0,5),_row(1,5),_row(2,9)]
        active=0;peak=0;lock=threading.Lock();barrier=threading.Barrier(2)
        def call(row,timeout):
            nonlocal active,peak
            cost=row['prompt_tokens_cpu']+1
            with lock:active+=cost;peak=max(peak,active)
            if row['source_index']<2:barrier.wait(timeout=2)
            with lock:active-=cost
            return {'id':row['id'],'contract_valid':True,'finish_reason':'stop'}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'input').mkdir();out=root/'output';out.mkdir()
            raw=('\n'.join(json.dumps(row) for row in rows)+'\n').encode()
            (root/'input/quality_requests.jsonl').write_bytes(raw)
            manifest=json.dumps({'model_id':'test-model','request_payload_sha256':hashlib.sha256(raw).hexdigest()}).encode()
            (root/'input/quality_manifest.json').write_bytes(manifest)
            request={'model_id':'test-model','quality_evaluation':{'requests':3,
                     'requests_sha256':hashlib.sha256(raw).hexdigest(),'client_concurrency':3,
                     'source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),
                     'client_token_budget':12,'max_seconds':601,'request_timeout_seconds':3}}
            with patch('scripts.controller.nextgen_quality._request',side_effect=call):
                result=run_core_generations(request,root,out,available_seconds=601)
            self.assertEqual(result['durable_responses'],3)
            self.assertTrue(result['all_response_contracts_valid'])
            self.assertEqual(peak,12)
            self.assertFalse(result['accuracy_scored'])
            self.assertEqual({json.loads(line)['id'] for line in (out/'quality_predictions.jsonl').read_text().splitlines()},
                             {row['id'] for row in rows})


if __name__=='__main__':unittest.main()
