"""Raw evidence must override optimistic cached worker summaries."""

import copy
import json
import unittest
import tempfile
from pathlib import Path

from scripts.audit_nextgen_validation import _jsonl, _stream


class NextgenIndependentAuditTests(unittest.TestCase):
    def test_literal_unicode_paragraph_separator_does_not_split_a_valid_JSON_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'responses.jsonl'
            row={'answer':'first\u2028second','id':'real-text-fixture'}
            path.write_bytes((json.dumps(row,ensure_ascii=False)+'\n').encode())
            self.assertEqual(_jsonl(path),[row])
            path.write_bytes(json.dumps(row,ensure_ascii=False).encode())
            with self.assertRaises(ValueError):_jsonl(path)

    def fixture(self):
        case={'messages':[{'role':'user','content':'17 times13?'}],'expected':'221'}
        request={'model':'test-model','messages':case['messages'],'temperature':0,'max_tokens':96,
                 'stream':True,'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
        usage={'prompt_tokens':27,'completion_tokens':4,'total_tokens':31}
        row={'request':request,'error':None,'answer':'221','usage':usage,'contract_valid':True,
             'expected_substring_present':True,'raw_SSE':[
                 'data: '+json.dumps({'choices':[{'delta':{'content':'221'},'finish_reason':None}]}),
                 'data: '+json.dumps({'choices':[{'delta':{},'finish_reason':'stop'}],'usage':usage}),
                 'data: [DONE]']}
        return case,row

    def test_complete_raw_stream_matches_frozen_request_and_tokenizer(self):
        case,row=self.fixture()
        self.assertEqual(_stream(row,case,27,'test-model'),'221')

    def test_optimistic_cached_flags_cannot_hide_changed_raw_or_incomplete_stream(self):
        case,row=self.fixture()
        for mutation in ('cached_answer','raw_answer','done','prompt_tokens'):
            broken=copy.deepcopy(row)
            if mutation=='cached_answer':broken['answer']='222'
            elif mutation=='raw_answer':
                broken['answer']='222';broken['raw_SSE'][0]=broken['raw_SSE'][0].replace('221','222')
            elif mutation=='done':broken['raw_SSE'].pop()
            else:broken['usage']['prompt_tokens']=28
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                _stream(broken,case,27,'test-model')
