"""Conversation failures remain raw evidence and cannot inject later questions."""

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.controller.native_tools import parse_sse_packets
from scripts.controller.nextgen_multiturn import generate_session


def response(*,calls=False,empty=False,prompt=30,status=200,identifier='call-1'):
    delta={'content':'' if empty else 'x'}
    if calls:delta={'tool_calls':[{'index':0,'id':identifier,'type':'function',
                                 'function':{'name':'add','arguments':'{"a":1,"b":2}'}}]}
    return {'status':status,'error':None,'raw_SSE':[
        'data: '+json.dumps({'choices':[{'index':0,'delta':delta,'finish_reason':None}]}),
        'data: '+json.dumps({'choices':[{'index':0,'delta':{},'finish_reason':'tool_calls' if calls else 'stop'}],
                            'usage':{'prompt_tokens':prompt,'completion_tokens':1,'total_tokens':prompt+1}}),
        'data: [DONE]']}


class MemoryChild:
    ready={'CPU_fixture_only':True}
    def __enter__(self):self.closed=False;return self
    def __exit__(self,*ignored):self.closed=True
    def call(self,name,args):return {'result':'{"result":3}','dispatch_valid':True}


class NativeConversationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.task={'id':'CPU_only','source_line_sha256_without_LF':'a'*64,
            'original_user_turns':[[{'role':'user','content':'first'}],[{'role':'user','content':'later'}]],
            'initial_payload':{'model':'fixture','messages':[],'max_tokens':1024,'tools':[]}}
        self.child=MemoryChild()

    def run_session(self,generate):
        with patch('scripts.controller.nextgen_multiturn.ToolChild',return_value=self.child):
            return generate_session(self.task,source=self.root,output=self.root/'out',generate=generate,
                                    count_tokens=lambda payload:30,deadline_ns=time.monotonic_ns()+10**10)

    def test_real_wire_history_uses_tool_results_before_revealing_the_next_user(self):
        offered=[]
        def generate(payload,*,turn,step,**ignored):
            offered.append(payload)
            return response(calls=turn==0 and step==0,empty=turn==0 and step==1)
        result=self.run_session(generate)
        self.assertEqual(result['state'],'ALL_ORIGINAL_USER_TURNS_COMPLETED')
        self.assertEqual(result['empty_final_assistant_turns'],1);self.assertEqual(result['model_API_requests'],3)
        self.assertEqual([row['content'] for row in offered[1]['messages'] if row['role']=='user'],['first'])
        self.assertEqual(offered[1]['messages'][-1]['role'],'tool')
        self.assertEqual(offered[1]['messages'][-1]['content'],'{"result":3}')
        self.assertEqual([row['content'] for row in offered[2]['messages'] if row['role']=='user'],['first','later'])
        self.assertIsInstance(offered[2]['messages'][1]['tool_calls'][0]['function']['arguments'],str)
        self.assertFalse(result['model_task_correctness_scored']);self.assertTrue(self.child.closed)

    def test_bad_usage_or_HTTP_keeps_raw_reply_and_stops_before_the_future_user(self):
        for name,reply in (('usage',response(prompt=31)),('HTTP',response(status=500))):
            # Each immutable attempt has its own local output.
            self.root=Path(self.temp.name)/name;self.root.mkdir()
            offered=[]
            def generate(payload,**ignored):offered.append(payload);return reply
            result=self.run_session(generate)
            with self.subTest(name=name):
                self.assertEqual(result['state'],'INCOMPLETE');self.assertFalse(result['actual_API_contracts_verified'])
                self.assertEqual(result['model_API_requests'],1);self.assertEqual(result['finished_user_turns'],0)
                self.assertNotIn('later',str(offered))
                rows=[json.loads(line) for line in (self.root/'out/requests.jsonl').read_text().splitlines()]
                self.assertEqual(rows[0]['status'],reply['status']);self.assertEqual(len(rows),1)

    def test_tool_loop_has_a_real_step_budget_and_cannot_continue_into_later_questions(self):
        offered=[]
        def generate(payload,*,step,**ignored):
            offered.append(payload);return response(calls=True,identifier='call-'+str(step))
        result=self.run_session(generate)
        self.assertEqual(result['model_API_requests'],10);self.assertEqual(result['tool_calls'],10)
        self.assertEqual(result['failure_stage'],'assistant_tool_step_budget')
        self.assertTrue(result['actual_API_contracts_verified']);self.assertEqual(result['finished_user_turns'],0)
        self.assertNotIn('later',str(offered));self.assertTrue(self.child.closed)

    def test_empty_final_is_explicitly_opted_in_and_not_a_tool_or_quality_success(self):
        row=response(empty=True);packets=[{'name':'data','value':line[5:].strip()} for line in row['raw_SSE']]
        with self.assertRaises(ValueError):parse_sse_packets(packets)
        parsed=parse_sse_packets(packets,allow_empty_final=True)
        self.assertEqual(parsed['content'],'');self.assertEqual(parsed['tool_calls'],[])
        self.assertEqual(parsed['finish_reason'],'stop')


if __name__=='__main__':unittest.main()
