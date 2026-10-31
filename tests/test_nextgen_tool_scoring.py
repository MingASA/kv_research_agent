"""Cached completion cannot hide wrong tool arguments, feedback or later questions."""

import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.controller.durable import digest
from scripts.score_nextgen_tools import replay_task


class ArithmeticEnvironment:
    instances={}
    def call(self,name,args):return {'result':json.dumps({'result':args['a']+args['b']}),
                                    'dispatch_valid':True,'tool_reported_error':False}
    def reference_call(self,unused):return {'result':json.dumps({'result':3})}


class CheckerFixture:
    checker_AST_sha256='CPU_checker_fixture'
    def __init__(self,source):pass
    def environment(self,task):return ArithmeticEnvironment()
    def state_checker(self,model,gold):return {'valid':True}
    def response_checker(self,actual,gold,turn):return {'valid':all(value in actual for value in gold)}


def stream(delta,finish):
    return ['data: '+json.dumps({'choices':[{'index':0,'delta':delta,'finish_reason':finish}],
        'usage':{'prompt_tokens':30,'completion_tokens':1,'total_tokens':31}}),'data: [DONE]']


class IndependentToolScoringTests(unittest.TestCase):
    def fixture(self,a=1):
        task={'id':'CPU_task','original_user_turns':[[{'role':'user','content':'Add1and2'}]],
              'initial_payload':{'model':'CPU_fixture','max_tokens':1024,'messages':[],'tools':[]}}
        call={'id':'call_1','type':'function','function':{'name':'add','arguments':json.dumps({'a':a,'b':2})}}
        reply={'result':json.dumps({'result':a+2}),'dispatch_valid':True,'tool_reported_error':False,
               'sequence':2,'task_id':task['id'],'CPU_call_monotonic_ns':1}
        messages=copy.deepcopy(task['original_user_turns'][0]);rows=[]
        for step,raw in enumerate((stream({'tool_calls':[{'index':0,**call}]},'tool_calls'),stream({'content':'x'},'stop'))):
            payload=copy.deepcopy(task['initial_payload']);payload['messages']=copy.deepcopy(messages)
            rows.append({'task_id':task['id'],'turn':0,'step':step,'request':payload,'request_sha256':digest(payload),
                         'status':200,'error':None,'raw_SSE':raw,'prompt_tokens_CPU_expected':30})
            if step==0:messages.extend([{'role':'assistant','content':'','tool_calls':[call]},
                {'role':'tool','tool_call_id':'call_1','content':reply['result']}])
        return {'task':task,'requests':rows,'tool_results':[{'task_id':task['id'],'turn':0,'step':0,
                    'call':call,'arguments':{'a':a,'b':2},'reply':reply}],'reference':[['trusted_reference_only']],
                'generation':{'state':'ALL_ORIGINAL_USER_TURNS_COMPLETED','tool_calls':1,
                              'model_API_requests':2,'finished_user_turns':1}}

    def run_replay(self,data):
        with patch('scripts.score_nextgen_tools.ToolFactory',CheckerFixture):return replay_task(Path('unused'),data)

    def test_correct_native_protocol_can_still_have_wrong_task_arguments(self):
        self.assertTrue(self.run_replay(self.fixture())['task_correct'])
        wrong=self.run_replay(self.fixture(a=2))
        self.assertFalse(wrong['task_correct']);self.assertTrue(wrong['raw_history_tool_feedback_and_reference_semantics_verified'])

    def test_changed_tool_reply_or_future_question_cannot_be_hidden_by_cached_complete(self):
        for change in ('reply','future'):
            data=self.fixture()
            if change=='reply':data['tool_results'][0]['reply']['result']='fabricated'
            else:
                row=data['requests'][1];row['request']['messages'].append({'role':'user','content':'future'})
                row['request_sha256']=digest(row['request'])
            with self.subTest(change=change),self.assertRaises(ValueError):self.run_replay(data)

    def test_optimistic_completed_turn_count_is_rejected(self):
        data=self.fixture();data['generation']['finished_user_turns']=2
        with self.assertRaises(ValueError):self.run_replay(data)


if __name__=='__main__':unittest.main()
