"""Performance inputs retain actual text, natural EOS and session cardinality."""

import copy
import json
from pathlib import Path
import unittest

from scripts.controller.nextgen_performance import aiperf_command,validate_payloads,validate_cli_protocol_proof


class NextgenPerformanceInputTests(unittest.TestCase):
    def payload(self):
        return {'model':'fixture','messages':[{'role':'user','content':'actual question'}],
                'temperature':0,'max_tokens':128,'stream':True,'stream_options':{'include_usage':True},
                'chat_template_kwargs':{'enable_thinking':False}}
    def data(self,payload=None):
        return json.dumps({'data':[{'session_id':'doc','payloads':[payload or self.payload(),self.payload()]}]}).encode()
    def test_original_two_turn_session_counts_two_requests(self):
        result=validate_payloads(self.data(),model_id='fixture')
        self.assertEqual((result['sessions'],result['requests']),(1,2));self.assertFalse(result['arrival_trace_claim'])
    def test_forced_lengths_stop_overrides_other_model_or_template_are_rejected(self):
        for key,value in [('ignore_eos',True),('stop',['end']),('model','other'),
                          ('chat_template_kwargs',{'enable_thinking':True}),('max_tokens',100000)]:
            payload=self.payload();payload[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_payloads(self.data(payload),model_id='fixture')
    def test_no_implicit_warmup_or_synthetic_timing_in_measured_command(self):
        command=aiperf_command('fixture',Path('/run/input.json'),Path('/run/profile'),requests=32,concurrency=16)
        self.assertEqual(command[command.index('--request-count')+1],'32')
        self.assertNotIn('--warmup-request-count',command);self.assertNotIn('--fixed-schedule',command)
    def test_invalid_demand_or_relative_paths_rejected(self):
        with self.assertRaises(ValueError):aiperf_command('fixture',Path('relative'),Path('/profile'),requests=32,concurrency=16)
        with self.assertRaises(ValueError):aiperf_command('fixture',Path('/input'),Path('/profile'),requests=32,concurrency=1000)
    def test_multiturn_session_limit_is_separate_from_wire_request_safety_cap(self):
        command=aiperf_command('fixture',Path('/input'),Path('/profile'),requests=32,concurrency=16,sessions=16)
        self.assertEqual(command[command.index('--request-count')+1],'32')
        self.assertEqual(command[command.index('--num-sessions')+1],'16')
        for sessions in (0,True,15,33):
            with self.subTest(sessions=sessions),self.assertRaises(ValueError):
                aiperf_command('fixture',Path('/input'),Path('/profile'),requests=32,concurrency=16,sessions=sessions)
    def test_old_synchronous_protocol_proof_cannot_qualify_new_dispatch(self):
        old={'all_checks_pass':True,'AIPerf_version':'0.13.0','observed_wire_request_total':194}
        validate_cli_protocol_proof(old)
        with self.assertRaisesRegex(ValueError,'heterogeneous-latency'):
            validate_cli_protocol_proof(old,require_session_limit=True)
        current=old|{'schema':'nextgen-full-cli-session-stop-v2','all_cases_session_limit_verified':True,
            'heterogeneous_latency_verified':True,'multi_turn_concurrency':16,
            'request_count_only_negative_control_reproduced':True}
        validate_cli_protocol_proof(current,require_session_limit=True)
        for field in ('all_cases_session_limit_verified','heterogeneous_latency_verified',
                      'request_count_only_negative_control_reproduced'):
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_cli_protocol_proof(current|{field:False},require_session_limit=True)


if __name__=='__main__':unittest.main()
