"""Actual tool-serving configuration stays separate from quality, perf and CPU fixtures."""

import copy
import unittest

from scripts.controller.nextgen_service import native_server_command
from scripts.controller.nextgen_tools_service import TOOLS_KIND,TOOLS_SCOPE,validate_cpu_protocol
from scripts.controller.policy import validate_request
import test_nextgen_native_service as fixtures


class NativeToolServiceContractTests(unittest.TestCase):
    def test_stale_template_probe_cannot_qualify_current_native_tool_runtime(self):
        proof={'state':'CPU32_CONVERSATIONS_AND_NATIVE_CPP224_HISTORIES_VERIFIED_NOT_MODEL',
            'tasks':32,'user_turns':112,'real_isolated_tool_calls':173,'fake_SSE_provider_invocations':224,
            'all_CPP_and_HF_raw_JSON_counts_match':True,
            'native_request_parameters_map_order':'protobuf_JsonTool_parameters_double_map_v2',
            'native_CPP_reference_sha256':'a'*64}
        validate_cpu_protocol(proof)
        for changes in ({'native_request_parameters_map_order':None},{'native_CPP_reference_sha256':None},
                        {'all_CPP_and_HF_raw_JSON_counts_match':False},{'fake_SSE_provider_invocations':223}):
            broken={**proof,**changes}
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_cpu_protocol(broken)

    def fixture(self):
        approval,request=fixtures._service('nvfp4')
        request.update(max_wall_seconds=7200,range_observation=False,
            service={**request['service'],'max_batch':16,'linear_state_slots':32,'max_memory_utilization':.9},
            workload={'kind':TOOLS_KIND,'scope':TOOLS_SCOPE},
            tool_evaluation={'tasks':32,'user_turns':112,'client_concurrency':4,'max_steps_per_turn':10,
                'max_new_tokens':1024,'max_seconds':3600,'tool_parser':'qwen3_coder',**{name:'a'*64 for name in
                ('manifest_sha256','tasks_sha256','tool_source_manifest_sha256','CPU_image_evidence_sha256',
                 'CPU_protocol_rollup_sha256','quality_scores_sha256')}})
        return approval,request

    def test_typed_real_tool_workload_keeps_production_cache_profile_and_native_parser(self):
        approval,request=self.fixture();validate_request(request,approval)
        argv=native_server_command(request,'nvfp4')
        self.assertIn('--tool_call_parser=qwen3_coder',argv)
        self.assertIn('--kv_cache_dtype=nvfp4',argv)
        self.assertEqual(sum(value.startswith('--tool_call_parser=') for value in argv),1)

    def test_unbounded_calls_wrong_parser_or_unqualified_image_cannot_start_tools(self):
        approval,request=self.fixture()
        for changes in ({'client_concurrency':128},{'max_steps_per_turn':10000},{'tool_parser':'qwen25'},
                        {'tasks':8},{'CPU_image_evidence_sha256':''},{'quality_scores_sha256':''},
                        {'arbitrary_execution_backend':'eval'}):
            broken=copy.deepcopy(request);broken['tool_evaluation'].update(changes)
            with self.subTest(changes=changes),self.assertRaises(ValueError):validate_request(broken,approval)

    def test_tool_generation_cannot_be_hidden_under_plain_service_or_quality(self):
        approval,tool=self.fixture();_,plain=fixtures._service('nvfp4');plain['tool_evaluation']=tool['tool_evaluation']
        with self.assertRaises(ValueError):validate_request(plain,approval)
        tool['quality_evaluation']={'requests':1720}
        with self.assertRaises(ValueError):validate_request(tool,approval)

    def test_tool_parser_does_not_silently_enable_for_ordinary_quality_service(self):
        _,request=fixtures._service('nvfp4')
        self.assertFalse(any(value.startswith('--tool_call_parser=') for value in native_server_command(request,'nvfp4')))


if __name__=='__main__':unittest.main()
