"""Reject unsupported source/storage identities at the real admission boundary."""

import copy
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.next_stage_deployment import validate_deployment, MX_CONFIGURATION, operator_acceptance_passed
from scripts.controller.native_next_stage import EXTRA_SOURCES
from scripts.controller.upstream_native import SOURCE_FILES, CONFIGURATIONS
from scripts.controller.policy import validate_request
from scripts.controller.upstream_qualification import BASELINE_COMMIT
from tests import test_upstream_qualification as qualification_fixtures


class NextStageAdmissionTest(unittest.TestCase):
    def setUp(self):
        fixture=qualification_fixtures.UpstreamQualificationTest();fixture.setUp()
        self.request=copy.deepcopy(fixture.request)
        self.approval=fixture.approval
        self.request.update(arms=['fp8_e4m3'],native_base_commit=BASELINE_COMMIT,
                            server_source_commit='7'*40,native_configuration=CONFIGURATIONS['fp8_e4m3'],
                            operators={'fp8_e4m3':{'planner':'fa3','decode':'xqa'}},
                            backend_source_sha256={name:'6'*64 for name in SOURCE_FILES},
                            next_stage_profile='fp8_optimization_20261105',
                            python_source_commit='1'*40,
                            operator_qualification_reference={'run_id':'operator-proof','result_sha256':'2'*64},
                            workload={'kind':'upstream_next_stage_model_qualification',
                                      'scope':'source_matched_native_model_prefix_Graph_qualification'})
        self.request['backend_source_sha256']['xllm/python/attention/mixed_native_kv.py']='3'*64

    def test_independent_python_commit_and_actual_binary_are_explicit(self):
        validate_request(self.request,self.approval)
        original=copy.deepcopy(self.request);original.pop('next_stage_profile')
        original['workload']={'kind':'upstream_native_C_model_qualification','scope':'native_model_prefix_graph_qualification'}
        with self.assertRaises(ApprovalError):
            validate_request(original,self.approval)

    def test_mxfp4_requires_new_binary_and_all_source_identities(self):
        request=copy.deepcopy(self.request)
        request.update(next_stage_profile='ocp_mxfp4_20261105',arms=['mxfp4'],
                       native_configuration=MX_CONFIGURATION,operators={'mxfp4':{'planner':'fa2','decode':'flashinfer'}},
                       backend_source_sha256={name:'4'*64 for name in (*SOURCE_FILES,*EXTRA_SOURCES)})
        with self.assertRaises(ApprovalError):
            validate_deployment(request,self.approval)
        request['python_source_commit']=request['server_source_commit']
        validate_deployment(request,self.approval)
        request['native_configuration']=dict(MX_CONFIGURATION,value_scale=.0625)
        with self.assertRaises(ApprovalError):
            validate_deployment(request,self.approval)

    def test_scale_change_needs_independent_calibration_and_cannot_change_resources(self):
        request=copy.deepcopy(self.request)
        request['native_configuration']=dict(request['native_configuration'],value_scale=.5)
        with self.assertRaises(ApprovalError):
            validate_deployment(request,self.approval)
        request['numeric_calibration_reference']={'run_id':'independent-calibration','result_sha256':'5'*64}
        validate_deployment(request,self.approval)
        for key,value in (('prefix_cache',False),('graph_mode','none'),('service',{'max_batch':64}),
                          ('next_stage_profile','unrecognized'),('numeric_diagnostics',True),
                          ('compiled_backend_source_sha256',{})):
            bad=copy.deepcopy(request);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ApprovalError):
                validate_deployment(bad,self.approval)

    def test_completed_operator_without_passed_gates_cannot_promote(self):
        self.assertFalse(operator_acceptance_passed({},'fp8_e4m3'))
        result={'schema':'FP8_finite_optimization_operator_v1','query_gate':True,'mixed_gate':False}
        self.assertFalse(operator_acceptance_passed(result,'fp8_e4m3'))
        result['mixed_gate']=True
        self.assertTrue(operator_acceptance_passed(result,'fp8_e4m3'))
        self.assertFalse(operator_acceptance_passed(result,'mxfp4'))


if __name__=='__main__':
    unittest.main()
