"""Reject silent card changes while permitting the explicit finite quality relocation."""
import unittest

from scripts.controller.upstream_tools import (
    TOOL_GPU, QUALITY_GPU, QUALITY_GPU_AFFINITY, validate_tool_gpu)
from scripts.controller.next_stage_deployment import P1_EXTENSION, P0_REPAIR_SOURCE, P0_REPAIR_BACKEND
from scripts.compare_nextgen_tools import validate_gpu_pair


class ToolGPUContractTests(unittest.TestCase):
    def relocated(self):
        return {'gpu_uuids':[QUALITY_GPU], 'tool_gpu_affinity':QUALITY_GPU_AFFINITY,
                'next_stage_profile':'ocp_mxfp4_20261105', 'p0_repair_only':True,
                'p1_extension':P1_EXTENSION, 'arms':['mxfp4'], 'python_source_commit':P0_REPAIR_SOURCE,
                'backend_source_sha256':{'xllm/python/attention/flashinfer_native_kv.py':P0_REPAIR_BACKEND}}

    def test_original_default_and_explicit_relocation(self):
        original={'gpu_uuids':[TOOL_GPU]}
        validate_tool_gpu(original)
        validate_tool_gpu(self.relocated())
        self.assertTrue(validate_gpu_pair(original, original, upstream=True))
        self.assertFalse(validate_gpu_pair(original, self.relocated(), upstream=True))

    def test_silent_or_unqualified_gpu_changes_are_rejected(self):
        for change in ({'tool_gpu_affinity':None}, {'tool_gpu_affinity':'unknown'},
                       {'gpu_uuids':['other']}, {'gpu_uuids':[TOOL_GPU]},
                       {'python_source_commit':'other'}, {'p0_repair_only':False},
                       {'arms':['fp8_e4m3']}, {'next_stage_profile':'other'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_tool_gpu(self.relocated() | change)
        with self.assertRaises(ValueError):
            validate_tool_gpu({'gpu_uuids':[QUALITY_GPU]})

    def test_other_hardware_pairs_and_nonupstream_are_rejected(self):
        original={'gpu_uuids':[TOOL_GPU]}
        with self.assertRaises(ValueError):
            validate_gpu_pair(original, self.relocated(), upstream=False)
        with self.assertRaises(ValueError):
            validate_gpu_pair({'gpu_uuids':['other']}, self.relocated(), upstream=True)


if __name__=='__main__':unittest.main()
