"""Official baseline admission and real trace parsing, without GPU execution."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.policy import validate_request
from scripts.controller.upstream_qualification import (
    KIND, MODEL, OFFICIAL_COMMIT, BASELINE_COMMIT, SERVICE, SOURCE_FILES, server_command, trace_summary,
)


class UpstreamQualificationTest(unittest.TestCase):
    def setUp(self):
        base = '/owned/project'
        self.approval = {'approved_workspace': base, 'read_only_model_path': '/models/' + MODEL,
                         'approved_image_id': 'sha256:' + 'a' * 64}
        run = base + '/agent_work/runs/upstream-A-qualification'
        source = base + '/agent_work/builds/official/source_repo'
        self.request = {
            'schema': 'kv-nextgen-run-v1',
            'workload': {'kind': KIND, 'scope': 'official_model_prefix_graph_qualification'},
            'arms': ['bf16'], 'model_id': MODEL, 'model_path': '/models/' + MODEL,
            'image_id': self.approval['approved_image_id'], 'upstream_commit': OFFICIAL_COMMIT,
            'server_source_commit': BASELINE_COMMIT, 'python_source_commit': BASELINE_COMMIT,
            'gpu_uuids': ['GPU-one'], 'max_wall_seconds': 2100, 'startup_seconds': 1500,
            'source_repo': source, 'python_source_repo': run + '/source_repo',
            'server_binary': base + '/agent_work/builds/official/build/xllm/xllm',
            'dataset_path': run + '/input/validation.json', 'remote_output': run,
            'runtime_library_directory': base + '/agent_work/builds/upstream-ac-261274e5-20261102b/zeromq-prefix/lib',
            'runtime_dependency_build_id': 'upstream-ac-261274e5-20261102b', 'zeromq_library_sha256': 'f' * 64,
            'binary_sha256': 'b' * 64, 'dataset_sha256': 'c' * 64, 'model_config_sha256': 'd' * 64,
            'backend_source_sha256': {name: 'e' * 64 for name in SOURCE_FILES},
            'service': dict(SERVICE), 'graph_mode': 'cudagraphs', 'prefix_cache': True,
            'operators': {'bf16': {'planner': 'official_auto', 'decode': 'official_auto'}},
            'server_binary_execution': 'required',
        }

    def test_real_policy_routes_official_A_and_rejects_experimental_source(self):
        validate_request(self.request, self.approval)
        for field in ('server_source_commit', 'python_source_commit', 'upstream_commit'):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.request)
                bad[field] = '0' * 40
                with self.assertRaises(ApprovalError):
                    validate_request(bad, self.approval)

    def test_rejects_quantization_operator_and_disabled_production_mechanisms(self):
        for field, value in [('operators', {'bf16': {'planner': 'fa3', 'decode': 'xqa'}}),
                             ('prefix_cache', False), ('arms', ['fp8_e4m3']),
                             ('server_binary_execution', 'none')]:
            with self.subTest(field=field):
                bad = copy.deepcopy(self.request)
                bad[field] = value
                with self.assertRaises(ApprovalError):
                    validate_request(bad, self.approval)

    def test_command_preserves_official_flashinfer_selection(self):
        command = server_command(self.request, Path('/results'))
        self.assertIn('--enable_graph=true', command)
        self.assertIn('--enable_chunked_prefill=true', command)
        self.assertFalse(any('kv_cache_backend' in argument or 'decode_backend' in argument for argument in command))
        self.assertFalse(any('linear_state_cache_slots' in argument for argument in command))

    def test_trace_counts_real_graph_launch_events_and_does_not_claim_full_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'worker.pt.trace.json').write_text(json.dumps({'traceEvents': [
                {'name': 'cudaGraphLaunch', 'cat': 'cuda_runtime'},
                {'name': 'cudaGraphLaunch_ptsz', 'cat': 'cuda_runtime'},
                {'name': 'BatchPrefillWithPagedKVCache', 'cat': 'kernel'},
                {'name': 'graph captured successfully', 'cat': 'user_annotation'},
            ]}))
            result = trace_summary(path)
        self.assertEqual(result['actual_cuda_graph_launches'], 2)
        self.assertFalse(result['all_requests_in_graph_claim'])


if __name__ == '__main__':
    unittest.main()
