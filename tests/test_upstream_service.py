"""Exercise real admission and empty/one-token Goodput semantics on CPU."""

import copy
import datetime
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_upstream_service import goodput, gauges
from scripts.controller.policy import validate_request
from scripts.controller.upstream_service import KIND, CASES
import test_upstream_qualification as qualification_fixtures


class UpstreamServiceTest(unittest.TestCase):
    def fixture(self):
        base = qualification_fixtures.UpstreamQualificationTest()
        base.setUp()
        request = copy.deepcopy(base.request)
        request.update(max_wall_seconds=7200,
            workload={'kind': KIND, 'scope': 'fixed512_sharegpt_and64_govreport_AC_v1'},
            benchmark_cases=[c[0] for c in CASES],
            service_benchmark={'input_files_sha256': {name: 'a'*64 for name in (
                'sharegpt/manifest.json', 'sharegpt/measured512.json', 'sharegpt/warmup16.json',
                'sharegpt/retokenized_Qwen38_CPU-a1.json', 'govreport/manifest.json',
                'govreport/measured64.json', 'govreport/warmup4.json')},
                'qualification_run_id': 'original-qualification', 'qualification_result_sha256': 'b'*64,
                'aiperf_cli_config_sha256': 'c'*64, 'full_CLI_protocol_CPU_audit_sha256': 'd'*64})
        return base.approval, request

    def test_real_nested_input_hash_contract_and_no_formal_observer(self):
        approval, request = self.fixture()
        validate_request(request, approval)
        for change in ({'graph_observer': {'kind': 'counter'}}, {'benchmark_cases': ['sharegpt-c128']}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request(request | change, approval)

    def test_empty_is_zero_goodput_and_one_token_needs_no_ITL(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifacts = directory/'artifacts'
            artifacts.mkdir()
            metrics, raw = [], []
            for identity, answer, tokens, itl in [('fast', 'ok', 1, None), ('empty', '', 1, None),
                                                  ('long', 'real answer', 3, 80)]:
                values = {'output_token_count': {'value': tokens}, 'time_to_first_token': {'value': 100}}
                if itl is not None:
                    values['inter_token_latency'] = {'value': itl}
                metrics.append({'metadata': {'x_request_id': identity}, 'metrics': values})
                raw.append({'metadata': {'x_request_id': identity}, 'responses': [{'packets': [
                    {'name': 'data', 'value': json.dumps({'choices': [{'delta': {'content': answer}}]})},
                    {'name': 'data', 'value': '[DONE]'}]}]})
            for name, rows in [('profile_export.jsonl', metrics), ('profile_export_raw.jsonl', raw)]:
                (artifacts/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            result = goodput(directory, {'observation_seconds': 2})
        self.assertEqual(result['empty_or_whitespace_answers'], 1)
        self.assertEqual(result['all_intended_requests_remain_denominator'], 3)
        points = {(r['TTFT_limit_ms'], r['request_mean_ITL_limit_ms']): r for r in result['points']}
        self.assertEqual(points[1000, 25]['goodput_rps'], .5)
        self.assertEqual(points[1000, 100]['goodput_rps'], 1)

    def test_residency_samples_exclude_client_preparation_and_shutdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'artifacts').mkdir()
            header = {'run_id': 'case', 'physical_pages': 10,
                      'accounting_source': 'upstream261_effective_usage_excludes_padding',
                      'units': {'num_running_requests': 'request_gauge'}}
            rows = [header]
            for index, (offset, running) in enumerate([(-1, 0), (.5, 7), (1.5, 9), (3, 100)]):
                rows.append({'utc': datetime.datetime.fromtimestamp(2000000000+offset,
                    datetime.timezone.utc).isoformat(), 'monotonic_ns': index+1,
                    'values': {'num_running_requests': running}, 'errors': {}})
            (directory/'application_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (directory/'artifacts/profile_export.jsonl').write_text(json.dumps({'metadata': {
                'request_start_ns': 2000000000*10**9, 'request_end_ns': 2000000002*10**9}})+'\n')
            (directory/'prefix_counter_window.json').write_text(json.dumps({
                'before': {'prefix_cache_hit_tokens_total': 0, 'prefix_cache_prompt_tokens_total': 0},
                'after': {'prefix_cache_hit_tokens_total': 0, 'prefix_cache_prompt_tokens_total': 128},
                'hit_tokens_delta': 0, 'prompt_tokens_delta': 128}))
            result = gauges(directory, {'run_id': 'case', 'service': {'block_size': 128}}, 10, {
                'worker_observation_started_monotonic_ns': 0, 'worker_observation_ended_monotonic_ns': 10,
                'isl_mean': 128, 'profiling_records': 1})
        self.assertEqual(result['samples'], 4)
        self.assertEqual(result['samples_during_client_window'], 2)
        self.assertEqual(result['ranges']['num_running_requests'], {'min': 7, 'max': 9, 'mean': 8})


if __name__ == '__main__':
    unittest.main()
