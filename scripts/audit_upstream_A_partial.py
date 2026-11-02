"""Recover only completed model/prefix components from the failed profiler job."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.controller.native_tools import parse_sse_packets
from scripts.controller.nextgen_multiturn import _packets


def audit(mirror: Path, ticket_path: Path, terminal_path: Path):
    manifest = json.loads((mirror/'NETWORK_SHA_MANIFEST.json').read_bytes())
    files = mirror/'files'
    identities = {}
    for item in manifest['files']:
        path = files/item['relative']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item['sha256']
        identities[item['relative']] = digest
    ticket = json.loads(ticket_path.read_bytes())
    request = ticket['request']
    terminal = json.loads(terminal_path.read_bytes())['files']['result.json']
    result = terminal['value']
    assert manifest['run_id'] == request['run_id'] == result['run_id']
    assert result['status'] == 'FAILED' and result['gpu_release_verified'] and not result['live_owned_containers']
    assert result['request_sha256'] == ticket['request_sha256']
    inputs = json.loads((files/'input/validation.json').read_bytes())
    assert identities['input/validation.json'] == request['dataset_sha256']
    output = files/'output/bf16'
    assert 'Unsupported CUDA profiling backend: torch' in (output/'server.log').read_text()
    assert '/start_profile' in (output/'container.log').read_text()
    environment = json.loads((output/'official_environment.json').read_bytes())
    assert not any(name.startswith('XLLM_') for name in environment)
    command = json.loads((output/'server_command.json').read_bytes())
    assert command[0] == request['server_binary'] and '--model_impl=python' in command
    assert '--python_model_path='+request['python_source_repo'] in command
    assert all(flag in command for flag in ('--enable_graph=true', '--enable_prefix_cache=true', '--enable_chunked_prefill=true'))
    assert not any(flag.startswith('--kv_cache_') for flag in command)

    def check_case(row, case):
        parsed = parse_sse_packets(_packets(row['raw_SSE']))
        assert parsed['done'] and parsed['finish_reason'] == 'stop' and parsed['content']
        assert row['request']['model'] == request['model_id'] and row['request']['messages'] == case['messages']
        assert row['request']['temperature'] == 0 and not row['request'].get('ignore_eos', False)
        assert parsed['content'] == row['answer'] and case['expected'] in parsed['content']
        assert parsed['usage'] == row['usage'] and 0 < parsed['usage']['completion_tokens'] <= 96
        return parsed

    assert len(inputs['cases']) == 16
    for case in inputs['cases']:
        check_case(json.loads((output/('case-'+case['id']+'.json')).read_bytes()), case)
    independent = check_case(json.loads((output/'prefix-intervening-independent-request.json').read_bytes()), inputs['cases'][0])
    assert independent['usage']['prompt_tokens'] < 128
    repeated = []
    for repeat in range(2):
        raw = json.loads((output/('prefix-raw-'+str(repeat)+'.json')).read_bytes())
        assert raw['status'] == 200 and raw['error'] is None
        parsed = parse_sse_packets(_packets(raw['raw_SSE']))
        assert parsed == json.loads((output/('prefix-parsed-'+str(repeat)+'.json')).read_bytes())
        assert parsed['done'] and parsed['finish_reason'] in ('stop', 'length')
        assert parsed['usage']['prompt_tokens'] == inputs['prefix_probe']['prompt_tokens_cpu'] == 11054
        assert 0 < parsed['usage']['completion_tokens'] <= 512
        repeated.append(parsed)
    assert repeated[0]['content'] == repeated[1]['content']
    prefix = json.loads((output/'cross_request_prefix_probe.json').read_bytes())
    snapshots = prefix['snapshots']
    assert prefix['API_cached_tokens_used_as_hit_evidence'] is False
    assert prefix['cold_hit_tokens'] == 0 and prefix['same_answer_after_independent_request']
    assert prefix['repeat_hit_tokens'] == snapshots[2]['prefix_cache_hit_tokens_total']-snapshots[1]['prefix_cache_hit_tokens_total'] == 8192
    assert snapshots[1]['prefix_cache_prompt_tokens_total']-snapshots[0]['prefix_cache_prompt_tokens_total'] == 11054
    assert snapshots[2]['prefix_cache_prompt_tokens_total']-snapshots[1]['prefix_cache_prompt_tokens_total'] == 11054+independent['usage']['prompt_tokens']
    assert re.search(r'\[  PASSED  \] 3 tests\.', (output/'GPU_context_cache_tests.log').read_text())
    return {'schema': 'upstream-A-completed-model-prefix-components-v1',
        'actual_model_and_prefix_verified': True, 'known_answers_checked': 16,
        'GPU_context_cache_tests_passed': 3, 'original_run_id': request['run_id'],
        'original_run_status': 'FAILED',
        'original_failure_scope': 'unsupported_CUDA_torch_profiler_after_model_prefix_pass',
        'graph_qualification_completed': False, 'original_failure_preserved': True,
        'prior_result_sha256': terminal['sha256'], 'request_sha256': ticket['request_sha256'],
        'source_commit': request['server_source_commit'], 'binary_sha256': request['binary_sha256'],
        'GPU_uuid': request['gpu_uuids'][0], 'cross_request_prefix_probe': prefix,
        'original_files_sha256': identities, 'performance_or_full_quality_claim': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mirror', type=Path, required=True)
    p.add_argument('--ticket', type=Path, required=True)
    p.add_argument('--terminal', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.mirror, a.ticket, a.terminal)
    a.out.open('x').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('original_files_sha256','cross_request_prefix_probe')}))
