"""Verify the actual raw model, cross-request prefix and serving Graph evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.audit_upstream_run import audit as audit_lifecycle
from scripts.controller.native_tools import parse_sse_packets
from scripts.controller.nextgen_multiturn import _packets
from scripts.controller.upstream_qualification import server_command


def native_runtime(log: str, native: dict | None) -> list[dict]:
    """Check the resident cache actually bound by the recorded serving process."""
    values = [json.loads(line.split('Native KV backend bound: ', 1)[1])
              for line in log.splitlines() if 'Native KV backend bound: ' in line]
    if native is None:
        if values:
            raise ValueError('BF16 unexpectedly selected the native quantized backend')
        return []
    if not values:
        raise ValueError('No actual native resident backend binding')
    fp8 = native['dtype'] == 'fp8_e4m3'
    expected = {'kv_cache_dtype': native['dtype'], 'prefill_backend': native['prefill'],
                'decode_backend': native['decode'], 'k_scale': native['key_scale'],
                'v_scale': native['value_scale'], 'full_BF16_KV_shadow': False,
                'VMM_virtual_address_capacity_not_residency': False,
                'decode_query_dtype': 'torch.bfloat16',
                'prefill_query_dtype': 'torch.float8_e4m3fn' if fp8 and native['prefill'] == 'fa3' else 'torch.bfloat16',
                'resident_global_scale_bytes': 0 if fp8 else 8,
                'resident_layout': 'NHD' if fp8 else 'NHD_page_packed_E2M1_then_E4M3_block16',
                'SM90_FP4_tensor_core_compute': False,
                'writer': 'fused_triton_quantize_scatter' if fp8 else 'flashinfer_native_nvfp4_slot_writer'}
    for info in values:
        if (any(info.get(key) != value for key, value in expected.items())
                or not isinstance(info.get('resident_KV_bytes'), int) or info['resident_KV_bytes'] <= 0
                or info.get('KV_tensor_address_bytes') != info['resident_KV_bytes']):
            raise ValueError('Actual resident native format, scales, query types or backend differs')
    return values


def audit(manifest_path: Path) -> dict:
    lifecycle = audit_lifecycle(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    root = Path(manifest['mirror_directory'])/'files'
    ticket = json.loads((root/'ticket.json').read_bytes())
    request = ticket['request']
    arm = request['arms'][0]
    output = root/'output'/arm
    inputs = json.loads((root/'input/validation.json').read_bytes())
    if request.get('model_proof_reuse') is not None:
        raise ValueError('Composite A proof uses its separate preserved partial-component audit')
    command = json.loads((output/'server_command.json').read_bytes())
    if command != server_command(request, output):
        raise ValueError('Actual model command differs from the pinned normal execution path')
    environment = json.loads((output/'official_environment.json').read_bytes())
    if any(k.startswith('XLLM_') for k in environment):
        raise ValueError('Legacy experimental environment reached the upstream model')

    def check_case(path, case):
        row = json.loads(path.read_bytes())
        parsed = parse_sse_packets(_packets(row['raw_SSE']))
        if (not parsed['done'] or parsed['finish_reason'] != 'stop'
                or row['request']['messages'] != case['messages']
                or row['request']['model'] != request['model_id']
                or row['request']['temperature'] != 0 or row['request'].get('ignore_eos', False)
                or parsed['content'] != row['answer'] or case['expected'] not in parsed['content']
                or parsed['usage'] != row['usage'] or not 0 < parsed['usage']['completion_tokens'] <= 96):
            raise ValueError('Raw known-answer payload/response/natural-stop mismatch: '+str(path))
        return parsed

    if len(inputs['cases']) != 16:
        raise ValueError('Known-answer inventory changed')
    for case in inputs['cases']:
        check_case(output/('case-'+case['id']+'.json'), case)
    intervening = check_case(output/'prefix-intervening-independent-request.json', inputs['cases'][0])
    repeated = []
    for number in range(2):
        raw = json.loads((output/('prefix-raw-'+str(number)+'.json')).read_bytes())
        parsed = parse_sse_packets(_packets(raw['raw_SSE']))
        if (raw['status'] != 200 or raw.get('error') is not None or not parsed['done']
                or parsed != json.loads((output/('prefix-parsed-'+str(number)+'.json')).read_bytes())
                or parsed['usage']['prompt_tokens'] != inputs['prefix_probe']['prompt_tokens_cpu']
                or not 0 < parsed['usage']['completion_tokens'] <= 512):
            raise ValueError('Actual long-prefix request/usage/completion mismatch')
        repeated.append(parsed)
    prefix = json.loads((output/'cross_request_prefix_probe.json').read_bytes())
    snapshots = prefix['snapshots']
    hits = snapshots[2]['prefix_cache_hit_tokens_total']-snapshots[1]['prefix_cache_hit_tokens_total']
    offered = inputs['prefix_probe']['prompt_tokens_cpu']
    if (repeated[0]['content'] != repeated[1]['content'] or not repeated[0]['content']
            or prefix['API_cached_tokens_used_as_hit_evidence'] is not False
            or hits != prefix['repeat_hit_tokens'] or hits <= 0
            or snapshots[1]['prefix_cache_prompt_tokens_total']-snapshots[0]['prefix_cache_prompt_tokens_total'] != offered
            or snapshots[2]['prefix_cache_prompt_tokens_total']-snapshots[1]['prefix_cache_prompt_tokens_total']
                != offered+intervening['usage']['prompt_tokens']):
        raise ValueError('Cross-request prefix/GDN restoration or admission counters differ')
    for name in ('graph-window-warmup.json', 'profile-repeat-0.json', 'profile-repeat-1.json'):
        check_case(output/name, inputs['cases'][0])
    trace = json.loads((output/'trace_summary.json').read_bytes())
    before, after = trace['before'], trace['after']
    counts = {name: after[name]-before[name] for name in
              ('attempts', 'cudaGraphLaunch_success', 'cudaGraphLaunch_ptsz_success', 'failures')}
    if (before['pid'] != after['pid'] or counts != trace['window_counts']
            or counts['failures'] != 0 or trace['actual_cuda_graph_launches'] <= 0
            or trace['actual_cuda_graph_launches'] != counts['cudaGraphLaunch_success']+counts['cudaGraphLaunch_ptsz_success']):
        raise ValueError('Actual serving Graph API counters differ')
    log = (output/'server.log').read_text()
    native_lines = native_runtime(log, request.get('native_configuration'))
    if not re.search(r'\[  PASSED  \] 3 tests\.', (output/'GPU_context_cache_tests.log').read_text()):
        raise ValueError('Deferred GPU-context cache tests did not pass')
    return dict(lifecycle, raw_known_answers_verified=16, GPU_context_cache_tests_passed=3,
                real_cross_request_prefix_hit_tokens=hits, GDN_restoration_same_answer=True,
                actual_serving_Graph_API_successes=trace['actual_cuda_graph_launches'],
                actual_serving_Graph_API_errors=0, native_runtime=native_lines,
                raw_model_prefix_Graph_verified=True,
                SHA_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                VMM_GPU_compatibility_verified=False, full_quality_or_service_claim=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.sha_manifest)
    args.out.open('x').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
