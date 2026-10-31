"""New-binary hybrid KV service gates and held-out BF16 range calibration.

All GPU ownership, deadlines, isolated containers and resource accounting stay
with the existing supervisor. This is not an AIPerf throughput experiment.
"""

from __future__ import annotations

import concurrent.futures
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import datetime

from .durable import digest, write_new_json
from .nextgen import GDN_OVERLAY, MODEL, QUALITY_KIND, SCHEMA, SERVICE_KIND, SOURCE_FILES, _require


SERVICE_SOURCE_FILES = tuple(sorted(set(SOURCE_FILES) | {
    'xllm/python/attention/flashinfer_nvfp4.py',
    'xllm/python/attention/kv_observation.py',
    'xllm/python/attention/quantized.py',
    'xllm/python/attention/quantized_triton.py',
    'xllm/python/attention/xqa_decode.py',
    'xllm/core/framework/config/kv_cache_config.cpp',
    'xllm/core/runtime/py_executor_impl.cpp',
    'xllm/core/distributed_runtime/master.cpp',
    'xllm/xllm.cpp',
    'tools/benchmark_kv_cache.py',
    'tools/validate_native_kv_pages.py',
    'tools/validate_kv_observation_graph.py',
}))
OBSERVATION_METHOD = 'native_BF16_valid_KV_write_ranges_v1'


def validate_service_request(request: dict, approval: dict) -> None:
    from .policy import _path

    _require(request.get('schema') == SCHEMA and request.get('model_id') == MODEL,
             'Unknown hybrid service contract/model')
    quality = request.get('workload', {}).get('kind') == QUALITY_KIND
    _require(request.get('model_path') == approval['read_only_model_path']
             and Path(request['model_path']).name == MODEL, 'Model escaped approved read-only scope')
    _require(request.get('arms') in (['bf16'], ['fp8_e4m3'], ['nvfp4']),
             'One explicit native format per new-binary validation')
    arm = request['arms'][0]
    _require(len(request.get('gpu_uuids', [])) == 1 and request.get('replicate') == 0,
             'Initial format service gate is one admitted GPU, no hidden retry')
    _require(type(request.get('max_wall_seconds')) is int and 1800 <= request['max_wall_seconds'] <= (7200 if quality else 3600)
             and request.get('startup_seconds') == 900, 'Unexpected service validation window')
    _require(request.get('image_id') == approval['approved_image_id'], 'Unpinned service image')
    for name in ('source_repo', 'python_source_repo', 'server_binary', 'dataset_path'):
        _require(_path(request.get(name)).startswith(approval['approved_workspace'] + '/'),
                 'Native service asset escaped project BASE')
    _require(request['python_source_repo'] == request['remote_output'] + '/source_repo'
             and request['dataset_path'] == request['remote_output'] + '/input/validation.json',
             'Native service source/input mapping differs')
    for name, length in [('server_source_commit', 40), ('python_source_commit', 40), ('binary_sha256', 64),
                         ('dataset_sha256', 64), ('model_config_sha256', 64)]:
        _require(re.fullmatch('[0-9a-f]{' + str(length) + '}', request.get(name, '')) is not None,
                 'Missing service provenance: ' + name)
    hashes = request.get('backend_source_sha256', {})
    _require(set(hashes) == set(SERVICE_SOURCE_FILES)
             and all(re.fullmatch('[0-9a-f]{64}', value) is not None for value in hashes.values()),
             'Incomplete new-binary hybrid/backend/observer source identities')
    _require(request.get('server_binary_execution') == 'required'
             and not request.get('startup_recovery') and not request.get('environment_recovery'),
             'New binary cannot reuse the old startup failure exception')
    dependency = request.get('runtime_dependency_overlay', {})
    _require(isinstance(dependency, dict) and dependency.get('overlay_name') == GDN_OVERLAY
             and set(dependency) == {'overlay_name', 'overlay_manifest_sha256', 'CPU_import_evidence_sha256'}
             and all(re.fullmatch('[0-9a-f]{64}', dependency.get(name, '')) is not None
                     for name in ('overlay_manifest_sha256', 'CPU_import_evidence_sha256')),
             'Private GDN runtime dependency must retain its CPU and file integrity proof')
    _require(request.get('graph_mode') == 'cudagraphs' and request.get('prefix_cache') is True,
             'Actual production Graph/prefix lifecycle required')
    _require(request.get('workload') == ({'kind':QUALITY_KIND,
                                        'scope':'natural_stop_core_quality_generations_not_service_performance'}
                                       if quality else {'kind': SERVICE_KIND,
                                        'scope': 'format_correctness_lifecycle_and_optional_untimed_calibration'}),
             'Unknown service validation workload')
    expected_service = {'max_batch':16 if quality else 8,'max_tokens_per_batch':2048,
                        'linear_state_slots':32 if quality else 16,'max_context':131072,
                        'max_memory_utilization':.90 if quality else .85,'page_size':64}
    _require(request.get('service') == expected_service,
             'Unexpected service validation configuration')
    _require(request.get('max_context_scope') == 'validation_input_budget_not_server_context_cap',
             'Do not confuse input budget with actual model context capacity')
    operator = request.get('operators', {}).get(arm, {})
    _require(set(request.get('operators', {})) == {arm}
             and operator.get('planner') == ('fa3' if arm == 'bf16' else 'fa2')
             and operator.get('decode') in (('flashinfer',) if arm == 'nvfp4' else ('flashinfer', 'xqa'))
             and set(operator) == {'planner', 'decode'}, 'Unsupported explicit native kernel combination')
    observation = request.get('range_observation')
    _require(type(observation) is bool and (not observation or arm == 'bf16'),
             'Untimed range calibration must observe unquantized BF16 only')
    _require(not quality or observation is False, 'Core quality must disable the diagnostic observer')
    if observation:
        _require(re.fullmatch('[0-9a-f]{64}', request.get('calibration_sha256', '')) is not None
                 and request.get('calibration_count') == 136 and request.get('calibration_concurrency') == 4,
                 'Missing frozen held-out calibration inputs or bounded independent long-context concurrency')
    if arm != 'bf16':
        scales = request.get('kv_scales', {})
        _require(set(scales) == {'key', 'value'} and all(
            type(value) in (int, float) and math.isfinite(value) and 2**-126 <= value <= (2-2**-23)*2**127
            for value in scales.values()), 'Low-precision KV needs positive normal FP32 calibrated scales')
        _require(re.fullmatch('[0-9a-f]{64}', request.get('scale_calibration_audit_sha256', '')) is not None,
                 'New-model scales require a traceable independent calibration audit')
    if quality:
        expected = {'requests':1720,'client_concurrency':16,'client_token_budget':160000,
                    'request_timeout_seconds':600,'max_seconds':4800}
        evaluation = request.get('quality_evaluation',{})
        _require(set(evaluation) == set(expected) | {'requests_sha256','source_manifest_sha256'}
                 and all(evaluation.get(key)==value for key,value in expected.items())
                 and all(re.fullmatch('[0-9a-f]{64}',evaluation.get(key,'')) is not None
                         for key in ('requests_sha256','source_manifest_sha256')),
                 'Core quality must use the frozen real cohort and bounded common client demand')
    else:
        _require(request.get('quality_evaluation') is None, 'Do not hide quality generation inside a service gate')


def native_server_command(request: dict, arm: str) -> list[str]:
    from .nextgen import server_command

    command = [argument for argument in server_command(request) if not argument.startswith('--kv_cache_dtype=')]
    operator = request['operators'][arm]
    command += ['--block_size=64', '--kv_cache_backend=flashinfer',
                '--kv_cache_dtype=' + ('auto' if arm == 'bf16' else arm),
                '--kv_cache_prefill_backend=' + operator['planner'],
                '--kv_cache_decode_backend=' + operator['decode']]
    if arm != 'bf16':
        command += ['--kv_cache_k_scale=' + str(request['kv_scales']['key']),
                    '--kv_cache_v_scale=' + str(request['kv_scales']['value'])]
    return command


def verify_scale_prerequisites(request: dict, base: Path, uid: int) -> None:
    """Bind quantized model scales to the actual released new-model BF16 run."""
    from .nextgen import _file_sha

    arm = request['arms'][0]
    if arm == 'bf16':
        return
    path = Path(request['remote_output'])/'input/scale_calibration_audit.json'
    _require(path.resolve(strict=True) == path and path.stat().st_uid == uid and path.stat().st_size <= 2**20,
             'Unowned/oversized scale audit')
    _require(_file_sha(path) == request['scale_calibration_audit_sha256'], 'Actual scale audit SHA differs')
    audit = json.loads(path.read_text())
    _require(audit.get('state') == 'BF16_CALIBRATION_INDEPENDENTLY_VERIFIED'
             and audit.get('model_id') == MODEL and audit.get('actual_observer_GPU_Graph_passed') is True
             and audit.get('candidate_scales', {}).get(arm) == request['kv_scales'],
             'Requested native scales differ from the independent BF16 candidates')
    _require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', audit.get('run_id', '')) is not None
             and audit['run_id'] != request['run_id'], 'Invalid calibration run identity')
    prior = base/'agent_work/runs'/audit['run_id']
    for name in ('result.json', 'ticket.json', 'output/bf16/kv_observer_snapshot.json',
                 'output/bf16/observer_Graph.json'):
        evidence = prior/name
        _require(evidence.resolve(strict=True) == evidence and evidence.stat().st_uid == uid
                 and evidence.stat().st_size <= 2**20, 'Unowned/oversized actual BF16 calibration evidence')
    _require(_file_sha(prior/'result.json') == audit['terminal_result_sha256']
             and _file_sha(prior/'output/bf16/kv_observer_snapshot.json') == audit['snapshot_sha256'],
             'Actual released calibration result/snapshot differs from audit')
    result = json.loads((prior/'result.json').read_text())
    old_ticket = json.loads((prior/'ticket.json').read_text())
    old = old_ticket['request']
    _require(digest(old) == old_ticket['request_sha256'], 'Prior calibration ticket content changed')
    _require(result['status'] == 'COMPLETE' and result['gpu_release_verified'] is True
             and not result['live_owned_containers'] and old['arms'] == ['bf16']
             and old.get('range_observation') is True
             and result['request_sha256'] == old_ticket['request_sha256'] == audit['calibration_request_sha256'],
             'Calibration is not the exact released BF16 observation run')
    for name in ('model_id', 'model_path', 'model_config_sha256', 'server_source_commit', 'binary_sha256'):
        _require(old[name] == request[name], 'Native scales refer to a different model/binary: ' + name)
    for name in ('xllm/python/models/qwen3_5.py', 'xllm/python/layers/cuda/qwen3_5/gated_delta_net.py'):
        _require(old['backend_source_sha256'][name] == request['backend_source_sha256'][name],
                 'New model K/V producer changed after BF16 calibration')
    snapshot = json.loads((prior/'output/bf16/kv_observer_snapshot.json').read_text())
    key = 'FP8_E4M3_candidate_scales' if arm == 'fp8_e4m3' else 'NVFP4_global_candidate_scales'
    _require(snapshot.get(key) == request['kv_scales'] and snapshot.get('request_sha256') == audit['calibration_request_sha256'],
             'Actual BF16 observation does not contain these format scales')


def _operator_environment(environment: dict) -> dict:
    """The tiny gate has a different layer inventory from the observed model."""
    return {name: value for name, value in environment.items() if name != 'XLLM_KV_OBSERVATION_CONFIG'}


def _verify_model_runtime(request: dict, output: Path, arm: str) -> tuple[dict, dict]:
    text = (output/'server.log').read_text()
    allocations = [json.loads(line.split('KV_ALLOCATION ',1)[1]) for line in text.splitlines()
                   if 'KV_ALLOCATION ' in line]
    backends = [json.loads(line.split('NATIVE_KV_BACKEND ',1)[1]) for line in text.splitlines()
                if 'NATIVE_KV_BACKEND ' in line]
    if not allocations or not backends:
        raise ValueError('Native model cache/backend evidence missing before evaluation')
    allocation, backend = allocations[-1], backends[-1]
    if (allocation.get('full_attention_layer_ids') != list(range(3,64,4))
            or len(allocation.get('linear_state_layers',[])) != 48
            or allocation.get('linear_state_resident_bytes',0) <= 0
            or not backend.get('graph_enabled') or 'NATIVE_KV_GRAPH_REPLAY' not in text
            or backend.get('graph_captured_batches') != [1<<index for index in range(request['service']['max_batch'].bit_length())]
            or backend.get('planner_backend') != request['operators'][arm]['planner']
            or backend.get('decode_impl') != request['operators'][arm]['decode']):
        raise ValueError('Actual hybrid allocation, requested native kernels or model Graph differs')
    if backend.get('KV_observation_enabled') is not request['range_observation']:
        raise ValueError('Actual range observer enablement differs from untimed configuration')
    if request['operators'][arm]['decode']=='xqa' and backend.get('xqa_max_context_tokens')!=262144:
        raise ValueError('Actual XQA metadata bound differs from the verified model context')
    if arm!='bf16':
        if backend.get('kv_cache_dtype')!=arm or backend.get('full_BF16_KV_shadow') is not False:
            raise ValueError('Low-precision service did not consume compressed resident pages')
        if any(backend.get(name+'_scale')!=request['kv_scales'][kind] for name,kind in [('k','key'),('v','value')]):
            raise ValueError('Actual native decode scales differ from the audited model candidates')
    if arm=='fp8_e4m3' and backend.get('fp8_views_alias_resident_pages') is not True:
        raise ValueError('FP8 attention does not alias resident compressed payloads')
    if arm=='nvfp4' and backend.get('scale_views_alias_resident_pages') is not True:
        raise ValueError('NVFP4 attention does not alias resident block scales')
    return allocation,backend


def _quiescent_command(output: Path, action: str, request_sha: str, server: subprocess.Popen) -> dict:
    write_new_json(output / ('kv_observer_' + action + '.json'), {'action': action, 'request_sha256': request_sha})
    deadline = time.monotonic() + 60
    path = output / ('kv_observer_' + action + '_ack.json')
    while not path.exists():
        if (output / 'kv_observer_error.json').exists() or server.poll() is not None:
            raise ValueError('Range observation or model process failed')
        if time.monotonic() >= deadline:
            raise TimeoutError('No quiescent range observation acknowledgement')
        time.sleep(.1)
    result = json.loads(path.read_text())
    if result['action'] != action or result['request_sha256'] != request_sha:
        raise ValueError('Range observation boundary acknowledgement differs')
    return result


def _calibration_request(row: dict, model_id: str) -> dict:
    payload = dict(row['payload'], model=model_id, stream=False)
    query = urllib.request.Request('http://127.0.0.1:19001/v1/chat/completions',
                                   data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    raw, value, error = b'', {}, None
    try:
        with urllib.request.urlopen(query, timeout=300) as response:
            raw = response.read(2**20 + 1)
            if len(raw) > 2**20:
                raise ValueError('Calibration response exceeds its bounded journal')
            value = json.loads(raw)
    except (OSError, ValueError) as failure:
        error = type(failure).__name__ + ': ' + str(failure)
    return {'id': row['id'], 'request': payload, 'response': value, 'error': error,
            'raw_HTTP_body_base64': base64.b64encode(raw).decode(),
            'contract_valid': error is None and 'error' not in value and bool(value.get('usage'))
            and value.get('usage', {}).get('prompt_tokens') == row['prompt_tokens_cpu']
            and len(value.get('choices', [])) == 1
            and value['choices'][0].get('finish_reason') in ('stop', 'length')
            and bool(value['choices'][0].get('message', {}).get('content'))}


def run_native_service(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from .nextgen import _stream_case, native_python_environment
    from .worker import stop_child

    request = ticket['request']
    env = {name: value for name, value in native_python_environment(request, env).items()
           if not name.startswith('XLLM_')}
    # The production backend/scales/kernels use typed service flags. The optional
    # diagnostic observer has one separately recorded, untimed configuration.
    if request['range_observation']:
        observer = output / 'kv_observer_config.json'
        write_new_json(observer, {'method': OBSERVATION_METHOD, 'layer_ids': list(range(3, 64, 4)),
                                 'heads': 4, 'dimension': 256, 'run_id': request['run_id'],
                                 'request_sha256': ticket['request_sha256'], 'output_dir': str(output)})
        env['XLLM_KV_OBSERVATION_CONFIG'] = str(observer)
    write_new_json(output / 'nextgen_environment.json', {name: value for name, value in env.items()
                   if name.startswith(('XLLM_', 'FLASHINFER_')) or name in ('PYTHONPATH', 'LD_LIBRARY_PATH')})
    if request['range_observation']:
        with (output / 'observer_Graph.log').open('x') as log:
            subprocess.run([sys.executable, '-B', '-m', 'tools.validate_kv_observation_graph',
                            '--out', str(output / 'observer_Graph.json')], cwd=request['python_source_repo'],
                           env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
    with (output / 'native_operator.log').open('x') as log:
        operator = request['operators'][arm]
        if arm == 'bf16':
            operator_command = [sys.executable, '-B', '-m', __name__, '--operator-output', str(output),
                                '--planner', operator['planner'], '--decode', operator['decode']]
        else:
            operator_command = [sys.executable, '-B', '-m', 'tools.validate_native_kv_pages',
                                '--out', str(output / 'native_pages'), '--formats', arm,
                                '--head-dim', '256', '--decode-impl', operator['decode'],
                                '--max-context-tokens', '262144']
        write_new_json(output / 'native_operator_command.json', operator_command)
        operator_env = _operator_environment(env)
        write_new_json(output / 'native_operator_environment.json', {name: value for name, value in operator_env.items()
                       if name.startswith(('XLLM_', 'FLASHINFER_')) or name in ('PYTHONPATH', 'LD_LIBRARY_PATH')})
        subprocess.run(operator_command, cwd=request['python_source_repo'], env=operator_env, stdout=log,
                       stderr=subprocess.STDOUT, timeout=900, check=True)
    command = native_server_command(request, arm)
    write_new_json(output / 'server_command.json', command)
    dataset = json.loads(Path(request['dataset_path']).read_text())
    if dataset.get('schema') != 'native-hybrid-validation-input-v1' or len(dataset.get('cases', [])) != 16:
        raise ValueError('Frozen known-answer/lifecycle input differs')
    server = None
    rows = []
    with (output / 'server.log').open('x') as log:
        try:
            server = subprocess.Popen(command, cwd=request['python_source_repo'], env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + request['startup_seconds']
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise ValueError('Native format service exited during startup')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:19001/v1/models', timeout=2) as response:
                        if any(row.get('id') == request['model_id'] for row in json.load(response).get('data', [])):
                            break
                except (OSError, ValueError):
                    time.sleep(1)
            else:
                raise TimeoutError('Native format service startup deadline exceeded')
            with (output / 'validation_raw.jsonl').open('x') as journal:
                for cases, parallel in ((dataset['cases'][:4], False), (dataset['cases'][4:], True),
                                        (dataset['cases'][:4], False)):
                    if parallel:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                            batch = list(pool.map(lambda case: _stream_case(case, request['model_id']), cases))
                    else:
                        batch = [_stream_case(case, request['model_id']) for case in cases]
                    for row in batch:
                        journal.write(json.dumps(row, ensure_ascii=False) + '\n')
                        journal.flush(); os.fsync(journal.fileno())
                        rows.append(row)
            if not all(row['contract_valid'] and row['expected_substring_present'] for row in rows):
                raise ValueError('Native format SSE/known answers failed; raw bytes preserved')
            if any(a['answer'] != b['answer'] for a, b in zip(rows[:4], rows[-4:])):
                raise ValueError('A/B/A recurrent/cache lifecycle changed repeated answers')
            allocation,backend = _verify_model_runtime(request,output,arm)
            write_new_json(output/'lifecycle_gate.json',{'known_answer_checks':len(rows),
                           'actual_model_Graph_verified':True,'native_backend':backend,
                           'model_accuracy_or_service_performance_accepted':False})
            calibration = []
            if request['range_observation']:
                path = root / 'input/calibration.json'
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != request['calibration_sha256']:
                    raise ValueError('Frozen calibration data changed')
                inputs = json.loads(raw)
                if len(inputs.get('requests', [])) != request['calibration_count']:
                    raise ValueError('Calibration request inventory differs')
                _quiescent_command(output, 'reset', ticket['request_sha256'], server)
                with concurrent.futures.ThreadPoolExecutor(max_workers=request['calibration_concurrency']) as pool:
                    with (output / 'calibration_raw.jsonl').open('x') as journal:
                        for row in pool.map(lambda item: _calibration_request(item, request['model_id']), inputs['requests']):
                            journal.write(json.dumps(row, ensure_ascii=False) + '\n')
                            journal.flush(); os.fsync(journal.fileno())
                            calibration.append(row)
                acknowledgement = _quiescent_command(output, 'export', ticket['request_sha256'], server)
                path = output / 'kv_observer_snapshot.json'
                if hashlib.sha256(path.read_bytes()).hexdigest() != acknowledgement['snapshot_sha256']:
                    raise ValueError('Exported calibration snapshot differs from acknowledgement')
                snapshot = json.loads(path.read_text())
                if (snapshot['layer_ids'] != list(range(3, 64, 4)) or snapshot['heads'] != 4
                        or snapshot['dimension'] != 256 or len(snapshot['rows']) != 256
                        or any(row['finite_values'] <= 0 or row['nonfinite_values'] for row in snapshot['rows'])
                        or any(min(calls) <= 0 for calls in snapshot['phase_calls_by_layer'])):
                    raise ValueError('Not every actual full-attention layer/head/prefill/decode was observed')
                if not all(row['contract_valid'] for row in calibration):
                    raise ValueError('Calibration service requests incomplete; do not derive accepted scales')
            allocation,backend = _verify_model_runtime(request,output,arm)
            result = {'classification': 'native_hybrid_' + arm + '_service_correctness_lifecycle',
                      'model_loaded': True, 'requests': len(rows), 'known_answers_checked': len(rows),
                      'A_B_A_repeated_answers_match': True, 'native_allocation': allocation,
                      'native_backend': backend, 'actual_model_Graph_replay_logged': True,
                      'calibration_requests': len(calibration), 'model_quality_validated': False,
                      'service_performance_measured': False, 'official_aiperf_complete': False}
            write_new_json(output / 'validation_result.json', result)
            if request['workload']['kind']==QUALITY_KIND:
                from .nextgen_quality import run_core_generations
                deadline = datetime.datetime.fromisoformat(ticket['deadline_utc'].replace('Z','+00:00'))
                available = (deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()-180
                quality = run_core_generations(request,root,output,available_seconds=available)
                result.update(classification='native_hybrid_'+arm+'_core_quality_generation_complete_scores_pending',
                              quality_generations=quality,model_quality_validated=False)
            return result
        finally:
            if server is not None:
                stop_child(server)


if __name__ == '__main__':
    import argparse
    from .nextgen import _operator_gate
    from xllm.python.attention.native_profile import NativeFlashInferConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-output', type=Path, required=True)
    parser.add_argument('--planner', choices=('fa2', 'fa3'), required=True)
    parser.add_argument('--decode', choices=('flashinfer', 'xqa'), required=True)
    args = parser.parse_args()
    _operator_gate(args.operator_output,
                   NativeFlashInferConfig(args.planner, args.decode, max_context_tokens=262144))
