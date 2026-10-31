# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Bounded resident-FP8/BF16-Q numerical and production Graph admission."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from .durable import write_new_json


def run_operator(ticket: dict, output: Path, arm: str, env: dict) -> dict:
    request=ticket['request'];workload=request['workload']
    if arm!='fp8_e4m3' or workload['kind']!='native_bf16q_decode_operator':
        raise ValueError('Explicit BF16-Q/FP8-KV operator-only scope required')
    manifest=json.loads(Path(request['dataset_path']).read_text())
    if any(manifest.get(k)!=workload[k] for k in ('method','geometry','relative_l2_limit','primary_sources_sha256')):
        raise ValueError('Frozen BF16-Q operator input differs from the declared scope')
    env.update(XLLM_FLASHINFER_NATIVE_PLANNER='fa2', XLLM_FLASHINFER_NATIVE_DECODE='flashinfer',
               XLLM_FLASHINFER_FP8_BF16Q_DECODE='1')
    write_new_json(output/'operator_environment.json',{k:v for k,v in env.items() if k.startswith('XLLM_')})
    command=[sys.executable,'-B',request['python_source_repo']+'/tools/check_fp8_bf16q_decode.py',
             '--input',request['dataset_path'],'--out',str(output/'operator_acceptance.json')]
    write_new_json(output/'operator_command.json',command)
    with (output/'operator_acceptance.log').open('x') as log:
        subprocess.run(command,env=env,cwd=request['python_source_repo'],stdout=log,
                       stderr=subprocess.STDOUT,timeout=1200,check=True)
    result=json.loads((output/'operator_acceptance.json').read_text())
    graph=result['graph128'];backend=graph['backend']
    native=result['native_operator']
    if (native.get('real_capture_and_replay') is not True or native.get('partial_page_and_replan') is not True
            or native.get('padded_graph_bucket_checked') is not True or native.get('noncontiguous_q_checked') is not True
            or not {'prefill','capture_replay','changed_context','padded_graph'}<=set(native.get('reference_comparisons',{}))
            or any(not 0<=r['relative_l2']<=.01 for r in native['reference_comparisons'].values())):
        raise ValueError('Native prefill/decode reference and real Graph evidence is incomplete')
    if (result.get('classification')!='confirmed_BF16Q_FP8KV_operator_Graph128_only'
            or result.get('model_loaded') is not False or result.get('AIPerf_ran') is not False
            or result.get('model_quality_validated') is not False
            or result.get('service_performance_measured') is not False
            or result.get('installed_sources_sha256')!=workload['primary_sources_sha256']):
        raise ValueError('Actual operator scope or installed primary sources differ')
    if any(graph.get(k) is not True for k in ('real_capture_and_replay','writer_in_same_graph',
            'noncontiguous_QKV','reserved_page0_unchanged','resident_storage_stable','rebind_preserves_live_cache')):
        raise ValueError('Actual production Graph128/stride/storage/padding lifecycle failed')
    if (graph.get('production_max_batch')!=128 or graph.get('reference_Q')!='BF16_without_FP8_QDQ'
            or graph.get('relative_l2_limit')!=.01
            or [(r['live_rows'],r['context']) for r in graph['dynamic_live_rows_and_context']]!=[(128,257),(126,130),(1,130),(127,257)]
            or any(not 0<=r['relative_l2']<=.01 for r in graph['dynamic_live_rows_and_context'])):
        raise ValueError('Production bucket/replan/reference precision is incomplete')
    required={'profile_scope':'production_bf16q_decode','planner_backend':'fa2','decode_impl':'flashinfer',
              'q_data_type':'torch.bfloat16','decode_internal_q_data_type':'torch.bfloat16',
              'kv_data_type':'torch.float8_e4m3fn','decode_tensor_cores':True,'q_scale':None,
              'fp8_views_alias_resident_pages':True,'resident_initialization_policy':'finite_full_initialization',
              'full_BF16_KV_shadow':False,'k_scale':.125,'v_scale':.0625}
    if any(backend.get(k)!=v for k,v in required.items()):
        raise ValueError('Actual resident KV / Q precision differs from the admitted candidate')
    return {'classification':result['classification'],'protocol':'production_BF16Q_FP8KV_operator_only_v1',
            'model_loaded':False,'official_aiperf_complete':False,'operator_only':True,
            'actual_operator_graph_verified':True,'quality_or_service_result':False,
            'backend':backend,'graph128':graph}
