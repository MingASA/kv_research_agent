#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze the recorded-geometry prefill repair gate without a model benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest, secure_directory, write_new_bytes, write_new_json
from scripts.controller.policy import PREFILL_WORKSPACE_OPERATORS, PREFILL_WORKSPACE_SOURCES

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-prefill-workspace-20261027')


def prepare(base: Path, run_id: str, out: Path, gpu: str, recorded: Path) -> dict:
    if out.exists():
        raise FileExistsError('New immutable operator package required')
    source_commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True):
        raise ValueError('Prefill repair source must be committed and clean')
    parent = json.loads(base.read_text())
    data = json.loads(recorded.read_text())
    windows = [window for window in data['arms']['bf16'] if window['index'] == 2]
    if len(windows) != 1:
        raise ValueError('Recorded production window 2 must be unique')
    row = windows[0]['forward_batches'][0]
    if (row['sequences'] != 1 or row['input_tokens'] != 510 or row['kv_context_tokens'] != [0,1022]
            or row['trace_annotation'] != 'xllm.forward.prefill' or row['completed'] is not True):
        raise ValueError('Recorded production geometry differs')
    secure_directory(out)
    geometry = {'method': 'recorded_production_prefill_repair_operator_gate',
        'geometry': {'sequences':1, 'query_tokens':510, 'kv_tokens':1022, 'layers':48},
        'recorded_source_sha256':hashlib.sha256(recorded.read_bytes()).hexdigest(),
        'recorded_source_path':str(recorded), 'recorded_run_id':data['run_id'],
        'recorded_arm':'bf16','recorded_window_index':2,'recorded_forward_index':0,
        'recorded_forward':row,'synthetic_tensor_values_not_real_model_activations':True,
        'samples':8,'timing_scope':'plan_plus_48_attention_calls_with_active_page_materialization_not_fullmodel',
        'seed':20261027,'model_loaded':False,'AIPerf_ran':False}
    geometry_path=out/'geometry.json';write_new_json(geometry_path,geometry)
    bundle=out/'source.bundle'
    subprocess.run(['git','-C',str(SOURCE),'bundle','create',str(bundle),'HEAD',
        '^ceff00bbd38054d2b6b4aa44513d773f2bd174b2'],check=True,capture_output=True)
    request=copy.deepcopy(parent['request']);remote=request['approved_workspace']+'/agent_work/runs/'+run_id
    request.update(run_id=run_id,remote_output=remote,replicate=0,arms=['fp8_e4m3'],concurrency=1,
        max_wall_seconds=1800,max_output_gib=3,python_source_commit=source_commit,
        python_source_repo=remote+'/source_repo',gpu_uuids=[gpu],
        dataset_path=remote+'/input/geometry.json',dataset_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),
        operators=copy.deepcopy(PREFILL_WORKSPACE_OPERATORS),
        backend_source_sha256={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in PREFILL_WORKSPACE_SOURCES},
        workload={'kind':'native_prefill_workspace_operator','ignore_eos':False,'model_loaded':False,'AIPerf_ran':False,
            'method':geometry['method'],'geometry':geometry['geometry'],
            'source_content_sha256':geometry['recorded_source_sha256']},
        research_reason='Production-directed prefill repair: resident FP8 and XQA Graph retained, one preallocated bounded BF16 workspace permits FA3 BF16-Q prefill. Gate shared-page dedup/strides/poison/multi-layer reuse and actual Graph, then time one recorded510-query/1022KV geometry including plan+48 attention calls. No14B/AIPerf/quality claim; no completed run repeated.')
    request.pop('diagnostic',None);request.pop('dataset_archive',None)
    request['experiment_key']=digest({k:request[k] for k in ('python_source_commit','binary_sha256','dataset_sha256',
        'backend_source_sha256','image_id','graph_mode','prefix_cache','fp8_scales','operators','service','workload')})
    files={'runtime/kv_controller/'+p.name:p for p in (ROOT/'scripts/controller').glob('*.py')}
    files['runtime/kv_controller/evidence.py']=ROOT/'scripts/audit_native_results.py'
    files.update({'input/geometry.json':geometry_path,'input/source.bundle':bundle})
    for name,path in list(files.items()):
        if name.startswith('runtime/'):
            frozen=out/'frozen_assets'/name;secure_directory(frozen.parent);write_new_bytes(frozen,path.read_bytes());files[name]=frozen
    package={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,'gpu_job_launched':False,
        'assets':{name:{'local_path':str(path.absolute()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in files.items()}}
    write_new_json(out/'package.json',package)
    return package


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base-package','out','recorded'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();p=prepare(args.base_package,args.run_id,args.out.absolute(),args.gpu_uuid,args.recorded)
    print(json.dumps({'run_id':args.run_id,'assets':len(p['assets']),'source':p['request']['python_source_commit'],'GPU_started':False}))


if __name__=='__main__':main()
