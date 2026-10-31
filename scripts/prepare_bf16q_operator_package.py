#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze a production-directed FP8 KV / BF16 Q operator and Graph128 gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest,secure_directory,write_new_bytes,write_new_json
from scripts.controller.policy import BF16Q_DECODE_OPERATORS,BF16Q_DECODE_SOURCES

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/mingasa/projects/xllm_quant_kvcache-agent-bf16q-decode-20261027')


def prepare(base: Path,run_id: str,out: Path,gpu: str,provenance: Path) -> dict:
    if out.exists():raise FileExistsError('New immutable BF16-Q operator package required')
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True):
        raise ValueError('Candidate source must be committed and clean')
    parent=json.loads(base.read_text());primary=json.loads(provenance.read_text())
    manifest={'method':'resident_FP8_BF16Q_production_Graph128_gate_v1',
        'geometry':{'batch':128,'heads':40,'kv_heads':8,'head_dim':128,'page_size':128},
        'relative_l2_limit':.01,'primary_sources_sha256':{r['path']:r['sha256'] for r in primary['files']},
        'primary_sources':primary,'synthetic_tensors_only':True,'model_loaded':False,'AIPerf_ran':False,
        'purpose':'Deployable no-Q-quantization candidate; actual padded/replanned Graph128 plus writer and dense reference before model quality or service work'}
    secure_directory(out);input_path=out/'operator_manifest.json';write_new_json(input_path,manifest)
    bundle=out/'source.bundle'
    subprocess.run(['git','-C',str(SOURCE),'bundle','create',str(bundle),'HEAD',
        '^ceff00bbd38054d2b6b4aa44513d773f2bd174b2'],check=True,capture_output=True)
    request=copy.deepcopy(parent['request']);remote=request['approved_workspace']+'/agent_work/runs/'+run_id
    sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
    request.update(run_id=run_id,remote_output=remote,replicate=0,arms=['fp8_e4m3'],concurrency=128,
        max_wall_seconds=1800,max_output_gib=3,python_source_commit=commit,python_source_repo=remote+'/source_repo',
        gpu_uuids=[gpu],dataset_path=remote+'/input/operator_manifest.json',dataset_sha256=sha,
        operators=copy.deepcopy(BF16Q_DECODE_OPERATORS),
        backend_source_sha256={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in BF16Q_DECODE_SOURCES},
        workload={'kind':'native_bf16q_decode_operator','ignore_eos':False,'source_content_sha256':sha,
            **{k:manifest[k] for k in ('method','geometry','relative_l2_limit','primary_sources_sha256','model_loaded','AIPerf_ran')}},
        research_reason='Production candidate preserving resident FP8 capacity while avoiding internal FP8 Q decode. Validate actual installed FA2 sources, BF16 Q numerical reference L2<=.01, real Graph128 writer+attention with128/126/1/127 live rows, dynamic context, packed QKV strides and stable prefix storage. No model/AIPerf/quality claim and no completed experiment repeated.')
    request.pop('diagnostic',None);request.pop('dataset_archive',None)
    request['experiment_key']=digest({k:request[k] for k in ('python_source_commit','binary_sha256','dataset_sha256',
        'backend_source_sha256','image_id','graph_mode','prefix_cache','fp8_scales','operators','service','workload')})
    files={'runtime/kv_controller/'+p.name:p for p in (ROOT/'scripts/controller').glob('*.py')}
    files['runtime/kv_controller/evidence.py']=ROOT/'scripts/audit_native_results.py'
    files.update({'input/operator_manifest.json':input_path,'input/source.bundle':bundle})
    for name,path in list(files.items()):
        if name.startswith('runtime/'):
            frozen=out/'frozen_assets'/name;secure_directory(frozen.parent);write_new_bytes(frozen,path.read_bytes());files[name]=frozen
    package={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,'gpu_job_launched':False,
        'assets':{n:{'local_path':str(p.absolute()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in files.items()}}
    write_new_json(out/'package.json',package);return package


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base-package','out','provenance'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();p=prepare(args.base_package,args.run_id,args.out.absolute(),args.gpu_uuid,args.provenance)
    print(json.dumps({'run_id':args.run_id,'assets':len(p['assets']),'source':p['request']['python_source_commit'],'GPU_started':False}))


if __name__=='__main__':main()
