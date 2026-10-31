#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze an untimed heldout BF16/Graph KV distribution diagnostic; no GPU use."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import digest, secure_directory, write_new_bytes, write_new_json
from scripts.controller.policy import BF16_XQA_OPERATORS, KV_RANGE_SOURCES
from scripts.controller.range_probe import METHOD, LONG_QA_SCOPE, validate_longcontext_inputs

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/mingasa/projects/xllm_quant_kvcache-agent-distribution-20261027')
DATA=ROOT/'results/prepared/calibration-sharegpt-heldout128-20261027a2'


def prepare(base: Path, out: Path, run_id: str, gpu: str, replicate: int = 0,
            retry_of: str | None = None, retry_reason: str | None = None,
            input_directory: Path | None = None) -> dict:
    if out.exists():raise FileExistsError('New immutable diagnostic package required')
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True):
        raise ValueError('Scientific source must be committed and clean')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    data_root = input_directory if input_directory is not None else DATA
    manifest=json.loads((data_root/'manifest.json').read_text());data=json.loads((data_root/'canonical_inputs.json').read_text())
    long_qa = manifest.get('content_scope') == LONG_QA_SCOPE
    if long_qa:
        validate_longcontext_inputs(data, manifest)
    if (sha(data_root/'canonical_inputs.json')!=manifest['canonical_sha256'] or manifest['requests']!=128
            or not long_qa and manifest.get('quality_labels_loaded') is not False or len(data['data'])!=128
            or len({r['session_id'] for r in data['data']})!=128):
        raise ValueError('Heldout unique128 input SHA/scope differs')
    for row in data['data']:
        if len(row['payloads'])!=1:raise ValueError('Single-turn original payload required')
        p=row['payloads'][0]
        if (p.get('ignore_eos',False) is not False or not long_qa and p.get('max_tokens')!=1024
                or p.get('stream') is not True):
            raise ValueError('Natural stopping and original cap/stream controls must remain unchanged')
    parent=json.loads(base.read_text());request=copy.deepcopy(parent['request'])
    remote=request['approved_workspace']+'/agent_work/runs/'+run_id
    secure_directory(out);bundle=out/'source.bundle'
    subprocess.run(['git','-C',str(SOURCE),'bundle','create',str(bundle),'HEAD',
                    '^ceff00bbd38054d2b6b4aa44513d773f2bd174b2'],capture_output=True,check=True)
    request.update(run_id=run_id,remote_output=remote,replicate=replicate,arms=['bf16'],concurrency=128,
        max_wall_seconds=1800,max_output_gib=3,gpu_uuids=[gpu],python_source_commit=commit,
        python_source_repo=remote+'/source_repo',dataset_path=remote+'/input/canonical_inputs.json',
        dataset_sha256=manifest['canonical_sha256'],operators={'bf16':copy.deepcopy(BF16_XQA_OPERATORS['bf16'])},
        backend_source_sha256={name:sha(SOURCE/name) for name in KV_RANGE_SOURCES},
        workload={'kind':'heldout_kv_range_probe','requests':128,'warmup':0,'ignore_eos':False,
            'method':METHOD,'quality_labels_loaded':False,'input_manifest':remote+'/input/production_manifest.json',
            'aiperf_zero_warmup_policy':'omit_explicit_option',
            'jsonl_framing':'physical_LF_UTF8','export_before_postprocessing':True,
            'aiperf_cli_config_sha256':sha(ROOT/'.deps/aiperf-runtime-20261026/aiperf/config/flags/cli_config.py'),
            'content_scope':'heldout_original_ShareGPT_natural_chat','source_content_sha256':manifest['source_content_sha256'],
            'arrival_mode':'closed_loop_concurrency','service_excluded_count':528},
        research_reason='Production quality risk diagnosis, not a throughput headline: original independent heldout128 natural ShareGPT requests on BF16 FA3/XQA/Graph; bounded per-layer/phase/head GPU KV range/clipping/underflow/QDQ moments. Actual small numerical/Graph/reset/export gate before model, reset after bootstrap, export after all AIPerf requests. No answer-derived scales or FP8 generation quality claim; no completed experiment repeated.')
    request.pop('diagnostic',None);request.pop('dataset_archive',None)
    if long_qa:
        workload = request['workload']
        workload.pop('quality_labels_loaded')
        workload.pop('service_excluded_count')
        workload.update(content_scope=LONG_QA_SCOPE, quality_labels_used_in_selection=False,
            answers_exported_to_probe=False, official_generation_caps_preserved=True,
            evaluation_excluded_count=400)
        request['research_reason'] = ('Resolve whether fixed FP8 scales clip actual long-document QA KV writes, '
            'after the real short-chat diagnostic had no clipping and BF16-Q did not restore overall quality. '
            'Four official real QA tasks,32 each,exact IDs/records/contexts disjoint from all measured LongBench400; '
            'original prompts,task caps,natural EOS,production BF16 FA3/XQA Graph128 and prefix cache. '
            'No answer/output-based selection,no FP8 generation or code coverage claim,no throughput headline. '
            'Keep existing scales if no clipping; only saturation can justify a predeclared coverage repair. '
            'No completed GPU experiment is repeated.')
    if retry_of is not None:
        request['workload']['retry_of']=retry_of
        request['research_reason']+=' Controller repair retry of '+retry_of+': '+(retry_reason or 'Explicit zero warmup is omitted after the prior CLI refusal.')+' This is not an independent statistical replicate.'
    request['experiment_key']=digest({k:request[k] for k in ('python_source_commit','binary_sha256',
        'dataset_sha256','backend_source_sha256','image_id','graph_mode','prefix_cache','fp8_scales',
        'operators','service','workload')})
    files={'runtime/kv_controller/'+p.name:p for p in (ROOT/'scripts/controller').glob('*.py')}
    files['runtime/kv_controller/evidence.py']=ROOT/'scripts/audit_native_results.py'
    files.update({'input/canonical_inputs.json':data_root/'canonical_inputs.json',
        'input/production_manifest.json':data_root/'manifest.json','input/source.bundle':bundle})
    files.update({name:Path(entry['local_path']) for name,entry in parent['assets'].items()
                  if name.startswith('runtime_support/')})
    for name,path in list(files.items()):
        if name.startswith(('runtime/','runtime_support/')):
            frozen=out/'frozen_assets'/name;secure_directory(frozen.parent)
            write_new_bytes(frozen,path.read_bytes());files[name]=frozen
    package={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,
        'gpu_job_launched':False,'assets':{name:{'local_path':str(path.absolute()),'sha256':sha(path)}
                                         for name,path in files.items()}}
    write_new_json(out/'package.json',package);return package


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base-package','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--replicate',type=int,default=0);parser.add_argument('--retry-of')
    parser.add_argument('--retry-reason')
    parser.add_argument('--input-directory', type=Path)
    args=parser.parse_args();p=prepare(args.base_package,args.out.absolute(),args.run_id,args.gpu_uuid,args.replicate,args.retry_of,args.retry_reason,args.input_directory)
    print(json.dumps({'run_id':args.run_id,'source':p['request']['python_source_commit'],
                      'assets':len(p['assets']),'GPU_started':False}))


if __name__=='__main__':main()
