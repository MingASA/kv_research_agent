#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Execute the exact candidate's entire dry-run on original local data, no GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.controller.cli import _local_package,validate_frozen_policy
from scripts.controller.approval import TRUSTED_APPROVAL,load_approval_snapshot
from scripts.controller.durable import write_new_json,digest
from scripts.controller.policy import validate_request
from scripts.controller.production import candidate_quality_command
from scripts.prepare_production_quality_batch import _relocate


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);a=parser.parse_args()
    approval,sha=load_approval_snapshot(TRUSTED_APPROVAL);r,h,_=_local_package(a.package)
    validate_request(r,approval);validate_frozen_policy(a.package,r,approval)
    if subprocess.check_output(['git','-C',str(a.source),'rev-parse','HEAD'],text=True).strip()!=r['python_source_commit']:
        raise ValueError('Actual dry-run source must equal the frozen candidate commit')
    p=json.loads(a.package.read_text());cmd=candidate_quality_command({'request':r},Path('/output'),'fp8_e4m3')
    cohort=str(Path(p['assets']['input/cohort/gsm8k.jsonl']['local_path']).parent)
    scorer=str(Path(p['assets']['input/scorer/LongBench/metrics.py']['local_path']).parents[1])
    mappings=[(r['python_source_repo'],str(a.source.resolve())),(str(Path(r['dataset_path']).parent),cohort),
        (r['workload']['longbench_repo'],scorer),
        (r['workload']['gsm8k_shots'],p['assets']['input/five_train_examples.json']['local_path'])]
    cmd[0]=sys.executable
    for old,new in mappings:cmd=[v.replace(old,new) for v in cmd]
    cmd+=['--dry-run'];environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=120,check=True,env=environment)
    plan=json.loads(result.stdout)
    if (plan['scope']!='one_unique_dataset_round' or plan['arms']!=['fp8_e4m3']
            or plan['environment']!=plan['environments_by_arm']['fp8_e4m3']
            or not plan.get('data_audit') or 'pending_input' in plan):
        raise ValueError('Actual complete candidate plan failed; no missing-data dry-run accepted')
    approved=json.loads(Path(p['assets']['input/approved_data_manifest.json']['local_path']).read_text())
    for old,new in mappings:approved=_relocate(approved,old,new)
    if plan['data_audit']!=approved.get('data_audit',approved):raise ValueError('Original full data/protocol SHA/count/order differs')
    write_new_json(a.out,{'classification':'confirmed_actual_frozen_full1719_runner_dry_run_CPU_not_GPU',
        'run_id':r['run_id'],'source_commit':r['python_source_commit'],'request_sha256':digest(r),
        'assets_SHA_verified':len(h),'actual_frozen_policy_valid':True,'approval_sha256':sha,
        'command':cmd,'plan':plan,'stderr':result.stderr,'original_approved_data_audit_matches':True,
        'CUDA_VISIBLE_DEVICES':'','GPU_started':False})
    print(json.dumps({'actual_entire_dry_run':True,'all_original_data_protocol_SHA_counts_order_match':True,
        'run_id':r['run_id'],'source':r['python_source_commit'],'assets':len(h),'GPU_started':False}))


if __name__=='__main__':main()
