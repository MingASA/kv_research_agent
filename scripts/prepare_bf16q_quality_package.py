#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze only the necessary full1719 BF16-Q candidate, retaining BF16 reference."""

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
from scripts.controller.policy import BF16Q_DECODE_OPERATORS,BF16Q_DECODE_SOURCES,QUALITY_SOURCES
from scripts.prepare_production_quality_batch import _relocate

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/mingasa/projects/xllm_quant_kvcache-agent-bf16q-quality-20261027')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(base: Path,operator: Path,reference: Path,out: Path,run: str,gpu: str,
            retry: Path | None = None) -> dict:
    if out.exists():raise FileExistsError('New immutable quality package required')
    parent=json.loads(base.read_text());previous=parent['request']
    if (previous['workload']['kind']!='native_quality_xqa_production_batch' or previous['concurrency']!=128
            or previous['workload']['samples_per_arm']!=1719):
        raise ValueError('Exact original full1719 production C128 package required')
    for n,e in parent['assets'].items():
        if _sha(Path(e['local_path']))!=e['sha256']:raise ValueError('Original frozen asset differs: '+n)
    op=json.loads(operator.read_text());ref=json.loads(reference.read_text())
    if (op['classification']!='confirmed_operator_evidence_integrity_and_BF16Q_Graph128_contract_only'
            or op['graph128']['production_max_batch']!=128 or op['model_loaded'] is not False
            or ref['classification']!='confirmed_full_production_batch_prediction_integrity_and_rescoring'
            or ref['run_id']!=previous['run_id'] or ref['samples_per_arm']!=1719
            or ref['audited_arms']!=['bf16','fp8_e4m3']):
        raise ValueError('Actual same-source operator gate and full modern BF16 reference must be independently audited')
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True):
        raise ValueError('Candidate source must be committed and clean')
    if set(op['backend_source_sha256'])!=set(BF16Q_DECODE_SOURCES) or any(
            _sha(SOURCE/n)!=op['backend_source_sha256'][n] for n in BF16Q_DECODE_SOURCES):
        raise ValueError('Do not reuse an operator gate for changed inference/operator code')
    secure_directory(out);remote=previous['approved_workspace']+'/agent_work/runs/'+run
    request=_relocate(copy.deepcopy(previous),previous['remote_output'],remote)
    request.update(run_id=run,replicate=0,arms=['fp8_e4m3'],concurrency=128,max_wall_seconds=5400,max_output_gib=5,
        python_source_commit=commit,python_source_repo=remote+'/source_repo',gpu_uuids=[gpu],
        backend_source_sha256={n:_sha(SOURCE/n) for n in BF16Q_DECODE_SOURCES+QUALITY_SOURCES},
        operators=copy.deepcopy(BF16Q_DECODE_OPERATORS),
        research_reason='Evaluate the concrete deployable resident-FP8/BF16-Q FA2 candidate after actual H800 Graph128 numerical admission. Same1719 original records/scorer, naturalEOS/C128/prefix/scheduler/scales; inference/operator code byte-identical to the audited gate. Retain existing modern BF16 FA3+XQA reference rather than repeat it. New precision path, not Q-causality or preaccepted quality.')
    request['workload'].update(kind='native_quality_bf16q_production_batch',cohort='full_production_batch',
        operator_validation={'run_id':op['run_id'],'source_commit':op['source_commit'],
            'terminal_result_sha256':op['terminal_result_sha256'],'audit_sha256':_sha(operator),
            'audit_path':remote+'/input/bf16q_operator_audit.json','source_sha256':op['backend_source_sha256']})
    files={n:Path(e['local_path']) for n,e in parent['assets'].items() if not n.startswith('runtime/')
        and n not in ('input/source.bundle','input/approved_data_manifest.json','input/production_batch_provenance.json')}
    approved=json.loads(Path(parent['assets']['input/approved_data_manifest.json']['local_path']).read_text())
    write_new_json(out/'approved_data_manifest.json',_relocate(approved,previous['remote_output'],remote))
    files['input/approved_data_manifest.json']=out/'approved_data_manifest.json'
    files['input/bf16q_operator_audit.json']=operator.absolute()
    files['input/existing_modern_BF16_reference_audit.json']=reference.absolute()
    if retry is not None:
        failed=json.loads(retry.read_text())
        if (failed.get('classification')!='confirmed_BF16Q_quality_failed_before_model_default_plan_arm'
                or failed.get('original_status')!='FAILED' or failed.get('model_loaded') is not False
                or failed.get('model_predictions')!=0 or failed.get('GPU_release_verified') is not True):
            raise ValueError('Retry only the independently preserved pre-model plan environment failure')
        request['replicate']=1
        request['workload']['retry_of']={'run_id':failed['run_id'],'terminal_result_sha256':failed['terminal_result_sha256'],
            'failure_audit_sha256':_sha(retry),'reason':'CPU_plan_environment_default_arm_fixed_before_any_model_prediction',
            'independent_statistical_replicate':False}
        request['research_reason']+=' Repair the separately preserved pre-model plan-environment failure; full actual CPU dry-run required before dispatch. No completed model predictions repeated.'
        files['input/previous_plan_failure_audit.json']=retry.absolute()
    proof={'classification':'proposal_CPU_only_not_new_quality','source_commit':commit,
        'operator_source_commit':op['source_commit'],'inference_and_operator_source_bytes_identical':True,
        'records':1719,'concurrency':128,'fixed_GPU_batch':False,'BF16_predictions_regenerated':0,
        'reference_run_id':ref['run_id'],'reference_audit_sha256':_sha(reference),
        'parent_package_sha256':_sha(base),'original_data_manifest_sha256':request['dataset_sha256'],
        'operator_audit_sha256':_sha(operator),'preparation_script_sha256':_sha(Path(__file__)),
        'selection':'Explicit production precision path; no scale/answer-based configuration search',
        'comparison_limit':'New FP8 run versus retained BF16 reference; scheduling/cache history may differ'}
    write_new_json(out/'bf16q_quality_provenance.json',proof);files['input/bf16q_quality_provenance.json']=out/'bf16q_quality_provenance.json'
    bundle=out/'source.bundle';subprocess.run(['git','-C',str(SOURCE),'bundle','create',str(bundle),'HEAD',
        '^ceff00bbd38054d2b6b4aa44513d773f2bd174b2'],check=True,capture_output=True);files['input/source.bundle']=bundle
    for p in (ROOT/'scripts/controller').glob('*.py'):
        name='runtime/kv_controller/'+p.name;original=ROOT/'scripts/audit_native_results.py' if p.name=='evidence.py' else p
        frozen=out/'frozen_assets'/name;secure_directory(frozen.parent);write_new_bytes(frozen,original.read_bytes());files[name]=frozen
    request['experiment_key']=digest({k:request[k] for k in ('python_source_commit','binary_sha256','dataset_sha256',
        'backend_source_sha256','image_id','graph_mode','prefix_cache','fp8_scales','operators','service','concurrency','workload')})
    package={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,'gpu_job_launched':False,
        'assets':{n:{'local_path':str(p.absolute()),'sha256':_sha(p)} for n,p in files.items()}}
    write_new_json(out/'package.json',package);return package


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('base','operator-audit','reference-audit','out'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--retry-failure',type=Path);a=parser.parse_args()
    p=prepare(a.base,a.operator_audit,a.reference_audit,a.out.absolute(),a.run_id,a.gpu_uuid,a.retry_failure)
    print(json.dumps({'run_id':a.run_id,'source_commit':p['request']['python_source_commit'],'assets':len(p['assets']),
        'original_records':1719,'BF16_repeated':False,'GPU_started':False}))


if __name__=='__main__':main()
