#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze one deterministic V-scale candidate, reusing the original BF16 reference."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.controller.durable import digest, secure_directory, write_new_bytes, write_new_json
from scripts.controller.policy import XQA_SOURCES, QUALITY_SOURCES
from scripts.controller.scale_coverage import METHOD, SCALES, TOOLS_CHANGED, validate_prerequisites
from scripts.prepare_production_quality_batch import _relocate

SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-vscale-coverage-20261027')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(base: Path, calibration: Path, proposal: Path, reference: Path,
            out: Path, run: str, gpu: str) -> dict:
    if out.exists():
        raise FileExistsError('Use a new immutable scale-candidate package')
    parent = json.loads(base.read_text())
    old = parent['request']
    ref = json.loads(reference.read_text())
    prior = json.loads(calibration.read_text())
    if (old['workload']['kind'] != 'native_quality_xqa_production_batch'
            or old['run_id'] != ref['run_id'] or ref['samples_per_arm'] != 1719
            or old['arms'] != ['bf16','fp8_e4m3']):
        raise ValueError('Use the independently audited original modern full1719 C128 reference')
    for name, entry in parent['assets'].items():
        if _sha(Path(entry['local_path'])) != entry['sha256']:
            raise ValueError('Frozen reference asset changed: '+name)
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True):
        raise ValueError('Commit the actual candidate tools before freezing')
    commit = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    hashes = {name:_sha(SOURCE/name) for name in XQA_SOURCES+QUALITY_SOURCES}
    if set(hashes) != set(old['backend_source_sha256']) or {
            n for n in hashes if hashes[n] != old['backend_source_sha256'][n]} != TOOLS_CHANGED:
        raise ValueError('Only numerical-gate scales and single-arm planning tools may change')
    old_runner = subprocess.check_output(['git','-C',str(SOURCE),'show',
        old['python_source_commit']+':tools/run_qwen25_joint_kv_quality.py'],text=True)
    old_ast, new_ast = [ast.parse(code) for code in (old_runner,(SOURCE/'tools/run_qwen25_joint_kv_quality.py').read_text())]
    for tree in (old_ast,new_ast):
        tree.body = [node for node in tree.body if not isinstance(node,ast.FunctionDef) or node.name != '_environment']
    if ast.dump(old_ast,include_attributes=False) != ast.dump(new_ast,include_attributes=False):
        raise ValueError('Runner changed outside the admitted plan-environment correction')
    remote = old['approved_workspace']+'/agent_work/runs/'+run
    request = _relocate(copy.deepcopy(old),old['remote_output'],remote)
    request.update(run_id=run,replicate=0,arms=['fp8_e4m3'],concurrency=128,max_wall_seconds=3600,
        max_output_gib=5,python_source_repo=remote+'/source_repo',python_source_commit=commit,
        gpu_uuids=[gpu],fp8_scales=dict(SCALES),backend_source_sha256=hashes,
        research_reason='One preregistered minimal V-scale coverage candidate after independently observed long-document saturation. K=.125 unchanged,V=.0625 to.125; exact same resident-FP8 FA2prefill/XQAdecode inference code/scorer/data/prompts/caps/Graph/prefix/runtime. Reuse existing modern BF16 reference; generate only the new FP8 full1719 arm. Actual numerical/Graph gate must use the new scales. No answer-based scale search or claim that68 rare clips caused overall quality loss.')
    request['workload']['scale_calibration'] = {
        'method':METHOD,'calibration_run_id':prior['run_id'],
        'calibration_audit_sha256':_sha(calibration),'calibration_terminal_sha256':prior['terminal_result_sha256'],
        'calibration_audit_path':remote+'/input/longcontext_range_independent_audit.json',
        'proposal_sha256':_sha(proposal),'proposal_path':remote+'/input/scale_coverage_candidate.json',
        'reference_run_id':ref['run_id'],'reference_audit_sha256':_sha(reference),
        'reference_audit_path':remote+'/input/existing_modern_reference_audit.json',
        'baseline_backend_source_sha256':old['backend_source_sha256']}
    secure_directory(out)
    prerequisites = out/'prerequisites';secure_directory(prerequisites)
    files = {name:Path(entry['local_path']) for name,entry in parent['assets'].items()
        if not name.startswith('runtime/') and name not in (
            'input/source.bundle','input/approved_data_manifest.json','input/production_batch_provenance.json')}
    for original, name in ((calibration,'longcontext_range_independent_audit.json'),
                           (proposal,'scale_coverage_candidate.json'),(reference,'existing_modern_reference_audit.json')):
        frozen = prerequisites/name;write_new_bytes(frozen,original.read_bytes());files['input/'+name] = frozen
    validate_prerequisites(request,input_root=prerequisites)
    approved = json.loads(Path(parent['assets']['input/approved_data_manifest.json']['local_path']).read_text())
    write_new_json(out/'approved_data_manifest.json',_relocate(approved,old['remote_output'],remote))
    files['input/approved_data_manifest.json'] = out/'approved_data_manifest.json'
    proof = {'classification':'confirmed_CPU_inference_and_scoring_bytes_only_new_scale_quality_pending',
        'source_commit':commit,'baseline_source_commit':old['python_source_commit'],
        'baseline_evaluator_sha256':old['backend_source_sha256']['tools/eval_kv_cache_quality.py'],
        'candidate_evaluator_sha256':hashes['tools/eval_kv_cache_quality.py'],
        'evaluator_and_scoring_source_bytes_identical':True,'all_inference_source_bytes_identical':True,
        'runner_AST_identical_except_plan_environment':True,'changed_tools':sorted(TOOLS_CHANGED),
        'source_sha256':hashes,'baseline_source_sha256':old['backend_source_sha256'],
        'parent_package_sha256':_sha(base),'reference_audit_sha256':_sha(reference),
        'calibration_audit_sha256':_sha(calibration),'proposal_sha256':_sha(proposal),
        'original_quality_records':1719,'BF16_predictions_regenerated':0,'candidate_scales':SCALES,
        'preparation_source_sha256':_sha(Path(__file__)),'new_GPU_started':False,
        'scope':'exact deployment recipe across runs,not isolated numerical causality or accepted quality'}
    write_new_json(out/'scale_quality_provenance.json',proof)
    files['input/scale_quality_provenance.json'] = out/'scale_quality_provenance.json'
    bundle = out/'source.bundle'
    subprocess.run(['git','-C',str(SOURCE),'bundle','create',str(bundle),'HEAD',
        '^ceff00bbd38054d2b6b4aa44513d773f2bd174b2'],check=True,capture_output=True)
    files['input/source.bundle'] = bundle
    for path in (ROOT/'scripts/controller').glob('*.py'):
        name = 'runtime/kv_controller/'+path.name
        actual = ROOT/'scripts/audit_native_results.py' if path.name == 'evidence.py' else path
        frozen = out/'frozen_assets'/name;secure_directory(frozen.parent)
        write_new_bytes(frozen,actual.read_bytes());files[name] = frozen
    request['experiment_key'] = digest({key:request[key] for key in ('python_source_commit','binary_sha256',
        'dataset_sha256','backend_source_sha256','image_id','graph_mode','prefix_cache','fp8_scales',
        'operators','service','concurrency','workload')})
    package = {'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,
        'gpu_job_launched':False,'assets':{name:{'local_path':str(path.absolute()),'sha256':_sha(path)}
                                        for name,path in files.items()}}
    write_new_json(out/'package.json',package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base','calibration-audit','proposal','reference-audit','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--gpu-uuid',required=True)
    args = parser.parse_args()
    result = prepare(args.base,args.calibration_audit,args.proposal,args.reference_audit,
        args.out.absolute(),args.run_id,args.gpu_uuid)
    print(json.dumps({'run_id':args.run_id,'source_commit':result['request']['python_source_commit'],
        'assets':len(result['assets']),'new_FP8_predictions_pending':1719,'BF16_repeated':False,'GPU_started':False}))


if __name__ == '__main__':main()
