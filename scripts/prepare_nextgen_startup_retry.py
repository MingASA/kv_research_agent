"""Freeze the single audited startup-window retry; never launches GPU work."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from scripts.controller.approval import TRUSTED_APPROVAL,load_approval_snapshot
from scripts.controller.durable import write_new_bytes,write_new_json
from scripts.controller.policy import validate_request


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    original=root/'plans/experiments/nextgen-bf16-qwen38-validation-r0-20261027b/package.json'
    prior=json.loads(original.read_text());request=copy.deepcopy(prior['request'])
    run_id='nextgen-bf16-qwen38-validation-r1-20261027d';output=root/'plans/experiments'/run_id
    output.mkdir(exist_ok=False)
    mirror=root/'results/mirrors/nextgen-bf16-qwen38-validation-r0-20261027b-failed-b1/files/support/agent_work/runs'/request['run_id']
    result=json.loads((mirror/'result.json').read_text())
    if result['status']!='FAILED' or not result['gpu_release_verified']:
        raise ValueError('Original run has not reached a verified released failure')
    request.update(run_id=run_id,remote_output=request['approved_workspace']+'/agent_work/runs/'+run_id,
                   max_wall_seconds=2700,startup_seconds=1500,replicate=1,
                   startup_recovery={'prior_run_id':prior['request']['run_id'],
                                     'prior_result_sha256':hashlib.sha256((mirror/'result.json').read_bytes()).hexdigest(),
                                     'reason':'weight_load_exceeded_initial900s_limit'})
    request['dataset_path']=request['remote_output']+'/input/validation.json'
    request['python_source_repo']=request['remote_output']+'/source_repo'
    evidence=root/'reports/operations/nextgen-native-kv-20261027a/new_model_validation_tokenization_CPU_a2.json'
    counts=json.loads(evidence.read_text());assert counts['input_sha256']==request['dataset_sha256']
    request['service']['max_context']=16384
    request['max_context_scope']='validation_input_budget_not_server_context_cap'
    request['startup_recovery']['context_budget_correction']={
        'prior_nominal_bound':8192,'new_input_budget':16384,
        'validated_input_max_prompt_plus_output':counts['max_prompt_plus_output'],
        'tokenization_evidence_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}
    approval,approval_sha=load_approval_snapshot(TRUSTED_APPROVAL);validate_request(request,approval)
    assets={}
    for name,entry in prior['assets'].items():
        if name.startswith('runtime/'):
            continue
        data=Path(entry['local_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Original input/source asset drift')
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);write_new_bytes(path,data)
        assets[name]={'local_path':str(path),'sha256':entry['sha256']}
    for original_path in sorted((root/'scripts/controller').glob('*.py')):
        name='runtime/kv_controller/'+original_path.name;path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        data=original_path.read_bytes();write_new_bytes(path,data)
        assets[name]={'local_path':str(path),'sha256':hashlib.sha256(data).hexdigest()}
    target=output/'input/validation_tokenization_CPU.json';data=evidence.read_bytes();write_new_bytes(target,data)
    assets['input/validation_tokenization_CPU.json']={'local_path':str(target),'sha256':hashlib.sha256(data).hexdigest()}
    write_new_json(output/'request.json',request)
    write_new_json(output/'package.json',{'schema':'kv-frozen-local-package-v1','classification':'proposal',
                                        'request':request,'approval_sha256_at_preparation':approval_sha,'assets':assets})


if __name__=='__main__':main()
