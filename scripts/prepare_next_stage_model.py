"""Freeze a source-matched model qualification from real build/operator receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from scripts.audit_upstream_run import audit
from scripts.controller.cli import validate_frozen_policy
from scripts.controller.durable import digest
from scripts.controller.next_stage_deployment import MODEL_KIND, MX_CONFIGURATION, PROFILES, operator_acceptance_passed
from scripts.controller.upstream_native import CONFIGURATIONS
from scripts.controller.policy import validate_request
from scripts.upstream_package_runtime import freeze_runtime, attach_graph_observer


ROOT=Path(__file__).resolve().parents[1]


def prepare(operator_manifest: Path, metadata_path: Path, run: str,
            profile: str, candidate_scale: bool, calibration_manifest: Path | None) -> dict:
    lifecycle=audit(operator_manifest)
    manifest=json.loads(operator_manifest.read_bytes())
    files_root=Path(manifest['mirror_directory'])/'files'
    prior=json.loads((files_root/'ticket.json').read_bytes())['request']
    request=dict(prior)
    request.pop('compiled_backend_source_sha256',None)
    mx=profile==PROFILES[0]
    if prior['workload']['kind'] != ('upstream_next_stage_native_operator' if mx else 'upstream_FP8_optimization_operator'):
        raise ValueError('Operator proof does not belong to the selected format')
    acceptance=json.loads((files_root/'output'/prior['arms'][0]/'operators/acceptance.json').read_bytes())
    if not operator_acceptance_passed(acceptance,prior['arms'][0]):
        raise ValueError('An operator candidate failed its frozen prerequisite gate')
    metadata=json.loads(metadata_path.read_bytes());result=metadata['build_result']
    repo=Path('/home/mingasa/projects/xllm-'+('mxfp4-kv' if mx else 'fp8-opt')+'-20261105a')
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip() or head!=prior['python_source_commit']:
        raise ValueError('Qualified operator source must still be the exact clean candidate')
    if (result.get('success') is not True or result.get('CPU_tests_executed') is not True
            or result.get('GPU_job') is not False or metadata.get('source_status')
            or metadata.get('container_exited_successfully') is not True):
        raise ValueError('Require a successful actual CPU build, never a proposed build manifest')
    if mx and metadata['source_commit']!=head:
        raise ValueError('MXFP4 must have its new allocator/configuration compiled')
    for name,sha in prior['backend_source_sha256'].items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Operator source drift: '+name)
        if not name.endswith('.py') and metadata['backend_source_sha256'].get(name)!=sha:
            raise ValueError('New Python source differs from actual compiled C++/headers: '+name)
    reference={'run_id':lifecycle['run_id'],'result_sha256':manifest['remote_result_sha256']}
    if candidate_scale and (mx or calibration_manifest is None):
        raise ValueError('Scale candidate needs an independent FP8 numerical receipt')
    if calibration_manifest is not None:
        numeric_lifecycle=audit(calibration_manifest)
        numeric_manifest=json.loads(calibration_manifest.read_bytes())
        calibration_root=Path(numeric_manifest['mirror_directory'])/'files'
        numeric=json.loads((calibration_root/'output/bf16/calibration_acceptance.json').read_bytes())
        if (numeric_lifecycle['kind']!='upstream_FP8_independent_numeric_calibration'
                or numeric.get('candidate_numerical_gate') is not True):
            raise ValueError('Independent numerical candidate gate did not pass')
        request['numeric_calibration_reference']={'run_id':numeric_lifecycle['run_id'],
            'result_sha256':numeric_manifest['remote_result_sha256']}
    plan=ROOT/'plans/experiments'/run;plan.mkdir();(plan/'input').mkdir()
    assets={}
    for name in ('validation.json','gdn_dependency_CPU_import.json','source.bundle'):
        target=plan/'input'/name;shutil.copyfile(files_root/'input'/name,target);assets['input/'+name]=target
    remote=request['approved_workspace']+'/agent_work/runs/'+run
    configuration=MX_CONFIGURATION if mx else dict(CONFIGURATIONS['fp8_e4m3'],value_scale=.5 if candidate_scale else .0625)
    arm=configuration['dtype']
    request.update(run_id=run,remote_output=remote,python_source_repo=remote+'/source_repo',
        dataset_path=remote+'/input/validation.json',next_stage_profile=profile,operator_qualification_reference=reference,
        server_source_commit=metadata['source_commit'],source_repo=metadata['source_repo'],
        server_binary=metadata['server_binary'],binary_sha256=result['binary_sha256'],
        GPU_context_test_binary=metadata['GPU_context_test_binary'],GPU_context_test_sha256=metadata['GPU_context_test_sha256'],
        native_configuration=configuration,server_binary_execution='required',max_wall_seconds=3600,
        gpu_uuids=['GPU-0c628a64-5f31-3501-fa3c-723f5b69254a'],arms=[arm],
        operators={arm:{'planner':configuration['prefill'],'decode':configuration['decode']}},
        workload={'kind':MODEL_KIND,'scope':'source_matched_native_model_prefix_Graph_qualification'},
        experiment_key=digest({'stage':'model_20261105','source':head,'compiled':metadata['source_commit'],
                               'operator_reference':reference,'configuration':configuration}))
    for name in ('runtime_library_directory','runtime_dependency_build_id','zeromq_library_sha256'):
        request[name]=metadata[name]
    assets.update(attach_graph_observer(plan,request));assets.update(freeze_runtime(plan))
    approval=json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request,approval)
    package={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,
             'request_sha256':digest(request),'assets':{name:{'local_path':str(path.resolve()),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in assets.items()}}
    path=plan/'package.json';path.open('x').write(json.dumps(package,indent=2)+'\n')
    (plan/'request.json').open('x').write(json.dumps(request,indent=2)+'\n')
    if mx:
        config=json.loads((ROOT/'state/UPSTREAM_AC_C.local.json').read_bytes())
        config.update(source_commit=metadata['source_commit'],remote_source_repo=metadata['source_repo'],
                      remote_new_binary_path=metadata['server_binary'],binary_sha256=result['binary_sha256'],
                      source_metadata_receipt=str(metadata_path.resolve()))
        target=ROOT/'state/UPSTREAM_MXFP4.local.json'
        if target.exists() and json.loads(target.read_bytes())!=config:
            raise ValueError('Existing MXFP4 identity cannot be overwritten')
        if not target.exists():
            target.open('x').write(json.dumps(config,indent=2)+'\n');target.chmod(0o600)
    validate_frozen_policy(path,request,approval)
    return {'package':str(path),'operator_proof_verified':True,'GPU_dispatched':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator-sha-manifest',type=Path,required=True)
    p.add_argument('--build-metadata',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--profile',choices=PROFILES,required=True)
    p.add_argument('--candidate-scale',action='store_true');p.add_argument('--calibration-sha-manifest',type=Path)
    a=p.parse_args()
    print(json.dumps(prepare(a.operator_sha_manifest,a.build_metadata,a.run_id,a.profile,
                             a.candidate_scale,a.calibration_sha_manifest)))
