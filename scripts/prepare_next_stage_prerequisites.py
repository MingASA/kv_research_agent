"""Freeze independent FP8 operator/calibration inputs; never dispatch a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from scripts.controller.cli import validate_frozen_policy
from scripts.controller.durable import digest
from scripts.controller.native_next_stage import KIND, FP8_KIND, CALIB_KIND, EXTRA_SOURCES
from scripts.controller.upstream_native import SOURCE_FILES
from scripts.controller.policy import validate_request
from scripts.upstream_package_runtime import freeze_runtime, attach_graph_observer


ROOT=Path(__file__).resolve().parents[1]


def prepare(mode: str) -> dict:
    mx=mode=='mxfp4'
    operator=mode in ('operator','mxfp4')
    repo=Path('/home/mingasa/projects/xllm-mxfp4-kv-20261105a' if mx else
              '/home/mingasa/projects/xllm-fp8-'+('opt' if operator else 'diagnostics')+'-20261105a')
    run=('upstream-mxfp4-operator-r2-20261105c' if mx else
         'upstream-fp8-'+('opt-operator' if operator else 'independent-calibration')+'-r0-20261105a')
    plan=ROOT/'plans/experiments'/run;plan.mkdir();(plan/'input').mkdir()
    previous=json.loads((ROOT/'plans/experiments/upstream-C-operators-qualification-r0-20261102a/package.json').read_bytes())
    request=dict(previous['request'])
    sha=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise ValueError('Prerequisite source must be committed and clean')
    files={}
    for name in ('validation.json','gdn_dependency_CPU_import.json'):
        path=plan/'input'/name;shutil.copyfile(previous['assets']['input/'+name]['local_path'],path);files['input/'+name]=path
    bundle=plan/'input/source.bundle'
    subprocess.run(['git','-C',str(repo),'bundle','create',str(bundle),'HEAD','^da7c6446ff4361c8060633b04d6bfb77e6f737ff'],check=True)
    files['input/source.bundle']=bundle
    remote=request['approved_workspace']+'/agent_work/runs/'+run
    extras=EXTRA_SOURCES if mx else ('xllm/python/attention/'+('mixed_native_kv.py' if operator else 'fp8_numeric_diagnostics.py'),)
    request.update(run_id=run,remote_output=remote,python_source_repo=remote+'/source_repo',
        python_source_commit=sha,dataset_path=remote+'/input/validation.json',
        arms=['mxfp4' if mx else 'fp8_e4m3' if operator else 'bf16'],max_wall_seconds=3600,
        gpu_uuids=['GPU-47211491-ba0d-78bf-5c84-f968c0648213' if operator else 'GPU-0c628a64-5f31-3501-fa3c-723f5b69254a'],
        backend_source_sha256={name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in (*SOURCE_FILES,*extras)},
        server_binary_execution='none' if operator else 'required',
        workload={'kind':KIND if mx else FP8_KIND if operator else CALIB_KIND,
                  'scope':'OCP_MXFP4_SM90_D256_bitwise_paged_Graph_gate' if mx else
                          'strided_query_and_mixed_XQA_finite_operator_gate' if operator else 'independent_BF16_QKV_ranges_clipping_query_error_136'},
        operators={'mxfp4':{'planner':'private_pinned_FA2_E8M0','decode':'private_pinned_FA2_E8M0'}} if mx else
                  {'fp8_e4m3':{'planner':'FA3_query_fusion_and_mixed_split','decode':'XQA'}} if operator else
                  {'bf16':{'planner':'official_auto','decode':'official_auto'}},
        research_reason='Finite same-base FP8 optimization prerequisites: independent calibration and real operator conformance before scored quality or service claims',
        experiment_key=digest({'mode':mode,'source':sha,'frozen_protocol':'20261105-prerequisites-v1'}))
    if mx:
        request['compiled_backend_source_sha256']={name:sha for name,sha in previous['request']['backend_source_sha256'].items() if not name.endswith('.py')}
    elif operator:
        path=plan/'input/query_reference_20261102.py'
        raw=subprocess.check_output(['git','-C',str(repo),'show','da7c6446ff4361c8060633b04d6bfb77e6f737ff:xllm/python/attention/query_quantization.py'])
        path.write_bytes(raw);files['input/'+path.name]=path
    else:
        path=plan/'input/calibration.json';shutil.copyfile(ROOT/'results/prepared/fp8-independent-calibration-20261105a/calibration.json',path)
        files['input/calibration.json']=path
        request.update(calibration_count=136,calibration_concurrency=16,
                       calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        files.update(attach_graph_observer(plan,request))
    approval=json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request,approval)
    files.update(freeze_runtime(plan))
    package={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,
             'request_sha256':digest(request),'assets':{name:{'local_path':str(path),
              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in files.items()}}
    path=plan/'package.json';path.write_text(json.dumps(package,indent=2)+'\n')
    (plan/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    validate_frozen_policy(path,request,approval)
    return {'package':str(path),'source':sha,'actual_frozen_policy_passed':True,'GPU_dispatched':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['operator','calibration','mxfp4'],required=True)
    print(json.dumps(prepare(parser.parse_args().mode)))
