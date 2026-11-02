"""One causal followup for the preserved FA3 dense-reference deviation."""

import hashlib
import json
from pathlib import Path
import shutil

from scripts.controller.durable import digest
from scripts.controller.policy import validate_request
from scripts.controller.upstream_native import DIAG_KIND
from scripts.upstream_package_runtime import freeze_runtime

ROOT=Path(__file__).resolve().parents[1]
RUN='upstream-C-fp8-direct-diagnosis-r0-20261102b'


def prepare():
    baseline=ROOT/'plans/experiments/upstream-C-operators-qualification-r0-20261102a/package.json'
    package=json.loads(baseline.read_bytes());request=dict(package['request'])
    terminal=json.loads((ROOT/'reports/operations/upstream-ac-20261102a/C_OPERATORS_TERMINAL-a1.json').read_bytes())['files']['result.json']
    result=terminal['value']
    if (result['status']!='COMPLETE' or result['gpu_release_verified'] is not True
            or result['live_owned_containers'] or result['run_id']!=request['run_id']):
        raise ValueError('Diagnosis needs the actual released original candidate evaluation')
    plan=ROOT/'plans/experiments'/RUN;plan.mkdir();files={}
    for name in ('input/validation.json','input/gdn_dependency_CPU_import.json','input/source.bundle'):
        target=plan/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(package['assets'][name]['local_path'],target);files[name]=target
    reference={'run_id':request['run_id'],'result_sha256':terminal['sha256'],
               'cause':'FA3prefill_dense_atol.02_rtol.02_failed_before_XQA_was_reached'}
    root=request['approved_workspace']+'/agent_work/runs/'+RUN
    request.update(run_id=RUN,remote_output=root,python_source_repo=root+'/source_repo',
        dataset_path=root+'/input/validation.json',max_wall_seconds=2400,operator_diagnosis_reference=reference,
        research_reason='Preserve original dense.02failure; isolate independent Torch quantization/directFI conformance and actual changed-input XQA Graph',
        workload={'kind':DIAG_KIND,'scope':'independent_quant_direct_FA3_XQA_conformance_preserve_dense_failure'},
        operators={'fp8_e4m3':{'planner':'paired_direct_FlashInfer','decode':'paired_direct_XQA'}},
        experiment_key=digest({'kind':DIAG_KIND,'source':request['server_source_commit'],'reference':reference,
                               'GPU':request['gpu_uuids'],'direct_tolerance':[.001,.001],'dense_tolerance':[.02,.02]}))
    files.update(freeze_runtime(plan))
    approval=json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    validate_request(request,approval)
    assets={name:{'local_path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            for name,path in files.items()}
    frozen={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,
            'request_sha256':digest(request),'assets':assets}
    (plan/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    (plan/'package.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({'run_id':RUN,'package':str(plan/'package.json'),'max_seconds':2400,'GPU':request['gpu_uuids'],'dispatched':False}))


if __name__=='__main__':prepare()
