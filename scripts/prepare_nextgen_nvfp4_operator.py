"""Freeze the native SM90 NVFP4 page/operator qualification, without launching."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.controller.approval import TRUSTED_APPROVAL, load_approval_snapshot
from scripts.controller.durable import write_new_bytes, write_new_json
from scripts.controller.nextgen import PAGES_KIND, PAGES_SOURCE_FILES
from scripts.controller.policy import validate_request


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-nextgen-native-kv-20261027')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True):
        raise ValueError('Operator source worktree must be committed and clean')
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    original = json.loads((root/'plans/experiments/nextgen-bf16-qwen38-validation-r2-20261027e/package.json').read_text())
    request = copy.deepcopy(original['request'])
    run_id = 'nextgen-nvfp4-pages-sm90-r0-20261027h'
    remote = request['approved_workspace']+'/agent_work/runs/'+run_id
    request.update(run_id=run_id, remote_output=remote, arms=['nvfp4'],replicate=0, max_wall_seconds=1800,
                   startup_seconds=900, max_output_gib=2, gpu_uuids=['GPU-0c628a64-5f31-3501-fa3c-723f5b69254a'],
                   python_source_repo=remote+'/source_repo', python_source_commit=commit,
                   dataset_path=remote+'/input/validation.json', operators={'nvfp4':{'planner':'fa2','decode':'flashinfer'}},
                   workload={'kind':PAGES_KIND,'scope':'SM90_native_compressed_pages_no_model_or_performance'},
                   server_binary_execution='none',
                   research_reason='Qualify mature FlashInfer direct packed NVFP4 on actualSM90/D256/GQA6: native writer, paged prefill/chunked/decode, changed-input Graph, resident-byte reference. No model, CXX allocator, quality or serving performance claim.')
    request.pop('startup_recovery',None)
    request.pop('environment_recovery',None)
    request['backend_source_sha256']={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in PAGES_SOURCE_FILES}
    dataset={'schema':'native-nvfp4-pages-input-v1','heads':24,'kv_heads':4,'head_dim':256,
             'page_size':64,'contexts':[129,130],'seed':20261027}
    output=root/'plans/experiments'/run_id
    output.mkdir(exist_ok=False)
    assets={}

    def add(name: str,data: bytes) -> None:
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);write_new_bytes(path,data)
        assets[name]={'local_path':str(path),'sha256':hashlib.sha256(data).hexdigest()}

    add('input/validation.json',(json.dumps(dataset,sort_keys=True)+'\n').encode())
    request['dataset_sha256']=assets['input/validation.json']['sha256']
    bundle=output/'input/source.bundle'
    subprocess.run(['git','-C',str(source),'bundle','create',str(bundle),'HEAD','^ceff00bbd38054d2b6b4aa44513d773f2bd174b2'],check=True,capture_output=True)
    assets['input/source.bundle']={'local_path':str(bundle),'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}
    for path in sorted((root/'scripts/controller').glob('*.py')):
        add('runtime/kv_controller/'+path.name,path.read_bytes())
    request['experiment_key']=hashlib.sha256(json.dumps({key:request[key] for key in
        ('python_source_commit','backend_source_sha256','dataset_sha256','image_id','operators','workload','graph_mode')},sort_keys=True).encode()).hexdigest()
    approval,approval_sha=load_approval_snapshot(TRUSTED_APPROVAL)
    validate_request(request,approval)
    write_new_json(output/'request.json',request)
    write_new_json(output/'package.json',{'schema':'kv-frozen-local-package-v1','classification':'proposal',
                                         'approval_sha256_at_preparation':approval_sha,'request':request,'assets':assets,
                                         'server_binary_provenance_checked_but_not_executed':True,'model_loaded':False})


if __name__=='__main__':main()
