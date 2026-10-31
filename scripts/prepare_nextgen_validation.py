"""Freeze a bounded new-model BF16 service/lifecycle validation, without SSH."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from scripts.controller.approval import TRUSTED_APPROVAL, load_approval_snapshot
from scripts.controller.durable import digest, write_new_bytes, write_new_json
from scripts.controller.nextgen import KIND, MODEL, SCHEMA, SOURCE_FILES
from scripts.controller.policy import validate_request


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-nextgen-native-kv-20261027')
RUN_ID = 'nextgen-bf16-qwen38-validation-r0-20261027b'


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases() -> list[dict]:
    records = '\n'.join(f'Record {i}: color=green; count={i % 37}; status=archived.' for i in range(380))
    long_text = records + '\nRecord TARGET: secret=azure_4187.\n' + records[:2600]
    cases = [
        {'id':'arithmetic','messages':[{'role':'user','content':'Compute 13 * 17. Reply with only the integer.'}], 'expected':'221'},
        {'id':'multi_turn','messages':[{'role':'user','content':'Remember the integer x=19.'},
                                     {'role':'assistant','content':'The stored integer x is 19.'},
                                     {'role':'user','content':'What is x+23? Reply with only the integer.'}], 'expected':'42'},
        {'id':'document_retrieval','messages':[{'role':'user','content':long_text+'\nReturn only the secret in Record TARGET.'}],
         'expected':'azure_4187'},
        {'id':'code_arithmetic','messages':[{'role':'user','content':'Python: sum([n*n for n in range(5)]). Return only its integer value.'}],
         'expected':'30'},
    ]
    prefix = '\n'.join(f'Inventory row {i}: item=standard; state=available.' for i in range(150))
    for i in range(12):
        value = f'orchid_{7319+i}'
        cases.append({'id':f'concurrent_state_{i:02d}',
                      'messages':[{'role':'user','content':prefix+f'\nCurrent request secret is {value}. Return only this secret.'}],
                      'expected':value})
    return cases


def main() -> None:
    approval, approval_sha = load_approval_snapshot(TRUSTED_APPROVAL)
    config = json.loads((ROOT/'state/NEXTGEN_CONFIG.local.json').read_text())
    output = ROOT/'plans/experiments'/RUN_ID
    output.mkdir(exist_ok=True)
    input_path = output/'validation.json'
    inputs = {'schema':'native-hybrid-validation-input-v1','cases':_cases(),
              'scope':'known-answer service and cache lifecycle checks; not a quality benchmark'}
    if input_path.exists():
        if json.loads(input_path.read_text()) != inputs or (output/'package.json').exists():
            raise ValueError('Existing frozen package/input must not be replaced')
    else:
        write_new_json(input_path, inputs)
    source_commit = subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():
        raise ValueError('Freeze only a clean source commit')
    changed = subprocess.check_output(['git','-C',str(SOURCE),'diff','--name-only',config['source_commit'],source_commit],text=True).splitlines()
    if any(name.endswith(('.cpp','.cc','.cu','.h','.cuh')) for name in changed):
        raise ValueError('Initial BF16 reuses the proven binary; C++ changes need a separately built binary')
    bundle = output/'source.bundle'
    if bundle.exists():
        raise FileExistsError('Existing source bundle must not be overwritten')
    subprocess.run(['git','-C',str(SOURCE),'bundle','create',str(bundle),'HEAD','^'+config['source_commit']],check=True)
    if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip() != source_commit:
        raise ValueError('Source HEAD changed during bundle preparation')
    remote = config['remote_base']+'/agent_work/runs/'+RUN_ID
    request = {key:approval[key] for key in ('ssh_target','approved_host_fingerprint','approved_workspace')}
    request.update(schema=SCHEMA,run_id=RUN_ID,model_id=MODEL,remote_output=remote,
                   model_path=approval['read_only_model_path'],image_id=approval['approved_image_id'],
                   gpu_uuids=[approval['allowed_gpu_uuids'][0]],max_wall_seconds=1800,max_output_gib=4,
                   arms=['bf16'],replicate=0,source_repo=config['remote_new_binary_path'].split('/build/',1)[0],
                   server_binary=config['remote_new_binary_path'],server_source_commit=config['source_commit'],
                   binary_sha256=config['binary_sha256'],python_source_repo=remote+'/source_repo',
                   python_source_commit=source_commit,dataset_path=remote+'/input/validation.json',dataset_sha256=_sha(input_path),
                   model_config_sha256=_sha(ROOT/'.deps/nextgen-official-models-20261027-a1'/MODEL/'config.json'),
                   backend_source_sha256={name:_sha(SOURCE/name) for name in SOURCE_FILES},
                   graph_mode='cudagraphs',prefix_cache=True,
                   service={'max_batch':8,'max_tokens_per_batch':2048,'linear_state_slots':16,
                            'max_context':8192,'max_memory_utilization':.85},
                   operators={'bf16':{'planner':'fa3','decode':'flashinfer'}},
                   workload={'kind':KIND,'scope':'correctness_cache_lifecycle_not_performance'},
                   research_reason='First native Qwen3.8-27B text-only service, concurrent GDN lifecycle and real Graph validation',
                   model_identity_scope='Official config/index/header mapping and18 shard sizes match; only first shard body SHA verified')
    request['experiment_key']=digest({k:request[k] for k in ('model_id','python_source_commit','dataset_sha256','operators','service')})
    validate_request(request,approval)
    assets={'input/validation.json':input_path,'input/source.bundle':bundle}
    runtime = output/'runtime/kv_controller'
    runtime.mkdir(parents=True, exist_ok=False)
    for path in sorted((ROOT/'scripts/controller').glob('*.py')):
        target = runtime/path.name
        write_new_bytes(target, path.read_bytes())
        assets['runtime/kv_controller/'+path.name] = target
    for name in ('model_assets_readonly_a3.json','model_assets_full_SHA_a4.json'):
        assets['input/'+name]=ROOT/'reports/operations/nextgen-native-kv-20261027a'/name
    package={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':request,
             'approval_sha256_at_preparation':approval_sha,
             'assets':{name:{'local_path':str(path),'sha256':_sha(path)} for name,path in assets.items()}}
    write_new_json(output/'request.json',request)
    write_new_json(output/'package.json',package)


if __name__=='__main__': main()
