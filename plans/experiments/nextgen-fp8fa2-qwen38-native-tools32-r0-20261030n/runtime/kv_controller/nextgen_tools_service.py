"""Typed native-tool evaluation gates; actual generations are independently scored."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

from .durable import digest,write_new_json
from .official_tokenization import NATIVE_TOOLS_RENDER_PROTOCOL


TOOLS_KIND='native_hybrid_multiturn_tools_quality'
TOOLS_SCOPE='native_tool_multiturn_generations_with_in_memory_APIs_not_service_performance'


def validate_cpu_protocol(proof: dict) -> None:
    if (proof.get('state')!='CPU32_CONVERSATIONS_AND_NATIVE_CPP224_HISTORIES_VERIFIED_NOT_MODEL'
            or proof.get('tasks')!=32 or proof.get('user_turns')!=112
            or proof.get('real_isolated_tool_calls')!=173 or proof.get('fake_SSE_provider_invocations')!=224
            or proof.get('all_CPP_and_HF_raw_JSON_counts_match') is not True
            or proof.get('native_request_parameters_map_order')!=NATIVE_TOOLS_RENDER_PROTOCOL
            or not isinstance(proof.get('native_CPP_reference_sha256'),str)
            or re.fullmatch('[0-9a-f]{64}',proof.get('native_CPP_reference_sha256','')) is None):
        raise ValueError('Complete current native schema-map tool-loop/CPP protocol fixture is unverified')


def validate_tools_contract(request: dict) -> None:
    config=request.get('tool_evaluation',{})
    expected={'tasks':32,'user_turns':112,'client_concurrency':4,'max_steps_per_turn':10,
              'max_new_tokens':1024,'max_seconds':3600,'tool_parser':'qwen3_coder'}
    hashes={'manifest_sha256','tasks_sha256','tool_source_manifest_sha256','CPU_image_evidence_sha256',
            'CPU_protocol_rollup_sha256','quality_scores_sha256'}
    if (set(config)!=set(expected)|hashes or any(config.get(key)!=value for key,value in expected.items())
            or any(re.fullmatch('[0-9a-f]{64}',config.get(key,'')) is None for key in hashes)):
        raise ValueError('Unknown native32 tool workload/parser/demand or missing CPU/quality identity')


def verify_tools_prerequisites(request: dict,root: Path,base: Path,uid: int) -> None:
    from .deployment_quality import verify_quality_reference
    from .nextgen_tools_env import SOURCE_COMMIT,verified_upstream_files

    config=request['tool_evaluation']
    def read(path,expected_sha=None):
        if (path.resolve(strict=True)!=path or path.stat().st_uid!=uid or not path.is_file()
                or path.stat().st_size>4*2**20):raise ValueError('Unsafe tool-evaluation prerequisite')
        raw=path.read_bytes()
        if expected_sha is not None and hashlib.sha256(raw).hexdigest()!=expected_sha:
            raise ValueError('Frozen native-tool prerequisite changed: '+path.name)
        return raw
    prepared=root/'input/tools32';manifest=json.loads(read(prepared/'manifest.json',config['manifest_sha256']))
    raw=read(prepared/'tasks.jsonl',config['tasks_sha256'])
    tasks=[json.loads(line) for line in raw.split(b'\n') if line]
    if (manifest.get('schema')!='nextgen-native-BFCL-multiturn-inputs-v1' or manifest.get('tasks')!=32
            or manifest.get('user_turns')!=112 or manifest.get('source_commit')!=SOURCE_COMMIT
            or manifest.get('ground_truth_used_for_selection_or_prompt') is not False
            or manifest.get('upstream_initial_state_dictionary_order_preserved') is not True
            or manifest.get('tasks_sha256')!=config['tasks_sha256'] or len(tasks)!=32
            or len({task['id'] for task in tasks})!=32):
        raise ValueError('Original source-order real multi-turn inventory differs')
    source=root/'input/tool_source'
    read(source/'source_manifest.json',config['tool_source_manifest_sha256']);verified_upstream_files(source)
    proof=json.loads(read(root/'input/tool_CPU_protocol_rollup.json',config['CPU_protocol_rollup_sha256']))
    validate_cpu_protocol(proof)
    if proof.get('source_tasks_sha256')!=config['tasks_sha256']:
        raise ValueError('Complete native tool-loop/CPP protocol fixture is unverified')
    image=json.loads(read(root/'input/tool_CPU_image_evidence.json',config['CPU_image_evidence_sha256']))
    identifier=image.get('id')
    if identifier not in ('nextgen-native-tools-cpu-20261028a','nextgen-native-tools-cpu-20261028b',
                           'nextgen-native-tools-cpu-20261028c','nextgen-native-tools-cpu-20261028d',
                           'nextgen-native-tools-cpu-20261030e','nextgen-native-tools-cpu-20261030f'):
        raise ValueError('Unreviewed H800 CPU-image validation')
    prior=base/'agent_work/scratch'/identifier
    actual=read(prior/'CPU_acceptance.json',config['CPU_image_evidence_sha256'])
    if actual!=read(root/'input/tool_CPU_image_evidence.json'):raise ValueError('Actual CPU image proof differs')
    if (image.get('schema')!='native-tools32-H800-CPU-image-proof-v1'
            or image.get('state')!='COMPLETE_CPU_TOOL_IMAGE_VALIDATION'
            or image.get('image_id')!=request['image_id'] or image.get('GPU_requested') is not False
            or image.get('GPU_compute_devices_visible') is not False or image.get('model_weights_loaded') is not False
            or image.get('source_tasks_sha256')!=config['tasks_sha256']
            or image.get('tool_source_manifest_sha256')!=config['tool_source_manifest_sha256']
            or image.get('native_CPP_reference_sha256')!=proof['native_CPP_reference_sha256']
            or image.get('native_CPP_prompt_counts_matched')!=224):
        raise ValueError('H800 CPU tool/tokenizer image dependencies are not independently qualified')
    old=json.loads(read(prior/'PREPARED.json'))
    container=json.loads(subprocess.check_output(['docker','inspect',old['container_id']],text=True,timeout=15))[0]
    if (container['State']['Running'] or container['State']['ExitCode']!=0 or container['State']['OOMKilled']
            or container['Config']['Labels'].get('kvra.native_tools_cpu_id')!=identifier
            or container['Image']!=request['image_id'] or container['HostConfig']['Runtime']!='runc'
            or container['HostConfig'].get('DeviceRequests') or container['HostConfig'].get('Devices')
            or container['HostConfig']['NetworkMode']!='none'):
        raise ValueError('Actual CPU validation container did not finish in its reviewed isolation')
    for name in ('official_tokenization.py','nextgen_multiturn.py','nextgen_tool_child.py','nextgen_tools_env.py','native_tools.py'):
        expected=image['runtime_sources_sha256']['pkg/scripts/controller/'+name]
        read(root/'runtime/kv_controller'/name,expected)
    if set(image.get('model_files_sha256',{}))!={'config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja'}:
        raise ValueError('CPU image proof omitted approved model tokenizer identities')
    for name,sha in image['model_files_sha256'].items():
        path=Path(request['model_path'])/name
        if (Path(name).name!=name or path.resolve(strict=True)!=path or not path.is_file()
                or path.stat().st_size>32*2**20 or hashlib.sha256(path.read_bytes()).hexdigest()!=sha):
            raise ValueError('Approved read-only shared tokenizer metadata changed')
    verify_quality_reference(request,root,uid,'input/tool_quality_scores.json',config['quality_scores_sha256'])


def run_tools(request: dict,root: Path,output: Path) -> dict:
    from .native_tools import parse_native_tool_calls,parse_sse_packets
    from .nextgen_multiturn import generate_core,owned_loopback_generate,_packets
    from .official_tokenization import OfficialJSONTokenizer

    model=Path('/models')/request['model_id'];tokenizer=OfficialJSONTokenizer(model)
    prepared=root/'input/tools32'
    tasks=[json.loads(line) for line in (prepared/'tasks.jsonl').read_bytes().split(b'\n') if line]
    addition=next(tool for task in tasks for tool in task['tools'] if tool['function']['name']=='add')
    payload={'model':request['model_id'],'messages':[{'role':'user','content':'Use the add tool to calculate 17 + 25.'}],
        'tools':[addition],'tool_choice':'auto','temperature':0,'max_tokens':128,'stream':True,
        'stream_options':{'include_usage':True},'chat_template_kwargs':{'enable_thinking':False}}
    raw=owned_loopback_generate(payload);write_new_json(output/'tool_API_admission_raw.json',{'request':payload,**raw})
    if raw.get('status')!=200 or raw.get('error') is not None:raise ValueError('Actual owned tool API admission failed')
    parsed=parse_sse_packets(_packets(raw['raw_SSE']))
    calls=parse_native_tool_calls(parsed['tool_calls'])
    if calls!=[{'add':{'a':17,'b':25}}] or parsed['usage']['prompt_tokens']!=tokenizer.count(payload):
        raise ValueError('Actual Qwen3.8 native XML parser/typed arguments/tokenization is not verified')
    write_new_json(output/'tool_API_admission.json',{'actual_native_function_API_verified':True,'parsed':parsed,
                    'not_in_scored32_tasks':True,'request_sha256':digest(payload)})
    summary=generate_core(prepared,root/'input/tool_source',output/'tool_generations',
                          generate=owned_loopback_generate,count_tokens=tokenizer.count)
    if summary['tasks_attempted']!=32 or summary['all_actual_API_contracts_verified'] is not True:
        raise ValueError('Incomplete/invalid actual native-tool API records; preserve all raw attempts')
    return {'classification':'actual_native_tool32_generations_complete_independent_task_scoring_pending',
            'actual_owned_model_HTTP_provider':True,'synthetic_provider_or_gold_reference_loaded':False,
            'generation':summary,'model_task_accuracy_scored':False,'service_performance_measured':False}
