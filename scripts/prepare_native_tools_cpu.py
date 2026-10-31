"""Freeze one no-GPU image validation, reusing actual local data and pinned image."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from scripts.controller.approval import TRUSTED_APPROVAL,load_approval_snapshot
from scripts.controller.durable import secure_directory,write_new_bytes,write_new_json


ROOT=Path(__file__).resolve().parents[1]


def prepare(identifier: str,native_cpp_proof: Path|None=None) -> Path:
    if identifier not in ('nextgen-native-tools-cpu-20261028a','nextgen-native-tools-cpu-20261028b',
                          'nextgen-native-tools-cpu-20261028c','nextgen-native-tools-cpu-20261028d',
                          'nextgen-native-tools-cpu-20261030e','nextgen-native-tools-cpu-20261030f'):
        raise ValueError('Unreviewed bounded CPU validation identifier')
    if identifier in ('nextgen-native-tools-cpu-20261030e','nextgen-native-tools-cpu-20261030f'):
        from scripts.controller.official_tokenization import NATIVE_TOOLS_RENDER_PROTOCOL
        if native_cpp_proof is None:raise ValueError('Corrected schema-map CPU image requires an explicit new CPP proof')
        proof=json.loads(native_cpp_proof.read_text())
        rows=proof.get('comparisons',[])
        if (proof.get('all_checks_pass') is not True or len(rows)!=224
                or proof.get('native_request_parameters_map_order')!=NATIVE_TOOLS_RENDER_PROTOCOL
                or any(row['native_CPP_tokens']!=row['HF_render_copy_tokens'] or
                       row['native_CPP_tokens']!=row['worker_expected_tokens'] for row in rows)):
            raise ValueError('Current complete native schema-map CPP224 proof required')
    output=ROOT/'plans/cpu-validation'/identifier
    if output.exists():raise FileExistsError('Original CPU validation must be recovered, never prepared twice')
    authority,sha=load_approval_snapshot(TRUSTED_APPROVAL)
    secure_directory(output);assets={}
    def add(name,path=None,data=None):
        raw=path.read_bytes() if path else data
        if len(raw)>4*2**20:raise ValueError('CPU asset exceeded4MiB')
        destination=output/'assets'/name;destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        write_new_bytes(destination,raw);assets[name]={'sha256':hashlib.sha256(raw).hexdigest(),
                                                      'base64':base64.b64encode(raw).decode()}
    add('driver.py',ROOT/'scripts/native_tools_cpu_driver.py')
    for name in ('__init__.py','nextgen_tokenization.py','validate_nextgen_multiturn_cpu.py','audit_native_results.py'):
        path=ROOT/'scripts'/name
        add('pkg/scripts/'+name,path if path.exists() else None,data=b'')
    for name in ('__init__.py','durable.py','native_tools.py','evidence.py','nextgen_tools_env.py',
                 'nextgen_tool_child.py','nextgen_multiturn.py','official_tokenization.py'):
        add('pkg/scripts/controller/'+name,ROOT/'scripts/controller'/name)
    prepared=ROOT/'results/prepared/nextgen-native-tools-multiturn32-20261028b'
    for name in ('manifest.json','tasks.jsonl','pre_gold_freeze_receipt.json'):add('input/prepared/'+name,prepared/name)
    source=ROOT/'results/prepared/bfcl-multiturn-native-source-20261028a'
    add('input/tool_source/source_manifest.json',source/'source_manifest.json')
    source_manifest=json.loads((source/'source_manifest.json').read_text())
    for entry in source_manifest['files']:
        add('input/tool_source/files/'+entry['path'],source/'files'/entry['path'])
    refs=ROOT/'results/prepared/nextgen-native-tools-multiturn32-references-20261028a'
    for name in ('manifest.json','BFCL_v4_multi_turn_base.json'):add('input/references/'+name,refs/name)
    add('input/native_CPP_counts.json',native_cpp_proof or ROOT/'results/cpu-validation/native-minja-template-20261028c/CPU_native_template_proof.json')
    if sum(len(base64.b64decode(entry['base64'])) for entry in assets.values())>16*2**20:
        raise ValueError('CPU asset total exceeded16MiB')
    model=ROOT/'.deps/nextgen-official-models-20261027-a1/Qwen3.8-27B'
    manifest={'schema':'owned-native-tools-CPU-validation-v1','id':identifier,
        'state':'CPU_ONLY_NATIVE_TOOL_RUNTIME_VALIDATION_NOT_MODEL_BENCHMARK','GPU_requested':False,
        'image_id':authority['approved_image_id'],'max_seconds':600,'CPU':2,'memory_GiB':8,
        'asset_sha256':{name:entry['sha256'] for name,entry in assets.items()},
        'model_files_sha256':{name:hashlib.sha256((model/name).read_bytes()).hexdigest()
                              for name in ('config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja')},
        'approval_sha256_at_preparation':sha,'local_only_at_preparation':True}
    write_new_json(output/'manifest.json',manifest)
    write_new_bytes(output/'rpc.py',(ROOT/'scripts/native_tools_cpu_rpc.py').read_bytes())
    # Private authority snapshot is never a tracked asset or public receipt.
    write_new_json(output/'local_payload.json',{'approval':authority,'manifest':manifest,'assets':assets})
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--id',required=True)
    parser.add_argument('--native-cpp-proof',type=Path)
    args=parser.parse_args();print(prepare(args.id,args.native_cpp_proof))
