"""Reproduce local C++/HF dynamic-history counts using pinned native frontend code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.controller.durable import write_new_bytes,write_new_json
from scripts.nextgen_tokenization import OfficialJSONTokenizer
from scripts.controller.official_tokenization import NATIVE_TOOLS_RENDER_PROTOCOL


def validate(source_repo: Path,minja_repo: Path,json_include: Path,model: Path,conversations: Path,output: Path) -> dict:
    if output.exists():raise FileExistsError('Use a fresh CPU template-proof directory')
    commit=subprocess.check_output(['git','-C',str(source_repo),'rev-parse','HEAD'],text=True).strip()
    link=subprocess.check_output(['git','-C',str(source_repo),'ls-tree','HEAD','third_party/minja'],text=True).split()[2]
    actual=subprocess.check_output(['git','-C',str(minja_repo),'rev-parse','HEAD'],text=True).strip()
    if actual!=link or subprocess.check_output(['git','--no-optional-locks','-C',str(minja_repo),'status','--porcelain']):
        raise ValueError('Minja headers differ from the exact clean model-source gitlink')
    name='xllm/core/framework/chat_template/jinja_chat_template.cpp';path=source_repo/name;raw=path.read_bytes()
    if subprocess.check_output(['git','-C',str(source_repo),'show',commit+':'+name])!=raw:
        raise ValueError('Native template source has uncommitted changes')
    text=raw.decode();start=text.index('void replace_undefined_tests(');end=text.index('\n}  // namespace',start)
    normalization=text[start:end]
    converter_name='xllm/core/framework/request/request_params.cpp'
    converter_raw=(source_repo/converter_name).read_bytes()
    if subprocess.check_output(['git','-C',str(source_repo),'show',commit+':'+converter_name])!=converter_raw:
        raise ValueError('Native protobuf conversion source has uncommitted changes')
    converter=converter_raw.decode()
    converter=converter[converter.index('nlohmann::json proto_value_to_json('):
                        converter.index('std::vector<xllm::JsonTool> parse_tools_from_proto(')]
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    write_new_bytes(output/'native_template_normalization_snapshot.h',normalization.encode())
    write_new_bytes(output/'native_request_conversion_snapshot.h',converter.encode())
    probe=Path(__file__).with_name('native_template_probe.cpp');binary=output/'native_template_probe'
    command=['/usr/bin/g++','-std=c++17','-O0','-I'+str(output),'-I'+str(minja_repo/'include'),
             '-I'+str(json_include),str(probe),str(json_include.parent/'lib/libprotobuf.a'),
             '-pthread','-o',str(binary)]
    write_new_json(output/'compile_command.json',command)
    with (output/'compile.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    rows=[]
    for path in sorted(conversations.glob('*/requests.jsonl')):
        rows.extend(json.loads(line) for line in path.read_bytes().split(b'\n') if line)
    if len(rows)!=224:raise ValueError('Require the complete independently inspectable32-task CPU loop fixture')
    wire=''.join(json.dumps(row['request'],ensure_ascii=False)+'\n' for row in rows)
    result=subprocess.run([str(binary),str(model/'chat_template.jinja'),str(model/'tokenizer_config.json')],
                          input=wire,text=True,capture_output=True,timeout=120)
    write_new_bytes(output/'rendered.jsonl',result.stdout.encode());write_new_bytes(output/'render.stderr.log',result.stderr.encode())
    if result.returncode:raise ValueError('Actual native C++ template failed; original CPU output retained')
    rendered=[json.loads(line) for line in result.stdout.splitlines()]
    if len(rendered)!=len(rows):raise ValueError('Native C++ did not render every original history')
    tokenizer=OfficialJSONTokenizer(model);comparisons=[]
    for row,native in zip(rows,rendered):
        count=len(tokenizer.encoder.encode(native['rendered_text'],add_special_tokens=True).ids)
        arguments=[call['function']['arguments'] for message in row['request']['messages'] for call in message.get('tool_calls',[])]
        comparisons.append({'task_id':row['task_id'],'turn':row['turn'],'step':row['step'],
            'native_CPP_tokens':count,'HF_render_copy_tokens':tokenizer.count(row['request']),
            'worker_expected_tokens':row['prompt_tokens_CPU_expected'],
            'HTTP_arguments_still_strings':all(isinstance(value,str) for value in arguments)})
    checks={'all224_prompt_counts_equal':all(row['native_CPP_tokens']==row['HF_render_copy_tokens']==row['worker_expected_tokens']
                                           for row in comparisons),
            'actual_HTTP_history_unchanged':all(row['HTTP_arguments_still_strings'] for row in comparisons)}
    proof={'schema':'nextgen-pinned-native-template-CPU-proof-v1','checks':checks,'all_checks_pass':all(checks.values()),
        'native_request_parameters_map_order':NATIVE_TOOLS_RENDER_PROTOCOL,
        'verbatim_protobuf_converter_source_sha256':hashlib.sha256(converter_raw).hexdigest(),
        'verbatim_protobuf_converter_snapshot_sha256':hashlib.sha256(converter.encode()).hexdigest(),
        'protobuf_library_sha256':hashlib.sha256((json_include.parent/'lib/libprotobuf.a').read_bytes()).hexdigest(),
        'source_commit':commit,'minja_gitlink':link,'native_template_source_sha256':hashlib.sha256(raw).hexdigest(),
        'verbatim_normalization_sha256':hashlib.sha256(normalization.encode()).hexdigest(),
        'probe_source_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),
        'probe_binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'comparisons':comparisons,
        'actual_model_or_H800_service_or_GPU_validated':False}
    write_new_json(output/'CPU_native_template_proof.json',proof);return proof


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-repo','minja-repo','json-include','tokenizer','conversations','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=validate(args.source_repo,args.minja_repo,args.json_include,args.tokenizer,args.conversations,args.out)
    print(json.dumps({'checks':result['checks'],'all_checks_pass':result['all_checks_pass'],'histories':len(result['comparisons'])}))
    if not result['all_checks_pass']:raise SystemExit(2)


if __name__=='__main__':main()
