"""Freeze genuine BFCL multi-turn tasks and Qwen3.8 native-tool schemas, before gold.

Only task data and official schema conversion are used here. Tool execution,
model generation, quality scoring and production-performance claims are absent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

from scripts.controller.durable import digest,secure_directory,write_new_bytes,write_new_json
from scripts.controller.nextgen_tools_env import SCHEMAS,verified_upstream_files
from scripts.nextgen_tokenization import OfficialJSONTokenizer
from scripts.prepare_tool_calls import PREFIX,SOURCE_COMMIT,_load_sources,_official_adapter


DOMAINS=('GorillaFileSystem','VehicleControlAPI','TradingBot','TravelAPI')
MODEL='Qwen3.8-27B'


def prepare(source: Path, tools_source: Path, tokenizer_dir: Path, output: Path) -> dict:
    if output.exists():raise FileExistsError('Use a new immutable native multi-turn input directory')
    original=_load_sources(source);files,upstream=verified_upstream_files(tools_source)
    convert,adapter=_official_adapter(original)
    source_name=PREFIX+'data/BFCL_v4_multi_turn_base.json';raw=original[source_name]
    rows=[(line,json.loads(line)) for line in raw.split(b'\n') if line]
    if len(rows)!=200:raise ValueError('Original200-entry BFCL base inventory differs')
    tokenizer=OfficialJSONTokenizer(tokenizer_dir)
    selected=[];inventory={name:0 for name in DOMAINS};all_ids=set();class_counts={}
    for index,(line,row) in enumerate(rows):
        if row['id'] in all_ids:raise ValueError('Duplicate original task ID')
        all_ids.add(row['id']);domain=[name for name in DOMAINS if name in row['involved_classes']]
        if len(domain)!=1:raise ValueError('Primary-domain partition changed; review selection before gold')
        primary=domain[0]
        if inventory[primary]>=8:continue
        inventory[primary]+=1
        if not 1<=len(row['question'])<=8 or any(not turn or any(message.get('role')!='user'
                or not isinstance(message.get('content'),str) for message in turn) for turn in row['question']):
            raise ValueError('Original user-turn lifecycle is outside the reviewed bounded scope')
        functions=[];mapping={}
        excluded=set(row.get('excluded_function',[]))
        for name in row['involved_classes']:
            if name not in SCHEMAS:raise ValueError('Unknown tool class; no implicit runtime import')
            class_counts[name]=class_counts.get(name,0)+1
            for value in files['data/multi_turn_func_doc/'+SCHEMAS[name]+'.json'].split(b'\n'):
                if not value:continue
                function=json.loads(value)
                if function['name'] in excluded:continue
                if function['name'] in mapping:raise ValueError('Ambiguous function name across original task classes')
                mapping[function['name']]=name;functions.append(function)
        tools=convert(functions)
        if any(re.fullmatch('[A-Za-z0-9_-]{1,64}',tool['function']['name']) is None for tool in tools):
            raise ValueError('Official tool-name conversion is not native-API compatible')
        messages=copy.deepcopy(row['question'][0])
        payload={'model':MODEL,'messages':messages,'tools':tools,'tool_choice':'auto',
                 'temperature':0,'max_tokens':1024,'stream':True,'stream_options':{'include_usage':True},
                 'chat_template_kwargs':{'enable_thinking':False}}
        tokens=tokenizer.count(payload)
        if not 0<tokens<=131072-1024:raise ValueError('Original tool documentation exceeds actual context budget')
        # A direct render of the real official XML tool template independently
        # checks that CPU counting includes the tool documentation, not just chat.
        rendered=tokenizer.renderer.apply_chat_template(messages,tools=tools,tokenize=False,
                                                        add_generation_prompt=True,enable_thinking=False)
        if (tokens!=len(tokenizer.encoder.encode(rendered,add_special_tokens=True).ids)
                or '<function=example_function_name>' not in rendered or '<tools>' not in rendered
                or tokens<=tokenizer.count({key:value for key,value in payload.items() if key!='tools'})):
            raise ValueError('Official XML tool rendering/counting is missing or inconsistent')
        selected.append({'id':row['id'],'primary_domain':primary,'original_user_turns':copy.deepcopy(row['question']),
            'initial_config':copy.deepcopy(row['initial_config']),'involved_classes':list(row['involved_classes']),
            'functions':functions,'tools':tools,'function_classes':mapping,'initial_payload':payload,
            'initial_prompt_tokens':tokens,'source_line_zero_based':index,
            'source_line_sha256_without_LF':hashlib.sha256(line).hexdigest(),
            'original_source_record_JSON':line.decode('utf-8'),
            'initial_payload_sha256':digest(payload)})
    if len(selected)!=32 or any(count!=8 for count in inventory.values()) or set(class_counts)!=set(SCHEMAS):
        raise ValueError('Expected32 original multi-turn tasks,8 per primary domain,covering all8 tool classes')
    # Preserve upstream dict insertion order: simulated listing outputs and
    # seeded state initialization can depend on it. Canonical sorted-key hashing
    # remains suitable for HTTP identities, not for rewriting simulator state.
    secure_directory(output);write_new_bytes(output/'tasks.jsonl',b''.join(
        (json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode() for row in selected))
    manifest={'schema':'nextgen-native-BFCL-multiturn-inputs-v1','classification':'confirmed_CPU_input_preparation_only',
        'source_repo':'https://github.com/ShishirPatil/gorilla','source_commit':SOURCE_COMMIT,
        'source_data_sha256':hashlib.sha256(raw).hexdigest(),'source_data_entries':200,
        'selection':'First8 original entries in source order per primary domain;no gold or prediction used',
        'domain_task_counts':inventory,'involved_class_task_counts':class_counts,'tasks':len(selected),
        'user_turns':sum(len(row['original_user_turns']) for row in selected),
        'multi_user_turn_tasks':sum(len(row['original_user_turns'])>=2 for row in selected),
        'single_user_turn_controls_retained':sum(len(row['original_user_turns'])==1 for row in selected),
        'tasks_sha256':hashlib.sha256((output/'tasks.jsonl').read_bytes()).hexdigest(),
        'source_manifest_sha256':hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest(),
        'tool_source_manifest_sha256':hashlib.sha256((tools_source/'source_manifest.json').read_bytes()).hexdigest(),
        'official_tool_schema_conversion':adapter,'tool_parser':'qwen3_coder','model_id':MODEL,
        'model_template_sha256':hashlib.sha256((tokenizer_dir/'chat_template.jinja').read_bytes()).hexdigest(),
        'tokenizer_json_sha256':hashlib.sha256((tokenizer_dir/'tokenizer.json').read_bytes()).hexdigest(),
        'initial_prompt_tokens':{row['id']:row['initial_prompt_tokens'] for row in selected},
        'model_receives_initial_state_or_future_questions_or_reference_calls':False,
        'original_user_messages_preserved':True,'ground_truth_used_for_selection_or_prompt':False,
        'upstream_initial_state_dictionary_order_preserved':True,
        'no_tool_or_model_executed':True,'GPU_used':False,'benchmark_overall_BFCL_score_claim':False,
        'limits':['Limited32-task base subset,not the complete BFCL leaderboard or a production arrival trace.',
                  'All8 tool APIs are candidates for reviewed in-memory simulation only;actual execution and evaluator still require validation.',
                  'Dynamic assistant/tool histories and actual server usage must be audited after model execution.']}
    write_new_json(output/'manifest.json',manifest);return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','tools-source','tokenizer','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=prepare(args.source,args.tools_source,args.tokenizer,args.out)
    print(json.dumps({key:result[key] for key in ('classification','tasks','user_turns','domain_task_counts','tasks_sha256')}))


if __name__=='__main__':main()
