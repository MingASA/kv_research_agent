"""Freeze genuine chat and long-document pressure inputs without GPU execution.

No repeated filler, original-arrival claim, model predictions or new downloads.
The RAG sessions retain source questions and complete distinct documents.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.controller.durable import canonical_bytes, write_new_bytes, write_new_json
from scripts.nextgen_tokenization import OfficialJSONTokenizer


def _payload(messages: list[dict], cap: int) -> dict:
    return {'model':'Qwen3.8-27B','messages':messages,'temperature':0,'top_p':1,'max_tokens':cap,
            'stream':True,'stream_options':{'include_usage':True},
            'chat_template_kwargs':{'enable_thinking':False}}


def main() -> None:
    root=Path(__file__).resolve().parents[1]
    out=root/'results/prepared/nextgen-production-real-inputs-20261028a'
    if out.exists():raise FileExistsError('Never replace frozen production inputs')
    tokenizer=OfficialJSONTokenizer(root/'.deps/nextgen-official-models-20261027-a1/Qwen3.8-27B')
    source_path=root/'results/prepared/sharegpt-source-1024-20261026.json'
    source=json.loads(source_path.read_text())
    heldout=json.loads((root/'results/prepared/calibration-sharegpt-heldout128-20261027a2/manifest.json').read_text())
    excluded={row['source_index'] for row in heldout['selected_lineage']}
    chats=[];chat_lineage=[]
    for item in source['records']:
        if item['source_index'] in excluded:continue
        record=item['record'];messages=[]
        for turn in record.get('conversations',[]):
            role={'human':'user','user':'user','system':'system','gpt':'assistant','assistant':'assistant'}.get(turn.get('from'))
            if role is None or not isinstance(turn.get('value'),str):messages=[];break
            messages.append({'role':role,'content':turn['value']})
        positions=[index for index,message in enumerate(messages) if message['role']=='user']
        if not positions:continue
        messages=messages[:positions[-1]+1]
        payload=_payload(messages,1024);count=tokenizer.count(payload)
        if count+1024>131072:continue
        identifier='sharegpt-source-'+str(item['source_index'])
        chats.append({'session_id':identifier,'payloads':[payload]})
        chat_lineage.append({'session_id':identifier,'source_index':item['source_index'],
                             'source_record_sha256':hashlib.sha256(canonical_bytes(record)).hexdigest(),
                             'prompt_tokens':count,'payload_sha256':hashlib.sha256(canonical_bytes(payload)).hexdigest()})
        if len(chats)==144:break
    if len(chats)!=144:raise ValueError('Insufficient unchanged independent natural chat inputs')
    quality=json.loads((root/'results/prepared/nextgen-quality-real-text-20261027a/manifest.json').read_text())
    archive=root/'.deps/longbench-official-5e628be-20261027-a1/data.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=quality['official_archive_sha256']:
        raise ValueError('Pinned cached LongBench archive changed')
    config=root/'results/mirrors/quality-protocols-20261026T075846Z/files/support/inputs/longbench-scorer/LongBench/config'
    prompt_path=config/'dataset2prompt.json';cap_path=config/'dataset2maxlen.json'
    if (hashlib.sha256(prompt_path.read_bytes()).hexdigest()!=quality['official_prompt_format_sha256']
            or hashlib.sha256(cap_path.read_bytes()).hexdigest()!=quality['official_generation_caps_sha256']):
        raise ValueError('Official long-document prompt/cap source changed')
    prompts=json.loads(prompt_path.read_text());caps=json.loads(cap_path.read_text())
    selected=[];long_lineage=[];warm_long=[];seen=set()
    with zipfile.ZipFile(archive) as data:
        for task in ('narrativeqa','gov_report','qmsum'):
            documents=OrderedDict()
            for index,line in enumerate(data.read('data/'+task+'.jsonl').split(b'\n')):
                if line:
                    record=json.loads(line);context_hash=hashlib.sha256(record['context'].encode()).hexdigest()
                    documents.setdefault(context_hash,[]).append((index,record))
            for context_hash,questions in documents.items():
                if context_hash in seen:continue
                index,record=questions[0]
                message={'role':'user','content':prompts[task].format(context=record['context'],input=record['input'])}
                payload=_payload([message],int(caps[task]));count=tokenizer.count(payload)
                if not 24576<=count<=73728:continue
                identifier='longdoc-'+task+'-'+str(index)
                if len(selected)<16:
                    seen.add(context_hash)
                    if len(questions)>1:
                        second_index,second=questions[1]
                        next_payload=_payload([{'role':'user','content':prompts[task].format(context=second['context'],input=second['input'])}],int(caps[task]))
                    else:
                        second_index=index;next_payload=payload
                    selected.append({'session_id':identifier,'payloads':[payload,next_payload]})
                    long_lineage.append({'session_id':identifier,'task':task,'source_indices':[index,second_index],
                        'context_sha256':context_hash,'prompt_tokens_first':count,
                        'prompt_tokens_second':tokenizer.count(next_payload),
                        'second_query_repeats_source_request':second_index==index,
                        'complete_untruncated_document':True})
                elif len(warm_long)<2:
                    seen.add(context_hash);warm_long.append({'session_id':'warm-'+identifier,'payloads':[payload]})
    if len(selected)!=16 or len(warm_long)!=2:
        raise ValueError('Real corpus cannot supply16 distinct long documents plus2 disjoint warmup documents')
    out.mkdir(mode=0o700)
    def save(name: str, sessions: list[dict]) -> dict:
        raw=canonical_bytes({'data':sessions});write_new_bytes(out/name,raw)
        return {'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'sessions':len(sessions),
                'requests':sum(len(row['payloads']) for row in sessions)}
    files={}
    for name,sessions in [('warmup_chat16.json',chats[:16]),('chat128.json',chats[16:]),
                          ('warmup_long2.json',warm_long),
                          ('long_unique16.json',[{'session_id':row['session_id'],'payloads':row['payloads'][:1]} for row in selected]),
                          ('long_prefix16x2.json',selected)]:files[name]=save(name,sessions)
    write_new_json(out/'manifest.json',{'schema':'nextgen-real-production-inputs-v1','state':'CPU_FROZEN_INPUTS_NOT_EXPERIMENT_RESULTS',
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'chat_source_selection_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'chat_source_content_sha256':source['source_sha256'],'calibration_chat_source_ids_excluded':128,
        'chat_lineage':chat_lineage,'long_archive_sha256':quality['official_archive_sha256'],
        'long_selection':'First16 input-length-eligible unique documents in frozen narrativeqa/gov_report/qmsum source order; no label/output selection',
        'long_lineage':long_lineage,'logical_first_wave_input_tokens':sum(row['prompt_tokens_first'] for row in long_lineage),
        'all_documents_untruncated':True,'natural_stopping':True,'original_dataset_generation_caps_preserved':True,
        'token_count_protocol':'Official tokenizer.json with chat rendering only','files':files,
        'proposed_core_pressure_points':{'chat':[4,16],'long_unique':[8,16],'long_prefix':[16]},
        'limitations':['Closed-loop controlled service demand, not an original production arrival replay.',
            'Unique-document and same-document repeat modes are different workloads, not extra independent users.',
            'These inputs have no throughput, latency, maximum admission or quantized quality result yet.',
            'AIPerf warmup/session cursor and actual payload bytes require CPU wire validation before GPU use.']})


if __name__=='__main__':main()
