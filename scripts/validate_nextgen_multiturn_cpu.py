"""Exercise the real32-task conversation loop with an explicit CPU-only fixture.

The fixture uses frozen official reference calls after prompt selection. It is
not a model, does not contact a server, and produces no accuracy/performance
claim. Actual bounded tool children execute the declared in-memory APIs.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import threading
import time

from scripts.controller.durable import write_new_json
from scripts.controller.nextgen_multiturn import generate_core
from scripts.controller.nextgen_tools_env import ToolFactory
from scripts.nextgen_tokenization import OfficialJSONTokenizer


def validate(prepared: Path,source: Path,references: Path,tokenizer_path: Path,output: Path) -> dict:
    tasks=[json.loads(line) for line in (prepared/'tasks.jsonl').read_bytes().split(b'\n') if line]
    freeze=json.loads((prepared/'pre_gold_freeze_receipt.json').read_text())
    refs_manifest=json.loads((references/'manifest.json').read_text())
    raw=(references/'BFCL_v4_multi_turn_base.json').read_bytes()
    if (hashlib.sha256((prepared/'tasks.jsonl').read_bytes()).hexdigest()!=freeze['tasks_sha256']
            or refs_manifest['pre_gold_tasks_sha256']!=freeze['tasks_sha256']
            or hashlib.sha256(raw).hexdigest()!=refs_manifest['reference_sha256']):
        raise ValueError('CPU fixture references do not bind the pre-gold frozen prompts')
    gold={row['id']:row['ground_truth'] for row in (json.loads(line) for line in raw.split(b'\n') if line)}
    factory=ToolFactory(source);by_id={task['id']:task for task in tasks}
    reference_states={task['id']:factory.environment(task) for task in tasks}
    tokenizer=OfficialJSONTokenizer(tokenizer_path);lock=threading.Lock();responses={};observations=[]
    def generate(payload,*,task_id,turn,step):
        task=by_id[task_id]
        users=[row for row in payload['messages'] if row['role']=='user']
        expected=[row for rows in task['original_user_turns'][:turn+1] for row in rows]
        if users!=expected or set(payload)!=set(task['initial_payload']):
            raise ValueError('Later questions or simulator state leaked into the original API payload')
        if payload['tools']!=task['tools']:raise ValueError('Native tool documentation changed during a session')
        key=(task_id,turn);calls=[]
        if step==0:
            if payload['messages'][-1]['role']!='user':raise ValueError('First assistant step missed the actual user turn')
            expected_results=[]
            with lock:
                env=reference_states[task_id]
                for index,expression in enumerate(gold[task_id][turn]):
                    node=ast.parse(expression,mode='eval').body
                    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name):
                        raise ValueError('CPU reference fixture requires a declared literal call')
                    method=env.methods[node.func.id]
                    args=[ast.literal_eval(value) for value in node.args]
                    kwargs={row.arg:ast.literal_eval(row.value) for row in node.keywords}
                    bound=inspect.signature(method).bind(*args,**kwargs)
                    calls.append({'index':index,'id':f'CPU_fixture_{task_id}_{turn}_{index}','type':'function',
                        'function':{'name':node.func.id,'arguments':json.dumps(dict(bound.arguments),ensure_ascii=False)}})
                    reply=env.reference_call(expression)
                    if not reply['dispatch_valid'] or reply['tool_reported_error']:
                        raise ValueError('CPU fixture reference did not execute cleanly')
                    expected_results.append(reply['result'])
                responses[key]=expected_results
            delta={'role':'assistant','tool_calls':calls};finish='tool_calls'
        elif step==1:
            replies=[]
            for row in reversed(payload['messages']):
                if row['role']!='tool':break
                replies.append(row['content'])
            if list(reversed(replies))!=responses[key]:
                raise ValueError('Actual isolated tool responses differ from independent reference execution')
            delta={'role':'assistant','content':'x'};finish='stop'
        else:raise ValueError('CPU fixture should finish after one tool-call batch and final response')
        prompt=tokenizer.count(payload);usage={'prompt_tokens':prompt,'completion_tokens':1,'total_tokens':prompt+1}
        chunks=[{'choices':[{'index':0,'delta':delta,'finish_reason':None}]},
                {'choices':[{'index':0,'delta':{},'finish_reason':finish}],'usage':usage}]
        now=time.time_ns()
        observations.append({'task_id':task_id,'turn':turn,'step':step,'prompt_tokens':prompt})
        return {'status':200,'error':None,'start_epoch_ns':now,'end_epoch_ns':time.time_ns(),
                'request_wire_sha256':hashlib.sha256(json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
                'raw_SSE':['data: '+json.dumps(chunk,ensure_ascii=False) for chunk in chunks]+['data: [DONE]']}
    summary=generate_core(prepared,source,output,generate=generate,count_tokens=tokenizer.count)
    checks={'all32_original_tasks_attempted':summary['tasks_attempted']==32,
            'all112_original_user_turns_completed':summary['user_turns_completed']==112,
            'all32_conversations_completed':summary['conversations_completed']==32,
            'all173_real_in_memory_tool_calls':summary['tool_calls']==173,
            'all224_fixture_SSE_API_records_verified':summary['model_API_requests']==224
                and summary['all_actual_API_contracts_verified'] is True and len(observations)==224,
            'all_tool_replies_match_independent_reference':len(responses)==112}
    proof={'schema':'nextgen-native32-actual-CPU-loop-fixture-v1','classification':'CPU_CONVERSATION_PROTOCOL_FIXTURE_NOT_MODEL_GENERATIONS',
        'checks':checks,'all_checks_pass':all(checks.values()),'frozen_tasks_sha256':freeze['tasks_sha256'],
        'reference_sha256':refs_manifest['reference_sha256'],'generation_summary_sha256':hashlib.sha256((output/'generation_summary.json').read_bytes()).hexdigest(),
        'observations':observations,'actual_bounded_CPU_tool_children':32,'real_tool_calls':173,
        'fake_SSE_provider_invocations':224,'HTTP_requests_or_GPU_or_model_generation':False,
        'native_tool_history_and_usage_contract_fixture_verified':True,'model_quality_or_service_performance_validated':False,
        'reference_based_fixture_is_not_a_model_prediction':True,'H800_runtime_validated':False}
    write_new_json(output/'CPU_protocol_proof.json',proof);return proof


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','source','references','tokenizer','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=validate(args.prepared,args.source,args.references,args.tokenizer,args.out)
    print(json.dumps({key:result[key] for key in ('classification','checks','all_checks_pass','actual_bounded_CPU_tool_children',
                                               'real_tool_calls','fake_SSE_provider_invocations')}))
    if not result['all_checks_pass']:raise SystemExit(2)


if __name__=='__main__':main()
