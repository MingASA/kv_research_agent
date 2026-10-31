"""Genuine native-tool conversation generation with isolated in-memory tools.

All source user turns stay ordered; later questions are revealed only after the
current assistant/tool loop finishes. No gold calls/answers are consumed here.
Quality scoring belongs to a separate independent CPU pass.
"""

from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .durable import digest,write_new_json
from .native_tools import parse_native_tool_calls,parse_sse_packets
from .nextgen_tool_child import ToolChild


def _append(stream,row: dict) -> None:
    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    stream.flush();os.fsync(stream.fileno())


def _packets(lines: list[str]) -> list[dict]:
    return [{'name':'data','value':line[5:].strip()} for line in lines if line.startswith('data:')]


def owned_loopback_generate(payload: dict,**metadata) -> dict:
    """Call only the already admitted server in its private network namespace."""
    started=time.time_ns();raw=[];wire=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
    result={'request_wire_sha256':hashlib.sha256(wire).hexdigest(),'start_epoch_ns':started,
            'raw_SSE':raw,'status':None,'error':None}
    request=urllib.request.Request('http://127.0.0.1:19001/v1/chat/completions',data=wire,
                                  headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=600) as response:
            result['status']=response.status;total=0
            while True:
                line=response.readline(256*1024+1)
                if not line:break
                total+=len(line)
                if len(line)>256*1024 or total>4*2**20:raise ValueError('Native-tool SSE exceeded bounded framing')
                raw.append(line.decode('utf-8').rstrip('\r\n'))
    except urllib.error.HTTPError as error:
        result.update(status=error.code,error=error.read(64*1024).decode('utf-8',errors='replace'))
    except (OSError,ValueError) as error:result['error']=type(error).__name__+': '+str(error)[:1000]
    result['end_epoch_ns']=time.time_ns();return result


def generate_session(task: dict,*,source: Path,output: Path,generate,count_tokens,
                     deadline_ns: int,max_steps_per_turn: int = 10) -> dict:
    """Persist raw replies before parsing/dispatch; never reset a failed tool state."""
    if max_steps_per_turn!=10:raise ValueError('Unreviewed per-turn tool-step budget')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    messages=[];records=0;verified_requests=0;calls=0;finished_turns=0;empty_final_turns=0;failure=None
    stage='tool_environment_initialization';tool_ready=False
    status='INCOMPLETE'
    with (output/'requests.jsonl').open('x') as requests,(output/'tool_results.jsonl').open('x') as results:
        try:
            with ToolChild(source,task,output/'tool_child.stderr.log') as tools:
                write_new_json(output/'tool_child_ready.json',tools.ready)
                tool_ready=True
                for turn,user_messages in enumerate(task['original_user_turns']):
                    messages.extend(copy.deepcopy(user_messages))
                    for step in range(max_steps_per_turn):
                        stage='request_budget_or_tokenization'
                        if time.monotonic_ns()>=deadline_ns:raise TimeoutError('Native-tool generation deadline reached')
                        payload=copy.deepcopy(task['initial_payload']);payload['messages']=copy.deepcopy(messages)
                        prompt=count_tokens(payload)
                        if type(prompt) is not int or not 0<prompt<=131072-payload['max_tokens']:
                            raise ValueError('Actual tool history exceeds declared model context; no truncation')
                        row=generate(payload,task_id=task['id'],turn=turn,step=step)
                        _append(requests,{**row,'task_id':task['id'],'turn':turn,'step':step,'request':payload,
                            'request_sha256':digest(payload),'prompt_tokens_CPU_expected':prompt})
                        records+=1
                        stage='actual_HTTP_or_native_SSE_contract'
                        if row.get('status')!=200 or row.get('error') is not None:
                            raise ValueError('Native-tool HTTP failed; original reply retained')
                        parsed=parse_sse_packets(_packets(row['raw_SSE']),allow_empty_final=True)
                        usage=parsed['usage']
                        if usage['prompt_tokens']!=prompt or usage['completion_tokens']>payload['max_tokens']:
                            raise ValueError('Actual tool API usage differs from official JSON tokenization/cap')
                        verified_requests+=1
                        assistant={'role':'assistant','content':parsed['content']}
                        if parsed['tool_calls']:assistant['tool_calls']=parsed['tool_calls']
                        messages.append(assistant)
                        if not parsed['tool_calls']:
                            stage='natural_generation_cap'
                            if parsed['finish_reason']=='length':
                                raise ValueError('Final assistant response exhausted its generation cap')
                            empty_final_turns+=not bool(parsed['content']);finished_turns+=1;break
                        arguments=parse_native_tool_calls(parsed['tool_calls'])
                        stage='bounded_in_memory_tool_execution'
                        for call,decoded in zip(parsed['tool_calls'],arguments):
                            name,values=next(iter(decoded.items()));reply=tools.call(name,values);calls+=1
                            _append(results,{'task_id':task['id'],'turn':turn,'step':step,'call':call,
                                             'arguments':values,'reply':reply})
                            messages.append({'role':'tool','tool_call_id':call['id'],'content':reply['result']})
                    else:
                        stage='assistant_tool_step_budget'
                        raise ValueError('Assistant/tool loop exceeded10steps; future user turns not injected')
                status='ALL_ORIGINAL_USER_TURNS_COMPLETED'
        except (OSError,ValueError,RuntimeError,TimeoutError) as error:
            failure=type(error).__name__+': '+str(error)[:1200]
    summary={'task_id':task['id'],'state':status,'error':failure,'original_user_turns':len(task['original_user_turns']),
             'finished_user_turns':finished_turns,'model_API_requests':records,'tool_calls':calls,
             'actual_API_contracts_verified':records>0 and verified_requests==records,
             'API_requests_independently_checkable':records,'API_requests_verified_by_worker':verified_requests,
             'tool_child_initialized':tool_ready,'failure_stage':stage if failure else None,
             'empty_final_assistant_turns':empty_final_turns,'source_line_sha256_without_LF':task['source_line_sha256_without_LF'],
             'model_task_correctness_scored':False,'future_questions_injected_before_previous_turn_finished':False,
             'external_tools_or_network_called_by_simulator':False}
    write_new_json(output/'generation_result.json',summary);return summary


def generate_core(prepared: Path,source: Path,output: Path,*,generate,count_tokens,
                  max_seconds: int = 3600,concurrency: int = 4) -> dict:
    if max_seconds!=3600 or concurrency!=4:raise ValueError('Unreviewed tool workload wall/concurrency budget')
    manifest=json.loads((prepared/'manifest.json').read_text());raw=(prepared/'tasks.jsonl').read_bytes()
    if (manifest.get('schema')!='nextgen-native-BFCL-multiturn-inputs-v1' or manifest.get('tasks')!=32
            or manifest.get('user_turns')!=112 or hashlib.sha256(raw).hexdigest()!=manifest['tasks_sha256']
            or manifest.get('ground_truth_used_for_selection_or_prompt') is not False
            or manifest.get('upstream_initial_state_dictionary_order_preserved') is not True):
        raise ValueError('Original real multi-turn inputs changed')
    tasks=[json.loads(line) for line in raw.split(b'\n') if line]
    if len(tasks)!=32 or len({task['id'] for task in tasks})!=32:raise ValueError('Original32task inventory changed')
    output.mkdir(mode=0o700,parents=True,exist_ok=False);deadline=time.monotonic_ns()+max_seconds*10**9
    def one(task):
        return generate_session(task,source=source,output=output/task['id'],generate=generate,count_tokens=count_tokens,
                                deadline_ns=deadline)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:results=list(pool.map(one,tasks))
    summary={'schema':'nextgen-native-tools32-generation-v1','tasks_attempted':len(results),'source_tasks_sha256':manifest['tasks_sha256'],
             'user_turns_expected':112,'user_turns_completed':sum(row['finished_user_turns'] for row in results),
             'conversations_completed':sum(row['state']=='ALL_ORIGINAL_USER_TURNS_COMPLETED' for row in results),
             'model_API_requests':sum(row['model_API_requests'] for row in results),
             'tool_calls':sum(row['tool_calls'] for row in results),'results':results,
             'all_actual_API_contracts_verified':all(row['actual_API_contracts_verified'] for row in results),
             'all_failed_tasks_retained_in_scoring_denominator':True,'model_quality_scored':False,
             'AIPerf_or_service_performance_measured':False,'reference_answers_loaded_on_serving_path':False}
    write_new_json(output/'generation_summary.json',summary);return summary
