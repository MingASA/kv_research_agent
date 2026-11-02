"""Independent BFCL32 state/response scoring from complete native-model mirrors.

Model-supplied calls are replayed only in bounded CPU children. Source questions,
wire/SSE/usage and chronological tool feedback are checked before an accuracy
claim; failed semantic/budget tasks remain in the original32-task denominator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys

from scripts.audit_nextgen_validation import _check,_jsonl,_mirror_paths,audit as audit_service
from scripts.controller.durable import digest,write_new_json
from scripts.controller.native_tools import parse_native_tool_calls,parse_sse_packets
from scripts.controller.nextgen_multiturn import _packets
from scripts.controller.nextgen_tools_env import ToolFactory
from scripts.controller.nextgen_tools_service import TOOLS_KIND
from scripts.nextgen_tokenization import OfficialJSONTokenizer


def replay_task(source: Path,data: dict) -> dict:
    """Run only inside a512MiB/30CPU-s child; no generated code is evaluated."""
    task,rows,trace,expected=data['task'],data['requests'],data['tool_results'],data['reference']
    generation=data['generation']
    factory=ToolFactory(source);model=factory.environment(task);gold=factory.environment(task)
    messages=copy.deepcopy(task['original_user_turns'][0]);turn=0;step=0;index=0;tool_calls=0
    cumulative=[];turn_results=[];budget_failure=False
    for row in rows:
        _check(turn<len(task['original_user_turns']) and row['turn']==turn and row['step']==step,
               'Original native-tool request order/turn changed')
        payload=copy.deepcopy(task['initial_payload']);payload['messages']=copy.deepcopy(messages)
        _check(row['task_id']==task['id'] and row['request']==payload and row['request_sha256']==digest(payload),
               'Native tool request changed source questions/schema,history or natural generation controls')
        _check(row.get('status')==200 and row.get('error') is None,'Actual native-tool HTTP did not complete')
        parsed=parse_sse_packets(_packets(row['raw_SSE']),allow_empty_final=True)
        _check(parsed['usage']['prompt_tokens']==row['prompt_tokens_CPU_expected']
               and parsed['usage']['completion_tokens']<=payload['max_tokens'],'Actual native-tool usage differs')
        assistant={'role':'assistant','content':parsed['content']}
        if parsed['tool_calls']:assistant['tool_calls']=parsed['tool_calls']
        messages.append(assistant)
        if parsed['tool_calls']:
            decoded=parse_native_tool_calls(parsed['tool_calls'])
            for call,arguments in zip(parsed['tool_calls'],decoded):
                if index==len(trace):
                    _check(generation['state']=='INCOMPLETE'
                           and generation.get('failure_stage')=='bounded_in_memory_tool_execution',
                           'Missing actual tool reply without an explicit bounded-execution failure')
                    budget_failure=True;break  # Never execute an unreturned/unbounded final call.
                record=trace[index];name,values=next(iter(arguments.items()))
                _check(record['task_id']==task['id'] and record['turn']==turn and record['step']==step
                       and record['call']==call and record['arguments']==values,'Tool reply identity differs')
                replay=model.call(name,values);actual=record['reply']
                _check(actual.get('task_id')==task['id'] and actual.get('sequence')==index+2
                       and type(actual.get('CPU_call_monotonic_ns')) is int and actual['CPU_call_monotonic_ns']>=0,
                       'Actual isolated tool reply has invalid session/CPU sequence evidence')
                _check(all(actual.get(key)==replay[key] for key in ('result','dispatch_valid','tool_reported_error')),
                       'Tool simulator replay differs from recorded execution;do not infer model accuracy')
                cumulative.append(actual['result']);messages.append({'role':'tool','tool_call_id':call['id'],'content':actual['result']})
                index+=1;tool_calls+=1
            if budget_failure:break
            step+=1
        else:
            if parsed['finish_reason']=='length':budget_failure=True;break
            gold_results=[gold.reference_call(expression)['result'] for expression in expected[turn]]
            state=factory.state_checker(model.instances,gold.instances)
            response=factory.response_checker(cumulative,gold_results,turn)
            turn_results.append({'turn':turn,'state_correct':state['valid'],'response_correct':response['valid'],
                                 'error_type':state.get('error_type') or response.get('error_type')})
            turn+=1;step=0
            if turn<len(task['original_user_turns']):messages.extend(copy.deepcopy(task['original_user_turns'][turn]))
    _check(index==len(trace) and tool_calls==generation['tool_calls'],'Tool trace count differs from actual generation')
    _check(len(rows)==generation['model_API_requests'] and turn==generation['finished_user_turns'],
           'Cached conversation completion differs from original raw requests')
    complete=turn==len(task['original_user_turns']) and generation['state']=='ALL_ORIGINAL_USER_TURNS_COMPLETED'
    if generation['state']=='ALL_ORIGINAL_USER_TURNS_COMPLETED':_check(complete,'Optimistic cached complete conversation')
    return {'task_id':task['id'],'original_user_turns':len(task['original_user_turns']),'completed_user_turns':turn,
            'task_correct':complete and all(row['state_correct'] and row['response_correct'] for row in turn_results),
            'turns':turn_results,'budget_or_incomplete_conversation':budget_failure or not complete,
            'original_failure_stage':generation.get('failure_stage'),'official_checker_AST_sha256':factory.checker_AST_sha256,
            'raw_history_tool_feedback_and_reference_semantics_verified':True}


def score_sessions(prepared: Path,source: Path,references: Path,conversations: Path,tokenizer_path: Path,output: Path) -> dict:
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    manifest=json.loads((prepared/'manifest.json').read_text());raw=(prepared/'tasks.jsonl').read_bytes()
    _check(hashlib.sha256(raw).hexdigest()==manifest['tasks_sha256'],'Original task bytes changed')
    tasks={row['id']:row for row in (json.loads(line) for line in raw.split(b'\n') if line)}
    _check(len(tasks)==32 and manifest['user_turns']==112,'Original32-task denominator changed')
    ref_manifest=json.loads((references/'manifest.json').read_text());ref_raw=(references/'BFCL_v4_multi_turn_base.json').read_bytes()
    _check(ref_manifest['pre_gold_tasks_sha256']==manifest['tasks_sha256']
           and hashlib.sha256(ref_raw).hexdigest()==ref_manifest['reference_sha256'],'Gold does not bind prompts frozen before reference fetch')
    gold={row['id']:row['ground_truth'] for row in (json.loads(line) for line in ref_raw.split(b'\n') if line)}
    generation=json.loads((conversations/'generation_summary.json').read_text())
    _check(generation['tasks_attempted']==32 and generation['all_actual_API_contracts_verified'] is True,
           'Incomplete/unverified real model API records cannot produce model accuracy')
    rows_by_id={row['task_id']:row for row in generation['results']}
    _check(set(rows_by_id)==set(tasks),'Generated task set differs')
    tokenizer=OfficialJSONTokenizer(tokenizer_path);scores=[]
    for identifier,task in tasks.items():
        _check(identifier.startswith('multi_turn_base_') and identifier.rsplit('_',1)[1].isdigit(),'Unsafe original task ID')
        directory=conversations/identifier;rows=_jsonl(directory/'requests.jsonl');trace=_jsonl(directory/'tool_results.jsonl')
        for row in rows:_check(tokenizer.count(row['request'])==row['prompt_tokens_CPU_expected'],'Independent official JSON token count differs')
        bundle=output/(identifier+'.input.json');result_path=output/(identifier+'.result.json')
        write_new_json(bundle,{'task':task,'requests':rows,'tool_results':trace,
                               'reference':gold[identifier],'generation':rows_by_id[identifier]})
        with (output/(identifier+'.stderr.log')).open('x') as stderr:
            process=subprocess.run([sys.executable,'-B','-m','scripts.score_nextgen_tools','--worker',
                '--source',str(source),'--input',str(bundle),'--out',str(result_path)],stderr=stderr,timeout=45)
        _check(process.returncode==0 and result_path.exists(),'Bounded independent replay failed;no accuracy conclusion')
        scores.append(json.loads(result_path.read_text()))
    summary={'schema':'nextgen-BFCL32-independent-state-response-scores-v1','tasks':32,
        'correct':sum(row['task_correct'] for row in scores),'task_success_percent':100*sum(row['task_correct'] for row in scores)/32,
        'source_tasks_sha256':manifest['tasks_sha256'],'reference_sha256':ref_manifest['reference_sha256'],'scores':scores,
        'all_original_tasks_retained_in_denominator':True,'all_actual_histories_SSE_usage_and_tool_replays_verified':True,
        'full_BFCL_leaderboard_score':False,'final_assistant_natural_language_quality_scored':False,
        'model_or_GPU_provenance_verified_by_this_lower_level_function':False}
    write_new_json(output/'summary.json',summary);return summary


def audit(manifest_path: Path,references: Path,tokenizer: Path,output: Path,*,upstream: bool=False) -> dict:
    if upstream:
        from scripts.audit_upstream_run import audit as audit_upstream
        lifecycle=audit_upstream(manifest_path)
    else:
        lifecycle=audit_service(manifest_path)
    manifest=json.loads(manifest_path.read_text());base,root=_mirror_paths(manifest)
    ticket=json.loads((root/'ticket.json').read_text());request=ticket['request'];arm=request['arms'][0]
    _check(request['workload']['kind']==('upstream_AC_real_native_tools32' if upstream else TOOLS_KIND),
           'This is not a real native-tool model run')
    verified={((root if upstream else base)/row['base_rel']).resolve(strict=True) for row in manifest['files']}
    for name,sha in ticket['assets_sha256'].items():
        path=root/name
        _check(path.resolve(strict=True) in verified and hashlib.sha256(path.read_bytes()).hexdigest()==sha,
               'Frozen actual tool input/runtime missing or changed: '+name)
    config=request['tool_evaluation']
    for name,key in (('input/tools32/manifest.json','manifest_sha256'),('input/tools32/tasks.jsonl','tasks_sha256'),
                     ('input/tool_source/source_manifest.json','tool_source_manifest_sha256')):
        _check(hashlib.sha256((root/name).read_bytes()).hexdigest()==config[key],'Actual tool source identity differs')
    measurement=json.loads((root/'output'/arm/'measurement.json').read_text())
    tool=measurement['tool_generations']
    _check(tool['actual_owned_model_HTTP_provider'] is True and tool['synthetic_provider_or_gold_reference_loaded'] is False,
           'Fixture output cannot stand in for actual model generations')
    summary=score_sessions(root/'input/tools32',root/'input/tool_source',references,root/'output'/arm/'tool_generations',
                           tokenizer,output/'independent_scores')
    result={'schema':'actual-native-model-BFCL32-quality-audit-v1','run_id':manifest['run_id'],'arm':arm,
            'lifecycle':lifecycle,'quality':summary,'actual_model_provenance_verified':True,
            'source_SHA_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'production_quality_accepted':False,'service_performance_measured':False}
    write_new_json(output/'actual_model_audit.json',result);return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--upstream',action='store_true')
    for name in ('source','input','out','sha-manifest','references','tokenizer'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.worker:
        resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
        resource.setrlimit(resource.RLIMIT_CPU,(30,30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        result=replay_task(args.source,json.loads(args.input.read_text()));write_new_json(args.out,result)
    else:
        if not all((args.sha_manifest,args.references,args.tokenizer,args.out)):
            parser.error('Actual model scoring requires --sha-manifest/--references/--tokenizer/--out')
        result=audit(args.sha_manifest,args.references,args.tokenizer,args.out,upstream=args.upstream)
        print(json.dumps({'run_id':result['run_id'],'quality':result['quality']['task_success_percent']}))


if __name__=='__main__':main()
