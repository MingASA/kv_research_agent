"""Audit actual raw API predictions and score the frozen new-model cohort.

No missing responses, cached worker scores or old model row ordering are used.
This produces measured scores, not automatic production quality acceptance.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.audit_nextgen_validation import _check, _jsonl, _mirror_paths, audit
from scripts.controller.durable import canonical_bytes, write_new_bytes, write_new_json
from scripts.controller.nextgen import MODEL, QUALITY_KIND


def _module(name: str, path: Path, expected_sha: str):
    _check(hashlib.sha256(path.read_bytes()).hexdigest()==expected_sha,'Scoring source identity changed')
    spec=importlib.util.spec_from_file_location(name,path)
    _check(spec is not None and spec.loader is not None,'Cannot load the pinned scoring source')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_prediction(
    row: dict, expected: dict, *, official_prompt_tokens: int | None = None,
    allow_empty_stop: bool = False,
) -> str:
    _check(row['id']==expected['id'] and row['request']==expected['request']
           and row['endpoint']==expected['endpoint'] and row.get('error') is None,
           'Prediction belongs to another payload or client failed')
    _check(row['request_body_sha256']==hashlib.sha256(canonical_bytes(expected['request'])).hexdigest(),
           'Actual request body identity differs')
    response=json.loads(base64.b64decode(row['raw_HTTP_body_base64'],validate=True))
    _check(response==row['response'] and row['HTTP_status']==200 and not response.get('error'),
           'Cached response differs from actual API body/status')
    _check(len(response.get('choices',[]))==1,'Quality response choice missing or duplicated')
    choice=response['choices'][0]
    answer=(choice.get('message',{}).get('content','') if expected['endpoint']=='/v1/chat/completions'
            else choice.get('text',''))
    usage=response['usage']
    legitimate_empty = allow_empty_stop and answer=='' and choice['finish_reason']=='stop' and usage['completion_tokens']==1
    _check(isinstance(answer,str) and (bool(answer) or legitimate_empty) and answer==row['answer']
           and choice['finish_reason']==row['finish_reason'] and choice['finish_reason'] in ('stop','length'),
           'Answer/finish contract differs from raw API bytes')
    expected_count=expected['prompt_tokens_cpu'] if official_prompt_tokens is None else official_prompt_tokens
    _check(usage==row['usage'] and usage['prompt_tokens']==expected_count
           and type(usage['completion_tokens']) is int and 0<usage['completion_tokens']<=expected['request']['max_tokens']
           and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'],
           'Actual logical input/output token usage differs')
    _check(row['request_started_monotonic_ns']<row['request_ended_monotonic_ns'],
           'Quality request monotonic interval invalid')
    return answer


def score(manifest_path: Path, cohort: Path, output: Path, recovery_path: Path | None = None,
          completion_anchor_manifest: Path | None = None, *, upstream: bool = False) -> dict:
    recovery=json.loads(recovery_path.read_text()) if recovery_path is not None else None
    if upstream:
        from scripts.audit_upstream_run import audit as audit_upstream
        _check(recovery is None and completion_anchor_manifest is None,
               'Upstream full-cohort scoring cannot hide a legacy recovery or partial aggregation')
        lifecycle=audit_upstream(manifest_path)
    else:
        lifecycle=audit(manifest_path,failed_quality_recovery=recovery)
    mirror_manifest=json.loads(manifest_path.read_text())
    _,root=_mirror_paths(mirror_manifest)
    ticket=json.loads((root/'ticket.json').read_text());request=ticket['request'];arm=request['arms'][0]
    expected_kind = 'upstream_AC_frozen_core_quality1720' if upstream else QUALITY_KIND
    _check(request['workload']['kind']==expected_kind and request['range_observation'] is False
           and (not upstream or request.get('graph_observer') is None),
           'Formal core scoring requires the observer-free quality workload')
    _check(request['model_id']==MODEL,'Quality source model differs')
    evaluation=request['quality_evaluation']
    source=(root/'input/quality_requests.jsonl').read_bytes()
    _check(hashlib.sha256(source).hexdigest()==evaluation['requests_sha256'],'Deployed quality input SHA differs')
    fragment_predictions=_jsonl(root/'output'/arm/'quality_predictions.jsonl')
    source_runs=[{'run_id':request['run_id'],'terminal':'COMPLETE' if recovery is None else 'FAILED',
                  'records':len(fragment_predictions),'sha_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}]
    origins={row['id']:request['run_id'] for row in fragment_predictions}
    anchor_audit=None
    if completion_anchor_manifest is not None:
        _check(recovery is None and request.get('quality_completion') is not None,
               'Only a declared disjoint completion may aggregate interrupted responses')
        anchor_audit=audit(completion_anchor_manifest,interrupted_quality_fragment=True)
        _,anchor=_mirror_paths(json.loads(completion_anchor_manifest.read_text()))
        old=json.loads((anchor/'ticket.json').read_text())['request']
        _check(old['run_id']==request['quality_completion']['anchor_run_id'],'Wrong completion anchor')
        for name in ('model_id','binary_sha256','server_source_commit','python_source_commit','backend_source_sha256',
                     'operators','service','graph_mode','prefix_cache','kv_scales'):
            _check(old.get(name)==request.get(name),'Disjoint quality sources changed deployment: '+name)
        completed=_jsonl(anchor/'output'/arm/'quality_predictions.jsonl')
        _check(not ({row['id'] for row in completed}&set(origins)),'Duplicated quality response ID across source runs')
        certificate_raw=(root/'input/partial_quality_certificate.json').read_bytes();certificate=json.loads(certificate_raw)
        _check(hashlib.sha256(certificate_raw).hexdigest()==request['quality_completion']['partial_raw_certificate_sha256']
               and certificate['parent_prediction_sha256']==hashlib.sha256((anchor/'output'/arm/'quality_predictions.jsonl').read_bytes()).hexdigest()
               and certificate['durable_count']==len(completed) and certificate['missing_count']==len(fragment_predictions),
               'Completion provenance differs from its independently audited partial source')
        _check((root/'input/scoring_protocol.json').read_bytes()==(anchor/'input/scoring_protocol.json').read_bytes(),
               'Logical cohort scoring rules changed between fragments')
        source=(anchor/'input/quality_requests.jsonl').read_bytes()
        _check(hashlib.sha256(source).hexdigest()==certificate['logical_full_requests_sha256'],
               'Logical full source differs')
        source_runs.insert(0,{'run_id':old['run_id'],'terminal':certificate['original_result_status'],
            'records':len(completed),'sha_manifest_sha256':hashlib.sha256(completion_anchor_manifest.read_bytes()).hexdigest()})
        origins.update({row['id']:old['run_id'] for row in completed});fragment_predictions=completed+fragment_predictions
    else:
        _check(request.get('quality_completion') is None,'A completion fragment alone is not the full quality cohort')
    inputs=[json.loads(line) for line in source.split(b'\n') if line]
    labels_raw=(cohort/'labels.jsonl').read_bytes()
    source_manifest=json.loads((root/'input/quality_manifest.json').read_text())
    _check(hashlib.sha256(labels_raw).hexdigest()==source_manifest['labels_sha256'],
           'Independent reference labels differ from frozen source manifest')
    labels={row['id']:row for row in (json.loads(line) for line in labels_raw.split(b'\n') if line)}
    predictions=fragment_predictions
    predicted={row['id']:row for row in predictions}
    _check(len(inputs)==len(labels)==len(predictions)==len(predicted)==1720
           and set(predicted)=={row['id'] for row in inputs}==set(labels),
           'Missing/duplicate/unexpected core quality predictions or labels')
    protocol_path=root/'input/scoring_protocol.json'
    protocol=json.loads(protocol_path.read_text())
    _check(protocol['model_id']==MODEL and protocol['acceptance']['automatic_quality_acceptance'] is False,
           'Scoring protocol changed model or invents acceptance')
    gsm=_module('nextgen_pinned_gsm_scorer',Path(protocol['GSM8K']['source']),protocol['GSM8K']['source_sha256'])
    lb=_module('nextgen_pinned_longbench_metrics',Path(protocol['LongBench']['metrics_source']),
               protocol['LongBench']['metrics_source_sha256'])
    records=[];values=defaultdict(list);caps=defaultdict(int);tokens=defaultdict(int);contexts=defaultdict(set)
    for expected in inputs:
        row=predicted[expected['id']];label=labels[expected['id']]
        _check(all(row[key]==expected[key]==label[key] for key in ('task','group','source_index')),
               'Prediction/reference metadata maps to another record')
        correction=(recovery.get('official_token_count_corrections',{}).get(expected['id']) if recovery else None)
        empty_allowed=(expected['id'] in recovery.get('empty_stop_response_ids',[]) if recovery else
                       row['answer']=='' and row['finish_reason']=='stop' and row['usage']['completion_tokens']==1)
        answer=_verify_prediction(row,expected,official_prompt_tokens=correction,allow_empty_stop=empty_allowed)
        task,group=expected['task'],expected['group'];reference=label['reference']
        item={'id':expected['id'],'task':task,'group':group,'source_index':expected['source_index'],
              'context_sha256':expected['context_sha256'],'finish_reason':row['finish_reason'],
              'prompt_tokens':row['usage']['prompt_tokens'],'completion_tokens':row['usage']['completion_tokens'],
              'source_run_id':origins[expected['id']]}
        if task=='gsm8k':
            strict,flexible=gsm._gsm_score(answer,reference['answer'])
            item.update(score=flexible,strict_score=strict)
            values[(group,'gsm8k_strict')].append(strict)
            caps[(group,'gsm8k_strict')]+=row['finish_reason']=='length'
            tokens[(group,'gsm8k_strict')]+=row['usage']['completion_tokens']
            contexts[(group,'gsm8k_strict')].add(expected['id'])
        else:
            metric=getattr(lb,protocol['LongBench']['task_metric_mapping'][task])
            references=reference['answers']
            references=references if isinstance(references,list) else [references]
            item['score']=(max(metric(answer,target,all_classes=reference.get('all_classes',[])) for target in references)
                           if answer else 0.0)
        _check(math.isfinite(item['score']) and 0<=item['score']<=1,'Official score outside its metric range')
        values[(group,task)].append(item['score']);caps[(group,task)]+=row['finish_reason']=='length'
        tokens[(group,task)]+=row['usage']['completion_tokens']
        contexts[(group,task)].add(expected['context_sha256'] or expected['id']);records.append(item)
    tasks={group+'::'+task:{'count':len(scores),'score_percent_unrounded':100*sum(scores)/len(scores),
             'official_rounded_percent':round(100*sum(scores)/len(scores),2),'capped_responses':caps[(group,task)],
             'actual_completion_tokens':tokens[(group,task)],'independent_context_clusters':len(contexts[(group,task)])}
           for (group,task),scores in values.items()}
    lb_tasks=protocol['LongBench']['task_metric_mapping']
    macro=sum(tasks['core::'+task]['score_percent_unrounded'] for task in lb_tasks)/len(lb_tasks)
    result={'schema':'nextgen_independently_scored_core_quality_v1','state':'SCORES_VERIFIED_QUALITY_ACCEPTANCE_NOT_AUTOMATIC',
            'run_id':request['run_id'],'model_id':MODEL,'arm':arm,'lifecycle_audit':lifecycle,
            'prediction_count':len(records),'request_payload_sha256':hashlib.sha256(source).hexdigest(),
            'canonical_API_payloads_sha256':hashlib.sha256(b''.join(canonical_bytes(
                {'id':row['id'],'endpoint':row['endpoint'],'request':row['request']}) for row in inputs)).hexdigest(),
            'binary_sha256':request['binary_sha256'],'server_source_commit':request['server_source_commit'],
            'python_source_commit':request['python_source_commit'],
            'labels_sha256':source_manifest['labels_sha256'],'scoring_protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            'service':request['service'],'operators':request['operators'],'kv_scales':request.get('kv_scales'),
            'tasks':tasks,'LongBench_core_macro_percent_unrounded':macro,
            'GSM8K_flexible_EM_percent':tasks['core::gsm8k']['score_percent_unrounded'],
            'GSM8K_strict_EM_percent':tasks['core::gsm8k_strict']['score_percent_unrounded'],
            'all_raw_API_and_usage_contracts_independently_verified':True,
            'code_similarity_is_execution_success':False,'user_quality_tolerance_provided':False,
            'production_quality_accepted':False,'AIPerf_or_service_performance_measured':False}
    if completion_anchor_manifest is not None:
        result.update(source_runs=source_runs,interrupted_anchor_lifecycle_audit=anchor_audit,
            disjoint_completion_aggregation=True,single_uninterrupted_run_claim=False,
            original_failed_or_conflict_results_overwritten=False,
            state='SCORES_VERIFIED_FULL_COHORT_FROM_DISJOINT_RUNS_NO_AUTOMATIC_ACCEPTANCE')
    if recovery is not None:
        result.update(original_run_terminal='FAILED',raw_recovery_certificate_sha256=hashlib.sha256(recovery_path.read_bytes()).hexdigest(),
                      recovered_CPU_token_count_corrections=len(recovery['official_token_count_corrections']),
                      empty_stop_outputs_scored_zero=len(recovery['empty_stop_response_ids']),
                      original_runtime_result_overwritten=False)
    if upstream:
        result.update(upstream_commit=request['upstream_commit'], native_configuration=request.get('native_configuration'),
                      same_version_upstream_A_C=True)
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    write_new_bytes(output/'per_record_scores.jsonl',b''.join(canonical_bytes(row) for row in records))
    write_new_json(output/'summary.json',result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest',type=Path,required=True)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--recovery-audit',type=Path)
    parser.add_argument('--completion-anchor-sha-manifest',type=Path)
    parser.add_argument('--upstream',action='store_true')
    args=parser.parse_args();score(args.sha_manifest,args.cohort,args.out,args.recovery_audit,args.completion_anchor_sha_manifest,upstream=args.upstream)


if __name__=='__main__':main()
