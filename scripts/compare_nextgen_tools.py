"""Pair complete fixed-source native-tool evaluations without discarding failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_nextgen_validation import _check,_mirror_paths
from scripts.controller.durable import write_new_json


def paired_scores(reference: dict,candidate: dict) -> dict:
    for score in (reference,candidate):
        _check(score.get('all_actual_histories_SSE_usage_and_tool_replays_verified') is True
               and score.get('all_original_tasks_retained_in_denominator') is True
               and score.get('tasks')==32 and len(score['scores'])==32,'Missing original task or replay evidence')
        _check(all(type(row['task_correct']) is bool for row in score['scores']), 'Invalid task result')
        correct=sum(row['task_correct'] for row in score['scores'])
        _check(correct==score['correct'] and score['task_success_percent']==100*correct/32,
               'Headline differs from complete per-task outcomes')
        _check(all(not row['task_correct'] for row in score['scores'] if row['budget_or_incomplete_conversation']),
               'An incomplete original task cannot be counted correct')
    for name in ('source_tasks_sha256','reference_sha256'):
        _check(reference[name]==candidate[name],'Tool task/reference cohort changed')
    left={row['task_id']:row for row in reference['scores']};right={row['task_id']:row for row in candidate['scores']}
    _check(len(left)==len(right)==32 and left.keys()==right.keys(),'Duplicated or different original tasks')
    groups={name:[] for name in ('both_correct','both_failed','BF16_correct_candidate_failed','BF16_failed_candidate_correct')}
    for identifier,l in left.items():
        r=right[identifier]
        _check(l['official_checker_AST_sha256']==r['official_checker_AST_sha256']
               and l['original_user_turns']==r['original_user_turns'],'Official checker or source turns differ')
        key=('both_correct' if l['task_correct'] else 'BF16_failed_candidate_correct') if r['task_correct'] else (
             'BF16_correct_candidate_failed' if l['task_correct'] else 'both_failed')
        groups[key].append(identifier)
    return {'BF16_correct':reference['correct'],'candidate_correct':candidate['correct'],'tasks_per_arm':32,
        'BF16_percent':reference['task_success_percent'],'candidate_percent':candidate['task_success_percent'],
        'change_percentage_points':candidate['task_success_percent']-reference['task_success_percent'],
        'paired_outcome_ids':groups,'all_original_tasks_in_denominator':True}


def bound(audit_path: Path,manifest_path: Path) -> tuple[dict,dict]:
    audit=json.loads(audit_path.read_text());manifest=json.loads(manifest_path.read_text());base,root=_mirror_paths(manifest)
    _check(audit.get('actual_model_provenance_verified') is True
           and audit['source_SHA_manifest_sha256']==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
           and audit['run_id']==manifest['run_id'],'Tool audit is not bound to the original actual-model mirror')
    for row in manifest['files']:
        raw=(base/row['base_rel']).read_bytes()
        _check(len(raw)==row['size'] and hashlib.sha256(raw).hexdigest()==row['snapshot_sha256']==row['remote_sha256'],
               'Original paired tool evidence changed')
    request=json.loads((root/'ticket.json').read_text())['request']
    _check(request['arms']==[audit['arm']],'Tool arm identity differs')
    return audit,request


def compare(reference_path: Path,reference_manifest: Path,candidate_path: Path,candidate_manifest: Path) -> dict:
    left,lreq=bound(reference_path,reference_manifest);right,rreq=bound(candidate_path,candidate_manifest)
    _check(left['arm']=='bf16' and right['arm'] in ('fp8_e4m3','nvfp4'),'Pair explicit BF16 and native low precision')
    for name in ('model_id','model_config_sha256','image_id','binary_sha256','server_source_commit',
                 'python_source_commit','service','gpu_uuids','prefix_cache','graph_mode'):
        _check(lreq[name]==rreq[name],'Different tool deployment: '+name)
    for name in ('tasks','user_turns','client_concurrency','max_steps_per_turn','max_new_tokens','max_seconds','tool_parser',
                 'manifest_sha256','tasks_sha256','tool_source_manifest_sha256','CPU_protocol_rollup_sha256','CPU_image_evidence_sha256'):
        _check(lreq['tool_evaluation'][name]==rreq['tool_evaluation'][name],'Different original tool demand/protocol: '+name)
    result=paired_scores(left['quality'],right['quality'])
    result.update(schema='nextgen-actual-native-tools32-pair-v1',reference_run_id=left['run_id'],candidate_run_id=right['run_id'],
        candidate_arm=right['arm'],service=lreq['service'],gpu_uuids=lreq['gpu_uuids'],
        operators={'BF16':lreq['operators'],right['arm']:rreq['operators']},
        reference_audit_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        candidate_audit_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        reference_manifest_sha256=hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
        candidate_manifest_sha256=hashlib.sha256(candidate_manifest.read_bytes()).hexdigest(),
        full_BFCL_leaderboard_score=False,final_natural_language_quality_scored=False,
        service_performance_or_production_acceptance=False,statistical_generalization_claim=False)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','reference-manifest','candidate','candidate-manifest','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();result=compare(args.reference,args.reference_manifest,args.candidate,args.candidate_manifest)
    write_new_json(args.out,result);print(json.dumps(result))
