"""Validate every durable interrupted response, freeze only missing request IDs.

No labels, answers, scores or predicted correctness select the completion set.
Original runs/results stay untouched. The final full-cohort score must retain
the source-run provenance; fragments are never a continuous performance run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_nextgen_validation import audit,_jsonl,_mirror_paths
from scripts.controller.durable import canonical_bytes,digest,write_new_bytes,write_new_json
from scripts.score_nextgen_core_quality import _verify_prediction


def prepare(manifest_path: Path, cohort: Path, output: Path) -> dict:
    lifecycle=audit(manifest_path,interrupted_quality_fragment=True)
    mirror=json.loads(manifest_path.read_text());_,root=_mirror_paths(mirror)
    ticket=json.loads((root/'ticket.json').read_text());request=ticket['request'];arm=request['arms'][0]
    result=json.loads((root/'result.json').read_text())
    full=(cohort/'requests.jsonl').read_bytes()
    if hashlib.sha256(full).hexdigest()!=request['quality_evaluation']['requests_sha256']:
        raise ValueError('Logical full quality input differs from interrupted deployed input')
    inputs=[json.loads(line) for line in full.split(b'\n') if line];expected={row['id']:row for row in inputs}
    predictions=_jsonl(root/'output'/arm/'quality_predictions.jsonl');seen=set()
    for row in predictions:
        identifier=row['id']
        if identifier in seen or identifier not in expected:raise ValueError('Duplicate/unexpected durable quality row')
        original=expected[identifier]
        if any(row[name]!=original[name] for name in ('task','group','source_index')):
            raise ValueError('Durable response belongs to another source item')
        allow_empty=row['answer']=='' and row['finish_reason']=='stop' and row['usage']['completion_tokens']==1
        _verify_prediction(row,original,allow_empty_stop=allow_empty)
        seen.add(identifier)
    if not 0<len(seen)<len(inputs)==1720:raise ValueError('Completion requires a genuine incomplete full cohort')
    missing=[row for row in inputs if row['id'] not in seen]
    raw=b''.join(canonical_bytes(row) for row in missing)
    certificate={'state':'INDEPENDENT_PARTIAL_RAW_QUALITY_VALIDATED_NOT_FULL_COHORT',
        'anchor_run_id':request['run_id'],'source_sha_manifest':str(manifest_path.resolve()),
        'source_sha_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_request':request,'source_request_sha256':digest(request),
        'original_result_status':result['status'],'original_result_sha256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
        'parent_prediction_sha256':hashlib.sha256((root/'output'/arm/'quality_predictions.jsonl').read_bytes()).hexdigest(),
        'logical_full_requests_sha256':hashlib.sha256(full).hexdigest(),
        'logical_full_input_count':1720,'durable_count':len(seen),'missing_count':len(missing),
        'durable_ids':sorted(seen),'durable_ids_sha256':digest(sorted(seen)),
        'missing_ids':[row['id'] for row in missing],'missing_ids_sha256':digest([row['id'] for row in missing]),
        'completion_requests_sha256':hashlib.sha256(raw).hexdigest(),
        'all_existing_raw_HTTP_usage_payloads_verified':True,'invalid_responses_filtered':False,
        'selection_uses_answers_labels_or_scores':False,'original_run_or_outputs_overwritten':False,
        'lifecycle_audit':lifecycle,'model_quality_or_performance_accepted':False}
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    write_new_bytes(output/'requests.jsonl',raw)
    source_manifest=json.loads((cohort/'manifest.json').read_text())
    source_manifest.update(request_payload_sha256=hashlib.sha256(raw).hexdigest(),
        completion_requests=len(missing),logical_full_requests_sha256=certificate['logical_full_requests_sha256'],
        completion_selection='All and only missing durable request IDs in original full input order; no correctness selection')
    write_new_json(output/'partial_raw_certificate.json',certificate)
    source_manifest['partial_raw_certificate_sha256']=hashlib.sha256((output/'partial_raw_certificate.json').read_bytes()).hexdigest()
    write_new_json(output/'manifest.json',source_manifest)
    write_new_bytes(output/'scoring_protocol.json',(cohort/'scoring_protocol.json').read_bytes())
    return certificate


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--sha-manifest',type=Path,required=True)
    parser.add_argument('--full-cohort',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=prepare(args.sha_manifest,args.full_cohort,args.out)
    print(json.dumps({k:result[k] for k in ('state','anchor_run_id','logical_full_input_count','durable_count','missing_count')}))


if __name__=='__main__':main()
