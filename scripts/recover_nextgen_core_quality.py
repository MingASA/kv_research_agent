"""Recover complete raw quality responses after a proven client-metadata error.

The original FAILED result is never changed. Empty stop outputs are retained
as zero-score observations; no response is regenerated, dropped or substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_nextgen_validation import _check, _jsonl, _mirror_paths, audit
from scripts.controller.durable import write_new_json
from scripts.score_nextgen_core_quality import _verify_prediction


def recover(manifest_path: Path, token_audit_path: Path, output: Path) -> dict:
    manifest=json.loads(manifest_path.read_text());_,root=_mirror_paths(manifest)
    for entry in manifest['files']:
        path=root/entry['base_rel'];raw=path.read_bytes()
        _check(len(raw)==entry['size'] and hashlib.sha256(raw).hexdigest()==entry['remote_sha256']==entry['snapshot_sha256'],
               'Failed-run raw mirror changed')
    ticket=json.loads((root/'ticket.json').read_text());request=ticket['request'];arm=request['arms'][0]
    terminal=json.loads((root/'result.json').read_text())
    _check(terminal['status']=='FAILED' and terminal['gpu_release_verified'] and not terminal['live_owned_containers'],
           'Recovery requires the original failed, released run')
    token_audit=json.loads(token_audit_path.read_text())
    inputs=_jsonl(root/'input/quality_requests.jsonl');rows=_jsonl(root/'output'/arm/'quality_predictions.jsonl')
    _check(len(inputs)==len(rows)==token_audit['cohort_records']==1720
           and token_audit['remaining_actual_API_mismatches']==[],
           'Official JSON tokenizer has not matched every actual request usage')
    corrections={row['id']:row['official_JSON_count'] for row in token_audit['corrections']}
    tokenizer_path=Path('.deps/nextgen-official-models-20261027-a1/Qwen3.8-27B/tokenizer.json')
    _check(hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()==token_audit['source_tokenizer_json_sha256'],
           'Official tokenizer JSON identity changed')
    expected={row['id']:row for row in inputs};predicted={row['id']:row for row in rows}
    _check(len(expected)==len(predicted)==1720 and set(expected)==set(predicted),
           'Missing/duplicate/unexpected generated response')
    empty=[]
    for identifier,row in predicted.items():
        reference=expected[identifier]
        allow_empty=row['answer']=='' and row['finish_reason']=='stop' and row['usage']['completion_tokens']==1
        if allow_empty:empty.append(identifier)
        _verify_prediction(row,reference,official_prompt_tokens=corrections.get(identifier),allow_empty_stop=allow_empty)
    generation=root/'output'/arm/'quality_generation_result.json'
    generated=json.loads(generation.read_text())
    _check(generated['cohort_requests']==generated['durable_responses']==1720
           and generated['actual_all_inputs_sent'] is True,
           'Worker did not actually submit and persist the whole cohort')
    result={'state':'INDEPENDENT_RAW_QUALITY_RECOVERY_ORIGINAL_RUN_FAILED','run_id':request['run_id'],
            'source_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'original_result_sha256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest(),
            'prediction_journal_sha256':hashlib.sha256((root/'output'/arm/'quality_predictions.jsonl').read_bytes()).hexdigest(),
            'all_raw_API_payloads_and_official_JSON_usage_verified':True,
            'official_tokenization_audit_sha256':hashlib.sha256(token_audit_path.read_bytes()).hexdigest(),
            'official_tokenizer_JSON_sha256':token_audit['source_tokenizer_json_sha256'],
            'official_token_count_corrections':corrections,'empty_stop_response_ids':sorted(empty),
            'responses_retained':1720,'empty_outputs_score_zero':True,'original_runtime_result_overwritten':False,
            'GPU_or_inference_repeated':False,'production_quality_accepted':False,
            'explanation':'The pinned official tokenizer JSON matches actual native API usage for all records; '
              'AutoTokenizer rebuilt a different pre-tokenizer regex for12 metadata counts. Six empty stop/completion1 '
              'responses remain in the scored cohort, not treated as successful nonempty answers or removed.'}
    result['lifecycle_audit']=audit(manifest_path,failed_quality_recovery=result)
    write_new_json(output,result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest',type=Path,required=True)
    parser.add_argument('--tokenization-audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();recover(args.sha_manifest,args.tokenization_audit,args.out)


if __name__=='__main__':main()
