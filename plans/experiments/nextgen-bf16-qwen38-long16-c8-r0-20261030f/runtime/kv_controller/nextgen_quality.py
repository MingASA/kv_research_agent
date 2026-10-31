"""Generate the frozen real-text quality cohort with bounded client demand.

Inputs contain no evaluation labels. This collects actual API bytes/usage;
official scoring happens independently after mirroring. It is not AIPerf.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .durable import atomic_json, canonical_bytes, write_new_json


def _request(row: dict, timeout_seconds: float) -> dict:
    payload = row['request']
    body = canonical_bytes(payload)
    query = urllib.request.Request('http://127.0.0.1:19001'+row['endpoint'], data=body,
                                   headers={'Content-Type':'application/json'})
    raw, response, error, status = b'', {}, None, None
    started = time.monotonic_ns()
    try:
        with urllib.request.urlopen(query, timeout=timeout_seconds) as connection:
            status = connection.status
            raw = connection.read(2**20+1)
            if len(raw)>2**20:
                raise ValueError('Quality response exceeds the one-MiB raw journal limit')
            decoded = json.loads(raw)
            if not isinstance(decoded,dict):
                raise ValueError('Quality API returned a non-object JSON body')
            response = decoded
    except urllib.error.HTTPError as failure:
        status = failure.code
        raw = failure.read(2**20)
        error = type(failure).__name__+': '+str(failure)
    except (OSError,ValueError) as failure:
        error = type(failure).__name__+': '+str(failure)
    ended = time.monotonic_ns()
    usage = response.get('usage', {})
    usage = usage if isinstance(usage,dict) else {}
    choices = response.get('choices', [])
    choice = choices[0] if isinstance(choices,list) and len(choices)==1 and isinstance(choices[0],dict) else {}
    message = choice.get('message', {})
    message = message if isinstance(message,dict) else {}
    answer = (message.get('content', '') if row['endpoint'].endswith('chat/completions')
              else choice.get('text', ''))
    empty_stop = answer=='' and choice.get('finish_reason')=='stop' and usage.get('completion_tokens')==1
    valid = (error is None and status==200 and 'error' not in response and isinstance(answer,str) and (bool(answer) or empty_stop)
             and choice.get('finish_reason') in ('stop','length')
             and usage.get('prompt_tokens')==row['prompt_tokens_cpu']
             and type(usage.get('completion_tokens')) is int
             and 0<usage['completion_tokens']<=payload['max_tokens']
             and usage.get('total_tokens')==usage['prompt_tokens']+usage['completion_tokens'])
    return {'id':row['id'],'task':row['task'],'group':row['group'],
            'source_index':row['source_index'],'endpoint':row['endpoint'],
            'request':payload,'request_body_sha256':hashlib.sha256(body).hexdigest(),
            'raw_HTTP_body_base64':base64.b64encode(raw).decode(), 'HTTP_status':status,
            'response':response,'answer':answer,'usage':usage,'finish_reason':choice.get('finish_reason'),
            'request_started_monotonic_ns':started,'request_ended_monotonic_ns':ended,
            'contract_valid':valid,'error':error,'model_accuracy_checked':False}


def _validate_inputs(rows: list[dict], expected_count: int, model_id: str, token_budget: int) -> None:
    if len(rows)!=expected_count or len({row['id'] for row in rows})!=expected_count:
        raise ValueError('Quality request count/identities differ from the immutable cohort')
    for row in rows:
        payload = row['request']
        if (row['endpoint'] not in ('/v1/chat/completions','/v1/completions')
                or payload.get('model')!=model_id or payload.get('temperature')!=0
                or payload.get('ignore_eos',False) or payload.get('stream',False)
                or type(payload.get('max_tokens')) is not int or not 1<=payload['max_tokens']<=2048
                or type(row['prompt_tokens_cpu']) is not int or row['prompt_tokens_cpu']<=0
                or row['prompt_tokens_cpu']+payload['max_tokens']>token_budget):
            raise ValueError('Unsupported/unbounded/unnatural quality API payload')
        if row['endpoint']=='/v1/chat/completions' and payload.get('chat_template_kwargs')!={'enable_thinking':False}:
            raise ValueError('Frozen quality chat template/thinking mode changed')


def run_core_generations(request: dict, root: Path, output: Path, *, available_seconds: float) -> dict:
    config = request['quality_evaluation']
    path = root/'input/quality_requests.jsonl'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=config['requests_sha256']:
        raise ValueError('Quality request payload SHA changed')
    manifest_raw = (root/'input/quality_manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if (hashlib.sha256(manifest_raw).hexdigest()!=config['source_manifest_sha256']
            or manifest.get('request_payload_sha256')!=config['requests_sha256']
            or manifest.get('model_id')!=request['model_id']):
        raise ValueError('Quality source manifest/model/payload identities differ')
    rows = [json.loads(line) for line in raw.splitlines()]
    _validate_inputs(rows,config['requests'],request['model_id'],config['client_token_budget'])
    deadline = time.monotonic()+min(config['max_seconds'],available_seconds)
    if deadline-time.monotonic()<600:
        raise ValueError('Startup left insufficient bounded time for the core quality cohort')
    index, reserved, completed, failures, capped, empty_outputs = 0, 0, 0, 0, 0, 0
    active = {}
    journal = output/'quality_predictions.jsonl'
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=config['client_concurrency'])
    try:
        with journal.open('x') as stream:
            while index<len(rows) or active:
                if time.monotonic()>=deadline:
                    raise TimeoutError('Core quality generation deadline exceeded; preserve partial raw journal')
                while index<len(rows) and len(active)<config['client_concurrency']:
                    row=rows[index];cost=row['prompt_tokens_cpu']+row['request']['max_tokens']
                    if active and reserved+cost>config['client_token_budget']:
                        break
                    timeout=min(config['request_timeout_seconds'],max(1,deadline-time.monotonic()))
                    future=pool.submit(_request,row,timeout)
                    active[future]=(row['id'],cost);reserved+=cost;index+=1
                done,_=concurrent.futures.wait(active,timeout=min(10,max(0,deadline-time.monotonic())),
                                                return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    identifier,cost=active.pop(future);reserved-=cost
                    value=future.result()
                    if value['id']!=identifier:
                        raise ValueError('Quality response associated with another request')
                    stream.write(json.dumps(value,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                    completed+=1;failures+=not value['contract_valid'];capped+=value['finish_reason']=='length'
                    empty_outputs+=value.get('answer')==''
                atomic_json(output/'quality_progress.json',{'submitted':index,'durable_responses':completed,
                            'contract_failures':failures,'capped_responses':capped,'active_client_requests':len(active),
                            'empty_stop_outputs':empty_outputs,
                            'reserved_client_token_budget':reserved,'accuracy_scored':False})
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    result={'classification':'core_quality_actual_API_generations_not_accuracy_or_performance_acceptance',
            'cohort_requests':len(rows),'durable_responses':completed,'contract_failures':failures,
            'capped_responses':capped,'actual_all_inputs_sent':index==len(rows),
            'empty_stop_outputs':empty_outputs,'empty_outputs_are_quality_zeros_not_nonempty_success':True,
            'all_response_contracts_valid':completed==len(rows) and failures==0,
            'prediction_journal_sha256':hashlib.sha256(journal.read_bytes()).hexdigest(),
            'model_quality_validated':False,'accuracy_scored':False,'official_aiperf_complete':False,
            'service_performance_measured':False,'client_concurrency':config['client_concurrency'],
            'client_token_budget':config['client_token_budget'],
            'scheduling_scope':'Original submission order; bounded logical prompt+cap reservations common to all formats; '
                               'no prefix-reuse or quantized capacity benefit claimed from this quality client'}
    write_new_json(output/'quality_generation_result.json',result)
    if not result['all_response_contracts_valid']:
        raise ValueError('Incomplete/invalid core quality responses; original bytes retained, no accuracy acceptance')
    return result
