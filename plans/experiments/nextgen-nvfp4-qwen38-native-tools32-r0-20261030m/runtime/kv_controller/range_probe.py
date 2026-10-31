# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Quiescent reset/export for a bounded BF16 KV distribution diagnostic."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import time

from .durable import digest, write_new_json

METHOD = 'heldout_BF16_KV_distribution_GPU_aggregate_v1'
COUNT_NAMES = ('finite','nonfinite','clipped','underflow_nonzero_to_zero','normal','subnormal','zero')
CHAT_SCOPE = 'heldout_original_ShareGPT_natural_chat'
LONG_QA_SCOPE = 'heldout_original_LongBench_long_document_QA'
LONG_QA_TASK_CAPS = {'qasper':128, 'multifieldqa_en':64, 'hotpotqa':32, 'musique':32}


def validate_longcontext_inputs(data: dict, manifest: dict) -> None:
    """Reject changed caps, padding, leaked labels and unsupported diagnostic scopes."""
    if (manifest.get('schema') != 'kv-real-heldout-longcontext-QA-input-v1'
            or manifest.get('content_scope') != LONG_QA_SCOPE
            or manifest.get('requests') != 128 or manifest.get('canonical_sha256') != digest(data)
            or manifest.get('quality_labels_used_in_selection') is not False
            or manifest.get('answers_exported_to_probe') is not False
            or manifest.get('previous_predictions_used_in_selection') is not False
            or manifest.get('official_generation_caps_preserved') is not True
            or manifest.get('messages_and_natural_stop_controls_unchanged') is not True
            or manifest.get('evaluation_excluded_count') != 400
            or manifest.get('overlap_with_previous_evaluation') != {
                'record_ids':0, 'exact_records':0, 'exact_contexts':0}
            or manifest.get('source_content_sha256') !=
                'cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64'):
        raise ValueError('Long-document QA source, exclusions, label separation or frozen controls differ')
    rows = data.get('data', [])
    lineage = manifest.get('selected_lineage', [])
    if (len(rows) != 128 or len(lineage) != 128
            or len({r['session_id'] for r in rows}) != 128
            or len({r['record_id'] for r in lineage}) != 128
            or len({r['context_sha256'] for r in lineage}) != 128):
        raise ValueError('Long-document QA needs128 unique records and contexts')
    counts = {task:0 for task in LONG_QA_TASK_CAPS}
    for row, origin in zip(rows, lineage):
        task = origin['task']
        if task not in LONG_QA_TASK_CAPS or row['session_id'] != origin['session_id'] or len(row['payloads']) != 1:
            raise ValueError('Long-document QA lineage/order/task changed')
        payload = row['payloads'][0]
        messages = payload.get('messages')
        if (not isinstance(messages, list) or len(messages) != 1
                or set(messages[0]) != {'role','content'} or messages[0]['role'] != 'user'
                or not isinstance(messages[0]['content'], str)):
            raise ValueError('Expected the original single document-QA user prompt')
        cap = LONG_QA_TASK_CAPS[task]
        expected = {'model':'Qwen2.5-14B-Instruct', 'messages':messages, 'stream':True,
                    'stream_options':{'include_usage':True}, 'temperature':0, 'top_p':1,
                    'seed':17, 'max_tokens':cap}
        if (payload != expected or digest(payload) != origin['payload_sha256']
                or hashlib.sha256(messages[0]['content'].encode()).hexdigest() != origin['prompt_sha256']
                or origin['max_tokens'] != cap or type(origin['prompt_tokens_cpu']) is not int
                or not 4096 <= origin['prompt_tokens_cpu'] <= 32768-cap):
            raise ValueError('Original QA prompt, natural stop, official cap or context bound differs')
        counts[task] += 1
    if any(count != 32 for count in counts.values()):
        raise ValueError('Long-document QA needs32 original requests per task')


def audit_export(data: dict, request_sha256: str, run_id: str, reset_ns: int,
                 content_scope: str = CHAT_SCOPE) -> dict:
    if content_scope not in (CHAT_SCOPE, LONG_QA_SCOPE):
        raise ValueError('Unsupported KV range observation content scope')
    expected={'method':METHOD,'request_sha256':request_sha256,'run_id':run_id,'command':'EXPORT',
        'reset_monotonic_ns':reset_ns,'layers':48,'heads':8,'dimension':128,
        'key_scale':.125,'value_scale':.0625,'host_counter_reads_in_forward':False,
        'activation_tensors_retained':False,'reserved_bytes':105216,
        'padding_policy':'exclude_all_nonpositive_slots_including_reserved_page0'}
    if any(data.get(k)!=v for k,v in expected.items()):
        raise ValueError('Diagnostic identity, reset epoch, geometry or storage differs')
    calls=data['device_writer_calls_by_layer_phase']
    if (len(calls)!=48 or any(len(r)!=2 or any(type(n)is not int or n<1 for n in r) for r in calls)
            or any(r!=calls[0] for r in calls)):
        raise ValueError('Missing actual all-layer prefill/decode writer calls')
    rows=data['rows'];seen=set();totals={};token_counts={}
    if len(rows)!=1536:raise ValueError('Missing complete layer/phase/KV/head grid')
    for row in rows:
        key=(row['layer'],row['phase'],row['kind'],row['head'])
        if (key in seen or type(key[0])is not int or not 0<=key[0]<48
                or key[1]not in ('prefill','decode') or key[2]not in ('key','value')
                or type(key[3])is not int or not 0<=key[3]<8):
            raise ValueError('Duplicate or invalid diagnostic group')
        seen.add(key)
        if (any(type(row[k])is not int or row[k]<0 for k in COUNT_NAMES)
                or row['nonfinite']!=0 or row['valid_values']!=row['finite']
                or row['finite']<1 or row['finite']%128
                or sum(row[k] for k in ('normal','subnormal','zero','clipped'))!=row['finite']
                or row['underflow_nonzero_to_zero']>row['subnormal']):
            raise ValueError('Nonfinite activation, incomplete tokens or counter partition mismatch')
        for field in ('absmax','squared_energy','squared_error','relative_l2'):
            if type(row[field])not in (float,int) or not math.isfinite(row[field]) or row[field]<0:
                raise ValueError('Invalid diagnostic moment')
        energy,error=row['squared_energy'],row['squared_error']
        if not math.isclose(row['relative_l2'],math.sqrt(error/energy) if energy else 0.,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('QDQ relative error differs from raw moments')
        phase=row['phase'];tokens=row['finite']//128
        if phase in token_counts and token_counts[phase]!=tokens:
            raise ValueError('Layers/heads/KV observed different valid token counts')
        token_counts[phase]=tokens
        group=totals.setdefault((phase,row['kind']),{k:0 for k in COUNT_NAMES}|{'absmax':0.,'squared_energy':0.,'squared_error':0.})
        for field in COUNT_NAMES:group[field]+=row[field]
        for field in ('squared_energy','squared_error'):group[field]+=row[field]
        group['absmax']=max(group['absmax'],row['absmax'])
    summaries={}
    for (phase,kind),row in totals.items():
        n=row['finite'];row.update(clipped_pct=100*row['clipped']/n,
            underflow_pct=100*row['underflow_nonzero_to_zero']/n,
            relative_l2=math.sqrt(row['squared_error']/row['squared_energy']) if row['squared_energy'] else 0.)
        summaries[phase+'/'+kind]=row
    result = {'classification':'BF16_actual_valid_KV_writes_fixed_FP8_QDQ_distribution_not_quality',
        'method':METHOD,'rows':len(rows),'writer_calls_per_layer':dict(zip(('prefill','decode'),calls[0])),
        'valid_tokens_written_per_layer_by_phase':token_counts,'aggregate':summaries,
        'heldout_chat_only_not_long_document_or_code_coverage':content_scope == CHAT_SCOPE,
        'FP8_generation_or_model_quality_certified':False,'headline_performance_result':False}
    if content_scope == LONG_QA_SCOPE:
        result.update(content_scope=LONG_QA_SCOPE, long_document_QA_only_not_code_coverage=True)
    return result


class RangeProbeCoordinator:
    def __init__(self, output: Path, ticket: dict) -> None:
        self.root=output/'range_probe';self.root.mkdir(mode=0o700)
        self.identity={'run_id':ticket['request']['run_id'],'request_sha256':ticket['request_sha256']}
        self.content_scope=ticket['request'].get('workload',{}).get('content_scope', CHAT_SCOPE)
        self.reset_ns=None
        write_new_json(self.root/'config.json',dict(self.identity,method=METHOD,layers=48,heads=8,
            dimension=128,key_scale=.125,value_scale=.0625,max_output_bytes=2*2**20))

    def _command(self, name: str) -> dict:
        command=dict(self.identity,command=name);write_new_json(self.root/(name+'.json'),command)
        path=self.root/(name+'_DONE.json');until=time.monotonic()+60
        while not path.exists():
            if (self.root/'ERROR.json').exists():raise ValueError((self.root/'ERROR.json').read_text())
            if time.monotonic()>=until:raise TimeoutError('KV range '+name+' acknowledgement missing')
            time.sleep(.1)
        if path.is_symlink() or path.stat().st_size>2*2**20:raise ValueError('Invalid acknowledgement path/size')
        result=json.loads(path.read_text())
        if any(result.get(k)!=v for k,v in command.items()) or result.get('command_sha256')!=digest(command):
            raise ValueError('Acknowledgement identity/raw command SHA differs')
        return result

    def reset_after_bootstrap(self) -> None:
        result=self._command('RESET');self.reset_ns=result['reset_monotonic_ns']
        if type(self.reset_ns)is not int or self.reset_ns<=0:raise ValueError('Missing reset epoch')

    def export_after_requests(self) -> dict:
        if self.reset_ns is None:raise ValueError('No acknowledged reset before requests')
        result=self._command('EXPORT');audit=audit_export(result,**self.identity,reset_ns=self.reset_ns,
                                                        content_scope=self.content_scope)
        write_new_json(self.root/'audit.json',audit)
        return audit
