# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Natural native-tool API evidence, retaining every original attempt and error."""

from __future__ import annotations

import collections
import datetime
import json
import math
from pathlib import Path
import re

from . import evidence
from .durable import digest


def _unique_object(items: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError('Duplicate JSON tool argument key')
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError('Nonfinite JSON tool argument: ' + value)


def _finite_values(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite JSON numeric argument')
    if isinstance(value, dict):
        for item in value.values():
            _finite_values(item)
    elif isinstance(value, list):
        for item in value:
            _finite_values(item)


def parse_native_tool_calls(calls: list[dict]) -> list[dict]:
    if not isinstance(calls, list):
        raise ValueError('Native tool_calls must be an array')
    output, identifiers = [], set()
    for call in calls:
        if (not isinstance(call, dict) or call.get('type') != 'function'
                or not isinstance(call.get('id'), str) or not call['id']
                or call['id'] in identifiers or not isinstance(call.get('function'), dict)):
            raise ValueError('Malformed or duplicate native tool-call identity')
        identifiers.add(call['id'])
        function = call['function']
        name, arguments = function.get('name'), function.get('arguments')
        if (not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name) is None
                or not isinstance(arguments, str) or len(arguments.encode()) > 256 * 2**10):
            raise ValueError('Malformed native function name or argument string')
        value = json.loads(arguments, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise ValueError('Native function arguments must be a JSON object')
        _finite_values(value)
        output.append({name: value})
    return output


def parse_sse_packets(packets: list[dict]) -> dict:
    calls, content, finish, usage, done = {}, [], None, None, False
    for packet in packets:
        if packet.get('name') != 'data':
            continue
        raw = packet.get('value')
        if done:
            raise ValueError('SSE data after DONE')
        if raw == '[DONE]':
            done = True
            continue
        item = json.loads(raw)
        if item.get('error'):
            raise ValueError('Server SSE error: ' + str(item['error'])[:512])
        if item.get('usage'):
            usage = item['usage']
        for choice in item.get('choices', []):
            if choice.get('index') != 0:
                raise ValueError('Unexpected multiple completion choices')
            finish = choice.get('finish_reason') or finish
            delta = choice.get('delta', {})
            if delta.get('content'):
                content.append(delta['content'])
            for piece in delta.get('tool_calls', []):
                index = piece.get('index')
                if type(index) is not int or not 0 <= index < 32:
                    raise ValueError('Missing or unbounded streamed tool-call index')
                call = calls.setdefault(index, {'id': '', 'type': 'function',
                                                'function': {'name': '', 'arguments': ''}})
                if piece.get('type') not in (None, 'function'):
                    raise ValueError('Unexpected streamed tool type')
                if piece.get('id'):
                    if call['id'] and call['id'] != piece['id']:
                        raise ValueError('Streamed tool-call identity changed')
                    call['id'] = piece['id']
                for field in ('name', 'arguments'):
                    value = piece.get('function', {}).get(field)
                    if value is not None:
                        if not isinstance(value, str):
                            raise ValueError('Native tool delta must contain strings')
                        call['function'][field] += value
    ordered = [calls[index] for index in sorted(calls)]
    if (not done or finish not in ('stop', 'length', 'tool_calls') or usage is None
            or sorted(calls) != list(range(len(calls))) or finish == 'tool_calls' and not calls):
        raise ValueError('Incomplete native-tool SSE termination/usage')
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        if type(usage.get(key)) is not int or usage[key] < 1:
            raise ValueError('Invalid server usage')
    if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
        raise ValueError('SSE total usage differs')
    # A syntactically malformed argument remains a recorded production failure.
    parse_native_tool_calls(ordered)
    if not ordered and not ''.join(content):
        raise ValueError('No visible content or native tool call')
    return {'tool_calls': ordered, 'content': ''.join(content), 'finish_reason': finish,
            'usage': usage, 'done': True}


def parse_buffered_chat_response(item: dict) -> dict:
    choices, usage = item.get('choices'), item.get('usage')
    if item.get('error') or not isinstance(choices, list) or len(choices) != 1 or usage is None:
        raise ValueError('Incomplete native-tool JSON response')
    choice = choices[0]
    if choice.get('index') != 0 or choice.get('finish_reason') not in ('stop', 'length', 'tool_calls'):
        raise ValueError('Invalid native-tool JSON choice/finish')
    message = choice.get('message', {})
    calls, content = message.get('tool_calls') or [], message.get('content') or ''
    parse_native_tool_calls(calls)
    if not calls and not content or choice['finish_reason'] == 'tool_calls' and not calls:
        raise ValueError('No complete native tool call or visible content')
    for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        if type(usage.get(name)) is not int or usage[name] < 1:
            raise ValueError('Invalid complete-JSON server usage')
    if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
        raise ValueError('Complete-JSON server usage mismatch')
    return {'tool_calls': calls, 'content': content, 'finish_reason': choice['finish_reason'],
            'usage': usage, 'complete_json_body': True}


def validate_native_inputs(source: Path, manifest_path: Path) -> None:
    frozen = json.loads(source.read_text())
    manifest = json.loads(manifest_path.read_text())
    if digest(frozen) != manifest['dataset_sha256'] or len(frozen['data']) != 274:
        raise ValueError('Frozen full native-tool request cohort changed')
    records = {row['session_id']: row for row in manifest['records']}
    buffered = manifest.get('response_mode', 'streaming') == 'buffered'
    if len(records) != 274 or len({row['session_id'] for row in frozen['data']}) != 274:
        raise ValueError('Duplicate or incomplete native-tool source session IDs')
    for index, session in enumerate(frozen['data']):
        record = records[session['session_id']]
        if len(session['payloads']) != 1:
            raise ValueError('Exactly one original tool request per session required')
        payload = session['payloads'][0]
        if (record['phase'] != ('warmup' if index < 16 else 'measurement')
                or digest(payload) != record['payload_sha256'] or not payload.get('tools')
                or payload.get('tool_choice') != 'auto' or payload.get('ignore_eos') is not False
                or payload.get('max_tokens') != 1024 or payload.get('stream') is not (not buffered)
                or payload.get('temperature') != 0 or payload.get('top_p') != 1
                or payload.get('stream_options') != (None if buffered else {'include_usage': True})):
            raise ValueError('Natural source tool protocol/order differs')


def audit_native_tools(artifacts: Path, source: Path, manifest_path: Path) -> dict:
    validate_native_inputs(source, manifest_path)
    frozen = json.loads(source.read_text())
    manifest = json.loads(manifest_path.read_text())
    buffered = manifest.get('response_mode', 'streaming') == 'buffered'
    if digest(frozen) != manifest['dataset_sha256'] or len(frozen['data']) != 274:
        raise ValueError('Frozen full native-tool request cohort changed')
    payloads = {row['session_id']: row['payloads'][0] for row in frozen['data']}
    phases = {row['session_id']: 'warmup' if row['phase'] == 'warmup' else 'profiling'
              for row in manifest['records']}
    metrics, sessions = {}, set()
    for line in (artifacts / 'profile_export.jsonl').read_text().splitlines():
        row = json.loads(line); meta = row['metadata']
        identity, session = meta['x_request_id'], meta['conversation_id']
        if (identity in metrics or session in sessions or session not in payloads
                or meta['benchmark_phase'] != phases[session] or meta['turn_index'] != 0
                or meta.get('was_cancelled') is not False or meta.get('context_overflow_skip') is not False):
            raise ValueError('Lost/duplicate/cancelled native-tool cohort or phase')
        metrics[identity] = row; sessions.add(session)
    if sessions != set(payloads):
        raise ValueError('Every frozen request must have exactly one actual attempt')
    results, seen, counts, outputs = [], set(), collections.Counter(), []
    for line in (artifacts / 'profile_export_raw.jsonl').read_text().splitlines():
        raw = json.loads(line); meta = raw['metadata']; identity = meta['x_request_id']
        if identity in seen or identity not in metrics:
            raise ValueError('Raw wire identity missing/duplicated')
        seen.add(identity); row = metrics[identity]; session = meta['conversation_id']
        if (any(meta.get(k) != row['metadata'].get(k) for k in ('conversation_id', 'benchmark_phase', 'turn_index', 'request_start_ns', 'request_end_ns'))
                or raw['payload'] != payloads[session] or digest(raw['payload']) != next(
                    item['payload_sha256'] for item in manifest['records'] if item['session_id'] == session)):
            raise ValueError('Actual native tools/messages/payload differs from frozen source')
        outcome = {'session_id': session, 'phase': meta['benchmark_phase'],
                   'http_status': raw.get('status'), 'aiperf_error': row.get('error'),
                   'transport_error': raw.get('error'), 'service_success': False}
        try:
            if buffered:
                if len(raw['responses']) != 1:
                    raise ValueError('Expected one complete JSON response')
                parsed = parse_buffered_chat_response(json.loads(raw['responses'][0]['text']))
            else:
                parsed = parse_sse_packets([p for response in raw['responses'] for p in response.get('packets', [])])
        except (ValueError, TypeError, KeyError, RecursionError) as error:
            outcome['native_contract_error'] = str(error)[:1024]
        else:
            outcome.update(parsed)
            if raw.get('status') == 200 and raw.get('error') is None and row.get('error') is None:
                actual = parsed['usage']; measured = row['metrics']
                for key, metric in (('prompt_tokens', 'input_sequence_length'),
                                    ('completion_tokens', 'output_token_count'), ('total_tokens', 'usage_total_tokens')):
                    evidence._close(actual[key], evidence._value(measured, metric), 'Native tool actual usage differs')
                if actual['completion_tokens'] > payloads[session]['max_tokens']:
                    raise ValueError('Actual native-tool output exceeds natural generation cap')
                outcome['service_success'] = True
                outcome['ttft_ms'] = None if buffered else evidence._value(measured, 'time_to_first_token')
                outcome['content_request_latency_ms'] = evidence._value(measured, 'request_latency')
                start, end = meta['request_start_ns'], meta['request_end_ns']
                if type(start) is not int or type(end) is not int or end <= start:
                    raise ValueError('Actual native-tool request start/last-SSE clock is invalid')
                outcome['complete_request_ms'] = (end - start) / 1e6
                outcome['request_start_ns'], outcome['request_end_ns'] = start, end
                if meta['benchmark_phase'] == 'profiling':
                    outputs.append(actual['completion_tokens'])
        if meta['benchmark_phase'] == 'profiling':
            counts['attempts'] += 1; counts['successes' if outcome['service_success'] else 'failures'] += 1
        results.append(outcome)
    if seen != set(metrics) or counts['attempts'] != 258 or len(results) != 274:
        raise ValueError('Native tool raw attempts are incomplete')
    summary = json.loads((artifacts / 'profile_export_aiperf.json').read_text())
    if (summary.get('is_complete') is not True or summary.get('was_cancelled') is not False
            or summary.get('aiperf_version') != '0.13.0'):
        raise ValueError('Actual native-tool AIPerf did not finish')
    duration = (datetime.datetime.fromisoformat(summary['end_time']) - datetime.datetime.fromisoformat(summary['start_time'])).total_seconds()
    if duration <= 0:
        raise ValueError('Invalid actual observation window')
    return {'protocol': 'native_tool_name_arguments_natural_API_v1', 'requests': results,
            'response_mode': 'buffered' if buffered else 'streaming',
            'measurement': dict(counts), 'warmup_attempts': 16, 'observation_seconds': duration,
            'successful_service_request_rps': counts['successes'] / duration,
            'actual_output_tokens': sum(outputs), 'actual_output_tps': sum(outputs) / duration,
            'official_error_summary': summary.get('error_summary'), 'task_quality_scored': False,
            'limits': ['Task-name/arguments matching is evaluated separately with the unchanged official checker',
                       'Natural tool/content text is not an executed external action or full multi-step agent task']}
