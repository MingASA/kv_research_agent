#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Verify real AIPerf error exports with a CPU loopback server, never a model."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.controller.durable import canonical_bytes, digest, write_new_json
from scripts.controller.trace_serving import audit_trace


async def validate(model: Path, out: Path, include_timeout: bool = False) -> dict:
    from aiohttp import web
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    wire = []

    async def serve(request: web.Request) -> web.StreamResponse:
        body = await request.json(); wire.append(body)
        if body['messages'][0]['content'] == 'CPU rejected fixture':
            return web.json_response({'error': {'message': 'CPU-only intentional HTTP429 fixture'}}, status=429)
        if body['messages'][0]['content'] == 'CPU timeout fixture':
            await asyncio.sleep(2)
            return web.json_response({'fixture':'intentional response after client deadline'})
        text = tokenizer.apply_chat_template(body['messages'], tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(text, add_special_tokens=False))
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        packet = {'id': 'cpu-fixture', 'object': 'chat.completion.chunk',
            'choices': [{'index': 0, 'delta': {'content': 'x'}, 'finish_reason': None}]}
        await response.write(('data: '+json.dumps(packet)+'\n\n').encode())
        last = {'id': 'cpu-fixture', 'object': 'chat.completion.chunk',
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}],
            'usage': {'prompt_tokens': count, 'completion_tokens': 1, 'total_tokens': count+1}}
        await response.write(('data: '+json.dumps(last)+'\n\n').encode())
        await response.write(b'data: [DONE]\n\n'); await response.write_eof()
        return response

    async def models(request: web.Request) -> web.Response:
        return web.json_response({'object': 'list', 'data': [{'id': 'Qwen2.5-14B-Instruct'}]})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', serve)
    app.router.add_get('/v1/models', models)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    rows, records = [], []
    # Accelerated fixture clock only: not the production allowed scaling range.
    for i, ms in enumerate((0, 70000, 100000, 120000, 150000) if include_timeout else (0,70000,100000,120000)):
        payload = {'model': 'Qwen2.5-14B-Instruct', 'messages': [{'role': 'user',
            'content': 'CPU timeout fixture' if i == 4 else 'CPU rejected fixture' if i == 3 else 'CPU successful fixture '+str(i)}],
            'ignore_eos': True, 'max_tokens': 1, 'stream': True, 'stream_options': {'include_usage': True}}
        text = tokenizer.apply_chat_template(payload['messages'], tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(text, add_special_tokens=False))
        identity = 'cpu-outcome-'+str(i)
        rows.append({'session_id': identity, 'timestamp': ms/100, 'output_length': 1, 'payload': payload})
        records.append({'replay_id': identity, 'source_timestamp_ms': ms, 'source_output_tokens': 1,
            'rendered_chat_tokens_cpu': count, 'logical_phase': 'warm_cache' if i == 0 else 'measurement',
            'payload_sha256': digest(payload)})
    source = out/'fixture.jsonl'; source.write_bytes(b''.join(canonical_bytes(row) for row in rows))
    manifest = out/'fixture_manifest.json'
    write_new_json(manifest, {'selected_requests': len(rows), 'records': records, 'arrival_scaling': 100,
        'first_retained_timestamp_ms': 0, 'window_begin_ms': 0, 'window_end_ms': 300000})
    command = [sys.executable, '-B', '-m', 'aiperf', 'profile', '--model', 'Qwen2.5-14B-Instruct',
        '--tokenizer', str(model), '--url', 'http://127.0.0.1:'+str(port), '--endpoint-type', 'chat',
        '--streaming', '--use-server-token-count', '--use-legacy-max-tokens', '--input-file', str(source),
        '--custom-dataset-type', 'mooncake_trace', '--fixed-schedule', '--fixed-schedule-auto-offset',
        '--request-count', str(len(rows)), '--concurrency', '16', '--workers-max', '1', '--no-gpu-telemetry', '--ui', 'none',
        '--request-timeout-seconds', '1' if include_timeout else '10', '--export-level', 'raw', '--output-artifact-dir', str(out/'artifacts')]
    try:
        with (out/'aiperf.log').open('x') as log:
            process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT)
            try:
                code = await asyncio.wait_for(process.wait(), 180)
            except asyncio.TimeoutError:
                process.terminate(); await process.wait()
                raise TimeoutError('CPU AIPerf outcome fixture timed out')
        write_new_json(out/'wire.json', wire)
        if code not in (0, 1):
            raise ValueError('Unexpected actual CPU AIPerf process exit')
        result = audit_trace(out/'artifacts', source, manifest)
        if (result['all_attempts_verified'] != len(rows) or result['http_status_counts'].get('200') != 3
                or result['http_status_counts'].get('429') != 1):
            raise ValueError('Actual AIPerf failed to retain both success and rejection')
        if result['phases']['measurement']['successes'] != 2 or result['phases']['measurement']['errors'] != (2 if include_timeout else 1):
            raise ValueError('Rejection was silently counted as completed useful service')
        return {'classification': 'real_AIPerf_CPU_loopback_only', 'gpu_used': False, 'model_loaded': False,
                'aiperf_exit_code': code, 'wire_requests': len(wire), 'command': command, 'audit': result,
                'limitation': 'Fixed one-token SSE fixture and accelerated fixture clock; no production performance result'}
    finally:
        await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--include-timeout',action='store_true')
    args = parser.parse_args()
    report = asyncio.run(validate(args.tokenizer.absolute(), args.out.absolute(),args.include_timeout))
    write_new_json(args.out/'acceptance.json', report)
    print(json.dumps({'cpu_only': True, 'http_statuses': report['audit']['http_status_counts'],
                      'actual_requests': report['wire_requests']}))


if __name__ == '__main__':
    main()
