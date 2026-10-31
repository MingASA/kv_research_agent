#!/usr/bin/env python3
"""Run real AIPerf against a loopback fixture to verify timestamp units and pacing.

The response is a fixed fixture, not a model prediction or performance result.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


async def validate(model: Path, out: Path, payload_source: Path | None = None) -> dict:
    from aiohttp import web
    from transformers import AutoTokenizer

    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    arrivals = []

    async def models(request):
        return web.json_response({'object': 'list', 'data': [{'id': 'Qwen2.5-14B-Instruct', 'object': 'model'}]})

    async def serve(request):
        body = await request.json()
        arrivals.append({'monotonic_ns': time.monotonic_ns(), 'body_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()})
        rendered = tokenizer.apply_chat_template(body['messages'], tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(rendered, add_special_tokens=False))
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        first = {'id': 'fixture', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': 'x'}, 'finish_reason': None}]}
        last = {'id': 'fixture', 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': count, 'completion_tokens': 1, 'total_tokens': count + 1}}
        for packet in [first, last]:
            await response.write(('data: ' + json.dumps(packet) + '\n\n').encode())
        await response.write(b'data: [DONE]\n\n'); await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get('/v1/models', models)
    app.router.add_get('/health', lambda request: web.Response(text='ok'))
    app.router.add_post('/v1/chat/completions', serve)
    runner = web.AppRunner(app); await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0); await site.start()
    port = site._server.sockets[0].getsockname()[1]
    cases = {}
    try:
        for kind in (['mooncake_payload'] if payload_source is not None else ['mooncake_trace', 'burst_gpt_trace']):
            directory = out / kind; directory.mkdir()
            times = [1.0, 1.1, 1.5, 2.1]
            source = directory / ('fixture.jsonl' if kind != 'burst_gpt_trace' else 'fixture.csv')
            expected_payloads = None
            if kind == 'mooncake_payload':
                original = [json.loads(line) for line in payload_source.read_text().splitlines()][:4]
                if len(original) != 4:
                    raise ValueError('Payload fixture needs four frozen production input records')
                source.write_text(''.join(json.dumps({**row, 'timestamp': int(t * 1000)}) + '\n'
                                          for row, t in zip(original, times)))
                expected_payloads = [hashlib.sha256(json.dumps(row['payload'], sort_keys=True).encode()).hexdigest()
                                     for row in original]
            elif kind == 'mooncake_trace':
                source.write_text(''.join(json.dumps({'timestamp': int(t * 1000), 'input_length': 1024,
                    'output_length': 8, 'hash_ids': [1, i + 10]}) + '\n' for i, t in enumerate(times)))
            else:
                with source.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['Timestamp', 'Request tokens', 'Response tokens'])
                    writer.writeheader()
                    for t in times: writer.writerow({'Timestamp': t, 'Request tokens': 1024, 'Response tokens': 8})
            command = [sys.executable, '-B', '-m', 'aiperf', 'profile', '--model', 'Qwen2.5-14B-Instruct',
                '--tokenizer', str(model), '--url', 'http://127.0.0.1:' + str(port), '--endpoint-type', 'chat',
                '--streaming', '--use-server-token-count', '--use-legacy-max-tokens', '--random-seed', '20261026',
                '--input-file', str(source), '--custom-dataset-type', 'mooncake_trace' if kind == 'mooncake_payload' else kind,
                '--fixed-schedule', '--fixed-schedule-auto-offset',
                '--request-count', '4', '--concurrency', '16', '--workers-max', '1', '--no-gpu-telemetry', '--ui', 'none',
                '--request-timeout-seconds', '30', '--export-level', 'raw', '--output-artifact-dir', str(directory / 'artifacts')]
            before = len(arrivals)
            with (directory / 'aiperf.log').open('x') as log:
                process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                try:
                    code = await asyncio.wait_for(process.wait(), 180)
                except asyncio.TimeoutError:
                    os.killpg(process.pid, signal.SIGTERM); await process.wait()
                    raise TimeoutError('Local CPU AIPerf fixture timed out; original logs retained')
            observed = arrivals[before:]
            (directory / 'wire_arrivals.json').write_text(json.dumps(observed, indent=2) + '\n')
            if code != 0:
                raise ValueError('Actual AIPerf CPU fixture failed; inspect ' + str(directory / 'aiperf.log'))
            if len(observed) != 4:
                raise ValueError('Actual timestamped fixture dropped or repeated requests')
            if expected_payloads is not None and expected_payloads != [row['body_sha256'] for row in observed]:
                raise ValueError('Actual AIPerf changed the frozen replay payload, order, or generation controls')
            relative = [(row['monotonic_ns'] - observed[0]['monotonic_ns']) / 1e9 for row in observed]
            expected = [t - times[0] for t in times]
            errors = [a - b for a, b in zip(relative, expected)]
            cases[kind] = {'command': command, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                           'received': 4, 'expected_relative_seconds': expected, 'observed_relative_seconds': relative,
                           'errors_seconds': errors, 'within_200ms_fixture_bound': max(map(abs, errors)) <= .2}
            if expected_payloads is not None:
                cases[kind].update(parent_payload_file_sha256=hashlib.sha256(payload_source.read_bytes()).hexdigest(),
                                   all_frozen_payloads_unchanged_on_wire=True,
                                   fixture_timestamps_override_source=True)
            if not cases[kind]['within_200ms_fixture_bound']:
                raise ValueError('Actual loopback arrivals did not match the declared fixture pacing bound')
    finally:
        await runner.cleanup()
    return {'schema': 'kv-aiperf-schedule-cpu-v1', 'classification': 'cpu_loopback_fixture_only',
            'model_inference_performed': False, 'gpu_used': False, 'cases': cases,
            'limitation': 'Short deterministic fixture, not a production trace or server performance result; GPU-load pacing still needs runtime audit'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--payload-source', type=Path)
    args = parser.parse_args()
    report = asyncio.run(validate(args.tokenizer.absolute(), args.out.absolute(), args.payload_source))
    (args.out / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'evidence': str(args.out / 'acceptance.json'), 'cpu_only': True, 'cases': list(report['cases'])}))


if __name__ == '__main__':
    main()
