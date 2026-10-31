#!/usr/bin/env python3
"""Replay frozen raw payloads through installed AIPerf code to a loopback fixture.

This validates CPU protocol transport, generates no model predictions and is
never a throughput/latency benchmark or an entry in results/INDEX.csv.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.durable import digest, write_new_json


async def validate(paths: list[Path], protocol: str = 'fixed_decode') -> dict:
    import aiohttp.web
    from aiperf.common.models import ModelEndpointInfo, RequestInfo
    from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
    from aiperf.workers.inference_client import InferenceClient
    from aiperf.common.models.record_models import MetricRecordMetadata, RawRecordInfo

    def validate_buffered_record(record, payload: bytes) -> None:
        from controller.native_tools import parse_buffered_chat_response

        if len(record.responses) != 1 or record.responses[0].__class__.__name__ != 'TextResponse':
            raise ValueError('Actual non-streaming transport did not preserve its single JSON TextResponse')
        parsed = parse_buffered_chat_response(json.loads(record.responses[0].text))
        # Exercise the actual raw-export model and the same JSON serialisation
        # used by RawRecordWriterProcessor; TextResponse itself is a dataclass.
        metadata = MetricRecordMetadata(session_num=0, request_start_ns=1, request_end_ns=2,
            worker_id='cpu-fixture', record_processor_id='cpu-fixture', benchmark_phase='profiling')
        exported = RawRecordInfo(metadata=metadata, start_perf_ns=record.start_perf_ns,
            payload_bytes=payload, status=record.status, responses=record.responses, error=record.error)
        dumped = exported.model_dump(exclude_none=True, mode='json')
        raw_response = json.loads(json.dumps(dumped))['responses'][0]
        if (raw_response['text'] != record.responses[0].text
                or raw_response['perf_ns'] != record.responses[0].perf_ns
                or parse_buffered_chat_response(json.loads(raw_response['text'])) != parsed):
            raise ValueError('Actual raw-export serialisation changed the buffered native response')

    received = []
    if protocol not in ('fixed_decode', 'native_tools', 'native_tools_buffered', 'heldout_natural_chat', 'heldout_long_document_QA'):
        raise ValueError('Unknown CPU wire protocol')
    tool_protocol = protocol.startswith('native_tools')
    buffered = protocol == 'native_tools_buffered'

    async def capture(request):
        received.append(await request.read())
        if tool_protocol:
            payload = json.loads(received[-1])
            name = payload['tools'][0]['function']['name']
            if buffered:
                return aiohttp.web.json_response({'object': 'chat.completion', 'choices': [{'index': 0,
                    'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'CPU_only',
                        'type': 'function', 'function': {'name': name, 'arguments': '{"cpu_fixture_only":true}'}}]},
                    'finish_reason': 'tool_calls'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}})
            chunks = [
                {'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'tool_calls': [
                    {'index': 0, 'id': 'cpu-fixture-only', 'type': 'function', 'function': {
                        'name': name, 'arguments': '{"cpu_fixture_only":'}}]}}]},
                {'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'tool_calls': [
                    {'index': 0, 'function': {'arguments': 'true}'}}]}}]},
                {'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}],
                 'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3}}]
            body = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks) + 'data: [DONE]\n\n'
            return aiohttp.web.Response(body=body.encode(), content_type='text/event-stream')
        return aiohttp.web.Response(body=b'data: {"choices":[],"usage":{}}\n\ndata: [DONE]\n\n',
                                    content_type="text/event-stream")

    application = aiohttp.web.Application(client_max_size=256 * 1024)
    application.router.add_post("/v1/chat/completions", capture)
    runner = aiohttp.web.AppRunner(application)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    endpoint = ModelEndpointInfo(models={"models": [{"name": "Qwen2.5-14B-Instruct"}], "model_selection_strategy": "round_robin"},
                                 endpoint={"type": "chat", "base_urls": [f"http://127.0.0.1:{port}"],
                                           "streaming": not buffered, "timeout": 10,
                                           "use_legacy_max_tokens": True, "use_server_token_count": True})
    client = InferenceClient(endpoint, "cpu-wire-fixture")
    await client.transport._init_aiohttp_client()
    reports = []
    try:
        for path in paths:
            parent = json.loads(path.read_text())
            if protocol == 'heldout_long_document_QA':
                from scripts.controller.range_probe import validate_longcontext_inputs
                validate_longcontext_inputs(parent, json.loads((path.parent / 'manifest.json').read_text()))
            loader = InputsJsonPayloadLoader(filename=path)
            conversations = loader.convert_to_conversations(loader.load_dataset())
            if len(conversations) != (128 if protocol in ('heldout_natural_chat', 'heldout_long_document_QA') else 274 if tool_protocol else 136):
                raise ValueError("AIPerf dropped/duplicated frozen sessions")
            expected = {row["session_id"]: row["payloads"][0] for row in parent["data"]}
            wire_hashes = []
            for index, conversation in enumerate(conversations):
                raw = expected[conversation.session_id]
                if conversation.turns[0].raw_payload != raw:
                    raise ValueError("Actual loader/model changed the raw payload")
                info = RequestInfo(model_endpoint=endpoint, turns=conversation.turns,
                                   credit_num=index, credit_phase="profiling", conversation_id=conversation.session_id,
                                   turn_index=0, x_request_id=f"fixture-{index}", x_correlation_id=f"fixture-{index}")
                before = len(received)
                record = await client._send_request_to_transport(info)
                if record.error or len(received) != before + 1:
                    raise ValueError("Actual transport did not complete one HTTP request: " + str(record.error))
                if buffered:
                    validate_buffered_record(record, info.payload_bytes)
                wire = received[-1]
                actual = json.loads(wire)
                if actual != raw or info.payload_bytes != wire:
                    raise ValueError("Serialized HTTP body differs from frozen payload")
                if protocol == 'fixed_decode':
                    if actual.get("ignore_eos") is not True or actual.get("max_tokens") != 512:
                        raise ValueError("Fixed decode controls were dropped from wire")
                elif protocol == 'heldout_natural_chat':
                    if (actual.get('ignore_eos', False) is not False or actual.get('max_tokens') != 1024
                            or actual.get('stream') is not True or not actual.get('messages') or actual.get('tools')):
                        raise ValueError('Heldout natural chat controls were changed on the actual wire')
                elif protocol == 'heldout_long_document_QA':
                    if (actual.get('ignore_eos', False) is not False or actual.get('max_tokens') not in (32,64,128)
                            or actual.get('stream') is not True or not actual.get('messages') or actual.get('tools')):
                        raise ValueError('Original long-document QA prompt/cap was changed on the wire')
                elif (actual.get('ignore_eos') is not False or actual.get('max_tokens') != 1024
                      or not actual.get('tools') or actual.get('tool_choice') != 'auto'
                      or actual.get('stream') is not (not buffered)):
                    raise ValueError('Native tools/natural stopping controls were changed')
                wire_hashes.append(hashlib.sha256(wire).hexdigest())
                # Also exercise the worker's already-encoded/mmap payload fast path.
                info.payload_bytes = wire
                record = await client._send_request_to_transport(info)
                if record.error or received[-1] != wire:
                    raise ValueError("Preencoded payload fast path changed bytes")
                if buffered:
                    validate_buffered_record(record, wire)
            reports.append({"input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "sessions_verified": len(conversations), "http_requests": len(conversations) * 2,
                            "wire_payload_sha256": wire_hashes})
    finally:
        await client.transport._close_aiohttp_client()
        await runner.cleanup()
    parsed_tool_fixture = None
    if tool_protocol:
        fixture = {'object': 'chat.completion.chunk', 'choices': [{'delta': {'tool_calls': [
            {'index': 0, 'function': {'name': 'cpu_fixture', 'arguments': '{"only":true}'}}]}}]}
        if buffered:
            fixture['object'] = 'chat.completion'
            fixture['choices'][0]['message'] = fixture['choices'][0].pop('delta')
        parsed = client.endpoint.extract_chat_response_data(fixture)
        if parsed.__class__.__name__ != 'ToolCallResponseData' or parsed.get_text() != 'cpu_fixture{"only":true}':
            raise ValueError('Actual installed chat parser dropped native tool-only content')
        fixture['choices'][0]['message' if buffered else 'delta']['content'] = 'CPU fixture prose.'
        if client.endpoint.extract_chat_response_data(fixture).get_text() != 'CPU fixture prose.cpu_fixture{"only":true}':
            raise ValueError('Actual installed chat parser dropped mixed prose/tool content')
        parsed_tool_fixture = {'tool_only_and_mixed_tool_text_preserved': True, 'real_model_or_quality_result': False}
    return {"schema": "kv-aiperf-cpu-wire-validation-v1", "classification": "confirmed_cpu_only",
            "aiperf_version": importlib.metadata.version("aiperf"), "gpu_used": False,
            "model_inference_performed": False, "loopback_fixture_only": True,
            "classes": ["InputsJsonPayloadLoader", "InputsJsonSession", "Conversation", "Turn", "RequestInfo",
                        "InferenceClient", "AioHttpTransport", "AioHttpClient"],
            "protocol": {"kind": protocol, "ignore_eos": protocol == 'fixed_decode',
                         "max_tokens": 'original_task_specific_32_64_128' if protocol == 'heldout_long_document_QA' else 512 if protocol == 'fixed_decode' else 1024,
                         "raw_dict_and_preencoded_bytes": True, 'parsed_tool_fixture': parsed_tool_fixture,
                         'buffered_TextResponse_and_actual_RawRecordInfo_JSON_export_verified': buffered},
            "inputs": reports,
            "implementation_sha256": {str(module.__file__): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                      for module in (__import__(InputsJsonPayloadLoader.__module__, fromlist=["x"]),
                                                     __import__(InferenceClient.__module__, fromlist=["x"]),
                                                     __import__(client.transport.__module__, fromlist=["x"]),
                                                     __import__(client.endpoint.__module__, fromlist=["x"]))},
            "limitation": "No model/server generation was tested; runtime must still verify every output token count and real Graph replay"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument('--protocol', choices=('fixed_decode', 'native_tools', 'native_tools_buffered', 'heldout_natural_chat', 'heldout_long_document_QA'), default='fixed_decode')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Evidence must be a new file")
    report = asyncio.run(validate(args.input, args.protocol))
    write_new_json(args.out, report)
    print(json.dumps({"evidence": str(args.out), "sessions_verified": sum(row["sessions_verified"] for row in report["inputs"]),
                      "gpu_used": False, "model_inference_performed": False}))


if __name__ == "__main__":
    main()
