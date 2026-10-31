"""Send frozen natural payloads through installed AIPerf to CPU loopback only.

No model/GPU inference or throughput measurement. Multi-turn document payloads
must remain byte-equivalent in both ordinary and pre-encoded transport paths.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path


async def validate(paths: list[Path]) -> dict:
    import aiohttp.web
    from aiperf.common.models import ModelEndpointInfo, RequestInfo
    from aiperf.dataset.loader.inputs_json import InputsJsonPayloadLoader
    from aiperf.workers.inference_client import InferenceClient

    received=[]
    async def capture(request):
        received.append(await request.read())
        body='data: '+json.dumps({'id':'CPU_FIXTURE_ONLY','object':'chat.completion.chunk',
            'choices':[{'index':0,'delta':{'role':'assistant','content':'CPU fixture'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3}})+'\n\ndata: [DONE]\n\n'
        return aiohttp.web.Response(body=body.encode(),content_type='text/event-stream')
    application=aiohttp.web.Application(client_max_size=8*2**20)
    application.router.add_post('/v1/chat/completions',capture)
    runner=aiohttp.web.AppRunner(application);await runner.setup()
    site=aiohttp.web.TCPSite(runner,'127.0.0.1',0);await site.start()
    port=site._server.sockets[0].getsockname()[1]
    endpoint=ModelEndpointInfo(models={'models':[{'name':'Qwen3.8-27B'}],'model_selection_strategy':'round_robin'},
        endpoint={'type':'chat','base_urls':[f'http://127.0.0.1:{port}'],'streaming':True,'timeout':30,
                  'use_legacy_max_tokens':True,'use_server_token_count':True})
    client=InferenceClient(endpoint,'new-model-CPU-wire-fixture')
    await client.transport._init_aiohttp_client()
    reports=[]
    try:
        for path in paths:
            source=json.loads(path.read_text())
            expected={row['session_id']:row['payloads'] for row in source['data']}
            if len(expected)!=len(source['data']):raise ValueError('Frozen dataset repeats a session identifier')
            loader=InputsJsonPayloadLoader(filename=path)
            conversations=loader.convert_to_conversations(loader.load_dataset())
            if {row.session_id for row in conversations}!=set(expected):raise ValueError('AIPerf changed session identities')
            hashes=[];turn_count=0
            for conversation in conversations:
                payloads=expected[conversation.session_id]
                if len(conversation.turns)!=len(payloads):raise ValueError('AIPerf dropped a document/session turn')
                for index,(turn,payload) in enumerate(zip(conversation.turns,payloads)):
                    if turn.raw_payload!=payload:raise ValueError('Actual loader changed a frozen raw payload')
                    if (payload.get('model')!='Qwen3.8-27B' or payload.get('ignore_eos',False)
                            or payload.get('stream') is not True
                            or payload.get('chat_template_kwargs')!={'enable_thinking':False}):
                        raise ValueError('Production model/natural stopping/template configuration changed')
                    # The worker exposes only the session prefix through the
                    # current turn. Passing future turns makes the raw-payload
                    # transport choose the last future request instead.
                    info=RequestInfo(model_endpoint=endpoint,turns=conversation.turns[:index+1],credit_num=turn_count,
                        credit_phase='profiling',conversation_id=conversation.session_id,turn_index=index,
                        x_request_id=f'fixture-{turn_count}',x_correlation_id=f'fixture-{turn_count}')
                    before=len(received)
                    record=await client._send_request_to_transport(info)
                    if record.error or len(received)!=before+1:raise ValueError('Actual CPU transport did not finish one request')
                    wire=received[-1]
                    if json.loads(wire)!=payload or wire!=info.payload_bytes:
                        raise ValueError('Actual HTTP body changed for '+conversation.session_id+' turn '+str(index))
                    info.payload_bytes=wire
                    replay=await client._send_request_to_transport(info)
                    if replay.error or received[-1]!=wire:raise ValueError('Pre-encoded/mmap path changed payload bytes')
                    hashes.append(hashlib.sha256(wire).hexdigest());turn_count+=1
            reports.append({'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'input':str(path),
                            'sessions':len(conversations),'turns':turn_count,'HTTP_requests':turn_count*2,
                            'wire_payload_sha256':hashes})
    finally:
        await client.transport._close_aiohttp_client();await runner.cleanup()
    return {'state':'CPU_ACTUAL_AIPERF_LOADER_TRANSPORT_WIRE_VERIFIED','AIPerf_version':importlib.metadata.version('aiperf'),
            'reports':reports,'model_predictions_generated':False,'GPU_used':False,
            'performance_measured':False,'scope':'Loader/transport and all original turns; full CLI warmup/session cursor still separately checked'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=asyncio.run(validate(args.input))
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')


if __name__=='__main__':main()
