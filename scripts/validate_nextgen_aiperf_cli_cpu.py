"""Exercise full installed AIPerf CLI on frozen inputs and CPU loopback only.

External warmup and measurement datasets are separate CLI invocations. Actual
wire hashes/counts and within-session ordering must match every frozen turn.
The fixed response is a fixture: this measures no model or server performance.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter,defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time


def encoded(value: dict) -> bytes:
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


async def validate(paths: list[Path], output: Path, *, tokenizer: str='/models/Qwen3.8-27B',
                   concurrency: int=16, session_limit: bool=True,
                   heterogeneous_latency: bool=False) -> dict:
    from aiohttp import web

    arrivals=[];delays={}
    if not 1<=concurrency<=16:raise ValueError('CPU concurrency must be bounded by16')

    async def models(request):
        return web.json_response({'object':'list','data':[{'id':'Qwen3.8-27B','object':'model'}]})

    async def serve(request):
        raw=await request.read();payload=json.loads(raw)
        arrivals.append({'monotonic_ns':time.monotonic_ns(),'payload_sha256':hashlib.sha256(encoded(payload)).hexdigest(),
                         'wire_sha256':hashlib.sha256(raw).hexdigest()})
        await asyncio.sleep(delays.get(hashlib.sha256(encoded(payload)).hexdigest(),0))
        first={'id':'CPU_FIXTURE_ONLY','object':'chat.completion.chunk','model':'Qwen3.8-27B',
               'choices':[{'index':0,'delta':{'role':'assistant','content':'x'},'finish_reason':None}]}
        last={'id':'CPU_FIXTURE_ONLY','object':'chat.completion.chunk','model':'Qwen3.8-27B',
              'choices':[{'index':0,'delta':{},'finish_reason':'stop'}],
              'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
        response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
        for packet in (first,last):await response.write(b'data: '+encoded(packet)+b'\n\n')
        await response.write(b'data: [DONE]\n\n');await response.write_eof();return response

    application=web.Application(client_max_size=8*2**20)
    application.router.add_get('/v1/models',models)
    application.router.add_get('/health',lambda request:web.Response(text='ok'))
    application.router.add_post('/v1/chat/completions',serve)
    runner=web.AppRunner(application);await runner.setup()
    site=web.TCPSite(runner,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1]
    root=output.parent/'full_cli_cases';root.mkdir(mode=0o700,exist_ok=False);reports=[]
    try:
        for path in paths:
            dataset=json.loads(path.read_text())['data'];total=sum(len(row['payloads']) for row in dataset)
            directory=root/path.stem;directory.mkdir(mode=0o700)
            hashes={row['session_id']:[hashlib.sha256(encoded(payload)).hexdigest() for payload in row['payloads']]
                    for row in dataset}
            if len(hashes)!=len(dataset):raise ValueError('Duplicate frozen session identity')
            expected=Counter(value for values in hashes.values() for value in values)
            delays.clear()
            if heterogeneous_latency:
                for session,values in hashes.items():
                    for value in values:delays[value]=2.5 if 'gov_report' in session else .02
            command=[sys.executable,'-B','-m','aiperf','profile','--model','Qwen3.8-27B',
                '--tokenizer',tokenizer,'--url',f'http://127.0.0.1:{port}',
                '--endpoint-type','chat','--streaming','--use-server-token-count','--use-legacy-max-tokens',
                '--input-file',str(path),'--custom-dataset-type','inputs_json',
                '--dataset-sampling-strategy','sequential','--request-count',str(total),
                '--concurrency',str(min(concurrency,len(dataset))),'--workers-max','1','--no-gpu-telemetry',
                '--random-seed','20261027','--request-timeout-seconds','30','--ui','none',
                '--export-level','raw','--output-artifact-dir',str(directory/'artifacts')]
            if session_limit:command.extend(['--num-sessions',str(len(dataset))])
            offset=len(arrivals)
            with (directory/'aiperf.log').open('x') as log:
                process=await asyncio.create_subprocess_exec(*command,stdout=log,stderr=asyncio.subprocess.STDOUT,
                                                             start_new_session=True)
                try:code=await asyncio.wait_for(process.wait(),180)
                except asyncio.TimeoutError:
                    process.terminate()
                    try:await asyncio.wait_for(process.wait(),10)
                    except asyncio.TimeoutError:process.kill();await process.wait()
                    raise TimeoutError('Own CPU AIPerf CLI fixture exceeded180s; original logs retained')
            observed=arrivals[offset:];observed_hashes=[x['payload_sha256'] for x in observed]
            proof={'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'command':command,'process_return_code':code,'sessions':len(dataset),'expected_turns':total,
                'actual_HTTP_requests':len(observed),'all_actual_payload_counts_match':Counter(observed_hashes)==expected,
                'session_limit_enabled':session_limit,'heterogeneous_latency':heterogeneous_latency,
                'response_delay_seconds_by_payload_sha256':dict(delays),
                'wire_observations':observed,'warmup_scope':'Separate frozen dataset/CLI; no embedded warmup cursor consumption'}
            (directory/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
            if code!=0:raise ValueError('Actual AIPerf CLI returned'+str(code)+'; '+str(directory/'aiperf.log'))
            if not proof['all_actual_payload_counts_match']:raise ValueError('Full CLI changed/dropped/repeated frozen payloads: '+path.name)
            owners=defaultdict(set)
            for session,values in hashes.items():
                for value in values:owners[value].add(session)
            checked=0
            for session,values in hashes.items():
                if len(values)>1:
                    if any(len(owners[value])!=1 for value in values):raise ValueError('Ambiguous multi-turn session fingerprint')
                    if [value for value in observed_hashes if value in values]!=values:
                        raise ValueError('Full CLI changed original within-session turn order')
                    checked+=1
            proof['multi_turn_sessions_with_order_verified']=checked;reports.append(proof)
    finally:await runner.cleanup()
    return {'state':'CPU_ACTUAL_AIPERF_FULL_CLI_ALL_FROZEN_TURNS_VERIFIED',
        'AIPerf_version':importlib.metadata.version('aiperf'),'reports':reports,
        'GPU_used':False,'model_inference':False,'performance_measured':False,
        'scope':'Exact frozen payload counts and session order through real full CLI; separate external warmup inputs. '
                'Fake usage/output are protocol fixtures, never model predictions or latency/throughput results.'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tokenizer',default='/models/Qwen3.8-27B')
    parser.add_argument('--concurrency',type=int,default=16)
    parser.add_argument('--heterogeneous-latency',action='store_true')
    parser.add_argument('--request-count-only-control',action='store_true',
                        help='CPU negative control only; never a production command')
    args=parser.parse_args()
    result=asyncio.run(validate(args.input,args.out,tokenizer=args.tokenizer,concurrency=args.concurrency,
                              session_limit=not args.request_count_only_control,
                              heterogeneous_latency=args.heterogeneous_latency))
    with args.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')


if __name__=='__main__':main()
