"""Independent same-base BF16 numerical calibration, without scored answers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

from .durable import write_new_json


METHOD = 'independent_BF16_QKV_FP8_clipping_error_v1'


def _graph_gate(output: Path) -> dict:
    import torch
    from xllm.python.attention.fp8_numeric_diagnostics import NumericFlashInferBackend, _RECIPES

    if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(9,0):
        raise ValueError('Numeric Graph gate requires one admitted SM90')
    backend = object.__new__(NumericFlashInferBackend)
    backend.head_dim = 256
    backend._metadata = SimpleNamespace(is_prefill=False,is_chunked_prefill=False,
                                       slot_mapping=torch.tensor([1,2,-1],dtype=torch.int32,device='cuda'))
    backend._stats = {name:(torch.zeros((1,2,heads,4),dtype=torch.int64,device='cuda'),
                           torch.zeros((1,2,heads,2),dtype=torch.float64,device='cuda'),
                           torch.zeros((1,2,heads),dtype=torch.float32,device='cuda'))
                      for name,_,heads in _RECIPES}
    backend._calls = torch.zeros((1,2),dtype=torch.int64,device='cuda')
    tensors = [t for group in backend._stats.values() for t in group]+[backend._calls]
    k=torch.zeros((3,4,256),device='cuda',dtype=torch.bfloat16)
    v=torch.zeros_like(k);q=torch.zeros((3,24,256),device='cuda',dtype=torch.bfloat16)
    v[0,0,0]=149;v[2].fill_(1000);q[0,0,0]=500
    for _ in range(2):backend._observe(q,k,v,3)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):backend._observe(q,k,v,3)
    for t in tensors:t.zero_()
    graph.replay();torch.cuda.synchronize()
    first=backend._stats['value_current'][0].clone()
    if first[0,1,0,1].item()!=1 or first[0,1,0,0].item()!=512:
        raise ValueError('Graph diagnostic counted padding or missed real clipping')
    for t in tensors:t.zero_()
    v[0,0,0]=0;q[0,0,0]=0
    graph.replay();torch.cuda.synchronize()
    if backend._stats['value_current'][0][0,1,0,1].item()!=0:
        raise ValueError('Graph diagnostic did not observe changed data')
    result={'actual_capture_replay':True,'changed_inputs':True,'padding_excluded':True,
            'quiescent_reset':True,'SM':90,'model_loaded':False,'performance_claim':False}
    write_new_json(output,result)
    return result


def _command(directory: Path, action: str, nonce: str, server: subprocess.Popen) -> dict:
    write_new_json(directory/('numeric_'+action+'.json'),{'action':action,'request_sha256':nonce})
    ack=directory/('numeric_'+action+'_ack.json');deadline=time.monotonic()+60
    while not ack.exists():
        if (directory/'numeric_ERROR.json').exists() or server.poll() is not None:
            raise RuntimeError('Numerical diagnostic or model failed')
        if time.monotonic()>deadline:raise TimeoutError('Numerical diagnostic boundary acknowledgement missing')
        time.sleep(.1)
    value=json.loads(ack.read_bytes())
    if value['action']!=action or value['request_sha256']!=nonce:
        raise ValueError('Numerical diagnostic boundary identity mismatch')
    if action=='export' and hashlib.sha256((directory/'numeric_snapshot.json').read_bytes()).hexdigest()!=value['snapshot_sha256']:
        raise ValueError('Numerical diagnostic snapshot SHA mismatch')
    return value


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict) -> dict:
    from .nextgen import native_python_environment, _stream_case
    from .upstream_qualification import server_command, MODEL
    from .upstream_graph_observer import configure, snapshot
    from .nextgen_service import _calibration_request
    from .worker import file_sha, stop_child

    request=ticket['request']
    env=native_python_environment(request,{k:v for k,v in env.items() if not k.startswith('XLLM_')})
    env['PYTHONPATH']=request['python_source_repo']+':'+env.get('PYTHONPATH','')
    library=request['runtime_library_directory']
    if file_sha(Path(library)/'libzmq.so.5.2.5')!=request['zeromq_library_sha256']:
        raise ValueError('Pinned runtime library changed')
    env['LD_LIBRARY_PATH']=library+':'+env.get('LD_LIBRARY_PATH','')
    command=[sys.executable,'-B','-m',__package__+'.fp8_numeric_model','--graph-gate',str(output/'numeric_graph_gate.json')]
    with (output/'numeric_graph_gate.log').open('x') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    numeric=output/'numerics';numeric.mkdir()
    config=output/'numeric_config.json'
    write_new_json(config,{'method':METHOD,'output_dir':str(numeric),
                           'run_id':request['run_id'],'request_sha256':ticket['request_sha256']})
    env['XLLM_FP8_RANGE_DIAGNOSTICS_CONFIG']=str(config)
    env=configure(request,root,output,env)
    command=server_command(request,output)
    write_new_json(output/'server_command.json',command)
    inputs=json.loads(Path(request['dataset_path']).read_bytes())
    raw=(root/'input/calibration.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=request['calibration_sha256']:
        raise ValueError('Independent frozen calibration inputs changed')
    rows=json.loads(raw)['requests']
    if len(rows)!=136:raise ValueError('Independent calibration count changed')
    started=time.monotonic()
    with (output/'server.log').open('x') as log:
        server=subprocess.Popen(command,env=env,cwd=output,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                if server.poll() is not None or time.monotonic()>started+request['startup_seconds']:
                    raise RuntimeError('Independent BF16 calibration server failed startup')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:19001/v1/models',timeout=2) as response:
                        if json.load(response).get('data'):break
                except OSError:pass
                time.sleep(1)
            warm=_stream_case(inputs['cases'][0],MODEL)
            if not warm['contract_valid'] or not warm['expected_substring_present']:
                raise ValueError('Independent BF16 warmup failed')
            before=snapshot(output/'cuda_graph_calls.bin',server.pid)
            for _ in range(2):
                item=_stream_case(inputs['cases'][0],MODEL)
                if not item['contract_valid'] or not item['expected_substring_present']:
                    raise ValueError('Actual BF16 Graph request failed')
            after=snapshot(output/'cuda_graph_calls.bin',server.pid)
            graph_calls=sum(after[k]-before[k] for k in ('cudaGraphLaunch_success','cudaGraphLaunch_ptsz_success'))
            if graph_calls<=0 or after['failures']!=before['failures']:
                raise ValueError('Instrumented BF16 actual model Graph replay was not proven')
            write_new_json(output/'actual_model_graph.json',{'before':before,'after':after,'successful_Graph_calls':graph_calls})
            _command(numeric,'reset',ticket['request_sha256'],server)
            results=[]
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool, (output/'calibration_raw.jsonl').open('x') as journal:
                for result in pool.map(lambda row:_calibration_request(row,MODEL),rows):
                    journal.write(json.dumps(result,ensure_ascii=False)+'\n');journal.flush();results.append(result)
                    if time.monotonic()>started+request['max_wall_seconds']-120:
                        raise TimeoutError('Independent calibration exhausted bounded job window')
            if not all(r['contract_valid'] for r in results):
                raise ValueError('Independent calibration HTTP/token contracts failed')
            _command(numeric,'export',ticket['request_sha256'],server)
            stats=json.loads((numeric/'numeric_snapshot.json').read_bytes())
            if any(r['nonfinite'] for r in stats['rows']) or not all(sum(r['finite'] for r in stats['rows'] if r['recipe']==name)>0 for name in ('key','value_current','value_candidate','query')):
                raise ValueError('Empty/nonfinite independent model observations')
        finally:
            stop_child(server)
    summary={}
    for name in ('key','value_current','value_candidate','query'):
        subset=[r for r in stats['rows'] if r['recipe']==name]
        summary[name]={key:sum(r[key] for r in subset) for key in ('finite','clipped','rounded_to_zero','nonfinite','squared_error','squared_signal')}
        summary[name]['absmax']=max(r['absmax'] for r in subset)
    numeric_gate=(summary['value_candidate']['clipped']<summary['value_current']['clipped']
                  and summary['value_candidate']['squared_error']<=summary['value_current']['squared_error'])
    result={'schema':'FP8_independent_same_base_BF16_calibration_v1','requests':136,
            'all_HTTP_and_official_prompt_counts_match':True,'numeric_summary':summary,
            'candidate_numerical_gate':numeric_gate,'frozen_candidate':{'K':.125,'V':.5,'Q':1.},
            'host_plan_counts':stats['host_plan_counts'],'actual_model_graph_success_calls':graph_calls,
            'performance_claim':False,'formal_quality_answers_used':False,'quality_acceptance_claim':False}
    write_new_json(output/'calibration_acceptance.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph-gate',type=Path,required=True)
    _graph_gate(parser.parse_args().graph_gate)
