"""Private bounded CPU child for reviewed in-memory BFCL tool sessions.

No model/provider code is loaded in this child. Model-generated calls stay JSON
data, dispatched only through the pinned per-task schemas. A timed-out session
fails explicitly; it is not silently restarted with lost simulator state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import selectors
import subprocess
import sys
import time


MEMORY_BYTES=512*2**20
CPU_SECONDS=30
CALL_TIMEOUT_SECONDS=2
INITIALIZATION_TIMEOUT_SECONDS=15
_MAX_INPUT_BYTES=2**20
_MAX_OUTPUT_BYTES=4*2**20


class ToolChild:
    def __init__(self,source: Path,task: dict,stderr: Path) -> None:
        self.task_id=task['id'];self.sequence=0;self.buffer=b'';self.selector=selectors.DefaultSelector()
        self.stderr=stderr.open('xb')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')
        command=[sys.executable,'-B','-m',__package__+'.nextgen_tool_child','--worker','--source',str(source)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
                                      env=env,start_new_session=True,bufsize=0)
        os.set_blocking(self.process.stdin.fileno(),False)
        os.set_blocking(self.process.stdout.fileno(),False)
        self.selector.register(self.process.stdout,selectors.EVENT_READ)
        try:
            ready=self._request({'operation':'initialize','task':task})
            expected=hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest()
            if (ready.get('status')!='READY' or ready.get('source_manifest_sha256')!=expected
                    or ready.get('task_id')!=self.task_id or ready.get('CUDA_visibility_hidden') is not True
                    or ready.get('CUDA_runtime_initialized') is not False
                    or ready.get('address_space_limit_bytes')!=MEMORY_BYTES
                    or ready.get('CPU_seconds_limit')!=CPU_SECONDS):
                raise ValueError('Bounded CPU tool child did not attest its declared source/task/limits: '+
                                 json.dumps({'observed':ready,'expected_source_sha256':expected,
                                             'expected_task_id':self.task_id},sort_keys=True))
            self.ready=ready
        except BaseException:
            self.close();raise

    def _receive(self,timeout: float) -> dict:
        until=time.monotonic()+timeout
        while b'\n' not in self.buffer:
            remaining=until-time.monotonic()
            if remaining<=0:raise TimeoutError('Owned CPU tool call exceeded2s; session state is not restarted')
            if not self.selector.select(remaining):continue
            data=os.read(self.process.stdout.fileno(),65536)
            if not data:raise RuntimeError('Owned CPU tool child exited before a complete response')
            self.buffer+=data
            if len(self.buffer)>_MAX_OUTPUT_BYTES:raise ValueError('CPU tool child response exceeded bounded framing')
        line,self.buffer=self.buffer.split(b'\n',1)
        return json.loads(line)

    def _request(self,body: dict) -> dict:
        self.sequence+=1
        raw=(json.dumps({**body,'sequence':self.sequence},ensure_ascii=False,allow_nan=False)+'\n').encode()
        if len(raw)>_MAX_INPUT_BYTES:raise ValueError('CPU tool request exceeded bounded framing')
        until=time.monotonic()+(INITIALIZATION_TIMEOUT_SECONDS if body['operation']=='initialize' else CALL_TIMEOUT_SECONDS)
        with selectors.DefaultSelector() as writable:
            writable.register(self.process.stdin,selectors.EVENT_WRITE);sent=0
            while sent<len(raw):
                remaining=until-time.monotonic()
                if remaining<=0:raise TimeoutError('Owned CPU tool command exceeded bounded write deadline')
                if not writable.select(remaining):continue
                try:sent+=os.write(self.process.stdin.fileno(),raw[sent:])
                except BlockingIOError:continue
        reply=self._receive(max(0,until-time.monotonic()))
        if reply.get('sequence')!=self.sequence or reply.get('task_id')!=self.task_id:
            raise ValueError('CPU tool response differs from its session/command identity')
        return reply

    def call(self,name: str,arguments: dict) -> dict:
        try:return self._request({'operation':'call','name':name,'arguments':arguments})
        except BaseException:
            self.close();raise  # A failed/partial response must not continue a stale session.

    def close(self) -> None:
        process=getattr(self,'process',None)
        if process is not None:
            # Only this direct Popen child is signalled, never a name/group scan.
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if process.stdin is not None:process.stdin.close()
            if process.stdout is not None:process.stdout.close()
        self.selector.close();self.stderr.close()

    def __enter__(self):return self
    def __exit__(self,*ignored):self.close()


def _limit_resources() -> None:
    # Limits apply only to the newly created CPU child, never the serving model.
    for kind,wanted in ((resource.RLIMIT_AS,MEMORY_BYTES),(resource.RLIMIT_CPU,CPU_SECONDS)):
        _,hard=resource.getrlimit(kind)
        if hard!=resource.RLIM_INFINITY and hard<wanted:raise ValueError('Existing child limit is below reviewed tool budget')
        resource.setrlimit(kind,(wanted,wanted))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def _cuda_runtime_initialized() -> bool:
    # is_initialized only reads PyTorch's process state; unlike is_available or
    # device_count it does not ask the CUDA driver to initialize anything.
    module=sys.modules.get('torch')
    cuda=getattr(module,'cuda',None) if module is not None else None
    return bool(cuda.is_initialized()) if cuda is not None else False


def worker(source: Path) -> None:
    _limit_resources()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU tool child must hide CUDA runtime visibility')
    if _cuda_runtime_initialized():raise ValueError('CPU tool child unexpectedly initialized CUDA runtime')
    from .nextgen_tools_env import ToolFactory
    factory=ToolFactory(source);environment=None;task_id=None;expected_sequence=1
    while True:
        raw=sys.stdin.buffer.readline(_MAX_INPUT_BYTES+1)
        if not raw:return
        if len(raw)>_MAX_INPUT_BYTES or not raw.endswith(b'\n'):raise ValueError('Malformed or excessive CPU command framing')
        body=json.loads(raw)
        if body.get('sequence')!=expected_sequence:raise ValueError('CPU tool command order changed')
        expected_sequence+=1;start=time.monotonic_ns()
        if environment is None:
            if body.get('operation')!='initialize':raise ValueError('Tool session requires one original initialization')
            task=body['task'];task_id=task['id'];environment=factory.environment(task)
            reply={'status':'READY','source_manifest_sha256':hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest(),
                   'address_space_limit_bytes':resource.getrlimit(resource.RLIMIT_AS)[0],
                   'CPU_seconds_limit':resource.getrlimit(resource.RLIMIT_CPU)[0],
                   'CPU_only_imports':not any(name in sys.modules for name in ('torch','cupy','pycuda','cuda')),
                   'preloaded_accelerator_modules':[name for name in ('torch','cupy','pycuda','cuda') if name in sys.modules],
                   'CUDA_visibility_hidden':os.environ.get('CUDA_VISIBLE_DEVICES')=='',
                   'CUDA_runtime_initialized':_cuda_runtime_initialized()}
        elif body.get('operation')=='call':
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or _cuda_runtime_initialized():
                raise ValueError('CPU tool runtime visibility/context changed')
            reply=environment.call(body['name'],body['arguments'])
            if _cuda_runtime_initialized():raise ValueError('A CPU tool unexpectedly initialized CUDA')
        else:raise ValueError('Unreviewed CPU operation; no reset or arbitrary command is available')
        reply.update(sequence=body['sequence'],task_id=task_id,CPU_call_monotonic_ns=time.monotonic_ns()-start)
        sys.stdout.write(json.dumps(reply,ensure_ascii=False,allow_nan=False)+'\n');sys.stdout.flush()


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--source',type=Path,required=True);args=parser.parse_args();worker(args.source)


if __name__=='__main__':main()
