# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Parse the exact service argv with AIPerf's own Cyclopts/CLIConfig, without running it."""

from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess

from .durable import write_new_json

PARSER_SOURCE = '''
import hashlib,importlib.metadata,json,sys
from typing import Annotated
from cyclopts import App,Parameter
from aiperf.config.flags.cli_config import CLIConfig
import aiperf.config.flags.cli_config as module
app=App()
@app.default
def parse_only(config:Annotated[CLIConfig,Parameter(name='*')]=CLIConfig()):
    return config
command,bound,unknown=app.parse_args(json.load(sys.stdin),exit_on_error=False,print_error=False)
if unknown:raise ValueError('Unparsed command arguments')
config=command(*bound.args,**bound.kwargs)
print(json.dumps({'actual_Cyclopts_CLIConfig_accepted':True,
    'aiperf_version':importlib.metadata.version('aiperf'),
    'cli_config_sha256':hashlib.sha256(open(module.__file__,'rb').read()).hexdigest(),
    'request_count':config.request_count,'concurrency':config.concurrency,
    'conversation_num':config.conversation_num,
    'warmup_request_count':config.warmup_request_count,
    'benchmark_started':False,'GPU_used':False}))
'''


def validate_service_cli(command: list[str], runtime_root: Path, output: Path,
                         env: dict, expected_config_sha: str) -> dict:
    with Path(command[0]).open() as stream:
        shebang=stream.readline().rstrip()
    interpreter=shlex.split(shebang[2:]) if shebang.startswith('#!') else []
    if not interpreter or not Path(interpreter[0]).name.startswith('python'):
        raise ValueError('Unknown AIPerf entrypoint interpreter; cannot attest actual CLI environment')
    cpu_env=dict(env,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(runtime_root/'runtime_support/tokenizer_compat'))
    with (output/'aiperf_cli_CPU_acceptance.log').open('x') as log:
        result=subprocess.run(interpreter+['-B','-c',PARSER_SOURCE],input=json.dumps(command[2:]),
            env=cpu_env,text=True,stdout=subprocess.PIPE,stderr=log,timeout=60,check=True)
    acceptance=json.loads(result.stdout)
    expected_sessions=int(command[command.index('--num-sessions')+1]) if '--num-sessions' in command else None
    if (acceptance.get('actual_Cyclopts_CLIConfig_accepted') is not True
            or acceptance.get('aiperf_version')!='0.13.0'
            or acceptance.get('cli_config_sha256')!=expected_config_sha
            or acceptance.get('warmup_request_count') is not None
            or acceptance.get('conversation_num')!=expected_sessions
            or acceptance.get('request_count')!=int(command[command.index('--request-count')+1])
            or acceptance.get('concurrency')!=int(command[command.index('--concurrency')+1])):
        raise ValueError('Actual installed CLI config/default differs from the frozen CPU acceptance')
    write_new_json(output/'aiperf_cli_CPU_acceptance.json',acceptance)
    return acceptance
