"""Inside one admitted GPU-free container: validate tool-loop image dependencies."""

from __future__ import annotations

import datetime
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def save(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())


def main(root: Path):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('GPU_requested') is not False or manifest.get('max_seconds')!=600:
        raise ValueError('Unreviewed CPU validation authority/budget')
    # This PID1 timer bounds even an interrupted controller/SSH connection.
    # Exiting namespace PID1 also retires its own remaining CPU children.
    deadline=threading.Timer(600,lambda:os._exit(124));deadline.daemon=True;deadline.start()
    if glob.glob('/dev/nvidia[0-9]*') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU validation unexpectedly sees GPU compute devices')
    for relative,sha in manifest['asset_sha256'].items():
        raw=(root/relative).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('CPU frozen asset changed: '+relative)
    model=Path('/models/Qwen3.8-27B')
    for name,sha in manifest['model_files_sha256'].items():
        if hashlib.sha256((model/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Approved existing tokenizer metadata differs: '+name)
    save(root/'CPU_ADMISSION.json',{'state':'CPU_ONLY_NATIVE_TOOL_RUNTIME_VALIDATION','GPU_requested':False,
        'python':sys.executable,'python_version':sys.version,'UID':os.getuid(),
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'max_seconds':600})
    task=json.loads((root/'input/prepared/tasks.jsonl').read_bytes().split(b'\n')[0])
    probe_command=[sys.executable,'-B','-m','scripts.controller.nextgen_tool_child','--worker',
                   '--source',str(root/'input/tool_source')]
    probe=subprocess.run(probe_command,input=json.dumps({'operation':'initialize','sequence':1,'task':task})+'\n',
                         text=True,capture_output=True,timeout=30)
    save(root/'CPU_CHILD_INITIAL_OBSERVATION.json',{'exit_code':probe.returncode,'stdout':probe.stdout[:12000],
            'stderr':probe.stderr[-8000:],'classification':'CPU_child_initial_fields_only_not_model_or_GPU'})
    if probe.returncode:raise ValueError('CPU child diagnostic exited before initial attestation')
    # Actual imports and native-tool children run in the existing image Python;
    # no environment installation, provider call or weight loading occurs.
    versions={name:importlib.metadata.version(name) for name in ('transformers','tokenizers','mpmath')}
    command=[sys.executable,'-B','-m','scripts.validate_nextgen_multiturn_cpu',
        '--prepared',str(root/'input/prepared'),'--source',str(root/'input/tool_source'),
        '--references',str(root/'input/references'),'--tokenizer',str(model),'--out',str(root/'loop')]
    with (root/'CPU_loop.log').open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=550)
    if result.returncode:raise ValueError('Actual image tool/tokenizer loop failed; retain CPU log')
    proof=json.loads((root/'loop/CPU_protocol_proof.json').read_text())
    if not proof.get('all_checks_pass'):raise ValueError('CPU tool loop did not complete all original tasks')
    expected=json.loads((root/'input/native_CPP_counts.json').read_text())
    from scripts.controller.official_tokenization import NATIVE_TOOLS_RENDER_PROTOCOL
    if (expected.get('all_checks_pass') is not True or
            expected.get('native_request_parameters_map_order')!=NATIVE_TOOLS_RENDER_PROTOCOL):
        raise ValueError('CPU image requires current native schema-map CPP proof')
    counts={(row['task_id'],row['turn'],row['step']):row['native_CPP_tokens'] for row in expected['comparisons']}
    actual=[]
    for path in sorted((root/'loop').glob('*/requests.jsonl')):
        for line in path.read_bytes().split(b'\n'):
            if not line:continue
            row=json.loads(line);key=(row['task_id'],row['turn'],row['step'])
            if key not in counts or row['prompt_tokens_CPU_expected']!=counts[key]:
                raise ValueError('Image tokenizer differs from independently executed pinned C++ frontend counts')
            actual.append(key)
    if len(actual)!=224 or len(set(actual))!=224 or set(actual)!=set(counts):
        raise ValueError('Incomplete or duplicated real32-task image protocol fixture')
    source_sha=hashlib.sha256((root/'input/tool_source/source_manifest.json').read_bytes()).hexdigest()
    save(root/'CPU_acceptance.json',{'schema':'native-tools32-H800-CPU-image-proof-v1',
        'state':'COMPLETE_CPU_TOOL_IMAGE_VALIDATION','id':manifest['id'],'image_id':manifest['image_id'],
        'GPU_requested':False,'GPU_compute_devices_visible':False,'model_weights_loaded':False,
        'actual_HTTP_or_model_predictions':False,'no_dependencies_installed':True,
        'original_tasks':32,'user_turns':112,'native_in_memory_tool_calls':173,
        'fake_SSE_provider_invocations':224,'native_CPP_prompt_counts_matched':224,
        'native_request_parameters_map_order':NATIVE_TOOLS_RENDER_PROTOCOL,
        'source_tasks_sha256':proof['frozen_tasks_sha256'],'tool_source_manifest_sha256':source_sha,
        'runtime_sources_sha256':{key:sha for key,sha in manifest['asset_sha256'].items()
                                  if key.startswith('pkg/scripts/controller/')},
        'model_files_sha256':manifest['model_files_sha256'],'versions':versions,
        'CPU_loop_proof_sha256':hashlib.sha256((root/'loop/CPU_protocol_proof.json').read_bytes()).hexdigest(),
        'native_CPP_reference_sha256':hashlib.sha256((root/'input/native_CPP_counts.json').read_bytes()).hexdigest(),
        'runtime_python':sys.executable,'CPU_only':True})
    deadline.cancel()


if __name__=='__main__':
    root=Path(sys.argv[1])
    try:main(root)
    except BaseException as error:
        save(root/'CPU_ERROR.json',{'error':type(error).__name__+': '+str(error)[:2000],
                                   'GPU_requested':False,'model_or_quality_success':False})
        raise
