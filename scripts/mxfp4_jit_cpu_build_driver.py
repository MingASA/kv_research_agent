"""Bounded GPU-invisible, same-image SM90a-only private FlashInfer build."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


CHILD=r'''
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]);m=json.loads((root/'manifest.json').read_bytes())
sys.path.insert(0,str(root/'source_repo'))
import torch,flashinfer
from xllm.python.attention.mxfp4_flashinfer import mxfp4_jit_spec
assert not torch.cuda.is_initialized() and not list(Path('/dev').glob('nvidia[0-9]*'))
source_manifest=m.get('native_jit_source_manifest')
manifest_sha=None
if source_manifest is not None:
 from xllm.python.attention.mxfp4_flashinfer import mxfp4_jit_source_manifest
 actual=mxfp4_jit_source_manifest()
 assert actual==source_manifest and m['build_purpose']=='MXFP4_SM90a_CPU_JIT_P0_P1'
 manifest_sha=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()
spec,_,_=mxfp4_jit_spec();spec.build(verbose=True,need_lock=True)
library=spec.get_library_path();module=spec.load()
exports=('plan','ragged_run','paged_run')
for name in exports:assert getattr(module,name) is not None
assert not torch.cuda.is_initialized()
raw=library.read_bytes()
result={'schema':'MXFP4_SM90a_CPU_JIT_build_v1','success':True,'source_commit':m['source_commit'],
 'CPU_compile_and_load':True,'CPU_tests_executed':False,'GPU_job':False,'model_or_GPU_validation':False,
 'module':spec.name,'library':str(library),'library_bytes':len(raw),'library_sha256':hashlib.sha256(raw).hexdigest(),
 'FLASHINFER_CUDA_ARCH_LIST':os.environ['FLASHINFER_CUDA_ARCH_LIST'],'torch':torch.__version__,
 'flashinfer':flashinfer.__version__,'exports':exports,'CUDA_initialized':False}
if source_manifest is not None:
 assert spec.name==source_manifest['module_name']
 cache=Path('/run_cache') if m.get('stable_JIT_cache_mount')=='/run_cache' else root/'cache'
 relative=library.relative_to(cache)
 cache_base=relative.parents[2]
 tree=[]
 for kind in ('generated','cached_ops'):
  for path in sorted((cache/cache_base/kind/spec.name).rglob('*')):
   if path.is_file():
    assert not path.is_symlink()
    data=path.read_bytes();info=path.stat()
    tree.append({'relative':str(path.relative_to(cache)),'sha256':hashlib.sha256(data).hexdigest(),
                 'size':len(data),'mtime_ns':info.st_mtime_ns})
 assert tree and str(relative) in [entry['relative'] for entry in tree]
 result.update(schema='MXFP4_SM90a_CPU_JIT_build_v2',source_manifest_sha256=manifest_sha,
               cache_relative=str(relative),cache_tree=tree)
with (root/'build_result.json').open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
'''


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root;m=json.loads((root/'manifest.json').read_bytes())
    if os.getuid()!=m['uid'] or list(Path('/dev').glob('nvidia[0-9]*')):
        raise ValueError('Private JIT build must have the approved UID and no GPU nodes')
    cache=root/'cache'
    if m.get('stable_JIT_cache_mount') is not None:
        if m['stable_JIT_cache_mount']!='/run_cache' or m.get('build_purpose')!='MXFP4_SM90a_CPU_JIT_P0_P1':
            raise ValueError('Unexpected private JIT cache mount')
        cache=Path('/run_cache')
        if not os.path.samestat(cache.stat(),(root/'cache').stat()):
            raise ValueError('Stable JIT path is not this build private cache')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',
        FLASHINFER_CUDA_ARCH_LIST='9.0a',FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
        XDG_CACHE_HOME=str(cache/'xdg'),MAX_JOBS='4',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name in ('flashinfer','xdg'):(cache/name).mkdir(parents=True)
    (root/'CPU_ONLY_ADMISSION.json').open('x').write(json.dumps({'uid':os.getuid(),'GPU_device_nodes':[],
        'source_commit':m['source_commit'],'network':'none','FLASHINFER_CUDA_ARCH_LIST':'9.0a'})+'\n')
    with (root/'compile.log').open('x') as log:
        process=subprocess.Popen([sys.executable,'-B','-c',CHILD,str(root)],env=env,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=m['build_max_seconds'])
            if code:raise RuntimeError('Offline private JIT build failed: '+str(code))
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


if __name__=='__main__':main()
