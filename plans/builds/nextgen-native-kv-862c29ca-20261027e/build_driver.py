"""Container-local offline build; writable state stays in its new build root."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sysconfig
import time


def _json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def _run(command: list[str], root: Path, stage: str, env: dict, seconds: int,
         budget_roots: tuple[Path, ...] = ()) -> None:
    _json(root/(stage+'_command.json'),{'argv':command,'max_seconds':seconds})
    with (root/(stage+'.log')).open('x') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+seconds
        try:
            while process.poll() is None:
                if time.monotonic()>deadline or sum(_tree_bytes(p) for p in (budget_roots or (root,)))>20*2**30:
                    raise TimeoutError('Bounded CPU build time/disk envelope exceeded')
                time.sleep(15)
            if process.returncode:
                raise RuntimeError(stage+' exited '+str(process.returncode))
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    manifest=json.loads((root/'manifest.json').read_text())
    if os.getuid()!=manifest['uid'] or list(Path('/dev').glob('nvidia[0-9]*')):
        raise ValueError('CPU build must have the approved UID and no GPU device nodes')
    _json(root/'CPU_ONLY_ADMISSION.json',{'uid':os.getuid(),'GPU_device_nodes':[],
                                        'source_commit':manifest['source_commit'],'network':'none'})
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',CARGO_NET_OFFLINE='true',
               LIBTORCH_ROOT='/usr/local/lib/python3.12/dist-packages/torch',
               PYTORCH_INSTALL_PATH='/usr/local/lib/python3.12/dist-packages/torch',
               PYTHON_INCLUDE_PATH=sysconfig.get_path('include'),
               PYTHON_LIB_PATH=sysconfig.get_config_var('LIBDIR'),CUDA_TOOLKIT_ROOT_DIR='/usr/local/cuda',
               CARGO_HOME=str(root/'cargo'),RUSTUP_HOME='/workspace/deps/rustup',
               CCACHE_DIR=str(root/'ccache'),CARGO_BUILD_JOBS='4',MAX_JOBS='4',CMAKE_BUILD_PARALLEL_LEVEL='4',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    cargo=root/'cargo';cargo.mkdir()
    for name in ('bin','registry','git'):
        source=Path('/workspace/deps/cargo')/name
        if source.exists():(cargo/name).symlink_to(source,target_is_directory=True)
    env['PATH']=str(cargo/'bin')+':'+env['PATH']
    source=root/'source_repo';build=root/'build';budget_roots=(root,)
    if manifest.get('incremental_reuse'):
        previous=Path('/workspace/agent_work/builds')/manifest['incremental_reuse']['build_id']
        source=previous/'source_repo';build=previous/'build';budget_roots=(root,previous)
        actual=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        if actual!=manifest['source_commit']:raise ValueError('Incremental mounted source differs')
        _json(root/'INCREMENTAL_ADMISSION.json',{'source_commit':actual,'source_mount':str(source),
              'reused_build':str(build),'parent_failed_build':manifest['incremental_reuse']['build_id'],
              'original_host_source_preserved':True,'combined_new_disk_GiB_limit':20})
    command=['cmake','-S',str(source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
             '-DUSE_CUDA=ON','-DUSE_NPU=OFF','-DUSE_MLU=OFF','-DUSE_CXX11_ABI=ON',
             '-DCMAKE_CUDA_ARCHITECTURES=90a','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
             '-DCMAKE_TOOLCHAIN_FILE=/workspace/deps/vcpkg/scripts/buildsystems/vcpkg.cmake',
             '-DVCPKG_MANIFEST_MODE=OFF','-DVCPKG_TARGET_TRIPLET=x64-linux',
             '-DVCPKG_INSTALLED_DIR=/workspace/xllm_quant_kvcache/build/cmake.linux-x86_64-cpython-312/vcpkg_installed',
             '-DCMAKE_PREFIX_PATH=/workspace/deps/prefix',
             '-Dyalantinglibs_DIR=/workspace/deps/prefix/lib/cmake/yalantinglibs',
             '-DTorch_DIR=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch',
             '-DPython_EXECUTABLE=/workspace/.venv/bin/python','-DFLASHINFER_FA2_0_6_18_ABI=OFF',
             '-DBUILD_TESTING=ON','-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
    result={'source_commit':manifest['source_commit'],'GPU_job':False,'success':False}
    try:
        for path in (Path(env['LIBTORCH_ROOT'])/'include/torch/extension.h',
                     Path(env['PYTORCH_INSTALL_PATH'])/'include/torch/csrc/api/include/torch/torch.h',
                     Path(env['PYTHON_INCLUDE_PATH'])/'Python.h'):
            if not path.is_file():raise ValueError('Missing existing toolchain header: '+str(path))
        _run(command,root,'configure',env,600,budget_roots)
        _run(['cmake','--build',str(build),'--target','xllm','kv_cache_estimation_test','kv_cache_test',
              'cuda_kv_cache_admission_test','--parallel','4'],root,'compile',env,4500,budget_roots)
        for target,pattern in [('kv_cache_estimation_test','*'),('cuda_kv_cache_admission_test','*'),
                               ('kv_cache_test','KVCacheTest.Nvfp4AllocationAndPageCopiesIncludeLinearBlockScales:KVCacheTest.QuantizedAllocationMatchesBudgetAndCopiesScales')]:
            matches=list(build.rglob(target))
            paths=[p for p in matches if p.is_file() and os.access(p,os.X_OK)]
            if len(paths)!=1:raise ValueError('Ambiguous/missing test executable: '+target)
            _run([str(paths[0]),'--gtest_filter='+pattern],root,target,env,120,budget_roots)
        binary=build/'bin/xllm'
        if manifest.get('incremental_reuse'):
            artifacts=root/'artifacts';artifacts.mkdir()
            frozen_binary=artifacts/'xllm'
            shutil.copy2(binary,frozen_binary)
            binary=frozen_binary
        hasher=hashlib.sha256()
        with binary.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*2**20),b''):hasher.update(chunk)
        result.update(success=True,binary=str(binary),binary_bytes=binary.stat().st_size,
                      binary_sha256=hasher.hexdigest(),CPU_tests_executed=True,model_or_GPU_validation=False)
    except (OSError,ValueError,RuntimeError,TimeoutError) as error:
        result.update(error=type(error).__name__+': '+str(error))
        raise
    finally:
        _json(root/'build_result.json',result)


if __name__=='__main__':main()
