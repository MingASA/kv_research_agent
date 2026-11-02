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
import tarfile
import time


def _json(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def _tree_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def _find_binary(build: Path) -> Path:
    """Support both explicit bin output and the current CMake target directory."""
    candidates=[build/'bin/xllm',build/'xllm/xllm']
    present=[path for path in candidates if path.is_file() and not path.is_symlink() and os.access(path,os.X_OK)]
    if len(present)!=1:
        raise ValueError('Ambiguous/missing compiled xLLM artifact: '+str(present))
    return present[0]


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
    import importlib.metadata
    packages={}
    for package in ('torch','flashinfer-python','triton','transformers'):
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    _json(root/'RUNTIME_PACKAGES.json',packages)
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
    if manifest.get('build_snapshot'):
        virtual = Path('/workspace/agent_work/builds') / manifest['build_snapshot'].get('virtual_build_id', manifest['build_snapshot']['object_build_id'])
        source = virtual/'source_repo'; build = virtual/'build'
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != manifest['source_commit']:
            raise ValueError('Isolated snapshot source differs')
        _json(root/'SNAPSHOT_ADMISSION.json', {'source_commit': actual,
              'private_build_mount': str(build), 'parent_objects_writable': False})
    command=['cmake','-S',str(source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
             '-DUSE_CUDA=ON','-DUSE_NPU=OFF','-DUSE_MLU=OFF','-DUSE_CXX11_ABI=ON',
             '-DUSE_ETCD=OFF','-DSTORE_USE_ETCD=OFF',
             '-DCMAKE_CUDA_ARCHITECTURES=90a','-DTORCH_CUDA_ARCH_LIST=9.0a','-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc',
             '-DCMAKE_TOOLCHAIN_FILE=/workspace/deps/vcpkg/scripts/buildsystems/vcpkg.cmake',
             '-DVCPKG_MANIFEST_MODE=OFF','-DVCPKG_TARGET_TRIPLET=x64-linux',
             '-DVCPKG_INSTALLED_DIR=/workspace/xllm_quant_kvcache/build/cmake.linux-x86_64-cpython-312/vcpkg_installed',
             '-DCMAKE_PREFIX_PATH=/workspace/deps/prefix',
             '-Dyalantinglibs_DIR=/workspace/deps/prefix/lib/cmake/yalantinglibs',
             '-DTorch_DIR=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch',
             '-DPython_EXECUTABLE=/workspace/.venv/bin/python',
             '-DBUILD_TESTING=ON','-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
    result={'source_commit':manifest['source_commit'],'GPU_job':False,'success':False}
    try:
        dependency = manifest.get('zeromq_source')
        if dependency:
            archive = root / dependency['filename']
            if hashlib.sha256(archive.read_bytes()).hexdigest() != dependency['sha256']:
                raise ValueError('Offline ZeroMQ source identity differs')
            with tarfile.open(archive) as package:
                package.extractall(root / 'dependencies', filter='data')
            dependency_source = root / 'dependencies' / 'zeromq-4.3.5'
            dependency_build = root / 'zeromq-build'
            dependency_prefix = root / 'zeromq-prefix'
            _run(['cmake', '-S', str(dependency_source), '-B', str(dependency_build), '-G', 'Ninja',
                  '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTS=OFF', '-DWITH_PERF_TOOL=OFF',
                  '-DBUILD_STATIC=OFF', '-DBUILD_SHARED=ON', '-DCMAKE_INSTALL_PREFIX='+str(dependency_prefix)],
                 root, 'zeromq_configure', env, 180, budget_roots)
            _run(['cmake', '--build', str(dependency_build), '--parallel', '4'],
                 root, 'zeromq_compile', env, 600, budget_roots)
            _run(['cmake', '--install', str(dependency_build)], root, 'zeromq_install', env, 60, budget_roots)
            command += ['-DZMQ_INCLUDE_DIR='+str(dependency_prefix / 'include'),
                        '-DZMQ_LIBRARY='+str(dependency_prefix / 'lib/libzmq.so')]
        reused_dependency = manifest.get('zeromq_reuse')
        if reused_dependency:
            dependency_prefix = Path('/workspace/agent_work/builds') / reused_dependency['build_id'] / 'zeromq-prefix'
            library = dependency_prefix / 'lib/libzmq.so.5.2.5'
            if hashlib.sha256(library.read_bytes()).hexdigest() != reused_dependency['library_sha256']:
                raise ValueError('Existing private ZeroMQ artifact changed')
            command += ['-DZMQ_INCLUDE_DIR='+str(dependency_prefix / 'include'),
                        '-DZMQ_LIBRARY='+str(dependency_prefix / 'lib/libzmq.so')]
        for path in (Path(env['LIBTORCH_ROOT'])/'include/torch/extension.h',
                     Path(env['PYTORCH_INSTALL_PATH'])/'include/torch/csrc/api/include/torch/torch.h',
                     Path(env['PYTHON_INCLUDE_PATH'])/'Python.h'):
            if not path.is_file():raise ValueError('Missing existing toolchain header: '+str(path))
        _run(command,root,'configure',env,600,budget_roots)
        targets = ['kv_cache_estimation_test', 'kv_cache_test']
        if manifest.get('native_KV_build'):
            targets += ['virtual_kv_cache_page_layout_test', 'vmm_kv_cache_layout_test']
        _run(['cmake','--build',str(build),'--target','xllm', *targets,
              '--parallel','4'],root,'compile',env,4500,budget_roots)
        for target in targets:
            paths=[p for p in build.rglob(target) if p.is_file() and os.access(p,os.X_OK)]
            if len(paths)!=1:raise ValueError('Ambiguous/missing official CPU test: '+target)
            arguments = [str(paths[0])]
            if target == 'kv_cache_test':
                arguments += ['--gtest_filter=-HostKVCacheTest.*']
            _run(arguments,root,target,env,180,budget_roots)
        binary=_find_binary(build)
        if manifest.get('incremental_reuse') or manifest.get('build_snapshot'):
            artifacts=root/'artifacts';artifacts.mkdir()
            frozen_binary=artifacts/'xllm'
            shutil.copy2(binary,frozen_binary)
            binary=frozen_binary
        hasher=hashlib.sha256()
        with binary.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*2**20),b''):hasher.update(chunk)
        result.update(success=True,binary=str(binary),binary_bytes=binary.stat().st_size,
                      binary_sha256=hasher.hexdigest(),CPU_tests_executed=True,model_or_GPU_validation=False)
        result['GPU_context_tests_deferred'] = ['HostKVCacheTest.*']
        if manifest.get('source_generated_write_subdirs'):
            generated=source/'third_party/brpc/src/butil/config.h'
            result['private_generated_header_sha256']={
                'third_party/brpc/src/butil/config.h':hashlib.sha256(generated.read_bytes()).hexdigest()}
    except (OSError,ValueError,RuntimeError,TimeoutError) as error:
        result.update(error=type(error).__name__+': '+str(error))
        raise
    finally:
        _json(root/'build_result.json',result)


if __name__=='__main__':main()
