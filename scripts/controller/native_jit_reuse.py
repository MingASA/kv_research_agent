"""Reuse one qualified MXFP4 library in each run's isolated writable cache."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

from .approval import ApprovalError
from .durable import write_new_bytes,write_new_json


MODULE='xllm_mxfp4_bf16_d256_b06c014c4f939393'
CACHE_RELATIVE=('flashinfer/.cache/flashinfer/0.6.18.post1/90a/'
                'cached_ops/'+MODULE+'/'+MODULE+'.so')
CACHE_BASE=Path(CACHE_RELATIVE).parents[2]
CACHE_PREFIXES=tuple(str(CACHE_BASE/k/MODULE)+'/' for k in ('generated','cached_ops'))


def validate(request: dict) -> None:
    artifact=request.get('native_jit_artifact')
    cpu=request.get('cpu_native_jit_artifact')
    if cpu is not None:
        if (artifact is not None or not isinstance(cpu,dict)
                or set(cpu)!={'schema','build_id','build_result_sha256','CPU_admission_sha256','sha256','bytes','source_commit','image_id'}
                or cpu.get('schema')!='MXFP4_SM90a_CPU_JIT_seed_v1'
                or request.get('arms')!=['mxfp4']
                or request.get('workload',{}).get('kind')!='upstream_next_stage_native_operator'
                or re.fullmatch(r'upstream-ac-[0-9a-f]{8}-20261105[a-z]',cpu.get('build_id','')) is None
                or cpu.get('source_commit')!=request.get('python_source_commit')
                or cpu.get('image_id')!=request.get('image_id')
                or any(re.fullmatch('[0-9a-f]{64}',cpu.get(k,'')) is None for k in
                       ('sha256','build_result_sha256','CPU_admission_sha256'))
                or type(cpu.get('bytes')) is not int or not 0<cpu['bytes']<=128*2**20):
            raise ApprovalError('CPU JIT seed is only a source/image-bound MXFP4 operator prerequisite')
    if artifact is None:return
    reference=request.get('operator_qualification_reference')
    if (not isinstance(artifact,dict)
            or set(artifact) not in ({'schema','module','cache_relative','sha256','bytes','source_commit','image_id','qualification_reference'},
                                    {'schema','module','cache_relative','sha256','bytes','source_commit','image_id','qualification_reference','cache_tree'})
            or artifact.get('schema')!='qualified_MXFP4_JIT_reuse_v1'
            or request.get('next_stage_profile')!='ocp_mxfp4_20261105' or request.get('arms')!=['mxfp4']
            or artifact.get('module')!=MODULE or artifact.get('cache_relative')!=CACHE_RELATIVE
            or artifact.get('source_commit')!=request.get('python_source_commit')
            or artifact.get('image_id')!=request.get('image_id')
            or reference is None or artifact.get('qualification_reference')!=reference
            or re.fullmatch('[0-9a-f]{64}',artifact.get('sha256','')) is None
            or type(artifact.get('bytes')) is not int or not 0<artifact['bytes']<=128*2**20):
        raise ApprovalError('JIT reuse requires the exact source/image-qualified MXFP4 artifact')
    if 'cache_tree' in artifact:
        tree=artifact['cache_tree'];seen=set();total=0
        if not isinstance(tree,list) or not 1<=len(tree)<=128:raise ApprovalError('Unbounded native JIT cache tree')
        for item in tree:
            if not isinstance(item,dict) or not isinstance(item.get('relative'),str):
                raise ApprovalError('Malformed native JIT cache member')
            relative=item.get('relative','');path=Path(relative)
            if (set(item)!={'relative','sha256','size','mtime_ns'} or not isinstance(relative,str)
                    or path.is_absolute() or '..' in path.parts or str(path)!=relative or relative in seen
                    or not any(relative.startswith(prefix) for prefix in CACHE_PREFIXES)
                    or re.fullmatch('[0-9a-f]{64}',item.get('sha256','')) is None
                    or type(item.get('size')) is not int or not 0<=item['size']<=64*2**20
                    or type(item.get('mtime_ns')) is not int or item['mtime_ns']<0):
                raise ApprovalError('Unsafe native JIT cache member')
            seen.add(relative);total+=item['size']
        if total>128*2**20 or CACHE_RELATIVE not in seen:
            raise ApprovalError('Incomplete/oversized native JIT cache tree')


def install(request: dict, output: Path, cache: Path=Path('/run_cache')) -> None:
    """Read the immutable predecessor; write only this run's fresh cache."""
    validate(request)
    artifact=request.get('native_jit_artifact')
    cpu=request.get('cpu_native_jit_artifact')
    if artifact is None and cpu is None:return
    from .next_stage_deployment import _completed_reference,operator_acceptance_passed
    import json

    if cpu is not None:
        prior=Path(request['approved_workspace'])/'agent_work/builds'/cpu['build_id']
        values={}
        for name,key in [('build_result.json','build_result_sha256'),('CPU_ONLY_ADMISSION.json','CPU_admission_sha256')]:
            path=prior/name
            if path.resolve(strict=True)!=path or path.stat().st_uid!=os.getuid():
                raise ValueError('CPU JIT proof is not owned/canonical')
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=cpu[key]:raise ValueError('CPU JIT proof SHA changed')
            values[name]=json.loads(raw)
        build=values['build_result.json'];admission=values['CPU_ONLY_ADMISSION.json']
        manifest=json.loads((prior/'manifest.json').read_bytes())
        if (build.get('schema')!='MXFP4_SM90a_CPU_JIT_build_v1' or build.get('success') is not True
                or build.get('CPU_compile_and_load') is not True or build.get('GPU_job') is not False
                or build.get('CUDA_initialized') is not False or build.get('FLASHINFER_CUDA_ARCH_LIST')!='9.0a'
                or admission.get('GPU_device_nodes')!=[] or admission.get('source_commit')!=cpu['source_commit']
                or build.get('source_commit')!=cpu['source_commit'] or manifest.get('image_id')!=cpu['image_id']
                or build.get('library_sha256')!=cpu['sha256'] or build.get('library_bytes')!=cpu['bytes']):
            raise ValueError('CPU build is not the exact offline MXFP4 prerequisite')
        source=prior/'cache'/CACHE_RELATIVE;artifact=cpu
        origin={'source_build_id':cpu['build_id'],'CPU_compile_only_before_GPU_qualification':True}
    else:
        prior,old=_completed_reference(request,artifact['qualification_reference'])
        acceptance=prior/'output/mxfp4/operators/acceptance.json'
        if (old.get('python_source_commit')!=artifact['source_commit']
                or old.get('image_id')!=artifact['image_id']
                or not operator_acceptance_passed(json.loads(acceptance.read_bytes()),'mxfp4')):
            raise ValueError('JIT predecessor did not pass this exact source/image operator gate')
        source=prior/'cache/mxfp4'/CACHE_RELATIVE
        origin={'source_run_id':artifact['qualification_reference']['run_id'],
                'CPU_compile_only_before_GPU_qualification':False}
    if (source.resolve(strict=True)!=source or not source.is_file() or source.stat().st_uid!=os.getuid()
            or source.stat().st_size!=artifact['bytes']):
        raise ValueError('Qualified JIT source ownership, path or size changed')
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=artifact['sha256']:
        raise ValueError('Qualified JIT source SHA changed')
    if cache.resolve(strict=True)!=cache or cache.stat().st_uid!=os.getuid():
        raise ValueError('JIT destination must be this run\'s owned canonical cache')
    tree=artifact.get('cache_tree',[])
    for item in tree:
        src=prior/'cache/mxfp4'/item['relative'];info=src.stat()
        if (src.resolve(strict=True)!=src or not src.is_file() or info.st_uid!=os.getuid()
                or info.st_size!=item['size'] or info.st_mtime_ns!=item['mtime_ns']):
            raise ValueError('Original JIT cache member ownership/path/size/mtime changed')
        data=src.read_bytes()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Original JIT cache member SHA changed')
        destination=cache/item['relative'];destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.parent.resolve(strict=True)!=destination.parent or destination.is_symlink():
            raise ValueError('Unsafe new JIT cache destination')
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Never overwrite a different cache member')
        else:write_new_bytes(destination,data)
        os.utime(destination,ns=(info.st_atime_ns,item['mtime_ns']))
        if destination.stat().st_mtime_ns!=item['mtime_ns']:
            raise ValueError('New cache cannot preserve original dependency timestamps')
    target=cache/CACHE_RELATIVE
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.parent.resolve(strict=True)!=target.parent or target.is_symlink():
        raise ValueError('JIT destination escaped the isolated cache')
    if target.exists():
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=artifact['sha256']:
            raise ValueError('Never overwrite a different cached JIT library')
    else:
        write_new_bytes(target,raw)
    write_new_json(output/'native_JIT_reuse.json',{'schema':'MXFP4_JIT_reuse_v1',**origin,'module':MODULE,
        'sha256':artifact['sha256'],'bytes':len(raw),'source_and_destination_SHA_verified':True,
        'copied_cache_tree_files':len(tree),'original_dependency_mtime_preserved':bool(tree),
        'ninja_noop_not_yet_proven':True,
        'predecessor_modified':False,'inference_performance_claim':False})


def verify_noop(receipt: Path,output: Path) -> None:
    """Use FI's real generator and Ninja scanner before loading the model."""
    import subprocess
    import torch
    from xllm.python.attention.mxfp4_flashinfer import mxfp4_jit_spec
    import json

    proof=json.loads(receipt.read_bytes());spec,_,_=mxfp4_jit_spec();spec.write_ninja()
    command=['ninja','-n','-d','explain','-C',str(spec.build_dir),'-f',str(spec.ninja_path)]
    result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
    text=result.stdout+'\n'+result.stderr
    if 'no work to do' not in text.lower() or ' is dirty' in text:
        raise ValueError('Copied cache requires compilation: '+text[-4000:])
    before=hashlib.sha256(spec.get_library_path().read_bytes()).hexdigest()
    if before!=proof['sha256']:raise ValueError('No-op cache library differs from qualified artifact')
    spec.build_and_load()
    after=hashlib.sha256(spec.get_library_path().read_bytes()).hexdigest()
    if after!=before:raise ValueError('Official build_and_load changed the precompiled library')
    write_new_json(output,{'module':spec.name,'ninja_dry_run':text,'actual_official_build_and_load':True,
        'ninja_no_work_to_do':True,'library_sha256_before':before,'library_sha256_after':after,
        'CUDA_initialized':torch.cuda.is_initialized(),'model_or_GPU_validation':False})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();verify_noop(a.receipt,a.out)
