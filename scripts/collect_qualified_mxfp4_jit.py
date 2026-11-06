"""Read a released MXFP4 operator's exact reusable JIT library identity."""

import argparse
import json
from pathlib import Path

from scripts.controller.durable import write_new_json
from scripts.controller.native_jit_reuse import CACHE_RELATIVE,MODULE
from scripts.controller.transport import PinnedSSH


SOURCE=r'''
import hashlib,json,os,sys
from pathlib import Path
p=json.load(sys.stdin);lib={'__name__':'qualified_mxfp4_jit_readonly'};exec(p['rpc_source'],lib)
t=p['ticket'];root=lib['check_ticket'](t,False);lib['existing'](root,t);r=t['request']
lib['require'](r['workload']['kind']=='upstream_next_stage_native_operator' and r['arms']==['mxfp4'],'Not the MXFP4 operator')
raw=lib['regular'](root/'result.json');result=json.loads(raw)
lib['require'](result['status']=='COMPLETE' and result['gpu_release_verified'] and not result['live_owned_containers'],'Operator not completed/released')
lib['require'](not lib['status'](root,t)['live_owned_containers'],'Original container remains live')
a=json.loads(lib['regular'](root/'output/mxfp4/operators/acceptance.json'))
lib['require'](a.get('schema')=='MXFP4_SM90_operator_gate_v1' and a.get('writer_CPU_bitwise') is True and a.get('whole_page_copy') is True and a.get('results',{}).get('prefill',{}).get('pass') is True and a.get('results',{}).get('decode',{}).get('pass') is True and a.get('results',{}).get('Graph')=={'changed_inputs':True,'bitwise_equal_direct':True},'MXFP4 operator gates did not pass')
relative=Path(p['cache_relative']);lib['require'](not relative.is_absolute() and '..' not in relative.parts and relative.name==p['module']+'.so','Invalid bounded cache artifact')
path=root/'cache/mxfp4'/relative
s=path.stat();lib['require'](path.resolve(strict=True)==path and path.is_file() and s.st_uid==os.getuid() and 0<s.st_size<=128*2**20,'Unowned/unbounded JIT artifact')
h=hashlib.sha256()
with path.open('rb') as f:
 for part in iter(lambda:f.read(4*2**20),b''):h.update(part)
after=path.stat();lib['require']((s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Artifact changed during read')
artifact={'schema':'qualified_MXFP4_JIT_reuse_v1','module':p['module'],'cache_relative':str(relative),'sha256':h.hexdigest(),'bytes':s.st_size,'source_commit':r['python_source_commit'],'image_id':r['image_id'],'qualification_reference':{'run_id':r['run_id'],'result_sha256':hashlib.sha256(raw).hexdigest()}}
tree=[];total=0;cache=root/'cache/mxfp4';workspace=relative.parents[2]
for kind in ('generated','cached_ops'):
 directory=cache/workspace/kind/p['module']
 lib['require'](directory.resolve(strict=True)==directory and directory.stat().st_uid==os.getuid(),'Unowned JIT cache directory')
 for member in sorted(directory.rglob('*')):
  info=member.stat();lib['require'](member.resolve(strict=True)==member and info.st_uid==os.getuid(),'Unsafe JIT cache member')
  if member.is_dir():continue
  lib['require'](member.is_file() and info.st_size<=64*2**20 and len(tree)<128,'Unbounded JIT cache member')
  data=member.read_bytes();total+=len(data);after=member.stat()
  lib['require'](total<=128*2**20 and (info.st_ino,info.st_size,info.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'JIT cache changed during collection')
  tree.append({'relative':str(member.relative_to(cache)),'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mtime_ns':info.st_mtime_ns})
artifact['cache_tree']=tree
print(json.dumps({'ok':True,'artifact':artifact,'remote_writes':0,'original_result_sha256':hashlib.sha256(raw).hexdigest(),'original_artifact_stat':{'inode':s.st_ino,'mtime_ns':s.st_mtime_ns},'CPU_collection_only_no_model_load':True}))
'''


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    ticket=json.loads((Path('state/controller/tickets')/(a.run_id+'.json')).read_bytes())
    result=PinnedSSH(ticket['approval'],a.out.parent/'jit-identity-pin').rpc(SOURCE,
        {'ticket':ticket,'rpc_source':Path('scripts/controller/remote_rpc.py').read_text(),
         'module':MODULE,'cache_relative':CACHE_RELATIVE},timeout=180)
    write_new_json(a.out,result['artifact'])
    write_new_json(a.out.with_name(a.out.stem+'-PROVENANCE.json'),result)
    print(json.dumps({'receipt':str(a.out),'sha256':result['artifact']['sha256'],
                      'bytes':result['artifact']['bytes'],'remote_writes':0}))


if __name__=='__main__':main()
