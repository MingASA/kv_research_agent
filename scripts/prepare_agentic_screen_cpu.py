"""Materialize one bounded, offline AIPerf Agentic Code screening trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.controller.transport import PinnedSSH


SOURCE = r'''
import hashlib,json,os,socket,subprocess,sys
from pathlib import Path
p=json.load(sys.stdin);a=p['approval'];e=a['approved_remote_identity']
assert (os.getuid(),os.getgid(),socket.gethostname())==(e['uid'],e['gid'],e['hostname'])
base=Path(a['approved_workspace']);assert base.resolve(strict=True)==base and base.stat().st_uid==os.getuid()
parent=base/'agent_work/scratch';assert parent.resolve(strict=True)==parent and parent.stat().st_uid==os.getuid()
root=parent/'agentic-screen-20261105b';root.mkdir(mode=0o700)
cmd=['docker','run','--rm','--name','kvra-cpu-agentic-screen-20261105b','--network','none','--read-only',
 '--user',str(e['uid'])+':'+str(e['gid']),'--cap-drop','ALL','--pids-limit','128','--memory','2g','--cpus','2',
 '--tmpfs','/tmp:rw,nosuid,size=256m','--mount','type=bind,src='+str(base)+',dst=/workspace,readonly',
 '--mount','type=bind,src='+str(base)+',dst='+str(base)+',readonly',
 '--mount','type=bind,src='+str(root)+',dst=/results']
for key,value in {'NVIDIA_VISIBLE_DEVICES':'void','PYTHONDONTWRITEBYTECODE':'1','HOME':'/tmp',
 'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}.items():
 cmd+=['-e',key+'='+value]
cmd+=['--entrypoint','/workspace/.venv-aiperf/bin/python',a['approved_image_id'],'-B','-m','aiperf',
 'synthesize','agentic-code','--num-sessions','32','--output','/results','--seed','20261105',
 '--max-isl','65536','--max-osl','1024']
with (root/'GENERATOR.stdout').open('x') as out,(root/'GENERATOR.stderr').open('x') as err:
 result=subprocess.run(cmd,stdout=out,stderr=err,timeout=180)
files=[];data=[]
for f in root.rglob('*'):
 if f.is_file():
  assert not f.is_symlink() and f.stat().st_uid==os.getuid()
  raw=f.read_bytes();files.append({'path':str(f.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  if f.name in ('dataset.jsonl','manifest.json','quality.json') and len(raw)<=4*2**20:
   data.append({'path':str(f.relative_to(root)),'content':raw.decode()})
print(json.dumps({'ok':True,'GPU_requested':False,'returncode':result.returncode,'root':str(root),
 'command':cmd,'files':files,'data':data,'stdout':(root/'GENERATOR.stdout').read_text()[-3000:],
 'stderr':(root/'GENERATOR.stderr').read_text()[-3000:]}))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    intent = args.out.with_name(args.out.name+'.intent.json')
    approval = json.loads((Path.home()/'.config/kv-research/approval.json').read_bytes())
    with intent.open('x') as stream:
        json.dump({'operation':'offline_CPU_synthetic_trace_generation','GPU_requested':False,
                   'remote_root_suffix':'agentic-screen-20261105b','do_not_repeat':True}, stream)
    response = PinnedSSH(approval, args.out.parent/'pin-agentic-cpu-b1').rpc(
        SOURCE, {'approval':approval}, mutation=True, timeout=220)
    with args.out.open('x') as stream:
        json.dump(response, stream, indent=2)
    print(json.dumps({'returncode':response['returncode'],'GPU_requested':False,
                      'files':response['files'],'stderr':response['stderr']}))


if __name__ == '__main__':
    main()
