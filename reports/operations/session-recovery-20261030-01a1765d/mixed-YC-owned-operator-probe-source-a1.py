import hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);t=p['ticket'];lib={'__name__':'owned_readonly_rpc'};exec(p['rpc_source'],lib)
root=lib['check_ticket'](t,False);lib['existing'](root,t);arm=t['request']['arms'][0];output=root/'output'/arm
identity=json.loads(lib['regular'](output/'container_identity.json'));lease=json.loads(lib['regular'](output/'lease.json'))
assert lease['request_sha256']==t['request_sha256'];sys.path.insert(0,str(root/'runtime'))
from kv_controller.supervisor import DockerBackend
container=DockerBackend(t,root)._inspect(identity['id']);assert container['State']['Running'] and container['Created']==identity['created']
assert lib['alive'](identity['pid_identity']);uid=t['approval']['approved_remote_identity']['uid'];processes=[]
for directory in Path('/proc').iterdir():
 if not directory.name.isdecimal():continue
 try:
  if directory.stat().st_uid!=uid:continue
  cgroup=(directory/'cgroup').read_text()
  if identity['id'] not in cgroup:continue
  raw=(directory/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
  argv=(directory/'cmdline').read_bytes().split(b'\0');cmd=[x.decode(errors='replace')[:180] for x in argv[:3] if x]
  processes.append({'pid':int(directory.name),'ppid':int(fields[1]),'state':fields[0],'utime_ticks':int(fields[11]),'stime_ticks':int(fields[12]),'start_ticks':int(fields[19]),'argv_first3':cmd})
 except (OSError,ValueError,IndexError):continue
assert len(processes)<=128
logs={}
for name in ('native_operator.log','container.log','server.log'):
 f=output/name
 if f.exists():
  data=lib['regular'](f);logs[name]={'bytes':len(data),'mtime_ns':f.stat().st_mtime_ns,'tail':data[-1500:].decode(errors='replace')}
print(json.dumps({'ok':True,'run_id':t['request']['run_id'],'container_id':identity['id'],'observed_epoch_ns':time.time_ns(),'own_processes':sorted(processes,key=lambda x:x['pid']),'logs':logs,'remote_file_writes':False,'GPU_or_model_or_start_stop_calls':False}))
