"""Hash the existing approved weight shards once, without model/GPU execution."""

import argparse
import json
from pathlib import Path

from scripts.controller.transport import PinnedSSH
from scripts.observe_nextgen_long_metrics import require_original_ack

SOURCE = '''import datetime,hashlib,json,os,resource,sys,time
from pathlib import Path
p=json.load(sys.stdin);lib={'__name__':'readonly_cached_model_identity'};exec(p['rpc_source'],lib)
t=p['ticket'];root=lib['check_ticket'](t,False);lib['existing'](root,t)
model=Path(t['request']['model_path']);assert str(model)==t['approval']['read_only_model_path'] and model.resolve()==model
os.nice(10);resource.setrlimit(resource.RLIMIT_CPU,(180,180));resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
started=time.monotonic();index=model/'model.safetensors.index.json';assert index.resolve()==index and index.stat().st_size<=2**20
raw=index.read_bytes();value=json.loads(raw);names=sorted(set(value['weight_map'].values()));assert 1<=len(names)<=64
assert all(Path(n).name==n and n.endswith('.safetensors') for n in names)
assert sum((model/n).stat().st_size for n in names)<=60*2**30
expected=p.get('expected')
if expected is not None:
 assert expected['model_path']==str(model) and expected['index_sha256']==hashlib.sha256(raw).hexdigest()
 assert names==[r['name'] for r in expected['files']]
 for record in expected['files']:
  q=model/record['name'];assert q.resolve()==q and q.is_file();s=q.stat()
  actual={'size':s.st_size,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'uid':s.st_uid}
  assert all(actual[k]==record[k] for k in actual),'Previously SHA-bound weight file identity changed'
 print(json.dumps({'ok':True,'classification':'file_identity_matches_prior_full_SHA_manifest_without_rehash','source_manifest_sha256':p['expected_sha256'],'files_verified':len(names),'index_sha256':expected['index_sha256'],'weight_payload_bytes_read':0,'index_bytes_read':len(raw),'remote_writes':0,'GPU_requested':False,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
 sys.exit(0)
files=[]
for name in names:
 q=model/name;assert q.resolve()==q and q.is_file();s=q.stat();h=hashlib.sha256()
 with q.open('rb') as f:
  for part in iter(lambda:f.read(4*2**20),b''):
   assert time.monotonic()-started<600
   h.update(part)
 after=q.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 files.append({'name':name,'sha256':h.hexdigest(),'size':s.st_size,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns,'uid':s.st_uid})
print(json.dumps({'ok':True,'classification':'confirmed_SHA256_of_existing_readonly_weights_no_model_or_GPU','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model_path':str(model),'index_sha256':hashlib.sha256(raw).hexdigest(),'index_bytes':len(raw),'files':files,'total_weight_bytes':sum(r['size'] for r in files),'elapsed_seconds':time.monotonic()-started,'remote_writes':0,'GPU_requested':False,'downloaded_weight_bytes':0,'nice':os.getpriority(os.PRIO_PROCESS,0)}))
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-existing', type=Path)
    args = parser.parse_args()
    run = 'upstream-C-fp8-qualification-r0-20261102a'
    require_original_ack(run)
    ticket = json.loads((Path('state/controller/tickets')/(run+'.json')).read_bytes())
    if args.out.exists():
        raise FileExistsError('Preserve original weight identity receipt')
    payload = {'ticket': ticket, 'rpc_source': Path('scripts/controller/remote_rpc.py').read_text()}
    if args.verify_existing is not None:
        import hashlib
        raw = args.verify_existing.read_bytes()
        payload.update(expected=json.loads(raw), expected_sha256=hashlib.sha256(raw).hexdigest())
    result = PinnedSSH(ticket['approval'], args.out.parent/'model-identity-readonly-pin').rpc(SOURCE, payload, timeout=660)
    args.out.open('x').write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}))
