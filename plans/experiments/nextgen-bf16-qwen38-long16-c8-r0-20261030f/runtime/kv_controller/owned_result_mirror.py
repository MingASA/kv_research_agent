"""Read-only, hash-checked mirroring of one terminal owned run.

No tar streaming, remote writes, signals, model execution or GPU admission.
Partial local files remain resumable evidence. Only a fully verified manifest
is published for the independent auditors.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import stat

from .durable import fsync_directory, secure_directory, write_new_json


SOURCE = r'''
import base64,hashlib,json,os,stat,sys
from pathlib import Path
p=json.load(sys.stdin);lib={'__name__':'owned_mirror_library'}
exec(p['rpc_source'],lib)
t=p['ticket'];root=lib['check_ticket'](t,False);lib['existing'](root,t)
result=json.loads(lib['regular'](root/'result.json'))
lib['require'](result['status'] in ('COMPLETE','FAILED','TIMED_OUT','CONFLICT','DISK_LIMIT','NOT_LAUNCHED')
 and result['gpu_release_verified'] is True and not result['live_owned_containers'],'Run is not terminal/released')
if result['status']=='NOT_LAUNCHED':
 lib['require'](result.get('error')=='TimeoutError: Budget expired while waiting for 120s genuinely idle GPUs'
  and result.get('allocation_started_monotonic_ns') is None and result.get('allocation_ended_monotonic_ns') is None
  and result.get('allocation_intervals_ns')==[] and result.get('arm_results')==[],
  'Unallocated mirror requires an exact zero-GPU idle-wait result')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(8*2**20),b''):h.update(data)
 return h.hexdigest()
if p['operation']=='inventory':
 files=[];total=0
 for directory,dirs,names in os.walk(root,followlinks=False):
  current=Path(directory)
  if current==root:dirs[:]=[d for d in dirs if d not in ('source_repo','cache')]
  for name in list(dirs):
   path=current/name
   lib['require'](not path.is_symlink() and path.resolve()==path and path.stat().st_uid==os.getuid(),'Foreign/symlink directory in owned output')
  for name in names:
   path=current/name;s=path.lstat()
   lib['require'](result['status']!='NOT_LAUNCHED' or 'output' not in path.relative_to(root).parts[:1],
    'Zero-allocation wait contains unexpected worker output')
   lib['require'](stat.S_ISREG(s.st_mode) and path.resolve()==path and s.st_uid==os.getuid(),'Nonregular/foreign owned output')
   total+=s.st_size
   lib['require'](s.st_size<=64*2**20 and total<=160*2**20,'Owned mirror size envelope exceeded')
   files.append({'relative':str(path.relative_to(root)),'size':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'sha256':sha(path)})
 response={'ok':True,'run_id':t['request']['run_id'],'result_sha256':sha(root/'result.json'),
           'terminal':result['status'],'files':files,'bytes':total,'remote_writes':False}
elif p['operation']=='chunk':
 item=p['file'];relative=Path(item['relative'])
 lib['require'](not relative.is_absolute() and '..' not in relative.parts,'Unsafe mirror relative path')
 path=root/relative;s=path.stat()
 lib['require'](path.resolve()==path and s.st_uid==os.getuid() and stat.S_ISREG(s.st_mode)
 and [s.st_size,s.st_ino,s.st_mtime_ns]==[item['size'],item['inode'],item['mtime_ns']],'Owned file changed after terminal inventory')
 offset=p['offset'];length=min(2**20,item['size']-offset)
 lib['require'](type(offset) is int and 0<=offset<=item['size'],'Invalid chunk offset')
 with path.open('rb') as f:f.seek(offset);data=f.read(length)
 lib['require'](len(data)==length,'Short owned mirror chunk')
 response={'ok':True,'offset':offset,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
           'base64':base64.b64encode(data).decode(),'remote_writes':False}
else:raise ValueError('Unknown readonly mirror operation')
print(json.dumps(response))
'''


def _sha(path: Path) -> str:
    hasher=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*2**20),b''):hasher.update(block)
    return hasher.hexdigest()


def _regular(path: Path) -> None:
    info=path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid()
            or path.resolve(strict=True)!=path.absolute() or info.st_mode & 0o022):
        raise ValueError('Unsafe local mirror file')


def mirror_terminal(transport, ticket: dict, rpc_source: str, destination: Path) -> Path:
    secure_directory(destination)
    inventory_path=destination/'REMOTE_INVENTORY.json'
    if inventory_path.exists():
        _regular(inventory_path)
        inventory=json.loads(inventory_path.read_text())
        if inventory['run_id']!=ticket['request']['run_id']:raise ValueError('Mirror belongs to another run')
    else:
        inventory=transport.rpc(SOURCE,{'operation':'inventory','ticket':ticket,'rpc_source':rpc_source},mutation=False,timeout=90)
        write_new_json(inventory_path,inventory)
    if (inventory.get('ok') is not True or inventory.get('remote_writes') is not False
            or inventory.get('run_id')!=ticket['request']['run_id']):
        raise ValueError('Unverified remote mirror inventory')
    files=destination/'files';secure_directory(files)
    manifest=[];seen=set();total=0
    for item in inventory['files']:
        relative=Path(item['relative'])
        if (relative.is_absolute() or '..' in relative.parts or str(relative)!=item['relative']
                or relative==Path('.') or item['relative'] in seen
                or type(item['size']) is not int or not 0<=item['size']<=64*2**20):
            raise ValueError('Invalid or duplicate bounded mirror entry')
        seen.add(item['relative']);total+=item['size']
        if total>160*2**20:raise ValueError('Total mirror envelope exceeded')
        path=files/relative;secure_directory(path.parent)
        if path.exists() or path.is_symlink():
            _regular(path)
            if path.stat().st_size!=item['size'] or _sha(path)!=item['sha256']:
                raise ValueError('Existing local mirrored file differs; never overwrite evidence')
        else:
            partial=path.with_name(path.name+'.partial')
            if partial.exists() or partial.is_symlink():_regular(partial)
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>item['size']:raise ValueError('Unsafe partial owned mirror')
            fd=os.open(partial,os.O_WRONLY|os.O_APPEND|os.O_CREAT|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'ab') as stream:
                while offset<item['size']:
                    chunk=transport.rpc(SOURCE,{'operation':'chunk','ticket':ticket,'rpc_source':rpc_source,
                                               'file':item,'offset':offset},mutation=False,timeout=60)
                    raw=base64.b64decode(chunk['base64'],validate=True)
                    if (chunk.get('ok') is not True or chunk.get('remote_writes') is not False
                            or chunk['offset']!=offset or len(raw)!=min(2**20,item['size']-offset)
                            or len(raw)!=chunk['bytes'] or hashlib.sha256(raw).hexdigest()!=chunk['sha256']):
                        raise ValueError('Corrupt readonly owned mirror chunk')
                    stream.write(raw);stream.flush();os.fsync(stream.fileno());offset+=len(raw)
            if partial.stat().st_size!=item['size'] or _sha(partial)!=item['sha256']:
                raise ValueError('Full local owned mirror SHA mismatch')
            os.link(partial,path,follow_symlinks=False)
            partial.unlink()  # Only the verified local partial created by this mirror.
            fsync_directory(path.parent)
        manifest.append({'base_rel':item['relative'],'size':item['size'],
                         'remote_sha256':item['sha256'],'snapshot_sha256':_sha(path)})
    output=destination/'SHA_MANIFEST.json'
    expected={'layout':'run_relative','mirror_directory':str(destination.resolve()),'run_id':inventory['run_id'],
              'files':manifest,'remote_result_sha256':inventory['result_sha256']}
    if output.exists():
        _regular(output)
        if json.loads(output.read_text())!=expected:raise ValueError('Existing audit manifest differs')
    else:write_new_json(output,expected)
    return output
