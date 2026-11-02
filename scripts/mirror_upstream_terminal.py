"""Batch small evidence files and bounded large chunks from one released upstream run."""

import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path

from scripts.controller.durable import secure_directory, write_new_bytes, write_new_json
from scripts.controller.transport import PinnedSSH

SOURCE = '''import base64,gzip,hashlib,json,os,re,sys
from pathlib import Path
p=json.load(sys.stdin);lib={'__name__':'bounded_upstream_readonly_mirror'};exec(p['rpc_source'],lib)
t=p['ticket'];root=lib['check_ticket'](t,False);lib['existing'](root,t)
lib['require'](t['request']['workload']['kind'].startswith('upstream_'),'Only frozen upstream runs may use this export')
r=json.loads(lib['regular'](root/'result.json'))
lib['require'](r['status'] in ('COMPLETE','FAILED','NOT_LAUNCHED','TIMED_OUT') and r['gpu_release_verified'] and not r['live_owned_containers'],'Original run not terminal/released')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(4*2**20),b''):h.update(part)
 return h.hexdigest()
if p['operation']=='inventory':
 lib['require'](not lib['status'](root,t)['live_owned_containers'],'Original owned container is live')
 files=[];total=0;omitted=[]
 if p.get('omit_large_core_dumps'):lib['require'](r['status'] in ('FAILED','TIMED_OUT'),'Only a preserved failure may omit a large core dump')
 for directory,dirs,names in os.walk(root,followlinks=False):
  here=Path(directory)
  if here==root:dirs[:]=[d for d in dirs if d not in ('source_repo','cache')]
  for d in dirs:lib['require']((here/d).resolve()==here/d and (here/d).stat().st_uid==os.getuid(),'Foreign/symlink evidence directory')
  for name in names:
   q=here/name;s=q.stat();total+=s.st_size
   lib['require'](q.resolve()==q and q.is_file() and s.st_uid==os.getuid() and total<=2*2**30,'Evidence export exceeds explicit bounds')
   if s.st_size>256*2**20 and p.get('omit_large_core_dumps') and re.fullmatch(r'core(?:[._-][A-Za-z0-9._-]+)?',name):
    lib['require'](s.st_size<=2**30,'Core digest exceeds its explicit one-GiB read bound')
    omitted.append({'relative':str(q.relative_to(root)),'size':s.st_size,'sha256':digest(q),'reason':'Core over256MiB kept immutable at original remote path; no unnecessary large transfer'})
    continue
   lib['require'](s.st_size<=256*2**20,'Evidence file exceeds explicit transfer bound')
   files.append({'relative':str(q.relative_to(root)),'size':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'sha256':digest(q)})
 out={'ok':True,'remote_writes':0,'run_id':t['request']['run_id'],'terminal':r['status'],'result_sha256':digest(root/'result.json'),'files':files,'bytes':sum(item['size'] for item in files),'remote_total_bytes':total,'omitted_remote_core_dumps':omitted}
else:
 items=p['files'];lib['require'](len(items)<=64,'Too many batch members');outfiles=[];total=0
 for item in items:
  rel=Path(item['relative']);lib['require'](not rel.is_absolute() and '..' not in rel.parts,'Unsafe evidence member');q=root/rel;s=q.stat()
  lib['require'](q.resolve()==q and q.is_file() and s.st_uid==os.getuid() and [s.st_size,s.st_ino,s.st_mtime_ns]==[item['size'],item['inode'],item['mtime_ns']],'Evidence changed after inventory')
  offset=item.get('offset',0);length=item.get('length',s.st_size)
  lib['require'](type(offset) is int and type(length) is int and 0<=offset<=s.st_size and 0<=length<=16*2**20 and offset+length<=s.st_size,'Unbounded chunk')
  total+=length;lib['require'](total<=16*2**20,'Batch bytes exceed limit')
  with q.open('rb') as f:f.seek(offset);raw=f.read(length)
  lib['require'](len(raw)==length,'Short evidence read')
  compressed=len(raw)>1048576
  outfiles.append({'relative':str(rel),'offset':offset,'sha256':hashlib.sha256(raw).hexdigest(),'encoding':'gzip-base64' if compressed else 'base64','base64':base64.b64encode(gzip.compress(raw,mtime=0) if compressed else raw).decode()})
 out={'ok':True,'remote_writes':0,'files':outfiles}
print(json.dumps(out))
'''


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4*2**20), b''):
            digest.update(part)
    return digest.hexdigest()


def decode_evidence(value: dict, expected_size: int) -> bytes:
    """Bound decompression before accepting the original uncompressed SHA."""
    import io
    wire = base64.b64decode(value['base64'], validate=True)
    if value.get('encoding', 'base64') == 'gzip-base64':
        if len(wire) > 16*2**20+2**16:
            raise ValueError('Compressed evidence transport exceeds its bounded chunk')
        with gzip.GzipFile(fileobj=io.BytesIO(wire)) as stream:
            raw = stream.read(expected_size+1)
    elif value.get('encoding', 'base64') == 'base64':
        raw = wire
    else:
        raise ValueError('Unknown evidence transport encoding')
    if len(raw) != expected_size:
        raise ValueError('Evidence decompression changed its declared uncompressed size')
    return raw


def mirror(run: str, destination: Path, *, omit_large_core_dumps: bool = False,
           reuse_local_frozen_assets: bool = False):
    secure_directory(destination)
    ticket = json.loads((Path('state/controller/tickets')/(run+'.json')).read_bytes())
    transport = PinnedSSH(ticket['approval'], destination/'pin')
    payload = {'ticket': ticket, 'rpc_source': Path('scripts/controller/remote_rpc.py').read_text()}
    inventory_path = destination/'REMOTE_INVENTORY.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_bytes())
    else:
        inventory = transport.rpc(SOURCE, dict(payload, operation='inventory',
            omit_large_core_dumps=omit_large_core_dumps), timeout=180)
        write_new_json(inventory_path, inventory)
    if inventory['run_id'] != run:
        raise ValueError('Mirror belongs to another original run')
    files = destination/'files'
    secure_directory(files)
    seen = set()
    for item in inventory['files']:
        relative = Path(item['relative'])
        if (relative.is_absolute() or '..' in relative.parts or relative == Path('.')
                or str(relative) != item['relative'] or item['relative'] in seen
                or type(item['size']) is not int or not 0 <= item['size'] <= 256*2**20):
            raise ValueError('Unsafe local evidence inventory')
        seen.add(item['relative'])
    if sum(x['size'] for x in inventory['files']) > 2*2**30 or len(seen) > 10000:
        raise ValueError('Local evidence inventory exceeds declared bounds')
    if reuse_local_frozen_assets:
        plan = Path('plans/experiments')/run
        package = json.loads((plan/'package.json').read_bytes())
        if package['request_sha256'] != ticket['request_sha256']:
            raise ValueError('Local frozen package belongs to another original request')
        reused = []
        for item in inventory['files']:
            name = item['relative']
            asset = package['assets'].get(name)
            if not name.startswith(('input/', 'runtime/')) or asset is None:
                continue
            if asset['sha256'] != item['sha256'] or ticket['assets_sha256'].get(name) != item['sha256']:
                raise ValueError('Original remote inventory differs from its frozen input asset')
            target = files/name
            if target.exists():
                continue
            source = Path(asset['local_path'])
            if source.is_symlink() or not source.resolve().is_relative_to(plan.resolve()):
                raise ValueError('Local reuse must stay inside the original frozen proposal')
            raw = source.read_bytes()
            if len(raw) != item['size'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Original local frozen input changed')
            secure_directory(target.parent)
            write_new_bytes(target, raw)
            reused.append({'relative': name, 'size': len(raw), 'sha256': item['sha256']})
        proof = destination/'LOCAL_FROZEN_ASSET_REUSE.json'
        if not proof.exists():
            write_new_json(proof, {'request_sha256': ticket['request_sha256'], 'files': reused,
                'fresh_original_remote_inventory_SHA_match': True,
                'model_outputs_or_responses_reused': False, 'remote_writes': 0})
    pending = []

    def fetch(items):
        result = transport.rpc(SOURCE, dict(payload, operation='read', files=items), timeout=180)
        if len(result['files']) != len(items):
            raise ValueError('Incomplete evidence batch')
        return result['files']

    def save_small(items):
        for item, value in zip(items, fetch(items)):
            raw = decode_evidence(value, item['size'])
            if (value['relative'] != item['relative'] or value['offset'] != 0
                    or len(raw) != item['size'] or hashlib.sha256(raw).hexdigest() != item['sha256']):
                raise ValueError('Corrupt small evidence batch')
            target = files/item['relative']
            secure_directory(target.parent)
            with target.open('xb') as stream:
                stream.write(raw)
            target.chmod(0o600)

    for item in inventory['files']:
        target = files/item['relative']
        if target.exists():
            if target.is_symlink() or sha(target) != item['sha256'] or target.stat().st_size != item['size']:
                raise ValueError('Never overwrite a different local evidence file')
            continue
        if item['size'] <= 2**20:
            if len(pending) == 64 or sum(x['size'] for x in pending)+item['size'] > 16*2**20:
                save_small(pending);pending=[]
            pending.append(item)
            continue
        if pending:
            save_small(pending);pending=[]
        secure_directory(target.parent)
        partial = target.with_name(target.name+'.partial')
        if partial.is_symlink():
            raise ValueError('Unsafe partial evidence path')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > item['size']:
            raise ValueError('Oversized partial evidence')
        with partial.open('ab') as stream:
            partial.chmod(0o600)
            while offset < item['size']:
                length = min(16*2**20, item['size']-offset)
                value = fetch([dict(item, offset=offset, length=length)])[0]
                raw = decode_evidence(value, length)
                if (value['relative'] != item['relative'] or value['offset'] != offset
                        or len(raw) != length or hashlib.sha256(raw).hexdigest() != value['sha256']):
                    raise ValueError('Corrupt evidence chunk')
                stream.write(raw);stream.flush();os.fsync(stream.fileno());offset+=length
        if sha(partial) != item['sha256']:
            raise ValueError('Full evidence file SHA mismatch')
        os.link(partial, target, follow_symlinks=False)
        partial.unlink()
    if pending:
        save_small(pending)
    manifest = {'layout': 'run_relative', 'mirror_directory': str(destination.resolve()), 'run_id': run,
        'remote_result_sha256': inventory['result_sha256'], 'files': [
            {'base_rel': x['relative'], 'size': x['size'], 'remote_sha256': x['sha256'],
             'snapshot_sha256': sha(files/x['relative'])} for x in inventory['files']]}
    if inventory.get('omitted_remote_core_dumps'):
        manifest['omitted_remote_core_dumps'] = inventory['omitted_remote_core_dumps']
    output = destination/'SHA_MANIFEST.json'
    if output.exists():
        if json.loads(output.read_bytes()) != manifest:
            raise ValueError('Existing mirror seal differs')
    else:
        write_new_json(output, manifest)
    print(json.dumps({'run_id': run, 'manifest': str(output), 'files': len(inventory['files']),
                      'bytes': inventory['bytes'], 'remote_writes': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--omit-large-core-dumps', action='store_true')
    parser.add_argument('--reuse-local-frozen-assets', action='store_true')
    args = parser.parse_args()
    mirror(args.run_id, args.out, omit_large_core_dumps=args.omit_large_core_dumps,
           reuse_local_frozen_assets=args.reuse_local_frozen_assets)
