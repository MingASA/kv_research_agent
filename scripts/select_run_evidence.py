#!/usr/bin/env python3
"""Inventory only a driver-released owned run for the read-only mirror collector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from controller.cli import _ticket, LEDGER_DIRECTORY
from controller.transport import PinnedSSH
from controller.durable import write_new_json


REMOTE = r'''
from pathlib import Path
import json,sys,os,stat,hashlib
p=json.load(sys.stdin);base=Path(p['base']);root=base/'agent_work/runs'/p['run_id']
assert os.getuid()==p['uid'] and base.resolve(strict=True)==base and root.resolve(strict=True)==root
assert base.stat().st_uid==p['uid'] and root.stat().st_uid==p['uid']
assert json.loads((root/'ticket.json').read_text())['reservation_id']==p['reservation_id']
b=(root/'result.json').read_bytes();assert hashlib.sha256(b).hexdigest()==p['result_sha256']
r=json.loads(b);assert r['gpu_release_verified'] and not r['live_owned_containers']
files=[]
for name in ('output','input','runtime','runtime_support'):
 for f in sorted((root/name).rglob('*')):
  st=f.lstat();assert not stat.S_ISLNK(st.st_mode)
  if stat.S_ISREG(st.st_mode) and '__pycache__' not in f.parts:
   assert f.resolve(strict=True)==f and st.st_uid==p['uid']
   files.append({'base_rel':str(f.relative_to(base)),'size':st.st_size})
for f in sorted(root.iterdir()):
 st=f.lstat()
 if stat.S_ISREG(st.st_mode):
  assert f.resolve(strict=True)==f and st.st_uid==p['uid'] and st.st_size<16*2**20
  files.append({'base_rel':str(f.relative_to(base)),'size':st.st_size})
assert sum(f['size'] for f in files)<512*2**20
print(json.dumps({'ok':True,'files':files,'total_bytes':sum(f['size'] for f in files),
                  'remote_writes':0,'run_id':p['run_id'],'terminal_result_sha256':p['result_sha256']}))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--status-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ticket = _ticket(args.run_id)
    status = json.loads(args.status_file.read_text())
    result = status['files']['result.json']
    if status['run_id'] != args.run_id or not result['value']['gpu_release_verified'] or status['live_owned_containers']:
        raise ValueError('Require an exact read-only terminal status with real driver release')
    transport = PinnedSSH(ticket['approval'], LEDGER_DIRECTORY / 'pins' / ticket['reservation_id'])
    selected = transport.rpc(REMOTE, {'base':ticket['approval']['approved_workspace'], 'run_id':args.run_id,
        'uid':ticket['approval']['approved_remote_identity']['uid'], 'reservation_id':ticket['reservation_id'],
        'result_sha256':result['sha256']})
    write_new_json(args.out, selected)
    print(json.dumps({'files':len(selected['files']), 'bytes':selected['total_bytes'], 'remote_writes':0}))


if __name__ == '__main__':
    main()
