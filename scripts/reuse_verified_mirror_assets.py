#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Reuse byte-identical input assets after a fresh, pinned remote SHA attestation.

Only input/runtime assets may be reused. Every output/raw file still crosses SSH.
The original partial transfer and network-subset snapshot remain immutable.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.cli import ROOT, LEDGER_DIRECTORY, _local_package, _ticket
from scripts.controller.durable import digest, secure_directory, write_new_json
from scripts.controller.transport import PinnedSSH

REMOTE = r'''
import datetime,hashlib,json,os,stat,sys
from pathlib import Path,PurePosixPath
p=json.load(sys.stdin);base=Path(p['base']);root=base/'agent_work/runs'/p['run_id']
assert os.getuid()==p['uid'] and base.resolve()==base and root.resolve()==root
assert base.stat().st_uid==p['uid'] and root.stat().st_uid==p['uid']
for f in (root/'ticket.json',root/'result.json'):
 assert f.resolve()==f and stat.S_ISREG(f.stat().st_mode) and f.stat().st_uid==p['uid'] and f.stat().st_size<8*2**20
ticket=json.loads((root/'ticket.json').read_text())
assert ticket['request_sha256']==p['request_sha256']
assert hashlib.sha256((json.dumps(ticket['request'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()).hexdigest()==p['request_sha256']
assert ticket['request']['remote_output']==str(root)
b=(root/'result.json').read_bytes();assert hashlib.sha256(b).hexdigest()==p['terminal_sha256']
r=json.loads(b);assert r['status']=='COMPLETE' and r['gpu_release_verified'] is True and not r['live_owned_containers']
rows=[]
for item in p['assets']:
 rel=PurePosixPath(item['asset']);assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in ('input','runtime','runtime_support')
 f=root/str(rel);assert f.resolve()==f
 before=f.stat();assert stat.S_ISREG(before.st_mode) and before.st_uid==p['uid'] and before.st_size==item['size'] and before.st_size<=64*2**20
 expected=(ticket['request']['dataset_sha256'] if item['asset']=='input/payload_trace.jsonl' else ticket['assets_sha256'][item['asset']]);assert expected==item['sha256']
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(2**20),b''):h.update(block)
 after=f.stat();assert all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns')) and h.hexdigest()==expected
 rows.append({'source':str(f),'relative_path':'support/'+str(f.relative_to(base)),'size':before.st_size,'sha256':expected})
print(json.dumps({'ok':True,'classification':'confirmed_fresh_remote_SHA_for_local_cached_input_runtime_assets_only','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':p['run_id'],'request_sha256':p['request_sha256'],'terminal_sha256':p['terminal_sha256'],'files':rows,'remote_writes':0}))
'''


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def destination(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or str(rel) != relative:
        raise ValueError('Unsafe mirror member')
    return root/relative


def copy_verified(source: Path, target: Path, size: int, expected: str) -> None:
    if (source.resolve(strict=True) != source or not source.is_file()
            or source.stat().st_uid != os.getuid() or source.stat().st_size != size or sha(source) != expected):
        raise ValueError('Local cache/network bytes differ from remote SHA')
    secure_directory(target.parent)
    h = hashlib.sha256()
    with source.open('rb') as inp, target.open('xb') as out:
        for block in iter(lambda:inp.read(2**20), b''):
            out.write(block);h.update(block)
        out.flush();os.fsync(out.fileno())
    if h.hexdigest() != expected or target.stat().st_size != size:
        raise ValueError('Mirror changed during assembly')


def prepare(package: Path, status: Path, selection: Path, out: Path) -> None:
    request, hashes, _ = _local_package(package)
    ticket = _ticket(request['run_id'])
    if digest(request) != ticket['request_sha256'] or hashes != ticket['assets_sha256']:
        raise ValueError('Package differs from actual issued ticket')
    current = json.loads(status.read_text());terminal = current['files']['result.json']
    if (current['run_id'] != request['run_id'] or terminal['value']['status'] != 'COMPLETE'
            or terminal['value']['gpu_release_verified'] is not True or current['live_owned_containers']):
        raise ValueError('Require actual completed driver-released run')
    selected = json.loads(selection.read_text());prefix = 'agent_work/runs/'+request['run_id']+'/'
    inventory = {row['base_rel']:row for row in selected['files']}
    if (len(inventory) != len(selected['files']) or any(not k.startswith(prefix) for k in inventory)
            or selected['terminal_result_sha256'] != terminal['sha256']):
        raise ValueError('Original selection differs from exact run')
    original = json.loads(package.read_text());cached = []
    for name, entry in original['assets'].items():
        if PurePosixPath(name).parts[0] not in ('input', 'runtime', 'runtime_support'):
            raise ValueError('Outputs cannot be reused from local cache')
        cached.append({'asset':name, 'local_path':entry['local_path'], 'sha256':entry['sha256']})
    cached.append({'asset':'input/payload_trace.jsonl', 'local_path':str(package.parent/'payload_trace.jsonl'),
                   'sha256':request['dataset_sha256']})
    for row in cached:
        row['base_rel'] = prefix+row['asset'];row['size'] = inventory[row['base_rel']]['size']
        p = Path(row['local_path'])
        if p.resolve() != p or p.stat().st_size != row['size'] or sha(p) != row['sha256']:
            raise ValueError('Local frozen cache does not match issued ticket')
    if out.exists():
        raise FileExistsError('New cache plan required')
    secure_directory(out)
    remote = PinnedSSH(ticket['approval'], LEDGER_DIRECTORY/'pins'/ticket['reservation_id']).rpc(
        REMOTE, {'base':request['approved_workspace'], 'uid':ticket['approval']['approved_remote_identity']['uid'],
            'run_id':request['run_id'], 'request_sha256':ticket['request_sha256'],
            'terminal_sha256':terminal['sha256'], 'assets':cached})
    write_new_json(out/'remote_cached_asset_SHA.json', remote)
    excluded = {row['base_rel'] for row in cached}
    network = dict(selected);network['files'] = [row for row in selected['files'] if row['base_rel'] not in excluded]
    network['total_bytes'] = sum(row['size'] for row in network['files'])
    write_new_json(out/'network_selection.json', network)
    write_new_json(out/'cache_plan.json', {'classification':'verified_cached_asset_reuse_plan_raw_outputs_must_transfer',
        'run_id':request['run_id'], 'base':request['approved_workspace'],
        'request_sha256':ticket['request_sha256'], 'terminal_sha256':terminal['sha256'],
        'original_selection':selected, 'cached':cached,
        'remote_proof':str(out/'remote_cached_asset_SHA.json'), 'remote_proof_sha256':sha(out/'remote_cached_asset_SHA.json')})
    print(json.dumps({'cached_input_runtime_files':len(cached), 'cached_bytes':sum(row['size'] for row in cached),
                      'raw_network_files':len(network['files']), 'network_bytes':network['total_bytes'], 'remote_writes':0}))


def assemble(plan_path: Path, network: Path | list[Path], out: Path) -> None:
    plan = json.loads(plan_path.read_text());proof_path = Path(plan['remote_proof'])
    if sha(proof_path) != plan['remote_proof_sha256']:
        raise ValueError('Fresh remote cache attestation changed')
    networks = [network] if isinstance(network, Path) else network
    snapshots = [json.loads((path/'SNAPSHOT.json').read_text()) for path in networks]
    if not snapshots:
        raise ValueError('Missing actual network snapshots')
    proof = json.loads(proof_path.read_text());snapshot = dict(snapshots[0])
    if (proof['run_id'] != plan['run_id'] or proof['terminal_sha256'] != plan['terminal_sha256']
            or proof['request_sha256'] != plan['request_sha256']
            or any(s['base'] != plan['base'] or s['uid'] != 1005 for s in snapshots)):
        raise ValueError('Cache/network run identity differs')
    expected = {'support/'+r['base_rel']:r['size'] for r in plan['original_selection']['files']}
    cache = {'support/'+r['base_rel']:r for r in plan['cached']}
    attested = {r['relative_path']:r for r in proof['files']}
    transferred, network_sources = {}, {}
    for path, part in zip(networks, snapshots):
        for row in part['files']:
            name = row['relative_path']
            if name in transferred:
                raise ValueError('Duplicate network evidence across transfer parts')
            transferred[name] = row;network_sources[name] = path
    if (set(cache) != set(attested) or set(cache)&set(transferred)
            or set(cache)|set(transferred) != set(expected)
            or len(transferred) != sum(len(s['files']) for s in snapshots)):
        raise ValueError('Missing, duplicate or extra cached/network evidence')
    if out.exists():
        raise FileExistsError('New assembled mirror directory required')
    secure_directory(out)
    rows = []
    for name in sorted(expected):
        row = attested[name] if name in cache else transferred[name]
        if row['size'] != expected[name] or row['source'] != str(Path(plan['base'])/name.removeprefix('support/')):
            raise ValueError('Remote source path/size differs')
        source = Path(cache[name]['local_path']) if name in cache else destination(network_sources[name], name)
        if name in cache and row['sha256'] != cache[name]['sha256']:
            raise ValueError('Cached bytes lack matching remote SHA')
        copy_verified(source, destination(out, name), row['size'], row['sha256']);rows.append(row)
    root = out/'support/agent_work/runs'/plan['run_id']
    if (sha(root/'result.json') != plan['terminal_sha256']
            or json.loads((root/'ticket.json').read_text())['request_sha256'] != plan['request_sha256']):
        raise ValueError('Assembled terminal/ticket differs')
    snapshot.update(files=rows, cached_asset_reuse={'files':len(cache), 'raw_files_reused':0,
        'fresh_remote_proof_sha256':plan['remote_proof_sha256'], 'asset_SHA_observed_utc':proof['observed_utc'],
        'network_snapshot_sha256':[sha(path/'SNAPSHOT.json') for path in networks],
        'original_partial_and_network_snapshot_preserved':True})
    write_new_json(out/'SNAPSHOT.json', snapshot)
    write_new_json(out.parent/'asset_reuse_assembly_receipt.json', {'classification':'confirmed_all_remote_SHA_with_declared_local_input_runtime_reuse_and_full_output_network_transfer',
        'run_id':plan['run_id'], 'files_SHA_verified':len(rows), 'bytes_SHA_verified':sum(r['size'] for r in rows),
        'input_runtime_files_reused_after_remote_SHA':len(cache), 'raw_output_files_reused':0,
        'terminal_sha256':plan['terminal_sha256'], 'plan_sha256':sha(plan_path), 'remote_writes':0})
    print(json.dumps({'files_SHA_verified':len(rows), 'bytes':sum(r['size'] for r in rows), 'raw_outputs_all_transferred':True}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__);sub = parser.add_subparsers(dest='mode', required=True)
    command = sub.add_parser('prepare')
    for name in ('package', 'status', 'selection', 'out'):
        command.add_argument('--'+name, type=Path, required=True)
    command = sub.add_parser('assemble')
    for name in ('plan', 'out'):
        command.add_argument('--'+name, type=Path, required=True)
    command.add_argument('--network', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.package.resolve(), args.status.resolve(), args.selection.resolve(), args.out.absolute())
    else:
        assemble(args.plan.resolve(), [p.resolve() for p in args.network], args.out.absolute())


if __name__ == '__main__':
    main()
