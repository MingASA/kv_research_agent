#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Close an uncertain, never-prepared dispatch using two read-only witnesses.

Never repeats prepare/start or writes a remote result. A settled run ID and its
scientific replicate remain consumed; a subsequent attempt needs a new ID.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.controller.approval import ApprovalError
from scripts.controller.durable import digest, write_new_json
from scripts.controller.quota import QuotaLedger
from scripts.controller.transport import PinnedSSH


AUDIT_SOURCE = r'''
"""S0 proof of an absent dispatch; no remote writes, signals or model loading."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin); t=p['ticket']; lib={'__name__':'kvra_absence_library'}
exec(p['rpc_source'],lib)
root=lib['check_ticket'](t,False)
writers=[]; workers=[]
for d in Path('/proc').iterdir():
    if not d.name.isdecimal() or int(d.name)==os.getpid():continue
    try:
        if d.stat().st_uid!=os.getuid():continue
        args=(d/'cmdline').read_bytes().split(b'\0')
        marker=b'Allowlisted stdin '+b'RPC, executed by pinned SSH'
        if any(marker in x for x in args):
            writers.append(lib['identity'](int(d.name)))
        if (str(root/'ticket.json').encode() in args and
                any(x in args for x in (b'kv_controller.supervisor',b'kv_controller.worker',b'kv_controller.watchdog'))):
            workers.append(lib['identity'](int(d.name)))
    except FileNotFoundError:pass
containers=[]
for label in ('kvra.reservation_id='+t['reservation_id'],'kvra.run_id='+t['request']['run_id']):
    containers.extend(subprocess.check_output(['docker','ps','-aq','--no-trunc','--filter','label='+label],text=True,timeout=10).split())
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
selected=[line for line in apps.splitlines() if line.split(',')[0].strip() in t['request']['gpu_uuids']]
result={'ok':True,'schema':'kvra-absent-dispatch-readonly-witness-v1','run_id':t['request']['run_id'],
 'reservation_id':t['reservation_id'],'request_sha256':t['request_sha256'],
 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
 'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic_ns':time.monotonic_ns(),
 'root_absent':not os.path.lexists(root),'remote_rpc_processes':writers,'run_processes':workers,
 'all_run_containers':sorted(set(containers)),'selected_compute_apps':selected,'remote_writes':0}
print(json.dumps(result))
'''


def terminal_from_witnesses(ticket: dict, witnesses: list[dict]) -> dict:
    if len(witnesses) != 2:
        raise ApprovalError('Two fresh read-only absence witnesses required')
    for value in witnesses:
        if (value.get('schema') != 'kvra-absent-dispatch-readonly-witness-v1'
                or value.get('ok') is not True or value.get('root_absent') is not True
                or value.get('remote_writes') != 0):
            raise ApprovalError('No trustworthy absent-root evidence')
        for key, expected in (('run_id', ticket['request']['run_id']),
                              ('reservation_id', ticket['reservation_id']),
                              ('request_sha256', ticket['request_sha256'])):
            if value.get(key) != expected:
                raise ApprovalError('Absence witness identity differs')
        for key in ('remote_rpc_processes', 'run_processes', 'all_run_containers'):
            if value.get(key) != []:
                raise ApprovalError('Possible old writer/process/container; keep UNKNOWN')
        # With a stably absent run root and no launcher/RPC/process/container,
        # another compute PID cannot be evidence that this run was prepared.
        # Preserve that busy resource evidence; this certificate grants no GPU
        # admission and never labels the shared device idle.
        apps = value.get('selected_compute_apps')
        if not isinstance(apps, list) or any(not isinstance(row, str) for row in apps):
            raise ApprovalError('Missing or malformed shared compute activity evidence')
    first, last = witnesses
    if (not first.get('boot_id') or first['boot_id'] != last.get('boot_id')
            or type(first.get('monotonic_ns')) is not int or type(last.get('monotonic_ns')) is not int
            or last['monotonic_ns'] - first['monotonic_ns'] < 30_000_000_000):
        raise ApprovalError('Absence is not stable for30seconds on the same boot')
    return {'run_id': ticket['request']['run_id'], 'reservation_id': ticket['reservation_id'],
            'request_sha256': ticket['request_sha256'], 'status': 'NOT_LAUNCHED',
            'evidence_origin': 'two_readonly_absence_witnesses_not_a_remote_result_file',
            'remote_result_file_created': False, 'gpu_release_verified': True,
            'gpu_release_scope': 'this_never_prepared_run_only',
            'shared_selected_compute_apps_present': any(w['selected_compute_apps'] for w in witnesses),
            'new_GPU_admission_authorized_by_this_proof': False,
            'allocation_started_monotonic_ns': None, 'allocation_ended_monotonic_ns': None,
            'live_owned_containers': [], 'new_output_bytes': 0,
            'readonly_witnesses': witnesses}


def _local_launcher_absent(run_id: str) -> bool:
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            args = (directory/'cmdline').read_bytes().split(b'\0')
            if (b'scripts.controller.cli' in args and b'launch' in args
                    and any(x.endswith(('/'+run_id+'/package.json').encode()) for x in args)):
                return False
        except FileNotFoundError:
            pass
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', args.run_id) is None:
        raise ApprovalError('Invalid run ID')
    directory = ROOT/'state/controller'
    ticket = json.loads((directory/'tickets'/(args.run_id+'.json')).read_text())
    if ticket['request']['run_id'] != args.run_id or not _local_launcher_absent(args.run_id):
        raise ApprovalError('Wrong ticket or original local launcher still exists')
    if args.out.exists():
        raise FileExistsError('Preserve earlier absence witnesses')
    with QuotaLedger(directory) as ledger:
        row = ledger.get(args.run_id)
        if (row['state'] != 'UNKNOWN' or row['reservation_id'] != ticket['reservation_id']
                or row['request_sha256'] != digest(ticket['request'])):
            raise ApprovalError('Only the exact uncertain dispatch may be reconciled')
    transport = PinnedSSH(ticket['approval'], directory/'pins'/ticket['reservation_id'])
    payload = {'ticket': ticket, 'rpc_source': (ROOT/'scripts/controller/remote_rpc.py').read_text()}
    witnesses = [transport.rpc(AUDIT_SOURCE, payload)]
    write_new_json(args.out.with_name(args.out.name+'.first_observation.json'),witnesses[0])
    time.sleep(30)
    witnesses.append(transport.rpc(AUDIT_SOURCE, payload))
    write_new_json(args.out.with_name(args.out.name+'.observations.json'),{'witnesses':witnesses})
    evidence = terminal_from_witnesses(ticket, witnesses)
    if not _local_launcher_absent(args.run_id):
        raise ApprovalError('Launcher appeared; keep UNKNOWN')
    write_new_json(args.out, evidence)
    with QuotaLedger(directory) as ledger:
        if ledger.get(args.run_id)['state'] != 'UNKNOWN':
            raise ApprovalError('Concurrent state change; inspect rather than settle')
        settled = ledger.settle(args.run_id, evidence)
    print(json.dumps({'run_id':args.run_id,'local_state':settled['state'],
                      'classification':'confirmed_never_prepared_not_launched',
                      'actual_gpu_us':settled['actual_gpu_us'],'remote_writes':0,
                      'shared_selected_compute_apps_present':evidence['shared_selected_compute_apps_present'],
                      'new_GPU_admission_authorized':False,
                      'relaunch_same_id':False,'evidence':str(args.out)}))


if __name__ == '__main__':
    main()
