"""Prove our stopped allocation ended while a foreign container keeps its GPU.

This performs no remote writes, signals or admission. The original UNKNOWN
result remains immutable. Two stable ownership witnesses prove our container
and worker are gone; every remaining driver PID must positively belong to a
different Docker cgroup. New launches still require the normal idle guard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from scripts.controller.approval import ApprovalError
from scripts.controller.durable import digest, write_new_json
from scripts.controller.quota import QuotaLedger
from scripts.controller.transport import PinnedSSH
from scripts.reconcile_absent_dispatch import _local_launcher_absent
from scripts.reconcile_unknown_release import SOURCE as BASE_SOURCE


ROOT = Path(__file__).resolve().parents[1]
_INSERT = '''
driver_identities=[]
for uuid,gpu in gpus.items():
 for pid in gpu['pids']:
  driver_identities.append({'GPU':uuid,'identity':lib['identity'](pid)})
'''
SOURCE = BASE_SOURCE.replace("print(json.dumps({", _INSERT + "\nprint(json.dumps({", 1).replace(
    "'supervisor_live':supervisor_live,", "'driver_identities':driver_identities,'supervisor_live':supervisor_live,", 1)


def terminal_from_witnesses(ticket: dict, witnesses: list[dict]) -> dict:
    if len(witnesses) != 2:
        raise ApprovalError('Two independent release witnesses required')
    first, last = witnesses
    for witness in witnesses:
        if (witness.get('ok') is not True or witness.get('remote_writes') is not False
                or witness.get('schema') != 'kvra-unknown-release-witness-v1'
                or witness.get('boot_id') != witness.get('allocation_boot_id')):
            raise ApprovalError('Invalid witness or boot drift')
        for key, expected in (('run_id',ticket['request']['run_id']),
                              ('reservation_id',ticket['reservation_id']),
                              ('request_sha256',ticket['request_sha256'])):
            if witness.get(key) != expected or witness.get('result',{}).get(key) != expected:
                raise ApprovalError('Witness/result identity mismatch')
        result = witness['result']
        if (result.get('status') != 'UNKNOWN' or result.get('gpu_release_verified') is not False
                or result.get('error') != 'UnsafeState: Foreign GPU activity/reservation; stop this run only'
                or result.get('arm_results') != [] or witness.get('supervisor_live') is not False
                or witness.get('run_workers') != []):
            raise ApprovalError('Not an exited incomplete conflict allocation')
        containers = witness.get('containers', [])
        if not containers or any(c.get('running') is not False or c.get('pid') != 0
                                 or c.get('status') not in ('exited','dead') for c in containers):
            raise ApprovalError('An owned container may still run')
        owned = {c['id'] for c in containers}
        if set(result.get('live_owned_containers', [])) != owned:
            raise ApprovalError('Owned registry does not cover the original unresolved containers')
        selected = witness.get('selected', {})
        if set(selected) != set(ticket['request']['gpu_uuids']):
            raise ApprovalError('Selected GPU set changed')
        expected = {(uuid,pid) for uuid,gpu in selected.items() for pid in gpu['pids']}
        proofs = witness.get('driver_identities', [])
        actual = {(p['GPU'],p['identity']['pid']) for p in proofs}
        if not expected or actual != expected or len(proofs) != len(expected):
            raise ApprovalError('All remaining driver PIDs need positive foreign ownership proofs')
        for proof in proofs:
            identity = proof['identity']
            cgroup = identity.get('cgroup','')
            foreign_ids = set(re.findall(r'docker[-/]([0-9a-f]{64})(?:\.scope|/|\n|$)', cgroup))
            if (identity.get('boot_id') != witness['boot_id'] or not foreign_ids
                    or foreign_ids & owned or any(cid in cgroup for cid in owned)):
                raise ApprovalError('Driver PID ownership is unknown or still ours')
    if (first['boot_id'] != last['boot_id'] or first['result_sha256'] != last['result_sha256']
            or first['result'] != last['result'] or first['driver_identities'] != last['driver_identities']
            or last['monotonic_ns'] - first['monotonic_ns'] < 30_000_000_000):
        raise ApprovalError('Release/result/foreign identity not stable for 30 seconds')
    result = first['result']
    intervals = result.get('allocation_intervals_ns', [])
    start = result.get('allocation_started_monotonic_ns')
    if (len(intervals) != 1 or type(start) is not int or intervals[0]['start_ns'] != start
            or intervals[0]['end_ns'] is not None or result.get('allocation_ended_monotonic_ns') is not None
            or last['monotonic_ns'] <= start):
        raise ApprovalError('Allocation accounting envelope is unknown')
    return {'run_id':ticket['request']['run_id'], 'reservation_id':ticket['reservation_id'],
            'request_sha256':ticket['request_sha256'], 'status':'CONFLICT',
            'gpu_release_verified':True, 'live_owned_containers':[],
            'allocation_started_monotonic_ns':start, 'allocation_ended_monotonic_ns':last['monotonic_ns'],
            'allocation_intervals_ns':[dict(intervals[0], end_ns=last['monotonic_ns'])],
            'new_output_bytes':max(result['new_output_bytes'], *(w['new_output_bytes'] for w in witnesses)),
            'evidence_origin':'two_readonly_own_container_exit_and_positive_foreign_driver_identity_witnesses',
            'original_result_status':'UNKNOWN', 'original_result_sha256':first['result_sha256'],
            'original_result_overwritten':False, 'GPU_is_idle':False,
            'accounting_scope':'conservative_upper_bound_until_last_witness',
            'new_GPU_admission_authorized':False, 'quality_or_performance_accepted':False,
            'readonly_witnesses':witnesses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', args.run_id) is None:
        raise ApprovalError('Invalid run ID')
    directory = ROOT/'state/controller'
    ticket = json.loads((directory/'tickets'/(args.run_id+'.json')).read_bytes())
    if args.out.exists() or not _local_launcher_absent(args.run_id):
        raise ApprovalError('Original launcher may still run or certificate exists')
    with QuotaLedger(directory) as ledger:
        row = ledger.get(args.run_id)
        if row['state'] != 'UNKNOWN' or row['request_sha256'] != digest(ticket['request']):
            raise ApprovalError('Only the original UNKNOWN reservation may be settled')
    transport = PinnedSSH(ticket['approval'], args.out.parent/'pin-conflict-certificate')
    payload = {'ticket':ticket,'rpc_source':(ROOT/'scripts/controller/remote_rpc.py').read_text()}
    witnesses = [transport.rpc(SOURCE, payload, timeout=90)]
    write_new_json(args.out.with_name(args.out.name+'.first.json'), witnesses[0])
    time.sleep(30)
    witnesses.append(transport.rpc(SOURCE, payload, timeout=90))
    write_new_json(args.out.with_name(args.out.name+'.observations.json'), {'witnesses':witnesses})
    evidence = terminal_from_witnesses(ticket, witnesses)
    if not _local_launcher_absent(args.run_id):
        raise ApprovalError('Original launcher reappeared')
    write_new_json(args.out, evidence)
    with QuotaLedger(directory) as ledger:
        if ledger.get(args.run_id)['state'] != 'UNKNOWN':
            raise ApprovalError('Concurrent ledger change')
        row = ledger.settle(args.run_id, evidence)
    print(json.dumps({'run_id':args.run_id,'state':row['state'],'remote_writes':0,
                      'GPU_launched':False,'original_UNKNOWN_unchanged':True,
                      'actual_gpu_us':row['actual_gpu_us'],'GPU_idle_claim':False}))


if __name__ == '__main__':
    main()
