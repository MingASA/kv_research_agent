"""Close an abandoned prepared run using two read-only no-start witnesses.

Never starts or changes the remote run. Its identifier remains consumed.
"""

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


ROOT = Path(__file__).resolve().parents[1]
MARKERS = ('START_INTENT.json', 'dispatch.lock', 'SUPERVISOR_PROCESS.json', 'SUPERVISOR_STARTED.json',
           'owned_containers.json', 'state.json', 'result.json', 'supervisor.log')
SOURCE = r'''
import datetime,hashlib,json,os,stat,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);t=p['ticket'];lib={'__name__':'prepared_no_start_readonly'}
exec(p['rpc_source'],lib)
root=lib['check_ticket'](t,False);lib['existing'](root,t)
prepared=lib['regular'](root/'PREPARED.json')
lib['require'](json.loads(prepared)['request_sha256']==t['request_sha256'],'Prepared identity differs')
markers={name:os.path.lexists(root/name) for name in p['markers']}
writers=[];workers=[]
for d in Path('/proc').iterdir():
 if not d.name.isdecimal() or int(d.name)==os.getpid():continue
 try:
  if d.stat().st_uid!=os.getuid():continue
  args=(d/'cmdline').read_bytes().split(b'\0')
  marker=b'Allowlisted stdin '+b'RPC, executed by pinned SSH'
  if any(marker in x for x in args):writers.append(lib['identity'](int(d.name)))
  if any(x in args for x in (b'kv_controller.supervisor',b'kv_controller.worker',b'kv_controller.watchdog')) and any(str(root).encode() in x for x in args):workers.append(lib['identity'](int(d.name)))
 except FileNotFoundError:pass
containers=set()
for label in ('kvra.reservation_id='+t['reservation_id'],'kvra.run_id='+t['request']['run_id']):
 containers.update(subprocess.check_output(['docker','ps','-aq','--no-trunc','--filter','label='+label],text=True,timeout=15).split())
entries=[];total=0
for directory,folders,files in os.walk(root,followlinks=False):
 for name in folders+files:
  path=Path(directory)/name;info=path.lstat()
  lib['require'](info.st_uid==os.getuid(),'Unowned member: '+str(path.relative_to(root)))
  # A Git checkout can contain owned symlinks. Inventory their link bytes;
  # never follow them or inspect the target, including symlink directories.
  if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
   total+=info.st_size;lib['require'](total<=2*2**30,'Unexpected prepared run size')
   entries.append([str(path.relative_to(root)),info.st_ino,info.st_size,info.st_mtime_ns,
                   os.readlink(path) if stat.S_ISLNK(info.st_mode) else None])
inventory=hashlib.sha256(json.dumps(sorted(entries),separators=(',',':')).encode()).hexdigest()
print(json.dumps({'ok':True,'schema':'kvra-prepared-no-start-witness-v1','remote_writes':0,
 'run_id':t['request']['run_id'],'reservation_id':t['reservation_id'],'request_sha256':t['request_sha256'],
 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'monotonic_ns':time.monotonic_ns(),
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'prepared_sha256':hashlib.sha256(prepared).hexdigest(),'markers':markers,
 'remote_rpc_processes':writers,'run_processes':workers,'all_run_containers':sorted(containers),
 'inventory_sha256':inventory,'new_output_bytes':total}))
'''


def terminal_from_witnesses(ticket: dict, witnesses: list) -> dict:
    if len(witnesses) != 2:
        raise ApprovalError('Two prepared/no-start witnesses required')
    for value in witnesses:
        if (value.get('schema') != 'kvra-prepared-no-start-witness-v1'
                or value.get('ok') is not True or value.get('remote_writes') != 0
                or value.get('markers') != dict.fromkeys(MARKERS, False)):
            raise ApprovalError('Incomplete no-start evidence or a launch marker exists')
        for key, expected in (('run_id', ticket['request']['run_id']),
                              ('reservation_id', ticket['reservation_id']),
                              ('request_sha256', ticket['request_sha256'])):
            if value.get(key) != expected:
                raise ApprovalError('No-start witness identity differs')
        if any(value.get(key) != [] for key in ('remote_rpc_processes', 'run_processes', 'all_run_containers')):
            raise ApprovalError('Possible original process/container; retain UNKNOWN')
        if (type(value.get('new_output_bytes')) is not int or not 0 <= value['new_output_bytes'] <= 2*2**30
                or type(value.get('monotonic_ns')) is not int
                or any(re.fullmatch('[0-9a-f]{64}', value.get(key, '')) is None
                       for key in ('prepared_sha256', 'inventory_sha256'))):
            raise ApprovalError('Missing inventory/accounting evidence')
    first, last = witnesses
    if (not first.get('boot_id') or any(first.get(k) != last.get(k) for k in
            ('boot_id', 'prepared_sha256', 'inventory_sha256', 'new_output_bytes'))
            or last['monotonic_ns'] - first['monotonic_ns'] < 30_000_000_000):
        raise ApprovalError('Prepared no-start state must remain stable for 30 seconds')
    return {'run_id': ticket['request']['run_id'], 'reservation_id': ticket['reservation_id'],
            'request_sha256': ticket['request_sha256'], 'status': 'NOT_LAUNCHED',
            'error': 'Abandoned prepared dispatch after uncertain SSH start; two no-start witnesses',
            'gpu_release_verified': True, 'live_owned_containers': [], 'arm_results': [],
            'allocation_started_monotonic_ns': None, 'allocation_ended_monotonic_ns': None,
            'allocation_intervals_ns': [], 'new_output_bytes': last['new_output_bytes'],
            'evidence_origin': 'two_readonly_prepared_no_start_witnesses',
            'remote_result_file_created': False, 'relaunch_same_id': False,
            'new_GPU_admission_authorized': False, 'readonly_witnesses': witnesses}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,127}', args.run_id) is None:
        raise ApprovalError('Invalid original run identity')
    directory = ROOT/'state/controller'
    ticket = json.loads((directory/'tickets'/(args.run_id+'.json')).read_bytes())
    if args.out.exists() or not _local_launcher_absent(args.run_id):
        raise ApprovalError('Existing certificate or live original launcher')
    finished = directory/'local_launch_once'/(args.run_id+'.finished.json')
    if not finished.is_file():
        raise ApprovalError('Original once-launcher has not recorded its exit')
    with QuotaLedger(directory) as ledger:
        row = ledger.get(args.run_id)
        if (row['state'] != 'UNKNOWN' or row['reservation_id'] != ticket['reservation_id']
                or row['request_sha256'] != digest(ticket['request'])):
            raise ApprovalError('Only the exact original UNKNOWN dispatch may be settled')
    transport = PinnedSSH(ticket['approval'], args.out.parent/'prepared-no-start-pin')
    payload = {'ticket': ticket, 'rpc_source': (ROOT/'scripts/controller/remote_rpc.py').read_text(),
               'markers': list(MARKERS)}
    witnesses = [transport.rpc(SOURCE, payload, timeout=120)]
    write_new_json(args.out.with_suffix('.first.json'), witnesses[0])
    time.sleep(30)
    witnesses.append(transport.rpc(SOURCE, payload, timeout=120))
    write_new_json(args.out.with_suffix('.observations.json'), {'witnesses': witnesses})
    evidence = terminal_from_witnesses(ticket, witnesses)
    if not _local_launcher_absent(args.run_id):
        raise ApprovalError('Original launcher reappeared')
    write_new_json(args.out, evidence)
    with QuotaLedger(directory) as ledger:
        if ledger.get(args.run_id)['state'] != 'UNKNOWN':
            raise ApprovalError('Concurrent state change; do not settle again')
        row = ledger.settle(args.run_id, evidence)
    print(json.dumps({'run_id': args.run_id, 'state': row['state'], 'actual_gpu_us': row['actual_gpu_us'],
                      'remote_writes': 0, 'relaunch_same_id': False, 'evidence': str(args.out)}))


if __name__ == '__main__':
    main()
