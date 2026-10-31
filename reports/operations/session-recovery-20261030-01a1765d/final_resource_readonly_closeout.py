"""Main-agent final read-only proof, only after every original run is settled.

No GPU dispatch, stop, retry, remote write or ledger mutation. Local evidence
uses a fresh exclusive directory. This does not stop the separate Luna monitor.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.controller.cli import _source, _ticket
from scripts.controller.durable import write_new_json, write_new_bytes
from scripts.controller.transport import PinnedSSH

LAST = 'nextgen-nvfp4-qwen38-mixed8192-tools32-r0-20261030zb'
MODEL_REPO = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-nextgen-native-kv-20261027')


def main(output):
    connection = sqlite3.connect('file:' + str(ROOT/'state/controller/ledger.sqlite3') + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute(
        'select run_id,state,reservation_id,actual_gpu_us,quota_breach,evidence_json from runs order by run_id')]
    assert rows and all(row['state'] == 'SETTLED' and row['quota_breach'] == 0 for row in rows)
    assert sum(row['run_id'] == LAST for row in rows) == 1
    history = {row[0] for row in connection.execute('select run_id from history')}
    connection.close()
    known = {row['run_id'] for row in rows} | history
    terminal_counts = {}
    for row in rows:
        terminal = json.loads(row.pop('evidence_json'))
        assert terminal['run_id'] == row['run_id']
        assert terminal['status'] in ('COMPLETE','FAILED','TIMED_OUT','CONFLICT','DISK_LIMIT','NOT_LAUNCHED')
        assert terminal['gpu_release_verified'] is True and terminal['live_owned_containers'] == []
        row['original_terminal'] = terminal['status']
        terminal_counts[terminal['status']] = terminal_counts.get(terminal['status'], 0) + 1
    actual_us = sum(row['actual_gpu_us'] for row in rows)
    assert actual_us <= 24 * 3600000000
    ticket = _ticket(LAST)
    approval_path = Path('/home/mingasa/.config/kv-research/approval.json')
    assert hashlib.sha256(approval_path.read_bytes()).hexdigest() == ticket['approval_sha256']
    request = ticket['request']
    expected = {request['server_binary']: request['binary_sha256']}
    for name, sha in request['backend_source_sha256'].items():
        repo = request['python_source_repo'] if name.endswith('.py') else request['source_repo']
        expected[repo + '/' + name] = sha
    config = json.loads((ROOT/'state/NEXTGEN_CONFIG.local.json').read_text())
    config.update(image_id=request['image_id'], hash_paths=sorted(expected))
    source = _source('discovery.py').split('if __name__ == "__main__":')[0]
    source += '''\nvalue=collect(json.load(sys.stdin))
value['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
value['GPU_platform']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
value['kernel_release']=os.uname().release
print(json.dumps({'ok':True,**value},sort_keys=True))
'''
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_new_bytes(output/'LOCAL_SOURCE.py', Path(__file__).read_bytes())
    write_new_bytes(output/'REMOTE_READONLY_SOURCE.py', source.encode())
    transport = PinnedSSH(ticket['approval'], output/'pin')
    remote = transport.rpc(source, config, timeout=120)
    assert remote.get('ok') is True and remote['remote_writes'] == 0
    write_new_json(output/'REMOTE_DISCOVERY.json', remote)
    assert remote['source_sha256'] == expected
    assert remote['model_config_sha256'] == request['model_config_sha256']
    assert remote['boot_id'] == '5afaf1c0-ecf6-4bda-a039-5341ef954efb'
    assert all(row['name'] in known and row['uid'] == config['remote_identity']['uid']
               and row['symlink'] is False for row in remote['run_inventory'])
    containers = [json.loads(line) for line in remote['owned_containers'].splitlines() if line.strip()]
    running = [row for row in containers if row.get('State') == 'running' or row.get('Status','').startswith('Up ')]
    assert running == [], 'A labelled container remains live; inspect ownership before closeout'
    status = transport.rpc(_source('remote_rpc.py'), {'operation':'status','ticket':ticket}, timeout=90)
    write_new_json(output/'LAST_ORIGINAL_STATUS.json', status)
    assert status.get('ok') is True and status['supervisor_alive'] is False
    assert status['live_owned_containers'] == []
    terminal = status['files']['result.json']['value']
    assert terminal['status'] in ('COMPLETE','FAILED') and terminal['gpu_release_verified'] is True
    assert terminal['live_owned_containers'] == []
    head = subprocess.check_output(['git','-C',str(MODEL_REPO),'rev-parse','HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(MODEL_REPO),'status','--porcelain'], text=True)
    assert head == request['python_source_commit'] == '03134b72ec904254890300a647c939095f7900e2'
    assert not dirty
    receipt = {'schema':'nextgen-final-original-run-resource-closeout-v1',
        'utc':datetime.now(timezone.utc).isoformat(), 'classification':'confirmed',
        'remote_readonly':True,'remote_writes':0,'GPU_start_stop_or_retry':False,
        'runs':rows,'settled_run_count':len(rows),'terminal_counts':terminal_counts,
        'all_runs_SETTLED':True,'current_reserved_gpu_us':0,'UNKNOWN_runs':0,
        'cumulative_actual_gpu_us':actual_us,'cumulative_actual_gpu_hours':actual_us/3600000000,
        'same_fixed_budget_not_reset':True,'approval_sha256':ticket['approval_sha256'],
        'private_approval_content_included':False,'remote_run_inventory_count':len(remote['run_inventory']),
        'unknown_or_unowned_run_subtrees':[], 'running_labelled_containers':[],
        'last_original_status':terminal['status'],'last_original_GPU_release_verified':True,
        'fresh_gpu_snapshot':remote['gpus'],'fresh_compute_apps':remote['compute_apps'],
        'GPU_platform':remote['GPU_platform'],'remote_kernel_release':remote['kernel_release'],
        'all_remote_source_and_binary_SHA_match':True,'model_worktree_clean':True,
        'model_source_commit':head,'Luna_monitor_stopped_by_this_script':False}
    write_new_json(output/'CLOSEOUT_ACCEPTANCE.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('runs','fresh_gpu_snapshot','fresh_compute_apps')}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out)
