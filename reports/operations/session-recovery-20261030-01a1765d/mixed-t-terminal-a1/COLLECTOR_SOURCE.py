"""Main-agent-only original-ticket reconciliation and bounded read-only mirror.

This operation never launches, restarts, stops or retries a GPU experiment.
Each invocation needs a fresh receipt tag. Inspect any old collector before
resuming its local partial mirror with another connection.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.controller.cli import _ticket,_source
from scripts.controller.durable import write_new_json
from scripts.controller.owned_result_mirror import mirror_terminal
from scripts.controller.readonly_connection import ReadOnlyConnection
from scripts.observe_nextgen_long_metrics import require_original_ack

RUNS={
    't':'nextgen-bf16-qwen38-mixed8192-quality1720-r0-20261030t',
    'u':'nextgen-fp8fa2-qwen38-mixed8192-quality1720-r0-20261030u',
    'v':'nextgen-nvfp4-qwen38-mixed8192-quality1720-r0-20261030v',
    'w':'nextgen-bf16-qwen38-mixed8192-prefix32-r0-20261030w',
    'x':'nextgen-fp8fa2-qwen38-mixed8192-prefix32-r0-20261030x',
    'y':'nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030y',
    'z':'nextgen-bf16-qwen38-mixed8192-tools32-r0-20261030z',
    'za':'nextgen-fp8fa2-qwen38-mixed8192-tools32-r0-20261030za',
    'zb':'nextgen-nvfp4-qwen38-mixed8192-tools32-r0-20261030zb',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',choices=RUNS,required=True)
    parser.add_argument('--receipt-tag',required=True)
    args=parser.parse_args();run=RUNS[args.run]
    if re.fullmatch(r'[a-z][a-z0-9-]{0,20}',args.receipt_tag) is None:
        raise ValueError('Expected a short fresh receipt tag')
    require_original_ack(run)
    operation=Path(__file__).parent/('mixed-'+args.run+'-terminal-'+args.receipt_tag)
    operation.mkdir(mode=0o700,exist_ok=False)
    write_new_json(operation/'LOCAL_INTENT.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'run_id':run,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'remote_file_writes':False,'GPU_start_stop_or_retry':False})
    with (operation/'reconcile.stdout.json').open('x') as out,(operation/'reconcile.stderr.log').open('x') as err:
        result=subprocess.run([sys.executable,'-B','-m','scripts.controller.cli','reconcile','--run-id',run],
                              stdout=out,stderr=err,cwd=ROOT)
    if result.returncode:raise RuntimeError('Original reconcile failed; inspect its exclusive receipt')
    status=json.loads((operation/'reconcile.stdout.json').read_text())
    terminal=status.get('files',{}).get('result.json',{}).get('value',{})
    if (terminal.get('status') not in ('COMPLETE','FAILED','TIMED_OUT','CONFLICT','DISK_LIMIT','NOT_LAUNCHED')
            or terminal.get('gpu_release_verified') is not True
            or terminal.get('live_owned_containers')!=[] or status.get('live_owned_containers')!=[]):
        raise ValueError('Original run is not terminal and freshly verified released; no mirror')
    ticket=_ticket(run)
    transport=ReadOnlyConnection(ticket['approval'],Path('/tmp')/('kv-mixed-'+args.run+'-'+args.receipt_tag))
    ready=transport.start();write_new_json(operation/'READONLY_MASTER.json',ready)
    if ready.get('ready') is not True:raise RuntimeError('New read-only master not ready; no dependent collection')
    output=mirror_terminal(transport,ticket,_source('remote_rpc.py'),ROOT/'results/mirrors'/(run+'-complete-a1'))
    manifest=json.loads(output.read_text());receipt={'run_id':run,'terminal':terminal['status'],
        'GPU_release_verified':True,'manifest':str(output),'files':len(manifest['files']),
        'bytes':sum(row['size'] for row in manifest['files']),
        'manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'remote_file_writes':False}
    write_new_json(operation/'COLLECTION_COMPLETE.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
