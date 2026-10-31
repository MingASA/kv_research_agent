"""Main-agent-only original-ticket reconciliation and bounded read-only mirror.

This operation never launches, restarts, stops or retries a GPU experiment.
Each invocation needs a fresh receipt tag. Inspect any old collector before
resuming its local partial mirror with another connection.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.controller.cli import _ticket,_source,_local_package
from scripts.controller.durable import write_new_json,write_new_bytes,secure_directory,digest
from scripts.controller.owned_result_mirror import mirror_terminal,SOURCE
from scripts.controller.readonly_connection import ReadOnlyConnection
from scripts.observe_nextgen_long_metrics import require_original_ack
from scripts.reuse_verified_mirror_assets import copy_verified

RUNS={
    't':'nextgen-bf16-qwen38-mixed8192-quality1720-r0-20261030t',
    'u':'nextgen-fp8fa2-qwen38-mixed8192-quality1720-r0-20261030u',
    'v':'nextgen-nvfp4-qwen38-mixed8192-quality1720-r0-20261030v',
    'w':'nextgen-bf16-qwen38-mixed8192-prefix32-r0-20261030w',
    'x':'nextgen-fp8fa2-qwen38-mixed8192-prefix32-r0-20261030x',
    'y':'nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030y',
    'wc':'nextgen-bf16-qwen38-mixed8192-prefix32-r1-20261030wc',
    'xc':'nextgen-fp8fa2-qwen38-mixed8192-prefix32-r0-20261030xc',
    'yc':'nextgen-nvfp4-qwen38-mixed8192-prefix32-r0-20261030yc',
    'z':'nextgen-bf16-qwen38-mixed8192-tools32-r0-20261030zc',
    'za':'nextgen-fp8fa2-qwen38-mixed8192-tools32-r0-20261030za',
    'zb':'nextgen-nvfp4-qwen38-mixed8192-tools32-r0-20261030zb',
}


def cached_assets(inventory,ticket_assets,package_assets):
    selected=[]
    for row in inventory['files']:
        name=row['relative'];relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative)!=name:
            raise ValueError('Unsafe fresh inventory path')
        if not relative.parts or relative.parts[0] not in ('input','runtime','runtime_support'):
            continue
        if name not in package_assets:continue
        entry=package_assets[name]
        if (row['sha256']!=ticket_assets.get(name) or row['sha256']!=entry['sha256']
                or type(row['size']) is not int or not 0<=row['size']<=64*2**20):
            raise ValueError('Fresh remote input differs from original frozen ticket/package')
        selected.append((row,Path(entry['local_path'])))
    return selected


def seed_frozen_inputs(transport,ticket,destination,operation,terminal_sha):
    intent=json.loads((ROOT/'state/controller/local_launch_once'/(ticket['request']['run_id']+'.json')).read_text())
    package_path=Path(intent['package'])
    if hashlib.sha256(package_path.read_bytes()).hexdigest()!=intent['package_sha256']:
        raise ValueError('Original once-launch package changed')
    request,hashes,_=_local_package(package_path)
    if digest(request)!=ticket['request_sha256'] or hashes!=ticket['assets_sha256']:
        raise ValueError('Frozen inputs do not belong to the original issued ticket')
    fresh=transport.rpc(SOURCE,{'operation':'inventory','ticket':ticket,'rpc_source':_source('remote_rpc.py')},
                        mutation=False,timeout=90)
    if (fresh.get('ok') is not True or fresh.get('remote_writes') is not False
            or fresh['result_sha256']!=terminal_sha or fresh['run_id']!=request['run_id']):
        raise ValueError('No fresh complete original remote SHA attestation')
    write_new_json(operation/'FRESH_REMOTE_INVENTORY.json',fresh)
    secure_directory(destination);inventory_path=destination/'REMOTE_INVENTORY.json'
    if inventory_path.exists():
        old=json.loads(inventory_path.read_text())
        if (old['run_id']!=fresh['run_id'] or old['result_sha256']!=fresh['result_sha256']
                or {r['relative']:r for r in old['files']}!={r['relative']:r for r in fresh['files']}):
            raise ValueError('Original terminal files changed after prior collection')
    else:write_new_json(inventory_path,fresh)
    package=json.loads(package_path.read_text());copied=[]
    for row,source in cached_assets(fresh,ticket['assets_sha256'],package['assets']):
        target=destination/'files'/row['relative']
        if not target.exists():
            copy_verified(source,target,row['size'],row['sha256']);copied.append(row['relative'])
    receipt={'run_id':request['run_id'],'classification':'fresh_remote_SHA_attested_frozen_input_runtime_reuse',
        'request_sha256':ticket['request_sha256'],'fresh_remote_inventory_sha256':digest(fresh),
        'copied_input_runtime_files':copied,'raw_model_outputs_reused_from_frozen_package':0,
        'original_partial_downloads_preserved':True,'remote_writes':False}
    write_new_json(operation/'FROZEN_INPUT_REUSE.json',receipt)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',choices=RUNS,required=True)
    parser.add_argument('--receipt-tag',required=True)
    parser.add_argument('--reuse-frozen-inputs',action='store_true')
    args=parser.parse_args();run=RUNS[args.run]
    if re.fullmatch(r'[a-z][a-z0-9-]{0,20}',args.receipt_tag) is None:
        raise ValueError('Expected a short fresh receipt tag')
    require_original_ack(run)
    operation=Path(__file__).parent/('mixed-'+args.run+'-terminal-'+args.receipt_tag)
    operation.mkdir(mode=0o700,exist_ok=False)
    write_new_bytes(operation/'COLLECTOR_SOURCE.py',Path(__file__).read_bytes())
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
    destination=ROOT/'results/mirrors'/(run+'-complete-a1')
    if args.reuse_frozen_inputs:
        seed_frozen_inputs(transport,ticket,destination,operation,status['files']['result.json']['sha256'])
    output=mirror_terminal(transport,ticket,_source('remote_rpc.py'),destination)
    manifest=json.loads(output.read_text());receipt={'run_id':run,'terminal':terminal['status'],
        'GPU_release_verified':True,'manifest':str(output),'files':len(manifest['files']),
        'bytes':sum(row['size'] for row in manifest['files']),
        'manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'remote_file_writes':False}
    write_new_json(operation/'COLLECTION_COMPLETE.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':main()
