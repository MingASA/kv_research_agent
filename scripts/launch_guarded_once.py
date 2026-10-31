"""One local launch intent, exclusive receipts and PID recovery per run ID.

This wraps the existing guarded CLI; it adds no GPU authority or remote action.
Any existing intent/ticket/ledger/history or output file refuses a second local
entry before Popen, without truncating the first dispatch's audit evidence.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from scripts.controller.durable import secure_directory,write_new_json
from scripts.controller.quota import QuotaLedger
from scripts.guarded_native_queue import _process_identity


ROOT=Path(__file__).resolve().parents[1]


def launch(package: Path, receipt: Path, errors: Path) -> dict:
    document=json.loads(package.read_text());run=document['request']['run_id']
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}',run) is None:raise ValueError('Invalid run ID')
    directory=ROOT/'state/controller';intents=directory/'local_launch_once';secure_directory(intents)
    lock=os.open(intents/(run+'.lock'),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent=intents/(run+'.json')
        with QuotaLedger(directory) as ledger:
            known=ledger.connection.execute('SELECT 1 FROM runs WHERE run_id=?',(run,)).fetchone()
            archived=ledger.connection.execute('SELECT 1 FROM history WHERE run_id=?',(run,)).fetchone()
        if intent.exists() or known or archived or (directory/'tickets'/(run+'.json')).exists():
            raise ValueError('Original run/intent already exists; reconcile only, never launch again')
        for path in (receipt,errors):
            secure_directory(path.parent)
            if path.exists() or path.is_symlink():raise FileExistsError('Original receipt must never be overwritten: '+str(path))
        argv=[sys.executable,'-B','-m','scripts.controller.cli','launch','--package',str(package.resolve())]
        write_new_json(intent,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run_id':run,
            'package':str(package.resolve()),'package_sha256':hashlib.sha256(package.read_bytes()).hexdigest(),
            'argv':argv,'receipt':str(receipt.resolve()),'errors':str(errors.resolve()),'do_not_repeat':True})
        with receipt.open('x') as stdout,errors.open('x') as stderr:
            child=subprocess.Popen(argv,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
            identity=_process_identity(child.pid);identity['argv']=argv
            write_new_json(intents/(run+'.process.json'),{'identity':identity,'operation':'one_original_CLI_child'})
            code=child.wait(timeout=2100)
        result={'run_id':run,'local_CLI_exit_code':code,'receipt':str(receipt),'GPU_success_claim':False}
        write_new_json(intents/(run+'.finished.json'),result)
        return result
    finally:os.close(lock)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--stderr',type=Path,required=True)
    args=parser.parse_args();result=launch(args.package,args.receipt,args.stderr);print(json.dumps(result))
    if result['local_CLI_exit_code']:raise SystemExit(result['local_CLI_exit_code'])


if __name__=='__main__':main()
