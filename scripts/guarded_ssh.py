#!/usr/bin/env python3
"""Read-only, deterministic SSH probe wrapper; never write/launch remotely.
Usage: python scripts/guarded_ssh.py --target <verified-user-ssh-alias> --probe identity
Requires normal known_hosts verification; no host-key bypass. Uses only ssh argv (no remote shell interpolation of user input).
"""
import argparse, subprocess, time, sys
PROBES={
 'identity':'id; hostname; pwd; date -u +%FT%TZ',
 'gpu':'nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits',
 'jobs':'tmux ls 2>&1 || true; docker ps --format "{{.ID}} {{.Names}} {{.Status}}" 2>&1 || true',
}
p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--probe',choices=PROBES,default='identity');p.add_argument('--attempts',type=int,default=5);a=p.parse_args()
if a.target.startswith('-') or any(c in a.target for c in ' ;|&\n\t') or not (1<=a.attempts<=10):p.error('Invalid target or attempts')
for n in range(a.attempts):
 cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-o','StrictHostKeyChecking=yes','--',a.target,PROBES[a.probe]]
 r=subprocess.run(cmd,timeout=40)
 if r.returncode==0:sys.exit(0)
 if n+1<a.attempts:
  delay=min(300,15*2**n);print('SSH read-only probe failed, retry in',delay,'seconds',file=sys.stderr);time.sleep(delay)
sys.exit(1)
