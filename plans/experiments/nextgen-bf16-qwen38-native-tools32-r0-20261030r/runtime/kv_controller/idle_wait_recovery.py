"""Explicit fresh attempt after a completed zero-allocation GPU-idle wait.

The prior run/result remains unchanged. This cannot recover numerical/model
failures or partial generations; it admits at most two operational attempts of
the exact same deployment after an original NOT_LAUNCHED idle deadline.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .durable import digest
from .processes import process_identity,process_lifecycle


REASON='idle_GPU_deadline_without_allocation'
ERROR='TimeoutError: Budget expired while waiting for 120s genuinely idle GPUs'
_IDENTITY_FIELDS={'run_id','remote_output','python_source_repo','dataset_path','replicate',
                  'idle_wait_recovery','transport_recovery','resource_recovery','orchestration_recovery'}


def verify_zero_allocation_wait(prior: dict,result: dict) -> None:
    if (result.get('status')!='NOT_LAUNCHED' or result.get('error')!=ERROR
            or result.get('run_id')!=prior['run_id'] or result.get('request_sha256')!=digest(prior)
            or result.get('gpu_release_verified') is not True or result.get('live_owned_containers')!=[]
            or result.get('allocation_started_monotonic_ns') is not None
            or result.get('allocation_ended_monotonic_ns') is not None
            or result.get('allocation_intervals_ns')!=[] or result.get('arm_results')!=[]):
        raise ValueError('Prior run is not an exact completed zero-GPU idle wait')


def verify_same_deployment(prior: dict,current: dict) -> None:
    if (type(current.get('replicate')) is not int or type(prior.get('replicate')) is not int
            or current['run_id']==prior['run_id'] or current.get('replicate')!=prior.get('replicate',-1)+1
            or not 1<=current['replicate']<=2
            or {k:v for k,v in prior.items() if k not in _IDENTITY_FIELDS}
               !={k:v for k,v in current.items() if k not in _IDENTITY_FIELDS}):
        raise ValueError('Idle-wait recovery changed actual deployment/data or explicit bounded attempt')


def verify_idle_wait_recovery(request: dict,base: Path,uid: int) -> None:
    recovery=request.get('idle_wait_recovery')
    if recovery is None:return
    root=base/'agent_work/runs'/recovery['prior_run_id']
    def read(path: Path) -> tuple[dict,str]:
        if (path.resolve(strict=True)!=path or path.stat().st_uid!=uid or not path.is_file()
                or path.stat().st_size>2**20):raise ValueError('Unsafe prior zero-GPU wait prerequisite')
        raw=path.read_bytes();return json.loads(raw),hashlib.sha256(raw).hexdigest()
    ticket,_=read(root/'ticket.json');prior=ticket['request'];result,sha=read(root/'result.json')
    if (sha!=recovery['prior_result_sha256'] or digest(prior)!=recovery['prior_request_sha256']
            or digest(prior)!=ticket['request_sha256'] or prior['run_id']!=recovery['prior_run_id']
            or prior['remote_output']!=str(root) or result['reservation_id']!=ticket['reservation_id']
            or (root/'COMPLETE').exists()):raise ValueError('Prior idle-wait identity/result changed')
    verify_zero_allocation_wait(prior,result);verify_same_deployment(prior,request)
    finished,_=read(root/'WATCHDOG_FINISHED.json')
    if (finished.get('request_sha256')!=digest(prior) or finished.get('gpu_release_verified') is not True
            or finished.get('result_sha256')!=sha):raise ValueError('Original wait lacks bound terminal watchdog evidence')
    started,_=read(root/'SUPERVISOR_STARTED.json');identity=started['identity']
    if identity['uid']!=uid or identity['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('Prior supervisor identity/host boot differs')
    try:
        current=process_identity(identity['pid'])
        if current==identity and process_lifecycle(identity['pid'])['state']!='Z':
            raise ValueError('Original idle-wait supervisor is still alive')
    except FileNotFoundError:pass
    registry=root/'owned_containers.json'
    if registry.exists() and read(registry)[0].get('containers')!=[]:
        raise ValueError('A zero-allocation wait cannot have registered any owned container')
    events=root/'events.jsonl'
    if events.resolve(strict=True)!=events or events.stat().st_uid!=uid or events.stat().st_size>2**20:
        raise ValueError('Unsafe original idle-wait event journal')
    rows=[json.loads(line) for line in events.read_bytes().split(b'\n') if line]
    if not any(row.get('action')=='WAIT_GPU' for row in rows) or any(
            row.get('action') in ('OWNED_CONTAINER_CREATED','ARM_VERIFIED','GPU_CONFLICT_OBSERVATION') for row in rows):
        raise ValueError('Prior events contradict an unallocated idle wait')
    output=root/'output'
    if output.exists():
        pending=[output];entries=0
        while pending:
            directory=pending.pop()
            if directory.resolve(strict=True)!=directory or directory.stat().st_uid!=uid:
                raise ValueError('Unsafe output directory in a zero-allocation wait')
            with os.scandir(directory) as children:
                for child in children:
                    entries+=1
                    if entries>64 or child.is_symlink() or not child.is_dir(follow_symlinks=False):
                        raise ValueError('No worker/model/prediction output may exist for a zero-allocation wait')
                    pending.append(Path(child.path))
