"""An occupied GPU may be reconciled only with positive foreign PID proof."""

import copy

import pytest

from scripts.controller.approval import ApprovalError
from scripts.reconcile_conflict_release import terminal_from_witnesses


def _fixture():
    own, foreign = 'a'*64, 'b'*64
    ticket = {'request':{'run_id':'conflict-fixture','gpu_uuids':['GPU-test']},
              'reservation_id':'reserved','request_sha256':'c'*64}
    result = {'run_id':'conflict-fixture','reservation_id':'reserved','request_sha256':'c'*64,
              'status':'UNKNOWN','gpu_release_verified':False,'arm_results':[],
              'error':'UnsafeState: Foreign GPU activity/reservation; stop this run only',
              'live_owned_containers':[own],'allocation_started_monotonic_ns':100,
              'allocation_ended_monotonic_ns':None,
              'allocation_intervals_ns':[{'arm':'mxfp4','start_ns':100,'end_ns':None}],
              'new_output_bytes':1024}
    witness = {'ok':True,'schema':'kvra-unknown-release-witness-v1','remote_writes':False,
               'run_id':'conflict-fixture','reservation_id':'reserved','request_sha256':'c'*64,
               'boot_id':'boot','allocation_boot_id':'boot','result':result,'result_sha256':'d'*64,
               'supervisor_live':False,'run_workers':[],
               'containers':[{'id':own,'running':False,'pid':0,'status':'exited'}],
               'selected':{'GPU-test':{'memory_mib':45000,'util_pct':99,'pids':[99]}},
               'driver_identities':[{'GPU':'GPU-test','identity':{'pid':99,'uid':0,'start_ticks':1000,
                                     'boot_id':'boot','cgroup':'0::/docker-'+foreign+'.scope\n'}}],
               'monotonic_ns':1000,'new_output_bytes':2048}
    later = copy.deepcopy(witness);later['monotonic_ns'] += 30_000_000_000
    return ticket, [witness,later]


def test_conflict_own_release_does_not_claim_gpu_idle_or_admission():
    ticket, witnesses = _fixture()
    result = terminal_from_witnesses(ticket, witnesses)
    assert result['status'] == 'CONFLICT'
    assert result['gpu_release_verified'] is True
    assert result['GPU_is_idle'] is False
    assert result['new_GPU_admission_authorized'] is False
    assert result['original_result_overwritten'] is False
    assert result['allocation_ended_monotonic_ns'] == witnesses[-1]['monotonic_ns']


@pytest.mark.parametrize('failure',['ours','no_cgroup','missing_pid','live_container','changed_identity','short_interval'])
def test_ambiguous_or_unreleased_allocation_stays_unknown(failure):
    ticket, witnesses = _fixture()
    first, last = witnesses
    if failure == 'ours':
        for w in witnesses:w['driver_identities'][0]['identity']['cgroup']='0::/docker-'+'a'*64+'.scope\n'
    elif failure == 'no_cgroup':
        for w in witnesses:w['driver_identities'][0]['identity']['cgroup']='0::/\n'
    elif failure == 'missing_pid':
        last['driver_identities']=[]
    elif failure == 'live_container':
        last['containers'][0].update(running=True,pid=45,status='running')
    elif failure == 'changed_identity':
        last['driver_identities'][0]['identity']['start_ticks'] += 1
    else:
        last['monotonic_ns'] -= 1
    with pytest.raises(ApprovalError):
        terminal_from_witnesses(ticket, witnesses)
