"""Verify a reviewed zero-GPU dispatch recovery; this never grants admission."""

from __future__ import annotations

from .approval import ApprovalError
from .durable import digest


def verify_absence_certificate(prior: dict, certificate: dict) -> None:
    if (certificate.get('status')!='NOT_LAUNCHED' or certificate.get('gpu_release_verified') is not True
            or certificate.get('live_owned_containers')!=[] or certificate.get('new_output_bytes')!=0
            or certificate.get('allocation_started_monotonic_ns') is not None
            or certificate.get('allocation_ended_monotonic_ns') is not None
            or certificate.get('remote_result_file_created') is not False
            or certificate.get('new_GPU_admission_authorized_by_this_proof') is not False
            or certificate.get('run_id')!=prior['run_id'] or certificate.get('request_sha256')!=digest(prior)):
        raise ApprovalError('Prior dispatch is not an independently verified zero-GPU non-launch')
    values=certificate.get('readonly_witnesses')
    if not isinstance(values,list) or len(values)!=2:raise ApprovalError('Two prior read-only absence witnesses required')
    for value in values:
        if (value.get('schema')!='kvra-absent-dispatch-readonly-witness-v1' or value.get('ok') is not True
                or value.get('remote_writes')!=0 or value.get('root_absent') is not True
                or value.get('run_id')!=prior['run_id'] or value.get('request_sha256')!=digest(prior)
                or value.get('reservation_id')!=certificate.get('reservation_id')
                or any(value.get(name)!=[] for name in ('remote_rpc_processes','run_processes','all_run_containers'))):
            raise ApprovalError('An old RPC/process/container or unbound absence witness remains')
    first,last=values
    if (not first.get('boot_id') or first['boot_id']!=last.get('boot_id')
            or type(first.get('monotonic_ns')) is not int or type(last.get('monotonic_ns')) is not int
            or last['monotonic_ns']-first['monotonic_ns']<30_000_000_000):
        raise ApprovalError('Prior root absence was not stable on one boot for30seconds')
