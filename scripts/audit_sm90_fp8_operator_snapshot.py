"""Independently recompute stored H800 decode arrays on CPU, without a GPU run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from scripts.controller.durable import write_new_json


def relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float((actual-reference).norm()/reference.norm().clamp_min(1e-12))


def audit(manifest_path: Path, ticket_path: Path, source: Path) -> dict:
    manifest=json.loads(manifest_path.read_text());ticket=json.loads(ticket_path.read_text())
    root=Path(manifest['mirror_path']);request=ticket['request'];run_id=request['run_id']
    if manifest['run_id']!=run_id:raise ValueError('Partial snapshot belongs to another run')
    files={row['path']:row for row in manifest['files']}
    for name,row in files.items():
        path=root/name
        if path.resolve(strict=True)!=path.absolute() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Partial snapshot hash/path changed')
    name='tools/sm90_xqa_fp8_reference.py';reference_source=source/name
    if hashlib.sha256(reference_source.read_bytes()).hexdigest()!=request['backend_source_sha256'][name]:
        raise ValueError('CPU reference differs from the deployed source')
    spec=importlib.util.spec_from_file_location('snapshot_cpu_reference',reference_source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    output=root/'output/fp8_e4m3/native_pages'
    metadata=json.loads((output/'fp8_e4m3-decode.json').read_text())
    if (metadata['writer_in_actual_decode_Graph'] is not True or not metadata['changed_decode_context_pages_slots']
            or metadata['format']!='fp8_e4m3' or metadata['head_dim']!=256):
        raise ValueError('Wrong operator or missing actual changed-input decode Graph proof')
    rows=[]
    for index,observation in enumerate(metadata['observations']):
        evidence=observation['CPU_reference_evidence'];path=output/evidence['file']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=evidence['sha256']:
            raise ValueError('Raw NPZ differs from the GPU observation')
        with np.load(path,allow_pickle=False) as archive:
            values={name:torch.from_numpy(archive[name]) for name in archive.files}
        for kind in ('key','value'):
            raw=values['resident_'+kind+'_bytes'];scales=values['resident_'+kind+'_scales']
            decoded=(raw.view(torch.float8_e4m3fn).float()*scales.unsqueeze(-1)).reshape(values['decoded_'+kind].shape)
            if not torch.equal(decoded,values['decoded_'+kind]):raise ValueError('Decoded reference does not match actual resident FP8 bytes/scales')
        computed=module.reference_decode(values['query'],values['decoded_key'],values['decoded_value'],
                                         split_count=observation['XQA_split_count'])
        backend_difference=relative(computed,values['backend_compute_reference'])
        error=relative(values['actual_output'],computed)
        old_error=relative(values['actual_output'],values['original_BF16_query_reference'])
        approximation=relative(computed,values['original_BF16_query_reference'])
        if not torch.isfinite(computed).all() or backend_difference>1e-5 or error>.02:
            raise ValueError('Independent CPU compute-contract correlation failed')
        if abs(error-observation['operator_relative_L2'])>1e-6 or abs(old_error-observation['actual_relative_L2_vs_original_BF16_query_decoded_KV'])>1e-6:
            raise ValueError('Worker error summary differs from actual stored GPU output arrays')
        rows.append({'replay':index,'context':values['decoded_key'].shape[1],
            'npz_sha256':evidence['sha256'],'resident_FP8_bytes_and_scales_decode_exact':True,
            'CPU_recompute_vs_stored_backend_reference_L2':backend_difference,
            'actual_GPU_output_vs_recomputed_Q_P_reference_L2':error,
            'actual_GPU_output_vs_original_BF16_Q_reference_L2':old_error,
            'additional_backend_Q_P_approximation_vs_BF16_Q_L2':approximation,
            'operator_tolerance_unchanged':.02,'operator_qualified':True})
    return {'state':'INDEPENDENT_CPU_RECOMPUTE_OF_ACTUAL_H800_OPERATOR_ARRAYS_VERIFIED','run_id':run_id,
        'source_commit':request['python_source_commit'],'partial_snapshot_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'partial_snapshot_files_reverified':len(files),'rows':rows,'GPU_experiment_repeated':False,
        'new_GPU_used_for_this_audit':False,'model_quality_or_service_performance_verified':False,
        'original_run_terminal':'UNKNOWN_UNCHANGED','scope':'Two actual SM90/D256/GQA6 decode Graph replays; '
        'CPU reference models documented FP8 Q/P arithmetic, not a bitwise GMMA emulator or full model quality.'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--ticket',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(2)
    write_new_json(args.out,audit(args.manifest,args.ticket,args.source))


if __name__=='__main__':main()
