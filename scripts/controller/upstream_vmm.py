"""Finite untimed VMM compatibility checks, separate from formal A/C service."""

import copy
from pathlib import Path

from .durable import write_new_json

PROFILES = (
    {'name': 'bf16-vmm', 'arm': 'bf16', 'native_configuration': None},
    {'name': 'fp8-vmm', 'arm': 'fp8_e4m3', 'native_configuration':
        {'dtype': 'fp8_e4m3', 'key_scale': .125, 'value_scale': .0625, 'prefill': 'fa3', 'decode': 'flashinfer'}},
    {'name': 'nvfp4-vmm', 'arm': 'nvfp4', 'native_configuration':
        {'dtype': 'nvfp4', 'key_scale': 1., 'value_scale': 1., 'prefill': 'fa2', 'decode': 'flashinfer'}},
)


def run(ticket: dict, root: Path, output: Path, arm: str, env: dict):
    from .upstream_qualification import run as qualify
    result = {}
    for profile in PROFILES:
        selected = copy.deepcopy(ticket)
        selected['request']['native_configuration'] = profile['native_configuration']
        selected['request']['cache_allocator'] = 'elastic_vmm'
        directory = output/profile['name']
        directory.mkdir()
        write_new_json(directory/'explicit_profile.json', profile)
        result[profile['name']] = qualify(selected, root, directory, profile['arm'], env)
    summary = {'classification': 'bounded_three_profile_real_model_VMM_compatibility_not_performance',
        'profiles': result, 'elastic_VMM_explicitly_enabled': True, 'default_A_C_service_changed': False,
        'physical_residency_inferred_from_virtual_tensor_numel': False,
        'suspend_resume_or_multimodel_reclaim_qualification': False,
        'performance_claim': False, 'quality_benchmark_claim': False}
    write_new_json(output/'VMM_compatibility_result.json', summary)
    return summary
