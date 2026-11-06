"""Offline SHA/lifecycle audit for the frozen upstream campaign; no model or SSH."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.controller.durable import digest


def audit(manifest_path: Path):
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get('layout') != 'run_relative':
        raise ValueError('Require the complete run-relative upstream mirror')
    root = Path(manifest['mirror_directory'])/'files'
    root = root.resolve(strict=True)
    covered = set()
    for item in manifest['files']:
        relative = Path(item['base_rel'])
        if relative.is_absolute() or '..' in relative.parts or str(relative) in covered:
            raise ValueError('Unsafe/duplicate mirror member')
        path = root/relative
        if path.resolve(strict=True) != path or not path.is_file() or path.stat().st_size != item['size']:
            raise ValueError('Mirror member path/type/size changed')
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for part in iter(lambda:stream.read(4*2**20),b''):
                hasher.update(part)
        if hasher.hexdigest() != item['remote_sha256'] or item['snapshot_sha256'] != item['remote_sha256']:
            raise ValueError('Mirror SHA mismatch: '+str(relative))
        covered.add(str(relative))
    ticket = json.loads((root/'ticket.json').read_bytes())
    request = ticket['request']
    result_raw = (root/'result.json').read_bytes()
    result = json.loads(result_raw)
    if (not request['workload']['kind'].startswith('upstream_')
            or request['run_id'] != result['run_id'] or request['run_id'] != manifest['run_id']
            or digest(request) != ticket['request_sha256'] or result['request_sha256'] != ticket['request_sha256']
            or hashlib.sha256(result_raw).hexdigest() != manifest['remote_result_sha256']
            or result['status'] != 'COMPLETE' or result['gpu_release_verified'] is not True
            or result['live_owned_containers'] != []):
        raise ValueError('Original upstream run was not completed and released with the bound request')
    for name, expected in ticket['assets_sha256'].items():
        if name not in covered or hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source/input/runtime asset missing or changed: '+name)
    if len(result['arm_results']) != 1 or request['arms'] != [result['arm_results'][0]['arm']]:
        raise ValueError('Upstream comparison must retain the actual single-format run')
    arm = request['arms'][0]
    measurement_path = 'output/'+arm+'/measurement.json'
    measurement = json.loads((root/measurement_path).read_bytes())
    if (digest(measurement) != result['arm_results'][0]['result_sha256']
            or measurement.get('measurement_success') is not True
            or measurement.get('request_sha256') != ticket['request_sha256']):
        raise ValueError('Worker result differs from the supervisor-bound measurement')
    for name, expected in measurement['evidence_sha256'].items():
        relative = 'output/'+arm+'/'+name
        if relative not in covered or hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Measured raw evidence missing or changed: '+relative)
    classification=measurement.get('classification')
    if classification is None:
        if (request['workload']['kind']!='upstream_FP8_independent_numeric_calibration'
                or measurement.get('schema')!='FP8_independent_same_base_BF16_calibration_v1'):
            raise ValueError('Unrecognized measurement classification')
        classification='independent_BF16_numeric_observer_untimed_no_quality_acceptance'
    return {'schema':'upstream-complete-raw-lifecycle-audit-v1','run_id':request['run_id'],
            'kind':request['workload']['kind'],'arm':arm,'source_commit':request['server_source_commit'],
            'binary_sha256':request['binary_sha256'],'all_mirror_SHA_verified':True,
            'all_frozen_assets_and_worker_evidence_verified':True,'files':len(covered),
            'GPU_release_verified':True,'original_status':'COMPLETE',
            'classification':classification,
            'all_operator_candidates_pass':measurement.get('all_candidates_pass'),
            'quality_or_performance_acceptance_inferred':False}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha-manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.sha_manifest)
    args.out.open('x').write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
