"""Freeze C's isolated CPU build from successful A objects, without dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from scripts.controller.upstream_qualification import BASELINE_COMMIT, OFFICIAL_COMMIT

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path('/home/mingasa/projects/xllm-upstream-native-kv-20261102a')


def prepare(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_bytes())
    result = metadata['build_result']
    if (result.get('success') is not True or result.get('CPU_tests_executed') is not True
            or result.get('GPU_job') is not False or metadata['source_commit'] != BASELINE_COMMIT
            or metadata['container_exited_successfully'] is not True or metadata['source_status']):
        raise ValueError('Native build requires the successful, clean, terminal A artifact receipt')
    native_sha = subprocess.check_output(['git', '-C', str(NATIVE), 'rev-parse', 'HEAD'], text=True).strip()
    clean = subprocess.check_output(['git', '-C', str(NATIVE), 'status', '--porcelain'], text=True)
    if clean.strip():
        raise ValueError('Commit and review the native source before freezing the build')
    subprocess.run(['git', '-C', str(NATIVE), 'merge-base', '--is-ancestor', BASELINE_COMMIT, native_sha], check=True)
    build_id = 'upstream-ac-' + native_sha[:8] + '-20261102f'
    directory = ROOT / 'plans/builds' / build_id
    directory.mkdir()
    old = json.loads((ROOT / 'plans/builds' / metadata['build_id'] / 'manifest.json').read_bytes())
    manifest = {k: v for k, v in old.items() if k not in ('incremental_reuse', 'repair_of', 'repair_reason',
                'last_object_writer_reconciled', 'official_A_has_no_KV_extensions', 'official_A_source_unmodified')}
    manifest.update(build_id=build_id, host_root=str(Path(old['host_root']).with_name(build_id)),
                    source_commit=native_sha, upstream_commit=OFFICIAL_COMMIT, native_KV_build=True,
                    object_build_id=build_id,
                    baseline_source=metadata['source_repo'], bundle_prerequisite_commit=BASELINE_COMMIT,
                    code_scope='same frozen upstream and common CUDA build fix, selectively integrated native KV C',
                    build_snapshot={'build_id': metadata['build_id'],
                        'object_build_id': old.get('object_build_id', old.get('incremental_reuse', {}).get('build_id')),
                        'virtual_build_id': old.get('build_snapshot', {}).get('virtual_build_id', 'upstream-ac-261274e5-20261102b'),
                        'source_commit': BASELINE_COMMIT, 'binary_sha256': result['binary_sha256'],
                        'immutable_files_sha256': metadata['immutable_files_sha256']},
                    source_generated_write_subdirs=['third_party/brpc/src/butil'])
    for name in ('build_driver.py', 'build_rpc.py'):
        shutil.copyfile(ROOT / 'scripts' / ('upstream_cpu_' + name), directory / name)
    subprocess.run(['git', '-C', str(NATIVE), 'bundle', 'create', str(directory / 'source.bundle'),
                    'HEAD', '^' + BASELINE_COMMIT], check=True)
    manifest['assets_sha256'] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                for name in ('source.bundle', 'build_driver.py')}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'build_id': build_id, 'manifest': str(directory / 'manifest.json'), 'source_commit': native_sha,
            'A_baseline_artifact_unchanged': True, 'GPU_requested': False, 'dispatched': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.metadata)))


if __name__ == '__main__':
    main()
