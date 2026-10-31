"""Create and independently verify a local evidence archive from explicit inputs.

No remote access. Only files named by run SHA manifests and explicit local
includes are archived. Model weights, installed environments and shared caches
are not needed for raw-evidence archival and are not implicitly collected.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from scripts.audit_nextgen_validation import _check,_mirror_paths
from scripts.controller.durable import write_new_json

ROOT=Path(__file__).resolve().parents[1]


def archive(spec_path: Path,output: Path) -> None:
    spec=json.loads(spec_path.read_text());files={};runs=[]

    def include(path: Path,*,expected_sha: str | None=None,expected_size: int | None=None) -> None:
        _check(path.is_file() and not path.is_symlink(),'Archive inputs must be existing regular files')
        relative=path.resolve(strict=True).relative_to(ROOT).as_posix()
        data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
        _check((expected_sha is None or sha==expected_sha) and
               (expected_size is None or len(data)==expected_size),'Archive source SHA or size changed')
        item={'source':str(path.resolve()),'size':len(data),'sha256':sha}
        _check(relative not in files or files[relative]==item,'Conflicting archive member identity')
        files[relative]=item

    for name in spec['sha_manifests']:
        path=ROOT/name;manifest=json.loads(path.read_text());base,_=_mirror_paths(manifest)
        _check(manifest['run_id'] not in runs,'Duplicate original run in archive specification')
        for row in manifest['files']:
            _check(row['remote_sha256']==row['snapshot_sha256'],'Unverified mirror SHA identity')
            include(base/row['base_rel'],expected_sha=row['remote_sha256'],expected_size=row['size'])
        include(path);runs.append(manifest['run_id'])
    for name in spec['include']:
        path=ROOT/name;path.resolve(strict=True).relative_to(ROOT)
        if path.is_dir():
            for child in sorted(path.rglob('*')):
                if child.is_file():include(child)
        else:include(path)
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    write_new_json(output/'ARCHIVE_INTENT.json',{'local_only':True,
        'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),'original_runs':runs})
    catalog={'schema':'nextgen-local-evidence-file-manifest-v1','original_runs':runs,'files':files,
        'scope':'Raw records and explicit analysis/source artifacts; not model weights or a runnable environment.',
        'relocation':'Original manifests retain original absolute paths as provenance; verification of archive members is path independent.'}
    catalog_path=output/'ARCHIVE_FILE_MANIFEST.json';write_new_json(catalog_path,catalog)
    catalog_data=catalog_path.read_bytes();catalog_sha=hashlib.sha256(catalog_data).hexdigest()
    expected={name:{'size':item['size'],'sha256':item['sha256']} for name,item in files.items()}
    expected['ARCHIVE_FILE_MANIFEST.json']={'size':len(catalog_data),'sha256':catalog_sha}
    archive_path=output/'evidence.tar.gz'
    with archive_path.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as compressed:
            with tarfile.open(mode='w',fileobj=compressed,format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(expected):
                    data=catalog_data if name=='ARCHIVE_FILE_MANIFEST.json' else Path(files[name]['source']).read_bytes()
                    _check(len(data)==expected[name]['size'] and hashlib.sha256(data).hexdigest()==expected[name]['sha256'],
                           'Source changed while constructing archive')
                    info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;info.mtime=0
                    tar.addfile(info,io.BytesIO(data))
    seen=set()
    with tarfile.open(archive_path,'r:gz') as tar:
        for member in tar:
            _check(member.isfile() and member.name in expected and member.name not in seen,'Unexpected or duplicate archive member')
            with tar.extractfile(member) as stream:
                digest=hashlib.sha256();size=0
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block);size+=len(block)
            _check(size==expected[member.name]['size'] and digest.hexdigest()==expected[member.name]['sha256'],
                   'Independent archive reread failed')
            seen.add(member.name)
    _check(seen==set(expected),'Archive is incomplete')
    digest=hashlib.sha256()
    with archive_path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    receipt={'schema':'nextgen-local-evidence-archive-acceptance-v1','local_only':True,
        'archive_path':str(archive_path),'archive_sha256':digest.hexdigest(),'archive_bytes':archive_path.stat().st_size,
        'file_manifest_sha256':catalog_sha,'original_runs':runs,'members':len(seen),
        'uncompressed_bytes':sum(item['size'] for item in expected.values()),
        'all_original_mirror_SHA_verified':True,'all_archive_members_independently_reread_SHA_verified':True,
        'remote_file_writes':False,'model_GPU_or_deployment_reproduction_claim':False}
    write_new_json(output/'ARCHIVE_ACCEPTANCE.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();archive(args.spec,args.out)
