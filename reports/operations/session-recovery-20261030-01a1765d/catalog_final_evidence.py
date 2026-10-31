"""Build a local delivery catalog from actual archives and pinned source bundles."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.controller.durable import write_new_json, write_new_bytes

NAMES = ['quality-a1','service-original-a2','tools-original-a2','mixed-pilot-a1',
         'local-failure-index-a1','mixed-quality-a1',
         'session-protocol-failure-and-repair-a1','mixed-pressure-a1','mixed-tools-a1']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(control_provenance):
    archive_root = ROOT/'results/archive/nextgen-native-kv-20261030'
    entries = []
    for name in NAMES:
        folder = archive_root/name
        receipt_path = folder/'ARCHIVE_ACCEPTANCE.json'
        receipt = json.loads(receipt_path.read_text())
        tar = folder/'evidence.tar.gz'; manifest = folder/'ARCHIVE_FILE_MANIFEST.json'
        assert tar.stat().st_size == receipt['archive_bytes'] and sha(tar) == receipt['archive_sha256']
        assert sha(manifest) == receipt['file_manifest_sha256']
        files = json.loads(manifest.read_text())['files']
        assert receipt['members'] == len(files)+1
        assert receipt['uncompressed_bytes'] == sum(item['size'] for item in files.values())+manifest.stat().st_size
        assert receipt['all_archive_members_independently_reread_SHA_verified'] is True
        assert receipt['local_only'] is True and receipt['remote_file_writes'] is False
        index_only = name == 'local-failure-index-a1'
        entries.append({'name':name,'scope':'Historical failure index and selected analyses only; not all historical failed raw runs.' if index_only else 'Original raw mirrors and explicit independent analyses, as named in the archive manifest.',
            'receipt':receipt_path.relative_to(ROOT).as_posix(),'receipt_sha256':sha(receipt_path),
            'archive':tar.relative_to(ROOT).as_posix(),'archive_sha256':receipt['archive_sha256'],
            'archive_bytes':receipt['archive_bytes'],'members':receipt['members'],
            'uncompressed_bytes':receipt['uncompressed_bytes'],'original_runs':receipt['original_runs'],
            'file_manifest_sha256':receipt['file_manifest_sha256'],'SHA_verified':True})
    model_path = ROOT/'reports/mainline/nextgen-native-kv-20261030/SOURCE_PROVENANCE.json'
    model = json.loads(model_path.read_text())
    for name, item in model['files'].items():
        path = model_path.parent/name
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    control = json.loads(control_provenance.read_text()); bundle = ROOT/control['bundle_path']
    assert bundle.stat().st_size == control['bundle_bytes'] and sha(bundle) == control['bundle_sha256']
    assert control['bundle_verified'] is True and control['prerequisite_commits'] == []
    result = {'schema':'nextgen-local-final-evidence-catalog-v1','utc':datetime.now(timezone.utc).isoformat(),
        'classification':'confirmed_local_artifact_integrity','archives':entries,
        'archive_count':len(entries),'total_archive_bytes':sum(row['archive_bytes'] for row in entries),
        'model_source_provenance':model_path.relative_to(ROOT).as_posix(),'model_source_provenance_sha256':sha(model_path),
        'controller_source_provenance':control_provenance.relative_to(ROOT).as_posix(),
        'controller_source_provenance_sha256':sha(control_provenance),
        'controller_bundle_frozen_commit':control['commit'],'final_catalog_and_status_metadata_in_bundle':False,
        'private_approval_source_file_weights_or_full_original_session_added_as_extra':False,
        'remote_writes_or_GPU_actions':False,'production_or_quality_acceptance_claim':False,
        'notes':'Archives retain original COMPLETE/FAILED outcomes. Catalog hashes the existing archives; each archive has a prior full member reread receipt. Original local run tickets retain their authorization snapshots. Final catalog/status metadata is recorded after the frozen controller bundle, avoiding a self-referential hash.'}
    write_new_json(archive_root/'CATALOG-a1.json', result)
    lines=['# 本地证据交付目录','',
           '以下九个分区的实际gzip与文件清单SHA均已重新核对；每个分区有写包后逐成员重读回执。原始FAILED保留。','',
           '| 分区 | 成员 | gzip字节 | SHA-256 |','|---|---:|---:|---|']
    lines += [f"| [{row['name']}]({row['name']}/ARCHIVE_ACCEPTANCE.json) | {row['members']} | {row['archive_bytes']} | `{row['archive_sha256']}` |" for row in entries]
    lines += ['', 'local-failure-index只包含历史失败索引与选定分析，不表示全部历史失败raw均打包。', '',
        '源码与归档分开：模型patch/增量bundle固定11e7cddb..03134b72，需要已有base；控制器完整bundle无prerequisite。',
        '控制器固定commit：`'+control['commit']+'`。最终catalog与状态收尾元数据在后续Git提交中，不冒充已包含在该bundle。',
        '不额外收集权重、安装环境、固定私有授权源文件或完整原会话内部日志；原票据中的授权快照随本地原始证据保留。这不是可直接部署镜像。']
    write_new_bytes(archive_root/'CATALOG-a1.md', ('\n'.join(lines)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k != 'archives'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-provenance', type=Path, required=True)
    main(parser.parse_args().control_provenance.resolve(strict=True))
