#!/usr/bin/env python3
"""Copy explicitly selected project files over pinned SSH without remote writes.

The remote process only reads regular files and streams an in-memory tar archive.
The local destination must be new. Failed downloads remain available for audit.
This is an S0 collector, not a GPU launcher or an authorization mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
from typing import BinaryIO


REMOTE_CODE = r'''
from pathlib import Path, PurePosixPath
import datetime, hashlib, io, json, os, re, sys, tarfile

class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.hasher = hashlib.sha256()
        self.count = 0
    def read(self, size):
        if not 0 <= size <= 1024 * 1024:
            raise ValueError('Unbounded archive source read')
        data = self.stream.read(size)
        self.hasher.update(data)
        self.count += len(data)
        return data

base = Path(payload['base'])
if not base.is_absolute() or base.resolve(strict=True) != base:
    raise ValueError('Base must be an existing canonical absolute directory')
manifest = {'schema': 'kv-readonly-mirror-v1', 'base': str(base),
            'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'uid': os.getuid(), 'files': [], 'archive_codec': payload.get('archive_codec', 'gzip')}
seen = set()
total = 0
max_file_mib = payload.get('max_file_mib', 16)
if type(max_file_mib) is not int or max_file_mib not in (16, 64, 128, 256):
    raise ValueError('Unsupported explicit per-file bound')
codec = payload.get('archive_codec', 'gzip')
if codec not in ('gzip', 'xz'):
    raise ValueError('Unsupported explicit archive codec')
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|xz' if codec == 'xz' else 'w|gz') as archive:
    for item in payload['files']:
        if sum(key in item for key in ('rel', 'base_rel', 'model_rel')) != 1:
            raise ValueError('Specify exactly one project-relative source')
        relative = PurePosixPath(item.get('rel', item.get('base_rel', item.get('model_rel'))))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unsafe relative path')
        if 'rel' in item:
            source = base / 'experiments/aiperf' / relative
            name = str(relative)
            allowed_root = base
        elif 'base_rel' in item:
            source = base / relative
            name = 'support/' + str(relative)
            allowed_root = base
        else:
            allowed_names = {'config.json', 'tokenizer.json', 'tokenizer_config.json',
                             'generation_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt'}
            if str(relative) not in allowed_names or not payload.get('model_root'):
                raise ValueError('Only discovered read-only tokenizer/model metadata may be copied')
            allowed_root = Path(payload['model_root'])
            if not allowed_root.is_absolute() or allowed_root.resolve(strict=True) != allowed_root:
                raise ValueError('Model metadata root must be canonical')
            source = allowed_root / relative
            name = 'model_metadata/' + str(relative)
        if source.resolve(strict=True) != source or not source.is_file():
            raise ValueError('Source contains a symlink or is not a regular file')
        if not source.is_relative_to(allowed_root) or name in seen:
            raise ValueError('Source escaped base or has a duplicate archive name')
        size = source.stat().st_size
        if size != item['size'] or size > max_file_mib * 1024 * 1024:
            raise ValueError('Source changed size or exceeds per-file limit')
        if size > 128 * 1024 * 1024 and re.fullmatch(
                r'support/agent_work/runs/[A-Za-z0-9_-]+/output/(bf16|fp8_e4m3)/artifacts/profile_export_raw.jsonl', name) is None:
            raise ValueError('256MiB opt-in is only for exact owned-run raw exports')
        sealed_hash = None
        if size > 128 * 1024 * 1024:
            run_id, arm = PurePosixPath(name).parts[3], PurePosixPath(name).parts[5]
            ticket_path = base / 'agent_work/runs' / run_id / 'ticket.json'
            measurement_path = source.parent.parent / 'measurement.json'
            for proof in (ticket_path, measurement_path):
                if (proof.resolve(strict=True) != proof or not proof.is_file()
                        or proof.stat().st_uid != os.getuid() or proof.stat().st_size > 8 * 1024 * 1024):
                    raise ValueError('Large raw export has no owned bounded run proof')
            ticket = json.loads(ticket_path.read_text())
            measurement = json.loads(measurement_path.read_text())
            if (source.stat().st_uid != os.getuid() or ticket['request']['run_id'] != run_id
                    or measurement.get('measurement_success') is not True or measurement.get('arm') != arm
                    or measurement.get('request_sha256') != ticket['request_sha256']):
                raise ValueError('Large raw export is not sealed by its exact owned run')
            sealed_hash = measurement['evidence_sha256']['artifacts/profile_export_raw.jsonl']
        total += size
        if total > 512 * 1024 * 1024:
            raise ValueError('Collection exceeds total limit')
        seen.add(name)
        info = tarfile.TarInfo(name)
        info.size = size
        info.mode = 0o600
        with source.open('rb') as stream:
            before = os.fstat(stream.fileno())
            reader = HashingReader(stream)
            archive.addfile(info, reader)
            after = source.stat()
            if reader.count != size or any(getattr(before, key) != getattr(after, key)
                    for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')):
                raise ValueError('Source changed during collection')
            if sealed_hash is not None and reader.hasher.hexdigest() != sealed_hash:
                raise ValueError('Large raw export differs from its sealed worker SHA')
        manifest['files'].append({'source': str(source), 'relative_path': name,
                                  'size': size, 'sha256': reader.hasher.hexdigest()})
    data = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    info = tarfile.TarInfo('SNAPSHOT.json')
    info.size = len(data)
    info.mode = 0o600
    archive.addfile(info, io.BytesIO(data))
'''


MAX_EXPANDED_TAR_BYTES = (512 + 8) * 1024 * 1024
XZ_MEMORY_LIMIT_BYTES = 128 * 1024 * 1024


def _stream_hash(source: BinaryIO, target: BinaryIO | None = None) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    while True:
        block = source.read(1024 * 1024)
        if not block:
            return size, hasher.hexdigest()
        size += len(block)
        hasher.update(block)
        if target is not None:
            target.write(block)


def _expand_xz_bounded(archive_path: Path) -> Path:
    """Retain a new local tar with bounded output and decoder memory; no remote writes."""
    expanded = archive_path.with_name(archive_path.name + '.expanded.tar')
    decoder = lzma.LZMADecompressor(memlimit=XZ_MEMORY_LIMIT_BYTES)
    total = 0
    with archive_path.open('rb') as source, expanded.open('xb') as output:
        while True:
            block = source.read(1024 * 1024) if decoder.needs_input else b''
            if decoder.needs_input and not block:
                raise ValueError('Truncated XZ mirror')
            data = decoder.decompress(block, max_length=min(1024 * 1024, MAX_EXPANDED_TAR_BYTES - total + 1))
            total += len(data)
            if total > MAX_EXPANDED_TAR_BYTES:
                raise ValueError('Expanded mirror tar exceeds bounded data/header allowance')
            output.write(data)
            if decoder.eof:
                if decoder.unused_data or source.read(1):
                    raise ValueError('Trailing or concatenated XZ mirror')
                return expanded


def extract_verified(archive_path: Path, destination: Path, max_file_mib: int = 16,
                     archive_codec: str = 'gzip') -> dict:
    """Validate all members and hashes before creating any extracted files."""
    if type(max_file_mib) is not int or max_file_mib not in (16, 64, 128, 256):
        raise ValueError("Unsupported explicit per-file bound")
    if archive_codec not in ('gzip', 'xz'):
        raise ValueError('Unsupported explicit archive codec')
    tar_path = _expand_xz_bounded(archive_path) if archive_codec == 'xz' else archive_path
    with tarfile.open(tar_path, 'r:' if archive_codec == 'xz' else 'r:gz') as archive:
        members = archive.getmembers()
        names = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or ".." in path.parts
                    or not path.parts or str(path) != member.name or member.name in names):
                raise ValueError(f"Unsafe or duplicate archive member: {member.name}")
            if member.size < 0 or member.size > max_file_mib * 1024 * 1024:
                raise ValueError("Archive member exceeds size limit")
            if member.size > 128 * 1024 * 1024 and re.fullmatch(
                    r'support/agent_work/runs/[A-Za-z0-9_-]+/output/(bf16|fp8_e4m3)/artifacts/profile_export_raw.jsonl', member.name) is None:
                raise ValueError('256MiB opt-in is only for exact owned-run raw exports')
            if member.name == 'SNAPSHOT.json' and member.size > 4 * 1024 * 1024:
                raise ValueError('Snapshot metadata exceeds bounded JSON allowance')
            names.add(member.name)
            total += member.size
        if total > 512 * 1024 * 1024:
            raise ValueError("Archive exceeds size limit")
        stream = archive.extractfile("SNAPSHOT.json")
        if stream is None:
            raise ValueError("Missing snapshot manifest")
        manifest = json.load(stream)
        if manifest.get('archive_codec', 'gzip') != archive_codec:
            raise ValueError('Snapshot archive codec differs from the explicit transport')
        expected = {row["relative_path"]: row for row in manifest["files"]}
        if len(expected) != len(manifest["files"]) or names != set(expected) | {"SNAPSHOT.json"}:
            raise ValueError("Archive inventory disagrees with snapshot manifest")
        for member in members:
            if member.name == "SNAPSHOT.json":
                continue
            stream = archive.extractfile(member)
            size, sha256 = _stream_hash(stream)
            row = expected[member.name]
            if size != row["size"] or sha256 != row["sha256"]:
                raise ValueError(f"Snapshot hash mismatch: {member.name}")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as target:
                _stream_hash(archive.extractfile(member), target)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--read-only-model", type=Path)
    parser.add_argument("--max-file-mib", type=int, choices=(16, 64, 128, 256), default=16,
                        help="Explicit bound;256 only for exact owned-run raw exports; total stays512MiB")
    parser.add_argument("--transfer-timeout-seconds", type=int, default=240,
                        help="Bounded read-only transfer timeout (30 through 900 seconds)")
    parser.add_argument('--archive-codec', choices=('gzip', 'xz'), default='gzip')
    args = parser.parse_args()
    if not 30 <= args.transfer_timeout_seconds <= 900:
        parser.error("Transfer timeout must be between 30 and 900 seconds")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.target):
        parser.error("Use a discovered SSH alias")
    selection = json.loads(args.selection.read_text())
    payload = {"base": args.base, "files": selection["files"], "max_file_mib": args.max_file_mib,
               'archive_codec': args.archive_codec}
    if any("model_rel" in item for item in selection["files"]):
        config = json.loads((Path(__file__).resolve().parents[1] / "state/CONFIG.local.json").read_text())
        if (args.read_only_model is None or config.get("remote_model_path_verified") is not True
                or str(args.read_only_model) != config.get("remote_model_path")):
            parser.error("Model metadata root must match the discovered verified local machine mapping")
        payload["model_root"] = str(args.read_only_model)
    script = "payload = " + repr(payload) + "\n" + REMOTE_CODE
    args.out.mkdir(parents=True, exist_ok=False)
    archive_path = args.out / ('snapshot.tar.xz' if args.archive_codec == 'xz' else 'snapshot.tar.gz')
    command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
               "-o", "ConnectionAttempts=1", "-o", "StrictHostKeyChecking=yes",
               "-o", "HostKeyAlgorithms=ssh-ed25519", "-o", "UpdateHostKeys=no",
               "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=6",
               "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
               args.target, "python3 -B -"]
    timed_out = False
    with archive_path.open("xb") as output:
        try:
            result = subprocess.run(command, input=script.encode(), stdout=output,
                                    stderr=subprocess.PIPE, timeout=args.transfer_timeout_seconds, check=False)
            returncode, stderr = result.returncode, result.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode, stderr = None, error.stderr or b""
    (args.out / "transport.json").write_text(json.dumps({
        "command": command, "rc": returncode,
        "stderr": stderr.decode(errors="replace"),
        "timed_out": timed_out, "timeout_seconds": args.transfer_timeout_seconds,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "remote_writes": 0,
        'archive_codec': args.archive_codec,
    }, indent=2) + "\n")
    if timed_out or returncode:
        raise SystemExit(f"Read-only copy failed (rc={returncode}, timeout={timed_out}); partial artifact retained")
    manifest = extract_verified(archive_path, args.out / "files", args.max_file_mib, args.archive_codec)
    print(json.dumps({"files": len(manifest["files"]), "destination": str(args.out),
                      "observed_utc": manifest["observed_utc"], "remote_writes": 0}))


if __name__ == "__main__":
    main()
