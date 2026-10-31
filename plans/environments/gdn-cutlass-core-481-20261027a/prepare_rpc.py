"""Create one private, immutable CuTe core overlay using installed native libs.

No package manager, global write, GPU operation or dependency replacement is
performed. The existing environment is read-only input to a new project path.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import socket
import stat
import sys
import zipfile


NAME = "gdn-cutlass-core-481-20261027a"
WHEEL = "nvidia_cutlass_dsl_libs_core-4.8.1-py3-none-any.whl"
WHEEL_SHA = "bacc6760e3286db447d45720d32ae58ac2e6cecade1152345ef784b2d5e44431"


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _wheel_files(data: bytes) -> dict[str, bytes]:
    _require(len(data) == 1635096 and hashlib.sha256(data).hexdigest() == WHEEL_SHA,
             "Expected the exact official, pinned 4.8.1 core wheel")
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        _require(len(archive.infolist()) == 208 and sum(i.file_size for i in archive.infolist()) < 8 * 2**20,
                 "Unexpected wheel size/layout")
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            _require(not name.is_absolute() and ".." not in name.parts and str(name) == info.filename
                     and info.filename not in files and not stat.S_ISLNK(info.external_attr >> 16),
                     "Unsafe/duplicate wheel member")
            _require(name.parts[0] in ("nvidia_cutlass_dsl", "nvidia_cutlass_dsl_libs_core-4.8.1.dist-info"),
                     "Unexpected wheel top-level path")
            files[str(name)] = archive.read(info)
    return files


def _hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


def _new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def run(payload: dict) -> dict:
    approval = payload["approval"]
    identity = approval["approved_remote_identity"]
    _require((os.getuid(), os.getgid(), socket.gethostname()) ==
             (identity["uid"], identity["gid"], identity["hostname"]), "Remote identity changed")
    base = Path(approval["approved_workspace"])
    _require(base.resolve(strict=True) == base and base.stat().st_uid == os.getuid(), "Unknown project BASE")
    work = base / "agent_work"
    _require(work.resolve(strict=True) == work and work.stat().st_uid == os.getuid(), "Unknown project work parent")
    _require(approval.get("enabled") is True and approval.get("allow_remote_user_workspace_writes") is True,
             "Project writes are not authorized")
    _require(payload.get("operation") == "prepare", "Only one new overlay preparation is supported")
    root = base / "agent_work/envs" / NAME
    _require(not root.exists() and not root.is_symlink(), "Existing overlay must never be replaced or prepared twice")
    existing = base / ".venv/lib/python3.12/site-packages/nvidia_cutlass_dsl"
    _require(existing.resolve(strict=True) == existing and existing.is_dir(), "Existing DSL prefix differs")
    _require(not (existing / "dsl_packages/cutlass/cute").exists(), "Core already exists; reconcile rather than duplicate")
    borrowed = []
    for path in sorted(existing.rglob("*")):
        _require(path.resolve(strict=True).is_relative_to(existing), "Borrowed file escaped existing DSL prefix")
        if path.is_file():
            borrowed.append({"relative_path": str(path.relative_to(existing)), "bytes": path.stat().st_size,
                             "sha256": _hash(path)})
    _require(len(borrowed) <= 512 and sum(row["bytes"] for row in borrowed) <= 2**30,
             "Unexpected installed native-library envelope")
    data = base64.b64decode(payload["wheel_base64"], validate=True)
    files = _wheel_files(data)
    _require(os.statvfs(base).f_bavail * os.statvfs(base).f_frsize >= 20 * 2**30, "Insufficient disk headroom")
    parent = root.parent
    if not parent.exists():
        parent.mkdir(mode=0o700)
    _require(parent.resolve(strict=True) == parent and parent.stat().st_uid == os.getuid(), "Unknown overlay parent")
    root.mkdir(mode=0o700)
    _new(root / "PREPARE_INTENT.json", json.dumps({"wheel_sha256": WHEEL_SHA, "GPU_requested": False}).encode())
    _new(root / WHEEL, data)
    site = root / "site-packages"
    for relative, content in files.items():
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _new(target, content)
    links = []

    def merge(source: Path, target: Path) -> None:
        if target.exists():
            if source.is_dir() and target.is_dir() and not target.is_symlink():
                for child in sorted(source.iterdir()):
                    merge(child, target / child.name)
            else:
                _require(source.is_file() and target.is_file() and _hash(source) == _hash(target),
                         "Core/native-library overlap differs; no replacement allowed")
        else:
            relative = os.path.relpath(source, target.parent)
            target.symlink_to(relative, target_is_directory=source.is_dir())
            links.append({"overlay_path": str(target.relative_to(root)), "relative_target": relative})

    merge(existing, site / "nvidia_cutlass_dsl")
    manifest = {"schema": "kv-private-GDN-DSL-overlay-v1", "name": NAME, "version": "4.8.1",
                "wheel": WHEEL, "wheel_sha256": WHEEL_SHA, "wheel_bytes": len(data),
                "official_metadata_source": "https://pypi.org/pypi/nvidia-cutlass-dsl-libs-core/4.8.1/json",
                "expanded_core_bytes": sum(map(len, files.values())), "new_file_count": len(files),
                "borrowed_prefix": str(existing), "borrowed_files": borrowed, "relative_readonly_links": links,
                "GPU_requested": False, "global_or_existing_files_changed": False,
                "import_validation_pending": True, "SM90_kernel_validation_pending": True}
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    _new(root / "manifest.json", raw)
    return {"ok": True, "root": str(root), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "new_bytes": len(data) + sum(map(len, files.values())) + len(raw), "borrowed_files": len(borrowed),
            "borrowed_native_bytes": sum(row["bytes"] for row in borrowed), "links": len(links),
            "GPU_requested": False, "global_files_changed": False, "import_validation_pending": True}


if __name__ == "__main__":
    try:
        result = run(json.load(sys.stdin))
    except Exception as error:
        result = {"ok": False, "error": type(error).__name__ + ": " + str(error)}
    print(json.dumps(result, sort_keys=True))
