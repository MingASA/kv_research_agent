"""Small durable-file primitives shared by local control and remote Python 3.9."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def secure_directory(path: Path) -> None:
    """Create a private directory; never change existing ownership or modes."""
    path = path.absolute()
    if not path.exists():
        path.mkdir(mode=0o700, parents=True, exist_ok=False)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or path.resolve(strict=True) != path
            or info.st_uid != os.getuid() or info.st_mode & 0o022):
        raise ValueError("Directory is symlinked, foreign-owned or writable by another identity")


def write_new_json(path: Path, value: object) -> None:
    write_new_bytes(path, canonical_bytes(value))


def write_new_bytes(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def atomic_json(path: Path, value: object) -> None:
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlink")
    data = canonical_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only the uniquely created local temporary file.
