"""Read-only live environment evidence. Does not create directories or jobs."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys


def collect(config: dict) -> dict:
    expected = config["remote_identity"]
    if (os.getuid(), os.getgid(), socket.gethostname()) != (expected["uid"], expected["gid"], expected["hostname"]):
        raise ValueError("Remote machine/UID/GID differs from discovered identity")
    base = Path(config["remote_base"])
    if base.resolve(strict=True) != base or base.stat().st_uid != os.getuid() or len(base.parts) < 5:
        raise ValueError("Project BASE is not the previously discovered owned canonical directory")
    ancestry = {str(path): [path.stat().st_uid, path.stat().st_gid, stat.S_IMODE(path.stat().st_mode)]
                for path in (base, *base.parents)}
    sources = {}
    for name in config.get("hash_paths", []):
        path = Path(name)
        if path.resolve(strict=True) != path or not path.is_file() or not str(path).startswith(str(base) + "/"):
            raise ValueError("Source escaped discovered BASE")
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                hasher.update(block)
        sources[name] = hasher.hexdigest()
    model = Path(config["remote_model_path"])
    if model.resolve(strict=True) != model:
        raise ValueError("Previously discovered model root changed")
    statfs = os.statvfs(base)
    commands = {
        "gpus": ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        "compute_apps": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        "owned_containers": ["docker", "ps", "-a", "--no-trunc", "--filter", "label=kvra.reservation_id", "--format", "{{json .}}"],
        "image": ["docker", "image", "inspect", config["image_id"]],
    }
    observations = {name: subprocess.check_output(argv, text=True, timeout=20).strip() for name, argv in commands.items()}
    runs = base / "agent_work/runs"
    # Unknown directories are reported, never opened as executable code or adopted.
    inventory = []
    if runs.exists():
        if runs.resolve(strict=True) != runs:
            raise ValueError("Agent runs subtree changed to a symlink")
        inventory = [{"name": path.name, "uid": path.lstat().st_uid, "mode": stat.S_IMODE(path.lstat().st_mode),
                      "symlink": path.is_symlink()} for path in runs.iterdir()]
    return {"schema": "kv-controller-readonly-discovery-v1", "can_launch": False, "remote_writes": 0,
            "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "identity": expected,
            "ancestor_identity": ancestry, "free_bytes": statfs.f_bavail * statfs.f_frsize,
            "source_sha256": sources, "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            "run_inventory": inventory, **observations}


if __name__ == "__main__":
    try:
        value = {"ok": True, **collect(json.load(sys.stdin))}
    except Exception as error:
        value = {"ok": False, "error": type(error).__name__ + ": " + str(error)}
    sys.stdout.write(json.dumps(value, sort_keys=True) + "\n")
