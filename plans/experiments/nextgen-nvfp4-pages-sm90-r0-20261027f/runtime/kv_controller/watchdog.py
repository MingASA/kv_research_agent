"""Independent whole-job deadline guard; signals only captured owned pidfds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

from .durable import atomic_json, write_new_json
from .processes import process_identity, signal_exact


def enforce_registry(config: dict, registry: dict, signaler: object = signal_exact) -> list[dict]:
    for field in ("reservation_id", "request_sha256"):
        if registry.get(field) != config[field]:
            raise ValueError("Watchdog registry identity mismatch")
    actions = []
    for item in registry.get("containers", []):
        if item.get("released") is True:
            continue
        identity = item["pid_identity"]
        identifier = item["container_id"]
        if (re.fullmatch(r"[0-9a-f]{64}", identifier) is None
                or identifier not in identity["cgroup"] or identity["uid"] != os.getuid()):
            raise ValueError("Watchdog refuses an unowned/foreign cgroup")
        try:
            sent = signaler(identity, signal.SIGKILL)
            actions.append({"container_id": identifier, "signal_sent": sent})
        except (OSError, ValueError) as error:
            actions.append({"container_id": identifier, "signal_sent": False, "error": str(error)})
    return actions


def run_watchdog(config_path: Path, clock_ns: object = time.monotonic_ns,
                 sleeper: object = time.sleep, signaler: object = signal_exact) -> dict:
    root = config_path.parent
    config = json.loads(config_path.read_text())
    write_new_json(root / "WATCHDOG_READY.json", {"identity": process_identity(os.getpid()),
                                                "request_sha256": config["request_sha256"]})
    last = {"reservation_id": config["reservation_id"], "request_sha256": config["request_sha256"], "containers": []}
    result = {"request_sha256": config["request_sha256"], "status": "UNKNOWN", "actions": []}
    try:
        while clock_ns() < config["kill_at_monotonic_ns"]:
            if (root / "owned_containers.json").exists():
                last = json.loads((root / "owned_containers.json").read_text())
            finish = root / "WATCHDOG_FINISHED.json"
            if finish.exists():
                value = json.loads(finish.read_text())
                if (value.get("request_sha256") != config["request_sha256"]
                        or value.get("gpu_release_verified") is not True
                        or hashlib.sha256((root / "result.json").read_bytes()).hexdigest() != value.get("result_sha256")):
                    raise ValueError("Invalid early watchdog shutdown proof")
                result["status"] = "RELEASE_CONFIRMED"
                return result
            sleeper(0.25)
        if (root / "owned_containers.json").exists():
            last = json.loads((root / "owned_containers.json").read_text())
        result.update(status="DEADLINE_ENFORCED", actions=enforce_registry(config, last, signaler))
        return result
    except Exception as error:
        result.update(status="GUARD_FAILURE", error=str(error), actions=enforce_registry(config, last, signaler))
        return result
    finally:
        # This report attests signalling attempts, not GPU release or experiment success.
        atomic_json(root / "watchdog_result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run_watchdog(args.config)
    raise SystemExit(0 if result["status"] == "RELEASE_CONFIRMED" else 1)


if __name__ == "__main__":
    main()
