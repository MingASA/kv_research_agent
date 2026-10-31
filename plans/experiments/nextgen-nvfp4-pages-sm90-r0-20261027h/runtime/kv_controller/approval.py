#!/usr/bin/env python3
"""Read-only approval/scope preflight. This module never launches GPU work.

Production CLI always reads the fixed user-issued approval location. A complete
runtime quota supervisor is still required; an eligible plan is not permission
to bypass that missing executor or live shared-resource checks.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat


TRUSTED_APPROVAL = Path.home() / ".config/kv-research/approval.json"
GPU_PATTERN = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


class ApprovalError(ValueError):
    """An approval or proposed scope fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ApprovalError(message)


def _number(value: object, name: str, positive: bool = False) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"Invalid finite numeric field: {name}")
    _require(value > 0 if positive else value >= 0, f"Invalid nonnegative/positive field: {name}")
    return float(value)


def _unique_object(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_approval_file(path: Path) -> dict:
    """Read secure local bytes. The CLI passes only TRUSTED_APPROVAL; tests use fixtures."""
    return load_approval_snapshot(path)[0]


def load_approval_snapshot(path: Path) -> tuple[dict, str]:
    """Parse and hash the same secure open file, avoiding a second-read race."""
    path = path.absolute()
    _require(path.resolve(strict=True) == path, "Approval path contains symlinks")
    for parent in path.parents:
        info = parent.stat()
        _require(info.st_uid in (0, os.getuid()), f"Untrusted approval ancestor owner: {parent}")
        _require(not info.st_mode & 0o022, f"Approval ancestor writable by another identity: {parent}")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(), "Approval must be a user-owned regular file")
        _require(not info.st_mode & 0o077 and bool(info.st_mode & stat.S_IRUSR), "Approval must be readable only by its owner")
        _require(info.st_size <= 65536, "Approval file exceeds size limit")
        with os.fdopen(fd, "rb") as stream:
            fd = -1
            data = stream.read(65537)
        _require(len(data) <= 65536, "Approval grew beyond the size limit")
        value = json.loads(data, object_pairs_hook=_unique_object)
        _require(isinstance(value, dict), "Approval must be a JSON object")
        return value, hashlib.sha256(data).hexdigest()
    finally:
        if fd >= 0:
            os.close(fd)


def validate_approval(approval: dict, now: datetime.datetime) -> None:
    _require(approval.get("schema") == "kv-agent-approval-v1", "Unsupported approval schema")
    for key in ("enabled", "allow_remote_user_workspace_writes", "allow_gpu_jobs"):
        _require(approval.get(key) is True, f"User has not enabled {key}")
    _require(isinstance(approval.get("ssh_target"), str)
             and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", approval["ssh_target"]) is not None,
             "Invalid approved SSH target")
    _require(isinstance(approval.get("approved_host_fingerprint"), str)
             and re.fullmatch(r"SHA256:[A-Za-z0-9+/]{43}", approval["approved_host_fingerprint"]) is not None,
             "Missing/invalid approved host fingerprint")
    workspace = approval.get("approved_workspace")
    _require(isinstance(workspace, str) and workspace.startswith("/")
             and str(PurePosixPath(workspace)) == workspace and ".." not in PurePosixPath(workspace).parts
             and workspace not in ("/", "/export", "/export/home", "/home"), "Invalid project workspace scope")
    try:
        expiry = datetime.datetime.fromisoformat(approval["valid_until_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ApprovalError("Missing/invalid UTC expiry") from error
    _require(expiry.utcoffset() == datetime.timedelta(0), "Approval expiry must explicitly use UTC")
    _require(now.tzinfo is not None and expiry > now, "Approval expired")
    maximum = approval.get("max_concurrent_gpus")
    _require(type(maximum) is int and 1 <= maximum <= 8, "Invalid GPU-count budget")
    uuids = approval.get("allowed_gpu_uuids")
    _require(isinstance(uuids, list) and bool(uuids)
             and all(isinstance(uuid, str) and GPU_PATTERN.fullmatch(uuid) for uuid in uuids), "Invalid GPU UUID whitelist")
    _require(len(set(uuids)) == len(uuids) and maximum <= len(uuids), "Duplicate/insufficient GPU UUID whitelist")
    for key in ("max_gpu_hours_per_job", "max_gpu_hours_total", "max_disk_gib_new_outputs"):
        _number(approval.get(key), key, positive=True)


def validate_scope(approval: dict, request: dict, usage: dict, now: datetime.datetime) -> dict:
    """Validate static identity/quotas; no reservation, dispatch or live idle check."""
    validate_approval(approval, now)
    for key in ("ssh_target", "approved_host_fingerprint", "approved_workspace"):
        _require(request.get(key) == approval[key], f"Discovered/requested identity does not match approval: {key}")
    run_id = request.get("run_id")
    _require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id) is not None,
             "Invalid run ID")
    _require(run_id not in usage.get("known_run_ids", []), "Run ID already exists; reconcile rather than relaunch")
    _require(usage.get("reconciled") is True and not usage.get("uncertain_runs"), "Unreconciled/UNKNOWN jobs retain their budget")
    output = request.get("remote_output")
    _require(isinstance(output, str) and str(PurePosixPath(output)) == output
             and ".." not in PurePosixPath(output).parts, "Noncanonical output path")
    prefix = approval["approved_workspace"] + "/agent_work/runs/"
    _require(output == prefix + run_id, "Output must be a unique isolated run under approved BASE")
    uuids = request.get("gpu_uuids")
    _require(isinstance(uuids, list) and bool(uuids) and all(isinstance(uuid, str) for uuid in uuids)
             and len(set(uuids)) == len(uuids)
             and set(uuids).issubset(approval["allowed_gpu_uuids"]), "Unapproved/duplicate GPU UUID requested")
    active = usage.get("active_gpu_uuids", [])
    _require(isinstance(active, list) and not set(uuids).intersection(active), "Selected GPU already reserved by an existing job")
    _require(len(active) + len(uuids) <= approval["max_concurrent_gpus"], "Parallel GPU-count budget exceeded")
    wall = _number(request.get("max_wall_seconds"), "max_wall_seconds", positive=True)
    hours = len(uuids) * wall / 3600
    _require(hours <= approval["max_gpu_hours_per_job"], "Per-job GPU-hours exceeded")
    spent = _number(usage.get("gpu_hours_consumed"), "gpu_hours_consumed")
    reserved = _number(usage.get("gpu_hours_reserved"), "gpu_hours_reserved")
    _require(spent + reserved + hours <= approval["max_gpu_hours_total"], "Total GPU-hours including reservations exceeded")
    disk = _number(request.get("max_output_gib"), "max_output_gib", positive=True)
    used_disk = _number(usage.get("disk_gib_consumed"), "disk_gib_consumed")
    reserved_disk = _number(usage.get("disk_gib_reserved"), "disk_gib_reserved")
    _require(used_disk + reserved_disk + disk <= approval["max_disk_gib_new_outputs"], "Total new-output disk budget exceeded")
    expiry = datetime.datetime.fromisoformat(approval["valid_until_utc"].replace("Z", "+00:00"))
    _require(now + datetime.timedelta(seconds=wall) <= expiry, "Proposed job exceeds approval expiry")
    return {"authorization_scope_valid": True, "required_gpu_hours_reservation": hours,
            "required_disk_gib_reservation": disk, "can_launch": False,
            "remaining_gates": ["atomic persistent budget reservation", "fresh live identity/ownership",
                                "stable GPU idle and reservation checks", "runtime quota supervisor and UNKNOWN recovery"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--usage", type=Path)
    args = parser.parse_args()
    result = {"approval_path": str(TRUSTED_APPROVAL), "can_launch": False,
              "runtime_executor_implemented": False, "os_resource_sandbox": False,
              "approval_file_valid": False}
    try:
        approval = load_approval_file(TRUSTED_APPROVAL)
        now = datetime.datetime.now(datetime.timezone.utc)
        validate_approval(approval, now)
        result["approval_file_valid"] = True
        if args.request and args.usage:
            result.update(validate_scope(approval, json.loads(args.request.read_text()),
                                         json.loads(args.usage.read_text()), now))
    except (ApprovalError, OSError, ValueError) as error:
        result["reason"] = str(error)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
