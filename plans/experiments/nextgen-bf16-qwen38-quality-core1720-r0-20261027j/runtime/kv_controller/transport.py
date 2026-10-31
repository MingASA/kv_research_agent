"""Pinned SSH RPC. Mutation failures become UNKNOWN and are never retried."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import random
from pathlib import Path
import shlex
import subprocess
import time

from .approval import ApprovalError
from .durable import canonical_bytes, secure_directory, write_new_bytes


class UncertainDispatch(RuntimeError):
    pass


MAX_RPC_JSON_BYTES = 96 * 2**20


def wire_request(source: str, payload: dict) -> tuple[str, bytes]:
    data = canonical_bytes(payload)
    if len(data) > MAX_RPC_JSON_BYTES:
        raise ApprovalError('RPC JSON exceeds the bounded transfer envelope')
    if len(data) > 256 * 1024:
        # Keep the marker visible to the absence detector while receiving stdin.
        # Compile original source separately so its future imports remain valid.
        wrapper = ('"""Allowlisted stdin RPC, executed by pinned SSH (bounded gzip transport)."""\n'
                   'import gzip,io,sys\n'
                   'data=gzip.GzipFile(fileobj=sys.stdin.buffer).read(' + str(MAX_RPC_JSON_BYTES+1) + ')\n'
                   'if len(data)>' + str(MAX_RPC_JSON_BYTES) + ':raise ValueError("Oversized RPC JSON")\n'
                   'sys.stdin=io.TextIOWrapper(io.BytesIO(data),encoding="utf-8")\n'
                   'exec(compile(' + repr(source) + ',"<pinned-rpc>","exec"))\n')
        return 'python3 -B -c ' + shlex.quote(wrapper), gzip.compress(data,compresslevel=6,mtime=0)
    return 'python3 -B -c ' + shlex.quote(source), data


def _error_text(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        detail = error.stderr or b''
        if isinstance(detail, bytes):
            detail = detail.decode(errors='replace')
        return 'SSH exit ' + str(error.returncode) + ': ' + detail[-2048:].strip()
    if isinstance(error, subprocess.TimeoutExpired):
        return 'SSH timed out after ' + str(error.timeout) + ' seconds'
    return type(error).__name__ + ': ' + str(error)[:2048]


def fingerprint(key: str) -> str:
    return "SHA256:" + base64.b64encode(hashlib.sha256(base64.b64decode(key, validate=True)).digest()).decode().rstrip("=")


def pinned_key(known_hosts: Path, endpoint: str, approved: str) -> str:
    result = subprocess.run(["ssh-keygen", "-F", endpoint, "-f", str(known_hosts)],
                            capture_output=True, text=True, timeout=10)
    matches = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] == "ssh-ed25519" and fingerprint(fields[2]) == approved:
            matches.append(fields[2])
    if not matches or len(set(matches)) != 1:
        raise ApprovalError("Approved ED25519 pin is absent/ambiguous in existing known_hosts")
    return matches[0]


class PinnedSSH:
    def __init__(self, approval: dict, local_directory: Path) -> None:
        target = approval["ssh_target"]
        configured = subprocess.check_output(["ssh", "-T", "-G", target], text=True, timeout=10)
        config = dict(line.split(" ", 1) for line in configured.splitlines() if " " in line)
        host, port = config["hostname"], config["port"]
        if not port.isdigit() or not host or any(char.isspace() for char in host):
            raise ApprovalError("Unknown SSH endpoint mapping")
        endpoint = host if port == "22" else "[" + host + "]:" + port
        key = pinned_key(Path.home() / ".ssh/known_hosts", endpoint, approval["approved_host_fingerprint"])
        secure_directory(local_directory)
        pin = local_directory / "known_hosts"
        content = (endpoint + " ssh-ed25519 " + key + "\n").encode()
        if pin.exists():
            if pin.is_symlink() or pin.read_bytes() != content:
                raise ApprovalError("Local pinned-host evidence changed")
        else:
            write_new_bytes(pin, content)
        self.argv = ["ssh", "-T", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                     "-o", "ConnectionAttempts=1", "-o", "StrictHostKeyChecking=yes",
                     "-o", "HostKeyAlgorithms=ssh-ed25519", "-o", "UpdateHostKeys=no",
                     "-o", "UserKnownHostsFile=" + str(pin), "-o", "GlobalKnownHostsFile=/dev/null",
                     "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
                     "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=6",
                     "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "RequestTTY=no",
                     "-o", "Hostname=" + host, "-p", port, "-l", config["user"], target]
        self.endpoint = {"host": host, "port": int(port), "user": config["user"], "fingerprint": approval["approved_host_fingerprint"]}

    def rpc(self, source: str, payload: dict, *, mutation: bool = False, timeout: int = 90) -> dict:
        # The static program is shell-quoted; payload paths/bytes are solely stdin JSON.
        command, wire = wire_request(source, payload)
        for attempt in range(3 if not mutation else 1):
            try:
                result = subprocess.run(self.argv + [command], input=wire,
                                        capture_output=True, timeout=timeout, check=True)
                response = json.loads(result.stdout)
                if not isinstance(response, dict):
                    raise ValueError("Malformed RPC response")
                if response.get("ok") is not True:
                    raise RuntimeError("Remote RPC refused: " + str(response.get("error")))
                return response
            except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as error:
                if mutation:
                    raise UncertainDispatch("Mutation reply uncertain; keep reservation and reconcile read-only: " + _error_text(error)) from error
                if attempt == 2:
                    raise RuntimeError('Read-only SSH recovery exhausted: ' + _error_text(error)) from error
                # Runbook's first finite recovery window:15s,30s plus jitter.
                # Mutations never enter this branch or get retried.
                time.sleep(15 * 2 ** attempt + random.uniform(0, 1))
        raise AssertionError("unreachable")
