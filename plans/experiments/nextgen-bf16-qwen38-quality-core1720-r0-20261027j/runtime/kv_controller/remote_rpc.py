"""Allowlisted stdin RPC, executed by pinned SSH's Python 3.9 (stdlib only).

status is read-only. prepare/start/stop require the immutable local reservation
ticket and touch only a new isolated run, or its proven supervisor process.
"""

from __future__ import annotations

import base64
import datetime
import fcntl
import hashlib
import gzip
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import stat
import subprocess
import sys


def encoded(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def identity(pid: int) -> dict:
    path = Path("/proc") / str(pid)
    text = (path / "stat").read_text()
    return {"pid": pid, "uid": path.stat().st_uid, "start_ticks": int(text[text.rfind(")") + 2:].split()[19]),
            "cgroup": (path / "cgroup").read_text(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def alive(expected: dict) -> bool:
    try:
        return identity(expected["pid"]) == expected
    except FileNotFoundError:
        return False


def regular(path: Path) -> bytes:
    require(path.resolve(strict=True) == path, "File path contains symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_size <= 64 * 1024 * 1024,
                "Unowned/nonregular/oversized run file")
        return stream.read(64 * 1024 * 1024 + 1)


def new_bytes(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def private_directory(path: Path) -> None:
    if not path.exists():
        path.mkdir(mode=0o700)
    info = path.lstat()
    require(path.resolve(strict=True) == path and stat.S_ISDIR(info.st_mode)
            and info.st_uid == os.getuid() and not info.st_mode & 0o022, "Unsafe existing project directory")


def check_ticket(ticket: dict, mutation: bool) -> Path:
    approval, request = ticket["approval"], ticket["request"]
    require(sha(encoded(request)) == ticket["request_sha256"], "Request hash mismatch")
    require(re.fullmatch(r"[0-9a-f]{32}", ticket["reservation_id"]) is not None, "Invalid reservation nonce")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", request["run_id"]) is not None, "Invalid run ID")
    remote = approval["approved_remote_identity"]
    require(os.getuid() == remote["uid"] and os.getgid() == remote["gid"]
            and socket.gethostname() == remote["hostname"], "Remote machine/UID/GID mismatch")
    base = Path(approval["approved_workspace"])
    require(str(base) == approval["approved_workspace"] and base.is_absolute()
            and len(base.parts) >= 5 and ".." not in base.parts
            and base.resolve(strict=True) == base and base.stat().st_uid == os.getuid(), "Invalid project BASE")
    # Shared ancestors may be group-writable, but none can be symlinked or foreign to the discovered tree.
    for path, expected in ticket["ancestor_identity"].items():
        parent = Path(path)
        require(parent in (base, *base.parents), "Ancestor evidence escaped BASE lineage")
        info = parent.stat()
        require(parent.resolve(strict=True) == parent and [info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)] == expected,
                "Discovered BASE ancestry changed")
    require(set(ticket["ancestor_identity"]) == {str(p) for p in (base, *base.parents)}, "Incomplete BASE ancestry evidence")
    root = base / "agent_work/runs" / request["run_id"]
    require(str(root) == request["remote_output"], "Run path escaped authorized subtree")
    if mutation:
        require(all(approval.get(key) is True for key in ("enabled", "allow_remote_user_workspace_writes", "allow_gpu_jobs")),
                "No user-issued remote-write/GPU authority")
        expiry = datetime.datetime.fromisoformat(approval["valid_until_utc"].replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        require(expiry > now and approval.get("execution_contract") == "kv-controller-v1", "Expired/unknown execution authority")
        require(now < datetime.datetime.fromisoformat(ticket["deadline_utc"].replace("Z", "+00:00")) <= expiry,
                "Run deadline outside authority")
    return root


def existing(root: Path, ticket: dict) -> None:
    require(root.resolve(strict=True) == root and root.stat().st_uid == os.getuid(), "Unknown run directory ownership")
    stored = json.loads(regular(root / "ticket.json"))
    require(stored == ticket, "Existing run ticket differs; never adopt/overwrite")


def status(root: Path, ticket: dict) -> dict:
    if not root.exists():
        return {"status": "ABSENT", "run_id": ticket["request"]["run_id"], "relaunch_allowed": False}
    existing(root, ticket)
    result = {"status": "PREPARED", "run_id": ticket["request"]["run_id"], "relaunch_allowed": False,
              "files": {}, "supervisor_alive": False}
    for name in ("START_INTENT.json", "SUPERVISOR_PROCESS.json", "SUPERVISOR_STARTED.json", "state.json",
                 "result.json", "COMPLETE", "owned_containers.json", "watchdog_result.json"):
        path = root / name
        if path.exists():
            data = regular(path)
            result["files"][name] = {"sha256": sha(data), "value": json.loads(data)}
    for name in ("SUPERVISOR_STARTED.json", "SUPERVISOR_PROCESS.json"):
        if name in result["files"]:
            result["supervisor_alive"] = alive(result["files"][name]["value"]["identity"])
            break
    identifiers = subprocess.check_output(["docker", "ps", "-aq", "--no-trunc", "--filter",
                                          "label=kvra.reservation_id=" + ticket["reservation_id"]], text=True, timeout=10).split()
    result["live_owned_containers"] = []
    result["owned_containers"] = []
    for identifier in identifiers:
        require(re.fullmatch(r"[0-9a-f]{64}", identifier) is not None, "Unknown Docker identity response")
        value = json.loads(subprocess.check_output(["docker", "inspect", identifier], text=True, timeout=10))[0]
        labels = value["Config"].get("Labels") or {}
        require(value["Id"] == identifier and labels.get("kvra.request_sha256") == ticket["request_sha256"]
                and labels.get("kvra.run_id") == ticket["request"]["run_id"], "Ambiguous run container ownership")
        result["owned_containers"].append({"id": identifier, "running": value["State"]["Running"], "created": value["Created"]})
        if value["State"]["Running"]:
            result["live_owned_containers"].append(identifier)
    if "result.json" in result["files"]:
        result["status"] = result["files"]["result.json"]["value"]["status"]
    elif result["supervisor_alive"]:
        result["status"] = "RUNNING"
    elif "START_INTENT.json" in result["files"]:
        result["status"] = "UNKNOWN"
    result["log_tail"] = {}
    for name in ("supervisor.log", "watchdog.log", "events.jsonl", "output/bf16/container.log", "output/fp8_e4m3/container.log",
                 "output/bf16/server.log", "output/fp8_e4m3/server.log", "output/bf16/aiperf.log", "output/fp8_e4m3/aiperf.log",
                 "output/bf16/operator_acceptance.log", "output/fp8_e4m3/operator_acceptance.log",
                 "output/bf16/component_profiler_acceptance.log", "output/fp8_e4m3/component_profiler_acceptance.log",
                 "output/bf16/quality.log", "output/fp8_e4m3/quality.log",
                 "output/bf16/quality/arms/auto/attempt-0001/server.log",
                 "output/fp8_e4m3/quality/arms/fp8_e4m3/attempt-0001/server.log",
                 "output/bf16/scorer_cpu_preflight.log", "output/fp8_e4m3/scorer_cpu_preflight.log"):
        path = root / name
        if path.exists():
            require(path.resolve(strict=True) == path and path.stat().st_uid == os.getuid(), "Unowned diagnostic path")
            with path.open("rb") as stream:
                stream.seek(max(0, path.stat().st_size - 8192))
                result["log_tail"][name] = stream.read(8192).decode(errors="replace")
    if ticket['request']['workload']['kind'] == 'heldout_kv_range_probe':
        result['range_probe_progress'] = {}
        for relative in ('aiperf_cli_CPU_acceptance.json', 'range_operator/acceptance.json', 'range_probe/RESET_DONE.json',
                         'range_probe/audit.json', 'range_probe/ERROR.json'):
            path = root / 'output/bf16' / relative
            if path.exists():
                data = regular(path)
                require(len(data) <= 65536, 'Oversized bounded diagnostic progress')
                try:
                    value = json.loads(data)
                except json.JSONDecodeError:
                    value = {'pending_file_write': True}
                result['range_probe_progress'][relative] = {'sha256': sha(data), 'value': value}
        path = root / 'output/bf16/range_operator/probe.log'
        if path.exists():
            require(path.resolve(strict=True) == path and path.stat().st_uid == os.getuid(),
                    'Unowned range operator diagnostic log')
            with path.open('rb') as stream:
                stream.seek(max(0,path.stat().st_size-8192))
                result['log_tail']['output/bf16/range_operator/probe.log'] = stream.read(8192).decode(errors='replace')
    result["gpu_snapshot"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                                     "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
    result["quality_progress"] = {}
    result["operator_acceptance"] = {}
    if ticket['request'].get('schema') == 'kv-nextgen-run-v1':
        result['nextgen_progress'] = {}
        for arm in ticket['request']['arms']:
            require(arm in ('bf16','fp8_e4m3','nvfp4'), 'Unknown native progress arm')
            progress = {}
            for filename in ('container.log','server.log','operator.log','native_operator.log','observer_Graph.log'):
                path = root/'output'/arm/filename
                if path.exists():
                    require(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid(),
                            'Unowned next-generation log')
                    with path.open('rb') as stream:
                        stream.seek(max(0,path.stat().st_size-8192))
                        result['log_tail']['output/'+arm+'/'+filename]=stream.read(8192).decode(errors='replace')
            for relative in ('observer_Graph.json','kv_observer_reset_ack.json','kv_observer_export_ack.json',
                             'kv_observer_error.json','validation_result.json','quality_progress.json',
                             'native_pages/acceptance.json','native_pages/'+arm+'-prefill.json',
                             'native_pages/'+arm+'-chunked_prefill.json','native_pages/'+arm+'-decode.json'):
                path = root/'output'/arm/relative
                if path.exists():
                    require(path.stat().st_size<=2**20,'Oversized native progress metadata')
                    data=regular(path)
                    try:value=json.loads(data)
                    except json.JSONDecodeError:value={'pending_file_write':True}
                    progress[relative]={'sha256':sha(data),'value':value}
            for filename in ('validation_raw.jsonl','calibration_raw.jsonl'):
                path=root/'output'/arm/filename
                if path.exists():
                    require(path.resolve(strict=True)==path and path.stat().st_uid==os.getuid()
                            and path.stat().st_size<=64*2**20,'Unowned/oversized native response journal')
                    with path.open('rb') as stream:
                        progress[filename+'_durable_rows']=sum(line.endswith(b'\n') for line in stream)
            result['nextgen_progress'][arm]=progress
    for arm, label in (("bf16", "auto"), ("fp8_e4m3", "fp8-e4m3")):
        acceptance = root / "output" / arm / "operator_acceptance.json"
        if acceptance.is_file():
            data = regular(acceptance)
            require(len(data) < 2**20, "Oversized operator evidence")
            try:
                value = json.loads(data)
            except json.JSONDecodeError:
                result["operator_acceptance"][arm] = {"pending_file_write": True}
            else:
                result["operator_acceptance"][arm] = {"sha256": sha(data), "value": value}
        counts = {}
        for task in ("gsm8k", "narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "hotpotqa", "musique", "lcc", "repobench-p"):
            path = root / "output" / arm / "quality/runs" / task / (label + ".jsonl")
            if path.is_file():
                require(path.resolve(strict=True) == path and path.stat().st_uid == os.getuid()
                        and path.stat().st_size < 64 * 2**20, "Unowned or oversized quality journal")
                with path.open("rb") as stream:
                    counts[task] = sum(line.endswith(b"\n") for line in stream)
        result["quality_progress"][arm] = counts
    if ticket['request']['workload']['kind'] == 'anonymous_trace_component_diagnostic':
        result['component_diagnostics'] = {}
        for arm in ticket['request']['arms']:
            observed = {}
            for relative in ('diagnostic_probe/probe.json', 'diagnostic/PROCESS.json', 'diagnostic/window_plan.json',
                             'diagnostic/window-00.json', 'diagnostic/window-01.json', 'diagnostic/window-02.json',
                             'diagnostic/activity_verification.json'):
                path = root / 'output' / arm / relative
                if not path.exists():
                    continue
                require(path.stat().st_size <= 64 * 2**10, 'Oversized owned component status metadata')
                data = regular(path)
                try:
                    value = json.loads(data)
                except json.JSONDecodeError:
                    value = {'pending_file_write': True}
                observed[relative] = {'sha256': sha(data), 'value': value}
            result['component_diagnostics'][arm] = observed
    if ticket['request']['workload']['kind'] == 'native_tools_xqa_bf16_reference':
        result['native_tool_progress'] = {}
        for arm in ticket['request']['arms']:
            observed = {}
            for relative in ('bootstrap.json', 'admission_probe.json', 'native_tool_admission.json'):
                path = root / 'output' / arm / relative
                if not path.exists():
                    continue
                require(path.stat().st_size <= 64 * 2**10, 'Oversized owned native tool status metadata')
                data = regular(path)
                try:
                    value = json.loads(data)
                except json.JSONDecodeError:
                    value = {'pending_file_write': True}
                observed[relative] = {'sha256': sha(data), 'value': value}
            result['native_tool_progress'][arm] = observed
    return result


def decode_dataset_asset(ticket: dict, assets: list) -> bytes | None:
    """Expand only a bounded, byte-pinned trace inside its new owned run."""
    request = ticket['request']
    archive = request.get('dataset_archive')
    if archive is None:
        return None
    require(request['workload']['kind'] in ('anonymous_trace_xqa', 'anonymous_trace_xqa_bf16_reference',
                                           'anonymous_trace_component_diagnostic'),
            'Compressed dataset is only a trace input')
    codec = archive.get('codec','gzip')
    require(codec in ('gzip','xz'), 'Unknown trace compression codec')
    require(archive.get('relative_path') == 'input/payload_trace.jsonl.' + ('xz' if codec=='xz' else 'gz')
            and request['dataset_path'] == request['remote_output']+'/input/payload_trace.jsonl',
            'Compressed dataset target escaped this run')
    size = archive.get('uncompressed_size_bytes')
    require(type(size) is int and 0 < size <= 64*2**20, 'Unbounded decompressed trace')
    entries = {str(name): data for name,data in assets}
    require(archive['relative_path'] in entries and 'input/payload_trace.jsonl' not in entries,
            'Missing or ambiguous compressed trace')
    if codec == 'xz':
        import lzma
        decoder = lzma.LZMADecompressor(memlimit=128*2**20)
        decoded = decoder.decompress(entries[archive['relative_path']],max_length=size+1)
        require(decoder.eof and not decoder.unused_data, 'Truncated or concatenated XZ trace')
    else:
        with gzip.GzipFile(fileobj=io.BytesIO(entries[archive['relative_path']]), mode='rb') as stream:
            decoded = stream.read(size+1)
    require(len(decoded) == size and sha(decoded) == request['dataset_sha256'], 'Expanded trace bytes/SHA differ')
    return decoded


def dispatch(payload: dict) -> dict:
    operation, ticket = payload["operation"], payload["ticket"]
    require(operation in ("prepare", "start", "status", "stop"), "Unknown operation")
    # stop is permitted after expiry: it only ends an already authorized owned job.
    root = check_ticket(ticket, operation in ("prepare", "start"))
    if operation == "status":
        return status(root, ticket)
    if operation == "prepare":
        require(not root.exists(), "Run already exists; reconcile read-only, never retry preparation")
        total, assets = 0, []
        for name, entry in payload["assets"].items():
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == name
                    and relative.parts[0] in ("runtime", "input", "tools_source", "runtime_support"), "Unsafe asset path")
            data = base64.b64decode(entry["base64"], validate=True)
            require(sha(data) == entry["sha256"] == ticket["assets_sha256"][name], "Asset SHA mismatch")
            total += len(data)
            require(len(data) <= 32 * 1024 * 1024 and total <= 64 * 1024 * 1024, "Asset size cap exceeded")
            assets.append((relative, data))
        require(set(payload["assets"]) == set(ticket["assets_sha256"]), "Incomplete asset bundle")
        require(total <= int(ticket["request"]["max_output_gib"] * 2**30 * .5), "Assets consume too much output reservation")
        expanded_dataset = decode_dataset_asset(ticket, assets)
        private_directory(root.parent.parent)
        private_directory(root.parent)
        locks = root.parent.parent / "controller_gpu_locks"
        if not locks.exists():
            private_directory(locks)
            new_bytes(locks / "IDENTITY.json", encoded({"schema": "kvra-cooperative-gpu-locks-v1", "uid": os.getuid(),
                                                       "base": ticket["approval"]["approved_workspace"]}))
        private_directory(locks)
        require(json.loads(regular(locks / "IDENTITY.json")) == {"schema": "kvra-cooperative-gpu-locks-v1", "uid": os.getuid(),
                    "base": ticket["approval"]["approved_workspace"]}, "Unknown existing lock directory; never adopt")
        for uuid in ticket["request"]["gpu_uuids"]:
            require(re.fullmatch(r"GPU-[0-9a-f-]{36}", uuid) is not None, "Invalid lock GPU UUID")
            path = locks / (uuid + ".lock")
            if not path.exists():
                try:
                    new_bytes(path, b"")
                except FileExistsError:
                    pass
            require(regular(path) == b"", "Unknown GPU lock file")
        root.mkdir(mode=0o700)  # Atomic exclusive claim, never exist_ok.
        new_bytes(root / "ticket.json", encoded(ticket))
        for parent in ("output", "cache", "input", "runtime", "tools_source", "runtime_support"):
            private_directory(root / parent)
        for arm in ticket["request"]["arms"]:
            private_directory(root / "output" / arm); private_directory(root / "cache" / arm)
        for relative, data in assets:
            target = root / relative
            for parent in reversed(target.parent.relative_to(root).parents):
                if str(parent) != ".":
                    private_directory(root / parent)
            private_directory(target.parent)
            new_bytes(target, data)
        if expanded_dataset is not None:
            new_bytes(root / 'input/payload_trace.jsonl', expanded_dataset)
        if "python_source_repo" in ticket["request"]:
            request = ticket["request"]
            require(request["python_source_repo"] == str(root / "source_repo")
                    and re.fullmatch(r"[0-9a-f]{40}", request["python_source_commit"]) is not None,
                    "Production checkout path/commit differs")
            source = Path(request["source_repo"])
            require(source.resolve(strict=True) == source and source.stat().st_uid == os.getuid()
                    and str(source).startswith(ticket["approval"]["approved_workspace"] + "/"), "Unowned baseline checkout")
            bundle = root / "input/source.bundle"
            require(sha(regular(bundle)) == ticket["assets_sha256"]["input/source.bundle"], "Bundle hash differs")
            env = {"PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
                   "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": "file"}
            templates = root / "empty_git_templates"; private_directory(templates)
            common = ["git", "-c", "core.hooksPath=/dev/null", "-c", "init.templateDir=" + str(templates)]
            target = request["python_source_repo"]
            subprocess.run(common + ["clone", "--no-checkout", "--shared", str(source), target],
                           env=env, check=True, capture_output=True, timeout=90)
            subprocess.run(common + ["-C", target, "fetch", "--no-tags", "--recurse-submodules=no", str(bundle), request["python_source_commit"]],
                           env=env, check=True, capture_output=True, timeout=60)
            subprocess.run(common + ["-C", target, "checkout", "--detach", request["python_source_commit"]],
                           env=env, check=True, capture_output=True, timeout=90)
        new_bytes(root / "PREPARED.json", encoded({"request_sha256": ticket["request_sha256"], "assets": len(assets)}))
        return {"status": "PREPARED", "run_id": ticket["request"]["run_id"]}
    existing(root, ticket)
    if operation == "start":
        require((root / "PREPARED.json").is_file(), "Incomplete preparation; do not restart")
        for name, expected in ticket["assets_sha256"].items():
            require(sha(regular(root / name)) == expected, "Prepared asset changed")
        with (root / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            new_bytes(root / "START_INTENT.json", encoded({"request_sha256": ticket["request_sha256"]}))
            env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL") if key in os.environ}
            env.update(PYTHONPATH=str(root / "runtime"), PYTHONDONTWRITEBYTECODE="1")
            with (root / "supervisor.log").open("xb") as output:
                process = subprocess.Popen([sys.executable, "-B", "-m", "kv_controller.supervisor", "--ticket", str(root / "ticket.json")],
                                           stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                           env=env, cwd=root, start_new_session=True)
            captured = identity(process.pid)
            new_bytes(root / "SUPERVISOR_PROCESS.json", encoded({"identity": captured}))
        return {"status": "DISPATCHED", "identity": captured}
    # STOP_REQUESTED and pidfd SIGTERM are scoped to this exact immutable ticket.
    path = root / "STOP_REQUESTED.json"
    if not path.exists():
        new_bytes(path, encoded({"request_sha256": ticket["request_sha256"]}))
    process_file = root / "SUPERVISOR_STARTED.json"
    if not process_file.exists():
        process_file = root / "SUPERVISOR_PROCESS.json"
    if process_file.exists():
        expected = json.loads(regular(process_file))["identity"]
        require(expected["uid"] == os.getuid() and expected["pid"] > 1, "Unowned supervisor identity")
        try:
            fd = os.pidfd_open(expected["pid"])
        except ProcessLookupError:
            return {"status": "STOP_REQUESTED", "supervisor_alive": False}
        try:
            require(identity(expected["pid"]) == expected, "PID reused; refusing to signal")
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
    return {"status": "STOP_REQUESTED", "release_verified": False}


def main() -> None:
    os.umask(0o077)
    try:
        payload = json.load(sys.stdin)
        response = {"ok": True, **dispatch(payload)}
    except Exception as error:
        response = {"ok": False, "error": type(error).__name__ + ": " + str(error)}
    sys.stdout.write(json.dumps(response, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
