"""Persistent per-run remote supervisor, portable to Python 3.9.

Only an existing, immutable, user-authorized ticket can enter this module.
No SSH/network clients, system configuration changes, global cleanup or
arbitrary shell commands are provided. Tests inject a CPU-only backend.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

from .durable import atomic_json, canonical_bytes, digest, write_new_json
from .policy import PRODUCTION_SOURCES, XQA_SOURCES, docker_command, validate_execution_authority, validate_request
from .processes import LINUX_PF_EXITING, pidfd_available, process_identity, process_lifecycle, signal_exact


class UnsafeState(RuntimeError):
    pass


def verify_quality_parent(request: dict, base: Path, run_root: Path, uid: int) -> None:
    workload = request["workload"]
    if workload.get("cohort") != "completion_fragment":
        return
    parent_id = workload["reuse_run_id"]
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", parent_id) is None:
        raise UnsafeState("Unsafe parent quality run identifier")
    parent = base / "agent_work/runs" / parent_id
    def read(path: Path) -> tuple[dict, str]:
        if path.resolve(strict=True) != path or path.stat().st_uid != uid or path.stat().st_size > 2**20:
            raise UnsafeState("Unowned or noncanonical quality prerequisite")
        data = path.read_bytes()
        return json.loads(data), hashlib.sha256(data).hexdigest()
    result, result_sha = read(parent / "result.json")
    complete, _ = read(parent / "COMPLETE")
    prior, _ = read(parent / "ticket.json")
    audit, audit_sha = read(run_root / "input/parent_quality_audit.json")
    if (result_sha != workload["parent_result_sha256"] or complete.get("result_sha256") != result_sha
            or result.get("status") != "COMPLETE" or result.get("gpu_release_verified") is not True
            or result.get("live_owned_containers") != [] or len(result.get("arm_results", [])) != 2
            or result.get("request_sha256") != digest(prior["request"])
            or audit_sha != workload["parent_audit_sha256"] or audit.get("terminal_result_sha256") != result_sha
            or audit.get("classification") != "confirmed_pilot_prediction_integrity_and_rescoring"
            or audit.get("samples_per_arm") != 80 or audit.get("run_id") != parent_id):
        raise UnsafeState("Prior 80-record quality evidence is not complete and independently verified")
    for key in ("binary_sha256", "model_config_sha256", "operators", "service", "graph_mode", "fp8_scales", "concurrency"):
        if prior["request"][key] != request[key]:
            raise UnsafeState("Completion changed the previously evaluated production configuration: " + key)
    if prior["request"]["workload"].get("kind") != workload.get("kind"):
        raise UnsafeState("Completion cannot reuse a different quality implementation")
    sources = XQA_SOURCES if workload.get("kind") == "native_quality_xqa" else PRODUCTION_SOURCES
    if any(prior["request"]["backend_source_sha256"].get(name) != request["backend_source_sha256"].get(name)
           or name not in request["backend_source_sha256"] for name in sources):
        raise UnsafeState("Completion changed the actual model/attention implementation")


def verify_quality_recovery(request: dict, base: Path, run_root: Path, uid: int) -> None:
    recovery = request['workload'].get('recovery')
    if recovery is None:
        return
    prior_root = base/'agent_work/runs'/recovery['prior_run_id']
    def read(path: Path) -> tuple[dict, str]:
        if path.resolve(strict=True) != path or path.stat().st_uid != uid or path.stat().st_size > 2**20:
            raise UnsafeState('Unowned/oversized quality recovery prerequisite')
        data = path.read_bytes()
        return json.loads(data), hashlib.sha256(data).hexdigest()
    old, result_sha = read(prior_root/'result.json')
    prior_ticket, _ = read(prior_root/'ticket.json')
    recovered, audit_sha = read(run_root/'input/recovered_fragment_audit.json')
    arm = recovery['verified_arm']
    measurement, measurement_sha = read(prior_root/'output'/arm/'measurement.json')
    if (request['arms'] != ['bf16'] or arm != 'fp8_e4m3' or result_sha != recovery['prior_terminal_result_sha256']
            or old.get('status') != 'CONFLICT' or old.get('gpu_release_verified') is not True
            or old.get('live_owned_containers') != [] or (prior_root/'COMPLETE').exists()
            or old.get('request_sha256') != digest(prior_ticket['request'])
            or recovered.get('classification') != 'confirmed_recovered_fragment_arm_prediction_integrity_and_rescoring'
            or recovered.get('run_id') != recovery['prior_run_id'] or recovered.get('audited_arms') != [arm]
            or recovered.get('samples_per_arm') != 1639 or recovered.get('root_complete') is not False
            or recovered.get('terminal_result_sha256') != result_sha or audit_sha != recovery['recovered_audit_sha256']
            or not recovered.get('conflict_seconds_after_last_prediction', -1) > 0
            or measurement_sha != recovery['arm_measurement_sha256']
            or recovered.get('arm_measurement_sha256',{}).get(arm) != measurement_sha
            or measurement.get('measurement_success') is not True
            or measurement.get('request_sha256') != old['request_sha256']):
        raise UnsafeState('The previous FP8 fragment is not independently recovered and driver released')
    for key in ('binary_sha256','model_config_sha256','operators','service','graph_mode','fp8_scales',
                'concurrency','python_source_commit','backend_source_sha256','dataset_sha256','model_path','image_id'):
        if prior_ticket['request'][key] != request[key]:
            raise UnsafeState('Missing-arm recovery changed the prior engine/input: '+key)
    if (prior_root/'output/bf16/measurement.json').exists() or (prior_root/'output/bf16/quality').exists():
        raise UnsafeState('BF16 may already have run; do not duplicate uncertain predictions')
    for name, expected in measurement['evidence_sha256'].items():
        path = prior_root/'output'/arm/name
        if (not path.resolve(strict=True).is_relative_to(prior_root/'output'/arm)
                or path.resolve(strict=True) != path or path.stat().st_uid != uid or path.stat().st_size > 64*2**20
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise UnsafeState('Prior recovered fragment seal changed')


def tree_bytes(root: Path) -> int:
    # Progress checkpoints and compiler caches legitimately rename temporary
    # files between walk's directory listing and lstat. Retry the complete
    # accounting pass; never return zero or omit a raced file from quota use.
    for _ in range(3):
        total = 0
        try:
            for directory, folders, files in os.walk(root, followlinks=False):
                folders[:] = [name for name in folders if not (Path(directory) / name).is_symlink()]
                for name in files:
                    total += (Path(directory) / name).lstat().st_size
        except FileNotFoundError:
            if not root.is_dir():raise
            continue
        return total
    raise UnsafeState('Output accounting changed during three complete scans; no quota estimate accepted')


def gpu_snapshot() -> dict:
    def query(fields: str, apps: bool = False) -> str:
        option = "--query-compute-apps=" if apps else "--query-gpu="
        result = subprocess.run(["nvidia-smi", option + fields, "--format=csv,noheader,nounits"],
                                text=True, capture_output=True, timeout=10, check=True)
        return result.stdout

    devices = {}
    for line in query("index,uuid,memory.used,utilization.gpu").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4 or not all(fields[index].isdigit() for index in (0, 2, 3)):
            raise UnsafeState("Unparseable GPU visibility/driver response")
        index, uuid, memory, util = fields
        devices[uuid] = {"index": int(index), "memory_mib": int(memory), "util_pct": int(util), "pids": []}
    for line in query("gpu_uuid,pid", apps=True).splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 2 or fields[0] not in devices or not fields[1].isdigit():
            raise UnsafeState("Unparseable/unknown GPU process visibility")
        devices[fields[0]]["pids"].append(int(fields[1]))
    return devices


class DockerBackend:
    def __init__(self, ticket: dict, root: Path) -> None:
        self.ticket, self.root = ticket, root
        self.handles = {}
        self.processes = {}
        self.proofs = {}
        self.gpu_locks = []
        self.gpu_process_proofs = {}
        self.last_gpu_observation = {}

    def _registry(self) -> None:
        atomic_json(self.root / "owned_containers.json", {
            "reservation_id": self.ticket["reservation_id"], "request_sha256": self.ticket["request_sha256"],
            "containers": [{"container_id": identifier, "pid_identity": proof["pid_identity"],
                             "released": proof.get("released", False)} for identifier, proof in self.proofs.items()
                            if proof["pid_identity"] is not None]})

    def watchdog_healthy(self) -> bool:
        path = self.root / "WATCHDOG_READY.json"
        if not path.is_file() or (self.root / "watchdog_result.json").exists():
            return False
        value = json.loads(path.read_text())
        return (value.get("request_sha256") == self.ticket["request_sha256"]
                and signal_exact(value["identity"], 0))

    def snapshot(self) -> dict:
        return gpu_snapshot()

    def reservation_clear(self) -> bool:
        policy = self.ticket["approval"]["shared_gpu_policy"]
        if policy["kind"] == "opportunistic_no_scheduler":
            return policy["user_confirms_no_external_reservations"] is True
        if policy["kind"] == "idle_only_standing_user":
            # Standing user authority permits opportunistic idle use; any discovered reservation still blocks it.
            return not any(Path(path).exists() for path in policy.get("known_reservation_markers", []))
        path = Path(policy["path"])
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise UnsafeState("Reservation source changed path")
        data = json.loads(path.read_text())
        observed = datetime.datetime.fromisoformat(data["observed_utc"].replace("Z", "+00:00"))
        age = (datetime.datetime.now(datetime.timezone.utc) - observed).total_seconds()
        if not 0 <= age <= 60 or data.get("complete") is not True:
            raise UnsafeState("Reservation source is stale/incomplete")
        return not set(data["reserved_gpu_uuids"]).intersection(self.ticket["request"]["gpu_uuids"])

    def verify_preflight(self) -> None:
        ticket = self.ticket
        approval, request = ticket["approval"], ticket["request"]
        validate_execution_authority(approval, datetime.datetime.now(datetime.timezone.utc))
        validate_request(request, approval)
        identity = approval["approved_remote_identity"]
        if os.getuid() != identity["uid"] or os.getgid() != identity["gid"] or socket.gethostname() != identity["hostname"]:
            raise UnsafeState("Remote identity changed")
        if not pidfd_available():
            raise UnsafeState("Identity-safe process deadline fallback unavailable")
        base = Path(approval["approved_workspace"])
        if base.resolve(strict=True) != base or base.stat().st_uid != identity["uid"]:
            raise UnsafeState("Approved project realpath/ownership changed")
        verify_quality_parent(request, base, self.root, identity["uid"])
        verify_quality_recovery(request, base, self.root, identity['uid'])
        from .nextgen import verify_startup_recovery
        verify_startup_recovery(request, base, identity['uid'])
        if not self.gpu_locks:
            for uuid in request["gpu_uuids"]:
                path = base / "agent_work/controller_gpu_locks" / (uuid + ".lock")
                if path.resolve(strict=True) != path or path.stat().st_uid != os.getuid():
                    raise UnsafeState("Cooperative GPU lock is unowned or symlinked")
                handle = path.open("r")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    handle.close()
                    raise UnsafeState("Another project controller holds this GPU; no model admitted")
                self.gpu_locks.append(handle)
        statmem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(statmem["MemAvailable"].strip().split()[0]) * 1024
        if available < (approval["container_limits"]["memory_gib"] + 16) * 2**30:
            raise UnsafeState("Insufficient host RAM headroom for a bounded container without pressuring other users")
        statfs = os.statvfs(base)
        if statfs.f_bavail * statfs.f_frsize < int(request["max_output_gib"] * 2**30) + 5 * 2**30:
            raise UnsafeState("Insufficient shared disk headroom")
        for field, expected in (("server_binary", "binary_sha256"), ("dataset_path", "dataset_sha256")):
            path = Path(request[field])
            if path.resolve(strict=True) != path:
                raise UnsafeState("Source path changed to symlink")
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != request[expected]:
                raise UnsafeState("Source/binary hash drift")
        for path, expected in [(Path(request["model_path"]) / "config.json", request["model_config_sha256"])] + [
            (Path(request.get("python_source_repo", request["source_repo"])) / name, expected)
            for name, expected in request["backend_source_sha256"].items()]:
            if path.resolve(strict=True) != path or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise UnsafeState("Model/backend/Graph source drift")
        head = subprocess.check_output(["git", "--no-optional-locks", "-C", request["source_repo"],
                                        "rev-parse", "HEAD"], text=True, timeout=10).strip()
        if head != request["server_source_commit"]:
            raise UnsafeState("Model runtime worktree HEAD drift")
        if "python_source_repo" in request:
            head = subprocess.check_output(["git", "--no-optional-locks", "-C", request["python_source_repo"],
                                            "rev-parse", "HEAD"], text=True, timeout=10).strip()
            if head != request["python_source_commit"]:
                raise UnsafeState("Pinned production Python checkout HEAD drift")
        image = json.loads(subprocess.check_output(["docker", "image", "inspect", request["image_id"]],
                                                   text=True, timeout=10))[0]
        if image["Id"] != request["image_id"]:
            raise UnsafeState("Pinned Docker image is absent/changed; never pull")

    def _inspect(self, identifier: str) -> dict:
        result = subprocess.run(["docker", "inspect", identifier], capture_output=True, text=True,
                                timeout=10, check=True)
        value = json.loads(result.stdout)[0]
        expected = self.proofs.get(identifier)
        if value["Id"] != identifier or (expected is not None and value["Created"] != expected["created"]):
            raise UnsafeState("Container identity was replaced")
        labels = value["Config"].get("Labels") or {}
        for key, selected in {"kvra.run_id": self.ticket["request"]["run_id"],
                              "kvra.reservation_id": self.ticket["reservation_id"],
                              "kvra.request_sha256": self.ticket["request_sha256"]}.items():
            if labels.get(key) != selected:
                raise UnsafeState("Container belongs to a different run; no mutation permitted")
        identity = self.ticket["approval"]["approved_remote_identity"]
        if value["Config"]["User"] != str(identity["uid"]) + ":" + str(identity["gid"]):
            raise UnsafeState("Container UID/GID changed")
        if value["Image"] != self.ticket["request"]["image_id"]:
            raise UnsafeState("Container image differs from pinned identity")
        host = value["HostConfig"]
        if host.get("Privileged") or host.get("NetworkMode") != "none" or host.get("IpcMode") != "private":
            raise UnsafeState("Container privilege/network/isolation drift")
        device_ids = [gpu for group in host.get("DeviceRequests", []) for gpu in (group.get("DeviceIDs") or [])]
        if device_ids != self.ticket["request"]["gpu_uuids"]:
            raise UnsafeState("Container device UUID visibility differs")
        for mount in value.get("Mounts", []):
            if mount.get("Type") == "tmpfs" and mount.get("Destination") in ("/tmp", "/dev/shm"):
                continue  # Private bounded tmpfs; not a writable host path.
            if mount.get("RW") and not str(mount["Source"]).startswith(str(self.root) + "/"):
                raise UnsafeState("Container writable mount escaped its unique run")
        return value

    def create(self, arm: str, device_index: int) -> str:
        command = docker_command(self.ticket, arm, device_index)
        write_new_json(self.root / "output" / arm / "docker_command.json", command)
        identifier = subprocess.check_output(command, text=True, timeout=30).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise UnsafeState("Docker create returned no exact immutable container ID")
        value = self._inspect(identifier)
        self.proofs[identifier] = {"created": value["Created"], "arm": arm, "pid_identity": None}
        return identifier

    def start(self, identifier: str, arm: str) -> None:
        self._inspect(identifier)
        output = (self.root / "output" / arm / "container.log").open("xb")
        self.handles[identifier] = output
        self.processes[identifier] = subprocess.Popen(["docker", "start", "--attach", identifier],
                                                      stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            value = self._inspect(identifier)
            if value["State"]["Running"]:
                proof = process_identity(value["State"]["Pid"])
                if proof["uid"] != os.getuid() or identifier not in proof["cgroup"]:
                    raise UnsafeState("Cannot bind the owned container to host PID/start/UID/cgroup")
                self.proofs[identifier]["pid_identity"] = proof
                self._registry()  # Durable watchdog target BEFORE permitting any model load.
                write_new_json(self.root / "output" / arm / "container_identity.json",
                               {"id": identifier, **self.proofs[identifier]})
                if not self.watchdog_healthy():
                    raise UnsafeState("Independent deadline watchdog is not alive")
                write_new_json(self.root / "output" / arm / "lease.json",
                               {"request_sha256": self.ticket["request_sha256"],
                                "reservation_id": self.ticket["reservation_id"]})
                return
            if value["State"].get("Status") in ("exited", "dead"):
                raise UnsafeState("Worker exited before supervised admission")
            time.sleep(0.1)
        raise UnsafeState("Docker start/admission became uncertain")

    def running(self, identifier: str) -> bool:
        return self._inspect(identifier)["State"]["Running"] is True

    def finish(self, identifier: str, arm: str) -> dict:
        value = self._inspect(identifier)
        if value["State"]["Running"]:
            raise UnsafeState("Cannot finalize a live owned container")
        process = self.processes.get(identifier)
        if process is not None:
            process.wait(timeout=10)
            self.handles[identifier].close()
        if value["State"]["ExitCode"] != 0:
            raise RuntimeError("Worker failed; inspect preserved per-arm container.log")
        result_path = self.root / "output" / arm / "measurement.json"
        result = json.loads(result_path.read_text())
        if (result.get("measurement_success") is not True or result.get("arm") != arm
                or result.get("request_sha256") != self.ticket["request_sha256"]):
            raise UnsafeState("Worker exit zero without matching complete measurement evidence")
        return {"arm": arm, "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()}

    def foreign_gpu(self, snapshot: dict, owned: list[str]) -> bool:
        self.last_gpu_observation = {"selected": {uuid: snapshot.get(uuid) for uuid in self.ticket["request"]["gpu_uuids"]},
                                     "owned_containers": list(owned)}
        return self._foreign_gpu_processes(snapshot, owned, allow_retired_recheck=bool(owned))

    def _foreign_gpu_processes(self, snapshot: dict, owned: list[str], *, allow_retired_recheck: bool) -> bool:
        for uuid in self.ticket["request"]["gpu_uuids"]:
            if uuid not in snapshot:
                raise UnsafeState("GPU UUID disappeared")
            for pid in snapshot[uuid]["pids"]:
                try:
                    proof = process_identity(pid)
                except FileNotFoundError as error:
                    prior = self.gpu_process_proofs.get((uuid, pid))
                    # NVML can return an owned server PID immediately after it exits.
                    # Only a previously proven owned process gets one fresh read.
                    # Unknown PIDs and all pre-launch races still fail closed.
                    if (allow_retired_recheck and prior is not None and prior["uid"] == os.getuid()
                            and any(identifier in prior["cgroup"] for identifier in owned)):
                        fresh = self.snapshot()
                        self.last_gpu_observation.update(retired_owned_pid=prior,
                            fresh_selected={key: fresh.get(key) for key in self.ticket["request"]["gpu_uuids"]})
                        if uuid in fresh and pid not in fresh[uuid]["pids"]:
                            return self._foreign_gpu_processes(fresh, owned, allow_retired_recheck=False)
                    self.last_gpu_observation.update(reason="unresolved_gpu_pid_visibility_race",
                        unresolved_gpu_pid=pid, previous_process_proof=prior,
                        missing_proc_file=error.filename)
                    return True
                if proof["uid"] != os.getuid() or not any(identifier in proof["cgroup"] for identifier in owned):
                    prior = self.gpu_process_proofs.get((uuid, pid))
                    self.last_gpu_observation.update(reason="live_unowned_gpu_process", process=proof,
                                                     previous_process_proof=prior)
                    # do_exit sets PF_EXITING before cgroup teardown, even while
                    # proc still reports R. An ordinary live cgroup change is
                    # never ownership proof. Only this exact previously owned
                    # identity, with positive kernel exit evidence, may wait
                    # briefly for actual driver removal.
                    matched = (prior is not None and all(prior.get(k) == proof.get(k)
                               for k in ('pid', 'uid', 'start_ticks', 'boot_id'))
                               and proof['uid'] == os.getuid()
                               and any(identifier in prior['cgroup'] for identifier in owned))
                    if allow_retired_recheck and matched:
                        try:
                            lifecycle = process_lifecycle(pid)
                        except FileNotFoundError:
                            lifecycle = None
                        self.last_gpu_observation['process_lifecycle'] = lifecycle
                        flags = lifecycle.get('kernel_flags', 0) if lifecycle is not None else 0
                        kernel_exiting = type(flags) is int and flags >= 0 and bool(flags & LINUX_PF_EXITING)
                        exited = (lifecycle is None or (lifecycle.get('state') == 'Z' or kernel_exiting) and
                                  all(lifecycle.get(k) == proof.get(k) for k in ('pid', 'uid', 'start_ticks')))
                        if exited:
                            # At most3 fresh reads/1s of waiting; driver removal
                            # remains mandatory. No new GPU work or signals here.
                            for attempt in range(3):
                                fresh = self.snapshot()
                                self.last_gpu_observation.update(retired_owned_pid=prior,
                                    fresh_selected={key:fresh.get(key) for key in self.ticket['request']['gpu_uuids']})
                                if uuid in fresh and pid not in fresh[uuid]['pids']:
                                    return self._foreign_gpu_processes(fresh, owned, allow_retired_recheck=False)
                                if uuid not in fresh or any(p != pid for p in fresh[uuid]['pids']):
                                    break
                                if attempt < 2:
                                    time.sleep(.5)
                    return True
                self.gpu_process_proofs[(uuid, pid)] = proof
        return False

    def stop(self, identifier: str) -> bool:
        try:
            value = self._inspect(identifier)
        except (subprocess.SubprocessError, OSError):
            expected = self.proofs[identifier]["pid_identity"]
            if expected is not None:
                signal_exact(expected, signal.SIGKILL)
            return False  # No Docker release proof: never free this reservation.
        if not value["State"]["Running"]:
            if not self._gpu_released():
                return False
            self.proofs[identifier]["released"] = True
            self._registry()
            return True
        expected = self.proofs[identifier]["pid_identity"]
        if expected is not None:
            # Validate before asking Docker to mutate this exact container ID.
            current = process_identity(value["State"]["Pid"])
            if current != expected:
                raise UnsafeState("Container PID was reused; refusing to signal any replacement")
        try:
            subprocess.run(["docker", "stop", "--time", "10", identifier], check=True,
                           capture_output=True, timeout=18)
        except (subprocess.SubprocessError, OSError):
            if expected is None:
                return False
            signal_exact(expected, signal.SIGKILL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if not self._inspect(identifier)["State"]["Running"]:
                    if not self._gpu_released():
                        return False
                    self.proofs[identifier]["released"] = True
                    self._registry()
                    return True
            except (subprocess.SubprocessError, OSError):
                return False
            time.sleep(0.2)
        if expected is not None:
            signal_exact(expected, signal.SIGKILL)
        released = not self._inspect(identifier)["State"]["Running"] and self._gpu_released()
        if released:
            self.proofs[identifier]["released"] = True
            self._registry()
        return released

    def _gpu_released(self) -> bool:
        until = time.monotonic() + 10
        while time.monotonic() < until:
            snapshot = self.snapshot()
            if all(uuid in snapshot and snapshot[uuid]["memory_mib"] == 0 and not snapshot[uuid]["pids"]
                   for uuid in self.ticket["request"]["gpu_uuids"]):
                return True
            time.sleep(.2)
        return False  # Container exit without actual driver release is not sufficient.

    def output_bytes(self) -> int:
        return tree_bytes(self.root)


class Supervisor:
    def __init__(self, ticket: dict, root: Path, backend: object,
                 clock_ns: object = time.monotonic_ns, sleeper: object = time.sleep,
                 wall_now: object = lambda: datetime.datetime.now(datetime.timezone.utc)) -> None:
        self.ticket, self.root, self.backend = ticket, root, backend
        self.clock_ns, self.sleep, self.wall_now = clock_ns, sleeper, wall_now
        self.containers = []
        self.intervals = []
        self.results = []
        self.released_containers = set()
        self.stop_requested = False

    def _event(self, action: str, data: dict) -> None:
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(canonical_bytes({"utc": self.wall_now().isoformat(), "action": action, **data}).decode())
            stream.flush()
            os.fsync(stream.fileno())

    def _idle(self, stop_at: int) -> dict:
        from .idle_admission import idle_parameters
        stable_seconds,poll_seconds=idle_parameters(self.ticket['request'])
        since = None
        previous=None
        last_event=None
        selected = self.ticket["request"]["gpu_uuids"]
        while self.clock_ns() < stop_at:
            if self.stop_requested or (self.root / "STOP_REQUESTED.json").exists():
                raise TimeoutError("Stop requested while waiting; no worker will be admitted")
            if not self.backend.watchdog_healthy():
                raise UnsafeState("Deadline watchdog died while waiting for idle")
            snapshot = self.backend.snapshot()
            clear = self.backend.reservation_clear()
            idle = clear and all(uuid in snapshot and snapshot[uuid]["memory_mib"] == 0
                                 and snapshot[uuid]["util_pct"] <= 5 and not snapshot[uuid]["pids"] for uuid in selected)
            now = self.clock_ns()
            since = (now if since is None else since) if idle else None
            signature=(clear,tuple((uuid,snapshot.get(uuid,{}).get('memory_mib',-1)==0,
                                    tuple(snapshot.get(uuid,{}).get('pids',[])),
                                    snapshot.get(uuid,{}).get('util_pct',100)<=5) for uuid in selected))
            if stable_seconds or signature!=previous or last_event is None or now-last_event>=1200*1000000000:
                self._event("WAIT_GPU", {"selected": {uuid: snapshot.get(uuid) for uuid in selected},
                                          "reservation_policy_clear": clear, "stable_since_ns": since,
                                          "stable_idle_seconds":stable_seconds,"poll_seconds":poll_seconds})
                previous,last_event=signature,now
            if since is not None and now - since >= stable_seconds * 1000000000:
                return snapshot
            self.sleep(poll_seconds)
        raise TimeoutError("Budget expired while waiting for "+str(stable_seconds)+"s genuinely idle GPUs")

    def run(self) -> dict:
        request = self.ticket["request"]
        end = datetime.datetime.fromisoformat(self.ticket["deadline_utc"].replace("Z", "+00:00"))
        remaining = min(request["max_wall_seconds"], (end - self.wall_now()).total_seconds())
        deadline = self.clock_ns() + int(max(0, remaining) * 1000000000)
        stop_at = deadline - 60 * 1000000000  # Reserve whole-run cleanup time before hard cap.
        status = "NOT_LAUNCHED"
        error = None
        try:
            for arm in request["arms"]:
                self._idle(stop_at)
                self.backend.verify_preflight()
                if self.clock_ns() >= stop_at:
                    raise TimeoutError("No runtime budget remains after preflight")
                snapshot = self.backend.snapshot()
                if (not self.backend.reservation_clear() or self.backend.foreign_gpu(snapshot, [])
                        or any(snapshot[uuid]["memory_mib"] != 0 or snapshot[uuid]["util_pct"] > 5
                               for uuid in request["gpu_uuids"])):
                    raise UnsafeState("GPU state changed between stable idle and worker create")
                identifier = self.backend.create(arm, snapshot[request["gpu_uuids"][0]]["index"])
                self.containers.append(identifier)
                interval = {"arm": arm, "start_ns": self.clock_ns(), "end_ns": None}
                self.intervals.append(interval)
                self._event("OWNED_CONTAINER_CREATED", {"id": identifier, "arm": arm})
                self.backend.start(identifier, arm)
                atomic_json(self.root / "state.json", {"status": "RUNNING", "arm": arm,
                                                       "containers": self.containers,
                                                       "request_sha256": self.ticket["request_sha256"]})
                while self.backend.running(identifier):
                    if self.stop_requested or self.clock_ns() >= stop_at or (self.root / "STOP_REQUESTED.json").exists():
                        status = "TIMED_OUT"
                        raise TimeoutError("Owned run deadline/stop requested")
                    if self.backend.output_bytes() >= int(request["max_output_gib"] * 2**30 * 0.9):
                        status = "DISK_LIMIT"
                        raise UnsafeState("New output reached stop threshold before disk reservation cap")
                    reservations_clear = self.backend.reservation_clear()
                    conflict = self.backend.foreign_gpu(self.backend.snapshot(), [identifier])
                    observation = getattr(self.backend, "last_gpu_observation", {})
                    if not reservations_clear or conflict:
                        self._event("GPU_CONFLICT_OBSERVATION", {"reservation_clear": reservations_clear, **observation})
                        status = "CONFLICT"
                        raise UnsafeState("Foreign GPU activity/reservation; stop this run only")
                    if observation.get("retired_owned_pid") is not None:
                        self._event("OWNED_GPU_PID_EXIT_RECHECK", observation)
                    if not self.backend.watchdog_healthy():
                        raise UnsafeState("Independent deadline watchdog died")
                    self.sleep(1)
                self.results.append(self.backend.finish(identifier, arm))
                if not self.backend.stop(identifier):
                    raise UnsafeState("Arm container exited without proven driver-memory release")
                self.released_containers.add(identifier)
                interval["end_ns"] = self.clock_ns()
                self._event("ARM_VERIFIED", {"arm": arm, "id": identifier})
            status = "COMPLETE"
        except Exception as exception:
            error = type(exception).__name__ + ": " + str(exception)
            if status == "NOT_LAUNCHED" and self.containers:
                status = "FAILED"
            self._event("STOP_OR_FAILURE", {"status": status, "error": error,
                                             "traceback": traceback.format_exc(limit=8)})
        released = True
        live = []
        for identifier in self.containers:
            if identifier in self.released_containers:
                continue
            try:
                if not self.backend.stop(identifier):
                    released = False
                    live.append(identifier)
            except Exception as exception:
                released = False
                live.append(identifier)
                self._event("OWNERSHIP_OR_RELEASE_UNKNOWN", {"id": identifier, "error": str(exception)})
        if released:
            for interval in self.intervals:
                if interval["end_ns"] is None:
                    interval["end_ns"] = self.clock_ns()
        else:
            status = "UNKNOWN"
        result = {"run_id": request["run_id"], "reservation_id": self.ticket["reservation_id"],
                  "request_sha256": self.ticket["request_sha256"], "status": status, "error": error,
                  "gpu_release_verified": released, "live_owned_containers": live,
                  "allocation_started_monotonic_ns": self.intervals[0]["start_ns"] if self.intervals else None,
                  "allocation_ended_monotonic_ns": self.intervals[-1]["end_ns"] if self.intervals else None,
                  "allocation_intervals_ns": self.intervals,
                  "new_output_bytes": self.backend.output_bytes(), "arm_results": self.results}
        # Metadata is small but its size must be included in final disk accounting.
        atomic_json(self.root / "result.json", result)
        result["new_output_bytes"] = self.backend.output_bytes() + 4096
        atomic_json(self.root / "result.json", result)
        atomic_json(self.root / "state.json", result)
        if released:
            write_new_json(self.root / "WATCHDOG_FINISHED.json", {
                "request_sha256": self.ticket["request_sha256"], "gpu_release_verified": True,
                "result_sha256": hashlib.sha256((self.root / "result.json").read_bytes()).hexdigest()})
        if status == "COMPLETE" and released:
            write_new_json(self.root / "COMPLETE", {"result_sha256": hashlib.sha256((self.root / "result.json").read_bytes()).hexdigest()})
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticket", type=Path, required=True)
    args = parser.parse_args()
    ticket = json.loads(args.ticket.read_text())
    root = args.ticket.parent.resolve(strict=True)
    if root != Path(ticket["request"]["remote_output"]) or digest(ticket["request"]) != ticket["request_sha256"]:
        raise UnsafeState("Ticket/root identity mismatch")
    if args.ticket.is_symlink() or root.stat().st_uid != os.getuid():
        raise UnsafeState("Unowned ticket directory")
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A supervisor can observe a prior result, never launch that run again.
        write_new_json(root / "SUPERVISOR_STARTED.json", {"identity": process_identity(os.getpid()),
                                                         "ticket_sha256": hashlib.sha256(args.ticket.read_bytes()).hexdigest()})
        remaining = (datetime.datetime.fromisoformat(ticket["deadline_utc"].replace("Z", "+00:00"))
                     - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        kill_at = time.monotonic_ns() + int(max(0, remaining - 5) * 1000000000)
        write_new_json(root / "watchdog_config.json", {"request_sha256": ticket["request_sha256"],
                       "reservation_id": ticket["reservation_id"], "kill_at_monotonic_ns": kill_at})
        with (root / "watchdog.log").open("xb") as output:
            guard = subprocess.Popen([sys.executable, "-B", "-m", "kv_controller.watchdog", "--config",
                                      str(root / "watchdog_config.json")], stdout=output,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        backend = DockerBackend(ticket, root)
        wait_until = time.monotonic() + 5
        while not (root / "WATCHDOG_READY.json").exists() and time.monotonic() < wait_until and guard.poll() is None:
            time.sleep(0.05)
        if not backend.watchdog_healthy():
            raise UnsafeState("Watchdog startup failed; no GPU worker admitted")
        supervisor = Supervisor(ticket, root, backend)
        signal.signal(signal.SIGTERM, lambda number, frame: setattr(supervisor, "stop_requested", True))
        result = supervisor.run()
    raise SystemExit(0 if result["status"] == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
