"""Transactional GPU-time/disk reservations. UNKNOWN never releases capacity.

The production controller uses one fixed ledger path. Tests use temporary paths.
SQLite FULL transactions and a durable identity anchor prevent silent resets.
This is application accounting, not an OS resource sandbox.
"""

from __future__ import annotations

import contextlib
import datetime
from decimal import Decimal, ROUND_FLOOR
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import uuid

from .approval import ApprovalError, validate_scope
from .durable import canonical_bytes, digest, secure_directory, write_new_json


ACTIVE = ("RESERVED", "DISPATCHING", "RUNNING", "UNKNOWN")
TERMINAL_REMOTE = ("COMPLETE", "FAILED", "TIMED_OUT", "CONFLICT", "DISK_LIMIT", "NOT_LAUNCHED")


def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ApprovalError("Invalid nonnegative integer: " + name)
    return value


def budget_key(approval: dict) -> str:
    """Notes/expiry edits never reset usage. A new budget_id requires user issuance."""
    identity = {key: approval[key] for key in
                ("ssh_target", "approved_host_fingerprint", "approved_workspace")}
    if "budget_id" in approval:
        identifier = approval["budget_id"]
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128:
            raise ApprovalError("Invalid user-issued budget_id")
        identity["budget_id"] = identifier
    return digest(identity)


def _gpu_limit_us(hours: object) -> int:
    # Avoid float rounding upward at an authorization boundary.
    value = Decimal(str(hours)) * Decimal(3600000000)
    if not value.is_finite() or value <= 0 or value > Decimal(2**63 - 1):
        raise ApprovalError("GPU-hours outside integer accounting range")
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


class QuotaLedger:
    def __init__(self, directory: Path) -> None:
        secure_directory(directory)
        self.directory = directory.resolve(strict=True)
        self.path = self.directory / "ledger.sqlite3"
        anchor = self.directory / "IDENTITY.json"
        lock_fd = os.open(self.directory / "setup.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(lock_fd, "r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if anchor.exists() and not self.path.is_file():
                raise ApprovalError("Ledger vanished after initialization; refusing to reset quota")
            if self.path.is_symlink() or anchor.is_symlink():
                raise ApprovalError("Ledger/identity anchor must not be symlinked")
            if self.path.exists() and not anchor.exists():
                raise ApprovalError("Unanchored existing ledger requires reconciliation")
            new = not self.path.exists()
            for path in (self.path, anchor):
                if path.exists():
                    info = path.lstat()
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                        raise ApprovalError("Existing ledger/anchor has unsafe ownership or permissions")
            ledger_id = uuid.uuid4().hex if new else json.loads(anchor.read_text())["ledger_id"]
            self.connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            if new:
                os.chmod(self.path, 0o600)
                self.connection.executescript("""
                    CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY, budget_key TEXT NOT NULL,
                        reservation_id TEXT NOT NULL UNIQUE, approval_sha256 TEXT NOT NULL,
                        request_sha256 TEXT NOT NULL, request_json TEXT NOT NULL,
                        experiment_key TEXT NOT NULL, replicate INTEGER NOT NULL,
                        state TEXT NOT NULL, created_utc TEXT NOT NULL,
                        reserved_gpu_us INTEGER NOT NULL, reserved_disk_bytes INTEGER NOT NULL,
                        actual_gpu_us INTEGER, actual_disk_bytes INTEGER,
                        evidence_json TEXT, quota_breach INTEGER NOT NULL DEFAULT 0,
                        UNIQUE(experiment_key, replicate)
                    );
                    CREATE TABLE history (run_id TEXT PRIMARY KEY, manifest_sha256 TEXT NOT NULL);
                    CREATE TABLE events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        utc TEXT NOT NULL, run_id TEXT, action TEXT NOT NULL, data_json TEXT NOT NULL
                    );
                """)
                self.connection.execute("INSERT INTO metadata VALUES ('ledger_id', ?)", (ledger_id,))
                write_new_json(anchor, {"schema": "kv-quota-anchor-v1", "ledger_id": ledger_id})
            stored = self.connection.execute("SELECT value FROM metadata WHERE key='ledger_id'").fetchone()
            if stored is None or stored[0] != ledger_id:
                raise ApprovalError("Ledger and durable identity anchor disagree")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> QuotaLedger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @contextlib.contextmanager
    def _transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise

    def _event(self, run_id: str, action: str, data: dict) -> None:
        self.connection.execute("INSERT INTO events(utc,run_id,action,data_json) VALUES(?,?,?,?)",
                                (_utc(), run_id, action, canonical_bytes(data).decode()))

    def register_history(self, manifests: dict[str, str]) -> None:
        with self._transaction():
            for run_id, sha in manifests.items():
                prior = self.connection.execute("SELECT manifest_sha256 FROM history WHERE run_id=?", (run_id,)).fetchone()
                if prior and prior[0] != sha:
                    raise ApprovalError("Historical manifest changed")
                self.connection.execute("INSERT OR IGNORE INTO history VALUES (?,?)", (run_id, sha))

    def _usage(self, key: str) -> dict:
        all_runs = self.connection.execute("SELECT * FROM runs").fetchall()
        active = [row for row in all_runs if row["state"] in ACTIVE]
        budget_runs = [row for row in all_runs if row["budget_key"] == key]
        budget_active = [row for row in budget_runs if row["state"] in ACTIVE]
        settled = [row for row in budget_runs if row["state"] == "SETTLED"]
        return {
            "known_run_ids": [row["run_id"] for row in all_runs]
                             + [row[0] for row in self.connection.execute("SELECT run_id FROM history")],
            "reconciled": True,
            "uncertain_runs": [row["run_id"] for row in active if row["state"] in ("UNKNOWN", "DISPATCHING")],
            "active_gpu_uuids": [gpu for row in active for gpu in json.loads(row["request_json"])["gpu_uuids"]],
            "gpu_hours_consumed": sum(row["actual_gpu_us"] for row in settled) / 3600000000,
            "gpu_hours_reserved": sum(row["reserved_gpu_us"] for row in budget_active) / 3600000000,
            "disk_gib_consumed": sum(row["actual_disk_bytes"] for row in settled) / 2**30,
            "disk_gib_reserved": sum(row["reserved_disk_bytes"] for row in budget_active) / 2**30,
            "quota_breach": any(row["quota_breach"] for row in all_runs),
        }

    def usage(self, approval: dict) -> dict:
        with self._transaction():
            return self._usage(budget_key(approval))

    def reserve(self, approval: dict, request: dict, approval_sha256: str,
                now: datetime.datetime) -> dict:
        with self._transaction():
            key = budget_key(approval)
            usage = self._usage(key)
            if usage["quota_breach"]:
                raise ApprovalError("Prior quota breach requires user review")
            validate_scope(approval, request, usage, now)
            seconds = request["max_wall_seconds"]
            if type(seconds) is not int or not 1 <= seconds <= 86400:
                raise ApprovalError("Runtime wall limit must be integer seconds within one day")
            gpu_us = seconds * len(request["gpu_uuids"]) * 1000000
            if gpu_us > _gpu_limit_us(approval["max_gpu_hours_per_job"]):
                raise ApprovalError("Per-job integer GPU budget exceeded")
            current = self.connection.execute(
                "SELECT state,reserved_gpu_us,actual_gpu_us FROM runs WHERE budget_key=?", (key,)).fetchall()
            charged = sum(row["reserved_gpu_us"] if row["state"] in ACTIVE else
                          (row["actual_gpu_us"] or 0) for row in current)
            if charged + gpu_us > _gpu_limit_us(approval["max_gpu_hours_total"]):
                raise ApprovalError("Total integer GPU budget exceeded")
            experiment = request.get("experiment_key")
            replicate = request.get("replicate")
            if (not isinstance(experiment, str) or len(experiment) != 64
                    or any(c not in "0123456789abcdef" for c in experiment)
                    or type(replicate) is not int or replicate < 0):
                raise ApprovalError("Scientific experiment key and explicit replicate required")
            reservation = uuid.uuid4().hex
            disk = int(Decimal(str(request["max_output_gib"])) * Decimal(2**30))
            disk_cap = int(Decimal(str(approval["max_disk_gib_new_outputs"])) * Decimal(2**30))
            disk_rows = self.connection.execute(
                "SELECT state,reserved_disk_bytes,actual_disk_bytes FROM runs WHERE budget_key=?", (key,)).fetchall()
            disk_charged = sum(item["reserved_disk_bytes"] if item["state"] in ACTIVE else
                               (item["actual_disk_bytes"] or 0) for item in disk_rows)
            if disk <= 0 or disk + disk_charged > disk_cap or disk > 2**63 - 1:
                raise ApprovalError("Integer output disk budget exceeded")
            try:
                self.connection.execute("""INSERT INTO runs
                    (run_id,budget_key,reservation_id,approval_sha256,request_sha256,request_json,
                     experiment_key,replicate,state,created_utc,reserved_gpu_us,reserved_disk_bytes)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (request["run_id"], key, reservation, approval_sha256, digest(request),
                     canonical_bytes(request).decode(), experiment, replicate, "RESERVED", _utc(), gpu_us, disk))
            except sqlite3.IntegrityError as error:
                raise ApprovalError("Run or scientific replicate already registered; never relaunch") from error
            self._event(request["run_id"], "RESERVED", {"gpu_us": gpu_us, "disk_bytes": disk})
            return self.get(request["run_id"])

    def get(self, run_id: str) -> dict:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ApprovalError("Unknown local run ID")
        return dict(row)

    def transition(self, run_id: str, state: str, evidence: dict) -> None:
        allowed = {"DISPATCHING": {"RESERVED"}, "RUNNING": {"DISPATCHING", "UNKNOWN", "RUNNING"},
                   "UNKNOWN": {"DISPATCHING", "RUNNING", "UNKNOWN"}, "CANCELLED": {"RESERVED"}}
        with self._transaction():
            row = self.get(run_id)
            if state not in allowed or row["state"] not in allowed[state]:
                raise ApprovalError("Illegal state transition; terminal runs cannot relaunch")
            self.connection.execute("UPDATE runs SET state=?,evidence_json=? WHERE run_id=?",
                                    (state, canonical_bytes(evidence).decode(), run_id))
            self._event(run_id, state, evidence)

    def settle(self, run_id: str, evidence: dict) -> dict:
        with self._transaction():
            row = self.get(run_id)
            if row["state"] not in ("DISPATCHING", "RUNNING", "UNKNOWN"):
                raise ApprovalError("Only a dispatched run can be settled exactly once")
            for key in ("run_id", "reservation_id", "request_sha256"):
                if evidence.get(key) != row[key]:
                    raise ApprovalError("Remote terminal evidence identity mismatch: " + key)
            if evidence.get("status") not in TERMINAL_REMOTE or evidence.get("gpu_release_verified") is not True:
                raise ApprovalError("No terminal proof of GPU release; reservation remains")
            if evidence.get("live_owned_containers") != []:
                raise ApprovalError("Owned container still running; reservation remains")
            started = evidence.get("allocation_started_monotonic_ns")
            ended = evidence.get("allocation_ended_monotonic_ns")
            if started is None:
                if evidence["status"] != "NOT_LAUNCHED" or ended is not None:
                    raise ApprovalError("Unaccounted allocation cannot be settled")
                gpu_us = 0
            else:
                started = _integer(started, "allocation start")
                ended = _integer(ended, "allocation end")
                if ended < started:
                    raise ApprovalError("GPU accounting clock moved backward")
                intervals = evidence.get("allocation_intervals_ns")
                if intervals is None:
                    elapsed = ended - started
                else:
                    if not intervals or intervals[0].get("start_ns") != started or intervals[-1].get("end_ns") != ended:
                        raise ApprovalError("Allocation interval envelope mismatch")
                    elapsed = 0
                    previous_end = started
                    for interval in intervals:
                        begin = _integer(interval.get("start_ns"), "interval start")
                        finish = _integer(interval.get("end_ns"), "interval end")
                        if begin < previous_end or finish < begin:
                            raise ApprovalError("Overlapping or invalid allocation intervals")
                        elapsed += finish - begin
                        previous_end = finish
                gpu_us = (elapsed * len(json.loads(row["request_json"])["gpu_uuids"]) + 999) // 1000
            actual_disk = _integer(evidence.get("new_output_bytes"), "new output bytes")
            breach = gpu_us > row["reserved_gpu_us"] or actual_disk > row["reserved_disk_bytes"]
            self.connection.execute("""UPDATE runs SET state='SETTLED',actual_gpu_us=?,actual_disk_bytes=?,
                evidence_json=?,quota_breach=? WHERE run_id=?""",
                (gpu_us, actual_disk, canonical_bytes(evidence).decode(), int(breach), run_id))
            self._event(run_id, "SETTLED", {"actual_gpu_us": gpu_us, "actual_disk_bytes": actual_disk,
                                           "remote_status": evidence["status"], "quota_breach": breach})
            return self.get(run_id)
