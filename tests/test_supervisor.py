"""CPU fault injection for the real supervisor loop and process ownership guards."""

import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.durable import digest
from scripts.controller.processes import pidfd_available, process_identity, signal_exact
from scripts.controller.supervisor import Supervisor, UnsafeState
from scripts.controller.watchdog import enforce_registry, run_watchdog


GPU = "GPU-00000000-0000-0000-0000-000000000001"


class Clock:
    def __init__(self) -> None:
        self.ns = 1_000_000_000_000
        self.origin = self.ns

    def now(self) -> int:
        return self.ns

    def sleep(self, seconds: float) -> None:
        self.ns += int(seconds * 1e9)

    def wall(self) -> datetime.datetime:
        return datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc) + datetime.timedelta(seconds=(self.ns-self.origin)/1e9)


class FakeBackend:
    """Does not call Docker, CUDA, NVML or any process-control API."""
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.live = {}
        self.created = []
        self.stopped = []
        self.release = True
        self.foreign = False
        self.guard = True
        self.disk = False
        self.fp8_seconds = 3
        self.busy_until = 0

    def snapshot(self) -> dict:
        return {GPU: {"index": 2, "memory_mib": 0 if self.clock.ns >= self.busy_until else 100,
                      "util_pct": 0, "pids": []}}

    def reservation_clear(self) -> bool: return True
    def verify_preflight(self) -> None: pass
    def create(self, arm: str, index: int) -> str:
        identifier = ("a" if arm == "bf16" else "b") * 64
        self.created.append((identifier, arm, self.clock.ns))
        return identifier
    def start(self, identifier: str, arm: str) -> None:
        self.live[identifier] = self.clock.ns + int((3 if arm == "bf16" else self.fp8_seconds) * 1e9)
    def running(self, identifier: str) -> bool: return self.clock.ns < self.live[identifier]
    def finish(self, identifier: str, arm: str) -> dict: return {"arm": arm, "fixture": True}
    def foreign_gpu(self, snapshot: dict, owned: list[str]) -> bool: return self.foreign and bool(owned)
    def watchdog_healthy(self) -> bool: return self.guard
    def stop(self, identifier: str) -> bool:
        self.stopped.append(identifier)
        return self.release
    def output_bytes(self) -> int: return int(2**30 * 2 * 0.91) if self.disk else 100


class SupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clock = Clock()
        self.backend = FakeBackend(self.clock)
        self.request = {"run_id": "fixture", "gpu_uuids": [GPU], "arms": ["bf16", "fp8_e4m3"],
                        "max_wall_seconds": 600, "max_output_gib": 2}
        self.ticket = {"request": self.request, "request_sha256": digest(self.request),
                       "reservation_id": "f" * 32,
                       "deadline_utc": (self.clock.wall() + datetime.timedelta(seconds=600)).isoformat()}

    def tearDown(self) -> None: self.temp.cleanup()

    def run_engine(self) -> dict:
        return Supervisor(self.ticket, self.root, self.backend, self.clock.now, self.clock.sleep, self.clock.wall).run()

    def test_complete_requires_both_arms_stopped_and_verified(self) -> None:
        result = self.run_engine()
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(result["arm_results"]), 2)
        self.assertEqual(len(self.backend.stopped), 2)
        self.assertTrue(result["gpu_release_verified"])
        self.assertTrue((self.root / "COMPLETE").is_file())
        self.assertGreaterEqual(self.backend.created[0][2] - self.clock.origin, 120_000000000)
        self.assertGreaterEqual(self.backend.created[1][2] - result["allocation_intervals_ns"][0]["end_ns"], 120_000000000)

    def test_busy_sample_resets_idle_window(self) -> None:
        self.backend.busy_until = self.clock.ns + 30_000000000
        self.run_engine()
        self.assertGreaterEqual(self.backend.created[0][2] - self.clock.origin, 150_000000000)

    def test_busy_until_deadline_has_no_allocation_and_settles_zero_GPU_time(self) -> None:
        from scripts.controller.quota import QuotaLedger
        from test_quota_ledger import _fixture
        approval,request,now=_fixture()
        request.update(arms=self.request['arms'],max_wall_seconds=600)
        self.ticket.update(request=request,request_sha256=digest(request))
        self.backend.busy_until=self.clock.ns+10_000_000000000
        with QuotaLedger(self.root/'ledger') as ledger:
            row=ledger.reserve(approval,request,digest(approval),now)
            self.ticket['reservation_id']=row['reservation_id']
            ledger.transition(request['run_id'],'DISPATCHING',{})
            result=self.run_engine()
            self.assertEqual(result['status'],'NOT_LAUNCHED')
            self.assertIn('expired while waiting',result['error'])
            self.assertIsNone(result['allocation_started_monotonic_ns'])
            self.assertIsNone(result['allocation_ended_monotonic_ns'])
            self.assertEqual(result['allocation_intervals_ns'],[])
            self.assertEqual(self.backend.created,[]);self.assertEqual(self.backend.stopped,[])
            self.assertFalse((self.root/'COMPLETE').exists())
            settled=ledger.settle(request['run_id'],result)
            self.assertEqual(settled['actual_gpu_us'],0)
            self.assertEqual(ledger.usage(approval)['gpu_hours_reserved'],0)

    def test_last_arm_timeout_preserves_first_arm_and_never_completes(self) -> None:
        self.backend.fp8_seconds = 10000
        result = self.run_engine()
        self.assertEqual(result["status"], "TIMED_OUT")
        self.assertEqual([row["arm"] for row in result["arm_results"]], ["bf16"])
        self.assertFalse((self.root / "COMPLETE").exists())
        self.assertLessEqual(self.clock.ns - self.clock.origin, 600_000000000)
        self.assertEqual(set(self.backend.stopped), {"a" * 64, "b" * 64})

    def test_foreign_activity_stops_only_owned_container(self) -> None:
        self.backend.foreign = True
        result = self.run_engine()
        self.assertEqual(result["status"], "CONFLICT")
        self.assertEqual(self.backend.stopped, ["a" * 64])
        self.assertFalse((self.root / "COMPLETE").exists())

    def test_missing_release_proof_is_unknown_and_keeps_watchdog_active(self) -> None:
        self.backend.release = False
        result = self.run_engine()
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertFalse(result["gpu_release_verified"])
        self.assertFalse((self.root / "WATCHDOG_FINISHED.json").exists())
        self.assertFalse((self.root / "COMPLETE").exists())

    def test_guard_death_does_not_allow_worker_to_continue(self) -> None:
        self.backend.watchdog_healthy = lambda: not self.backend.live
        result = self.run_engine()
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("watchdog died", result["error"])
        self.assertEqual(self.backend.stopped, ["a" * 64])

    def test_dead_guard_before_idle_never_admits_any_worker(self) -> None:
        self.backend.guard = False
        result = self.run_engine()
        self.assertEqual(result["status"], "NOT_LAUNCHED")
        self.assertEqual(self.backend.created, [])

    def test_stop_while_idle_never_admits_worker(self) -> None:
        (self.root / "STOP_REQUESTED.json").write_text("{}")
        result = self.run_engine()
        self.assertEqual(result["status"], "NOT_LAUNCHED")
        self.assertEqual(self.backend.created, [])

    def test_disk_stop_leaves_outputs_and_failed_state(self) -> None:
        self.backend.disk = True
        result = self.run_engine()
        self.assertEqual(result["status"], "DISK_LIMIT")
        self.assertTrue((self.root / "result.json").exists())
        self.assertFalse((self.root / "COMPLETE").exists())

    def test_worker_exit_zero_still_needs_measurement_evidence(self) -> None:
        def bad_finish(identifier: str, arm: str) -> dict:
            raise UnsafeState("exit zero but Graph evidence missing")
        self.backend.finish = bad_finish
        result = self.run_engine()
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("Graph evidence missing", result["error"])


class ProcessOwnershipTests(unittest.TestCase):
    def test_watchdog_deadline_uses_only_recorded_owned_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); clock = Clock(); calls = []
            config = {"request_sha256": "a" * 64, "reservation_id": "b" * 32,
                      "kill_at_monotonic_ns": clock.ns + 1_000_000_000}
            (root / "watchdog_config.json").write_text(json.dumps(config))
            identity = {"pid": 12345, "uid": os.getuid(), "cgroup": "0::/docker/" + "c" * 64,
                        "start_ticks": 1234, "boot_id": "fixture"}
            registry = {**config, "containers": [{"container_id": "c" * 64, "pid_identity": identity, "released": False}]}
            (root / "owned_containers.json").write_text(json.dumps(registry))
            result = run_watchdog(root / "watchdog_config.json", clock.now, clock.sleep,
                                  lambda proof, number: calls.append((proof, number)) or True)
            self.assertEqual(result["status"], "DEADLINE_ENFORCED")
            self.assertEqual(calls, [(identity, signal.SIGKILL)])
            self.assertTrue((root / "watchdog_result.json").exists())

    def test_forged_watchdog_finish_cannot_skip_deadline_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); clock = Clock()
            config = {"request_sha256": "a" * 64, "reservation_id": "b" * 32,
                      "kill_at_monotonic_ns": clock.ns + 1_000_000_000}
            (root / "watchdog_config.json").write_text(json.dumps(config))
            (root / "WATCHDOG_FINISHED.json").write_text(json.dumps({"request_sha256": "wrong"}))
            result = run_watchdog(root / "watchdog_config.json", clock.now, clock.sleep, lambda *args: False)
            self.assertEqual(result["status"], "GUARD_FAILURE")

    @unittest.skipUnless(pidfd_available(), "Kernel pidfd unavailable")
    def test_real_cpu_child_can_be_stopped_by_captured_identity(self) -> None:
        child = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(10)"])
        try:
            proof = process_identity(child.pid)
            self.assertTrue(signal_exact(proof, signal.SIGTERM))
            self.assertEqual(child.wait(timeout=3), -signal.SIGTERM)
        finally:
            if child.poll() is None:
                signal_exact(process_identity(child.pid), signal.SIGKILL)
                child.wait(timeout=3)

    @unittest.skipUnless(pidfd_available(), "Kernel pidfd unavailable")
    def test_reused_pid_or_changed_start_time_never_receives_signal(self) -> None:
        proof = process_identity(os.getpid())
        proof["start_ticks"] -= 1
        with patch("scripts.controller.processes.signal.pidfd_send_signal") as send:
            with self.assertRaisesRegex(ValueError, "reused"):
                signal_exact(proof, signal.SIGKILL)
            send.assert_not_called()

    def test_watchdog_filters_foreign_cgroups_before_signalling(self) -> None:
        config = {"reservation_id": "fixture", "request_sha256": "d" * 64}
        registry = {**config, "containers": [{"container_id": "a" * 64,
                    "pid_identity": {"uid": os.getuid(), "cgroup": "/foreign", "pid": 123}, "released": False}]}
        calls = []
        with self.assertRaisesRegex(ValueError, "foreign cgroup"):
            enforce_registry(config, registry, lambda identity, sig: calls.append(identity))
        self.assertEqual(calls, [])


if __name__ == "__main__": unittest.main()
