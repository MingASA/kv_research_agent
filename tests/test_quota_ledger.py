"""CPU quota/recovery/concurrency fixtures, never actual GPU experiments."""

import copy
import datetime
import multiprocessing
from pathlib import Path
import tempfile
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.durable import digest
from scripts.controller.quota import QuotaLedger, budget_key


def _fixture() -> tuple[dict, dict, datetime.datetime]:
    now = datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc)
    gpu = "GPU-00000000-0000-0000-0000-000000000001"
    approval = {"schema": "kv-agent-approval-v1", "enabled": True,
                "allow_remote_user_workspace_writes": True, "allow_gpu_jobs": True,
                "ssh_target": "fixture", "approved_host_fingerprint": "SHA256:" + "A" * 43,
                "approved_workspace": "/fixture/project", "allowed_gpu_uuids": [gpu],
                "max_concurrent_gpus": 1, "max_gpu_hours_per_job": 2,
                "max_gpu_hours_total": 3, "max_disk_gib_new_outputs": 10,
                "valid_until_utc": "2026-10-27T00:00:00Z"}
    request = {key: approval[key] for key in
               ("ssh_target", "approved_host_fingerprint", "approved_workspace")}
    request.update({"run_id": "fixture-run", "remote_output": "/fixture/project/agent_work/runs/fixture-run",
                    "gpu_uuids": [gpu], "max_wall_seconds": 3600, "max_output_gib": 2,
                    "experiment_key": "b" * 64, "replicate": 0})
    return approval, request, now


def _race(directory: str, name: str, result: object) -> None:
    approval, request, now = _fixture()
    request.update(run_id=name, remote_output="/fixture/project/agent_work/runs/" + name,
                   experiment_key=digest(name))
    try:
        with QuotaLedger(Path(directory)) as ledger:
            ledger.reserve(approval, request, digest(approval), now)
        result.put("reserved")
    except ApprovalError:
        result.put("denied")


class QuotaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "ledger"
        self.ledger = QuotaLedger(self.path)
        self.approval, self.request, self.now = _fixture()

    def tearDown(self) -> None:
        self.ledger.close()
        self.temp.cleanup()

    def _reserve(self) -> dict:
        return self.ledger.reserve(self.approval, self.request, digest(self.approval), self.now)

    def _terminal(self, reserved: dict, seconds: int = 1200, status: str = "FAILED") -> dict:
        return {key: reserved[key] for key in ("run_id", "reservation_id", "request_sha256")} | {
            "status": status, "gpu_release_verified": True, "live_owned_containers": [],
            "allocation_started_monotonic_ns": 1000000000,
            "allocation_ended_monotonic_ns": (seconds + 1) * 1000000000, "new_output_bytes": 100}

    def test_reserve_and_reopen_keeps_full_reservation(self) -> None:
        self._reserve()
        with QuotaLedger(self.path) as reopened:
            self.assertEqual(reopened.usage(self.approval)["gpu_hours_reserved"], 1)
            self.assertEqual(reopened.usage(self.approval)["active_gpu_uuids"], self.request["gpu_uuids"])

    def test_dispatch_disconnect_blocks_all_new_launches(self) -> None:
        self._reserve()
        self.ledger.transition("fixture-run", "DISPATCHING", {"intent": "before SSH send"})
        self.ledger.transition("fixture-run", "UNKNOWN", {"error": "connection lost"})
        self.request.update(run_id="second", remote_output="/fixture/project/agent_work/runs/second",
                            experiment_key="c" * 64)
        with self.assertRaisesRegex(ApprovalError, "UNKNOWN"):
            self._reserve()
        self.assertEqual(self.ledger.usage(self.approval)["gpu_hours_reserved"], 1)

    def test_failure_time_is_charged_and_settlement_is_exactly_once(self) -> None:
        row = self._reserve()
        self.ledger.transition(row["run_id"], "DISPATCHING", {})
        settled = self.ledger.settle(row["run_id"], self._terminal(row))
        self.assertEqual(settled["actual_gpu_us"], 1200_000000)
        self.assertAlmostEqual(self.ledger.usage(self.approval)["gpu_hours_consumed"], 1 / 3)
        self.assertEqual(self.ledger.usage(self.approval)["gpu_hours_reserved"], 0)
        with self.assertRaises(ApprovalError):
            self.ledger.settle(row["run_id"], self._terminal(row))

    def test_terminal_without_gpu_release_cannot_free_quota(self) -> None:
        row = self._reserve(); self.ledger.transition(row["run_id"], "DISPATCHING", {})
        proof = self._terminal(row); proof["gpu_release_verified"] = False
        with self.assertRaisesRegex(ApprovalError, "GPU release"):
            self.ledger.settle(row["run_id"], proof)
        self.assertEqual(self.ledger.usage(self.approval)["gpu_hours_reserved"], 1)

    def test_foreign_identity_terminal_is_rejected(self) -> None:
        row = self._reserve(); self.ledger.transition(row["run_id"], "DISPATCHING", {})
        proof = self._terminal(row); proof["reservation_id"] = "foreign"
        with self.assertRaisesRegex(ApprovalError, "identity mismatch"):
            self.ledger.settle(row["run_id"], proof)

    def test_surviving_container_cannot_be_settled(self) -> None:
        row = self._reserve(); self.ledger.transition(row["run_id"], "DISPATCHING", {})
        proof = self._terminal(row); proof["live_owned_containers"] = ["owned-still-alive"]
        with self.assertRaisesRegex(ApprovalError, "still running"):
            self.ledger.settle(row["run_id"], proof)

    def test_only_never_dispatched_reservation_can_cancel(self) -> None:
        row = self._reserve(); self.ledger.transition(row["run_id"], "DISPATCHING", {})
        with self.assertRaises(ApprovalError):
            self.ledger.transition(row["run_id"], "CANCELLED", {})

    def test_replicate_dedupe_survives_terminal_failure(self) -> None:
        row = self._reserve(); self.ledger.transition(row["run_id"], "DISPATCHING", {})
        self.ledger.settle(row["run_id"], self._terminal(row))
        self.request.update(run_id="second", remote_output="/fixture/project/agent_work/runs/second")
        with self.assertRaisesRegex(ApprovalError, "replicate already"):
            self._reserve()

    def test_historical_id_is_not_relaunched(self) -> None:
        self.ledger.register_history({"fixture-run": "d" * 64})
        with self.assertRaisesRegex(ApprovalError, "already exists"):
            self._reserve()

    def test_expiry_or_notes_edit_does_not_reset_budget(self) -> None:
        self._reserve()
        key = budget_key(self.approval)
        self.approval.update(notes="user refreshed comments", valid_until_utc="2026-10-28T00:00:00Z")
        self.assertEqual(budget_key(self.approval), key)
        self.assertEqual(self.ledger.usage(self.approval)["gpu_hours_reserved"], 1)

    def test_lost_database_is_not_reinitialized(self) -> None:
        separate = self.path.parent / "lost"
        with QuotaLedger(separate):
            pass
        (separate / "ledger.sqlite3").rename(separate / "preserved.sqlite3")
        with self.assertRaisesRegex(ApprovalError, "vanished"):
            QuotaLedger(separate)

    def test_overrun_is_recorded_without_capping_and_blocks_future(self) -> None:
        row = self._reserve(); self.ledger.transition(row["run_id"], "DISPATCHING", {})
        settled = self.ledger.settle(row["run_id"], self._terminal(row, seconds=3601))
        self.assertEqual(settled["actual_gpu_us"], 3601_000000)
        self.assertTrue(settled["quota_breach"])
        self.request.update(run_id="second", remote_output="/fixture/project/agent_work/runs/second", replicate=1)
        with self.assertRaisesRegex(ApprovalError, "quota breach"):
            self._reserve()

    def test_two_processes_cannot_reserve_same_gpu(self) -> None:
        context = multiprocessing.get_context("spawn")
        result = context.Queue()
        children = [context.Process(target=_race, args=(str(self.path), name, result)) for name in ("a", "b")]
        for child in children: child.start()
        for child in children:
            child.join(timeout=10)
            self.assertEqual(child.exitcode, 0)
        self.assertCountEqual([result.get(timeout=2), result.get(timeout=2)], ["reserved", "denied"])

    def test_device_time_excludes_inter_arm_idle_windows(self) -> None:
        row = self._reserve(); self.ledger.transition(row["run_id"], "DISPATCHING", {})
        proof = self._terminal(row, seconds=1500)
        proof["allocation_intervals_ns"] = [{"start_ns": 1_000000000, "end_ns": 601_000000000},
                                            {"start_ns": 901_000000000, "end_ns": 1501_000000000}]
        settled = self.ledger.settle(row["run_id"], proof)
        self.assertEqual(settled["actual_gpu_us"], 1200_000000)


if __name__ == "__main__":
    unittest.main()
