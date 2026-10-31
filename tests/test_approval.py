"""CPU-only fixture checks; never create the actual trusted approval file."""

import datetime
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.controller.approval import ApprovalError, load_approval_file, validate_scope


class ApprovalScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc)
        self.uuid = "GPU-00000000-0000-0000-0000-000000000001"
        self.approval = {"schema": "kv-agent-approval-v1", "enabled": True,
                         "allow_remote_user_workspace_writes": True, "allow_gpu_jobs": True,
                         "ssh_target": "fixture-host", "approved_host_fingerprint": "SHA256:" + "A" * 43,
                         "approved_workspace": "/fixture/user/project", "allowed_gpu_uuids": [self.uuid],
                         "max_concurrent_gpus": 1, "max_gpu_hours_per_job": 2, "max_gpu_hours_total": 8,
                         "max_disk_gib_new_outputs": 20, "valid_until_utc": "2026-10-27T00:00:00Z"}
        self.request = {key: self.approval[key] for key in
                        ("ssh_target", "approved_host_fingerprint", "approved_workspace")}
        self.request.update({"run_id": "fixture-run", "remote_output": "/fixture/user/project/agent_work/runs/fixture-run",
                             "gpu_uuids": [self.uuid], "max_wall_seconds": 3600, "max_output_gib": 2})
        self.usage = {"known_run_ids": [], "reconciled": True, "uncertain_runs": [], "active_gpu_uuids": [],
                      "gpu_hours_consumed": 0, "gpu_hours_reserved": 0, "disk_gib_consumed": 0, "disk_gib_reserved": 0}

    def _check(self) -> dict:
        return validate_scope(self.approval, self.request, self.usage, self.now)

    def test_valid_static_scope_still_never_allows_launch(self) -> None:
        result = self._check()
        self.assertTrue(result["authorization_scope_valid"])
        self.assertEqual(result["required_gpu_hours_reservation"], 1)
        self.assertFalse(result["can_launch"])
        self.assertIn("runtime quota supervisor and UNKNOWN recovery", result["remaining_gates"])

    def test_missing_or_disabled_scope_is_not_permission(self) -> None:
        for key in ("enabled", "allow_gpu_jobs", "allow_remote_user_workspace_writes"):
            with self.subTest(key=key):
                saved = self.approval.pop(key)
                with self.assertRaises(ApprovalError):
                    self._check()
                self.approval[key] = saved

    def test_expired_approval(self) -> None:
        self.approval["valid_until_utc"] = "2026-10-25T00:00:00Z"
        with self.assertRaisesRegex(ApprovalError, "expired"):
            self._check()

    def test_job_cannot_outlive_approval(self) -> None:
        self.approval["valid_until_utc"] = "2026-10-26T00:30:00Z"
        with self.assertRaisesRegex(ApprovalError, "job exceeds approval expiry"):
            self._check()

    def test_host_identity_change(self) -> None:
        self.request["approved_host_fingerprint"] = "SHA256:" + "B" * 43
        with self.assertRaisesRegex(ApprovalError, "identity"):
            self._check()

    def test_uuid_change(self) -> None:
        self.request["gpu_uuids"] = [self.uuid[:-1] + "2"]
        with self.assertRaisesRegex(ApprovalError, "UUID"):
            self._check()

    def test_path_prefix_confusion_and_traversal(self) -> None:
        for path in ("/fixture/user/project-other/agent_work/runs/fixture-run",
                     "/fixture/user/project/agent_work/runs/../fixture-run"):
            with self.subTest(path=path):
                self.request["remote_output"] = path
                with self.assertRaises(ApprovalError):
                    self._check()

    def test_boolean_and_nonfinite_budgets_rejected(self) -> None:
        for number in (True, float("nan"), float("inf"), -1):
            with self.subTest(number=number):
                self.approval["max_gpu_hours_total"] = number
                with self.assertRaises(ApprovalError):
                    self._check()

    def test_failed_and_reserved_hours_count_toward_total(self) -> None:
        self.usage["gpu_hours_consumed"] = 6.5
        self.usage["gpu_hours_reserved"] = 1
        with self.assertRaisesRegex(ApprovalError, "Total GPU-hours"):
            self._check()

    def test_per_job_budget(self) -> None:
        self.request["max_wall_seconds"] = 7201
        with self.assertRaisesRegex(ApprovalError, "Per-job"):
            self._check()

    def test_disk_budget_includes_existing_and_reserved_outputs(self) -> None:
        self.usage["disk_gib_consumed"] = 18
        self.usage["disk_gib_reserved"] = 1
        with self.assertRaisesRegex(ApprovalError, "disk budget"):
            self._check()

    def test_unknown_run_blocks_dispatch_without_releasing_budget(self) -> None:
        self.usage["uncertain_runs"] = ["old-fixture-job"]
        with self.assertRaisesRegex(ApprovalError, "UNKNOWN"):
            self._check()

    def test_completed_run_id_is_not_relaunched(self) -> None:
        self.usage["known_run_ids"] = ["fixture-run"]
        with self.assertRaisesRegex(ApprovalError, "already exists"):
            self._check()

    def test_own_active_gpu_is_not_double_booked(self) -> None:
        self.usage["active_gpu_uuids"] = [self.uuid]
        with self.assertRaisesRegex(ApprovalError, "already reserved"):
            self._check()

    def test_file_trust_boundary(self) -> None:
        # Workspace ancestors are private enough; /tmp is intentionally rejected.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            path = root / "fixture-approval.json"
            path.write_text(json.dumps(self.approval))
            os.chmod(path, 0o600)
            self.assertEqual(load_approval_file(path), self.approval)
            link = root / "fixture-link.json"
            link.symlink_to(path)
            with self.assertRaisesRegex(ApprovalError, "symlink"):
                load_approval_file(link)
            os.chmod(path, 0o644)
            with self.assertRaisesRegex(ApprovalError, "only by its owner"):
                load_approval_file(path)

    def test_duplicate_json_keys_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "fixture.json"
            path.write_text('{"enabled":false,"enabled":true}')
            os.chmod(path, 0o600)
            with self.assertRaisesRegex(ApprovalError, "Duplicate JSON key"):
                load_approval_file(path)

    def test_foreign_owned_ancestor_is_not_a_trusted_approval_location(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            path = Path(temp) / "fixture.json"
            path.write_text(json.dumps(self.approval))
            os.chmod(path, 0o600)
            original = Path.stat

            def fake_owner(candidate: Path, *args: object, **kwargs: object) -> object:
                info = original(candidate, *args, **kwargs)
                if candidate == path.parent:
                    return SimpleNamespace(st_mode=info.st_mode, st_uid=os.getuid() + 1)
                return info

            with patch.object(Path, "stat", new=fake_owner):
                with self.assertRaisesRegex(ApprovalError, "ancestor owner"):
                    load_approval_file(path)


if __name__ == "__main__":
    unittest.main()
