"""CPU-only execution/SSH/ownership faults; no actual SSH, Docker or CUDA."""

import base64
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.controller.approval import ApprovalError
from scripts.controller.dispatch import Dispatcher
from scripts.controller.durable import canonical_bytes, digest
from scripts.controller.policy import SOURCE_FILES, docker_command, validate_execution_authority, validate_request
from scripts.controller.quota import QuotaLedger
from scripts.controller import remote_rpc
from scripts.controller.transport import PinnedSSH, UncertainDispatch, fingerprint, pinned_key
from scripts.controller.worker import environment, fixed_lengths, fixed_server_command


def fixture():
    now = datetime.datetime(2026, 10, 26, tzinfo=datetime.timezone.utc)
    base = "/fixture/project"
    approval = {"schema": "kv-agent-approval-v1", "enabled": True, "allow_remote_user_workspace_writes": True,
                "allow_gpu_jobs": True, "ssh_target": "fixture", "approved_host_fingerprint": "SHA256:" + "A" * 43,
                "approved_workspace": base, "approved_remote_identity": {"uid": 1000, "gid": 1000, "hostname": "fixture"},
                "allowed_gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000001"],
                "max_concurrent_gpus": 1, "max_gpu_hours_per_job": 2, "max_gpu_hours_total": 3,
                "max_disk_gib_new_outputs": 10, "valid_until_utc": "2026-10-27T00:00:00Z",
                "execution_contract": "kv-controller-v1", "approved_image_id": "sha256:" + "a" * 64,
                "read_only_model_path": "/fixture/model", "container_limits": {"cpus": 4, "memory_gib": 64, "shm_gib": 8},
                "authorization_origin": {"kind": "explicit_user_message"},
                "shared_gpu_policy": {"kind": "idle_only_standing_user", "no_foreign_gpu_use": True, "never_interrupt_others": True}}
    request = {key: approval[key] for key in ("ssh_target", "approved_host_fingerprint", "approved_workspace")}
    request.update(schema="kv-native-run-v1", run_id="fixture-run", remote_output=base + "/agent_work/runs/fixture-run",
                   gpu_uuids=approval["allowed_gpu_uuids"], max_wall_seconds=3600, max_output_gib=2,
                   experiment_key="b" * 64, replicate=0, arms=["bf16", "fp8_e4m3"],
                   source_repo=base + "/source", server_binary=base + "/source/xllm", dataset_path=base + "/input.json",
                   model_path="/fixture/model", image_id=approval["approved_image_id"], server_source_commit="c" * 40,
                   binary_sha256="d" * 64, dataset_sha256="e" * 64, model_config_sha256="f" * 64,
                   backend_source_sha256={name: "1" * 64 for name in SOURCE_FILES},
                   graph_mode="cudagraphs", prefix_cache=False, fp8_scales={"key": .125, "value": .0625}, concurrency=4,
                   workload={"kind": "fixed_decode", "requests": 128, "warmup": 8, "isl": 16384, "output_tokens": 512, "ignore_eos": True},
                   research_reason="CPU fixture only, not a real experiment")
    return approval, request, now


class PolicyTests(unittest.TestCase):
    def test_concrete_authority_and_protocol_are_accepted(self):
        a, r, now = fixture(); validate_execution_authority(a, now); validate_request(r, a)

    def test_standing_grant_preserves_shared_server_conditions(self):
        a, _, now = fixture(); a["shared_gpu_policy"]["never_interrupt_others"] = False
        with self.assertRaises(ApprovalError): validate_execution_authority(a, now)

    def test_eager_and_natural_stopping_cannot_masquerade_as_fixed_decode(self):
        for field, value in (("graph_mode", "off"), ("prefix_cache", True)):
            a, r, _ = fixture(); r[field] = value
            with self.assertRaises(ApprovalError): validate_request(r, a)
        a, r, _ = fixture(); r["workload"]["ignore_eos"] = False
        with self.assertRaises(ApprovalError): validate_request(r, a)

    def test_missing_graph_source_and_mount_encoding_are_rejected(self):
        a, r, _ = fixture(); r["backend_source_sha256"].pop(SOURCE_FILES[-1])
        with self.assertRaises(ApprovalError): validate_request(r, a)
        a, r, _ = fixture(); r["source_repo"] += ",rw"
        with self.assertRaises(ApprovalError): validate_request(r, a)

    def test_docker_has_only_new_run_writable_and_exact_gpu(self):
        a, r, _ = fixture()
        command = docker_command({"approval": a, "request": r, "reservation_id": "a" * 32, "request_sha256": digest(r)}, "bf16", 2)
        self.assertEqual(command[command.index("--gpus") + 1], "device=" + r["gpu_uuids"][0])
        mounts = [command[index + 1] for index, option in enumerate(command) if option == "--mount"]
        self.assertEqual(len([value for value in mounts if not value.endswith(",readonly")]), 2)
        self.assertTrue(all(r["remote_output"] in value for value in mounts if not value.endswith(",readonly")))
        self.assertNotIn("--privileged", command)
        self.assertIn("--read-only", command)
        self.assertEqual(command[command.index("--network") + 1], "none")
        self.assertTrue(all(not value.startswith("LD_LIBRARY_PATH=") for value in command))

    def test_worker_env_strips_inherited_backend_and_preserves_cuda_compat(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/image/compat", "XLLM_QUANTIZED_BACKEND": "triton", "XLLM_FP8_CALIBRATION_OUTPUT": "/old/result"}):
            bf = environment("bf16"); fp = environment("fp8_e4m3")
        self.assertNotIn("XLLM_QUANTIZED_BACKEND", bf)
        self.assertNotIn("XLLM_FP8_CALIBRATION_OUTPUT", fp)
        self.assertTrue(fp["LD_LIBRARY_PATH"].endswith(":/image/compat"))
        self.assertEqual(fp["XLLM_QUANTIZED_BACKEND"], "flashinfer")
        self.assertEqual(fp["XLLM_FP8_K_SCALE"], "0.125")

    def test_worker_rejects_early_eos_even_if_average_looks_good(self):
        fixed_lengths({"profiling": [{"metrics": {"output_token_count": {"value": 512}}}]})
        with self.assertRaises(ValueError):
            fixed_lengths({"profiling": [{"metrics": {"output_token_count": {"value": 511}}}]})


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.ledger = QuotaLedger(Path(self.temp.name) / "ledger")
        a, r, self.now = fixture()
        row = self.ledger.reserve(a, r, digest(a), self.now)
        self.ticket = {"approval": a, "approval_sha256": digest(a), "request": r, "request_sha256": digest(r),
                       "reservation_id": row["reservation_id"], "deadline_utc": "2026-10-26T01:00:00Z"}
        self.calls = []
        self.response = {"status": "RUNNING", "supervisor_alive": True, "files": {}, "live_owned_containers": ["a" * 64]}
        outer = self
        class Transport:
            def rpc(self, source, payload, **kwargs):
                outer.calls.append((payload["operation"], kwargs))
                if kwargs.get("mutation") and outer.lose == payload["operation"]:
                    raise UncertainDispatch("fixture SSH reply lost")
                return outer.response
        self.lose = None
        self.dispatch = Dispatcher(self.ledger, Transport(), "fixture-only", lambda: self.now,
                                   lambda: (a, digest(a)))

    def tearDown(self): self.ledger.close(); self.temp.cleanup()

    def test_lost_prepare_reply_becomes_unknown_and_cannot_relaunch(self):
        self.lose = "prepare"
        with self.assertRaises(UncertainDispatch): self.dispatch.launch(self.ticket, {})
        self.assertEqual(self.ledger.get("fixture-run")["state"], "UNKNOWN")
        with self.assertRaises(ApprovalError): self.dispatch.launch(self.ticket, {})
        self.assertEqual([item[0] for item in self.calls], ["prepare"])

    def test_lost_start_reply_recovers_readonly_without_second_start(self):
        self.lose = "start"
        with self.assertRaises(UncertainDispatch): self.dispatch.launch(self.ticket, {})
        self.dispatch.reconcile(self.ticket)
        self.assertEqual(self.ledger.get("fixture-run")["state"], "RUNNING")
        self.assertEqual([item[0] for item in self.calls], ["prepare", "start", "status"])
        self.assertFalse(self.calls[-1][1].get("mutation", False))

    def test_changed_or_revoked_authority_never_dispatches(self):
        self.dispatch.authority_loader = lambda: ({}, "different")
        with self.assertRaises(ApprovalError): self.dispatch.launch(self.ticket, {})
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger.get("fixture-run")["state"], "RESERVED")

    def test_absent_remote_after_lost_reply_keeps_unknown_reservation(self):
        self.lose = "prepare"
        with self.assertRaises(UncertainDispatch): self.dispatch.launch(self.ticket, {})
        self.response = {"status": "ABSENT", "files": {}}
        self.dispatch.reconcile(self.ticket)
        self.assertEqual(self.ledger.get("fixture-run")["state"], "UNKNOWN")
        self.assertEqual(self.ledger.usage(self.ticket["approval"])["gpu_hours_reserved"], 1)


class TransportTests(unittest.TestCase):
    def test_mutations_are_attempted_exactly_once(self):
        ssh = PinnedSSH.__new__(PinnedSSH); ssh.argv = ["fixture-ssh"]
        with patch("scripts.controller.transport.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 1)) as run:
            with self.assertRaises(UncertainDispatch): ssh.rpc("pass", {}, mutation=True)
            self.assertEqual(run.call_count, 1)

    def test_readonly_ssh_has_three_bounded_retries(self):
        ssh = PinnedSSH.__new__(PinnedSSH); ssh.argv = ["fixture-ssh"]
        with patch("scripts.controller.transport.subprocess.run", side_effect=subprocess.TimeoutExpired("fixture", 1)) as run, patch("scripts.controller.transport.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, 'recovery exhausted'): ssh.rpc("pass", {})
            self.assertEqual(run.call_count, 3); self.assertEqual(sleep.call_count, 2)
            self.assertTrue(15 <= sleep.call_args_list[0].args[0] <= 16)
            self.assertTrue(30 <= sleep.call_args_list[1].args[0] <= 31)

    def test_lost_reply_records_actual_stderr_without_dumping_rpc_source(self):
        ssh = PinnedSSH.__new__(PinnedSSH); ssh.argv = ['fixture-ssh']
        error = subprocess.CalledProcessError(255, ['fixture', 'very-long-rpc-source'],
                                             stderr=b'Connection closed by remote host')
        with patch('scripts.controller.transport.subprocess.run', side_effect=error) as run:
            with self.assertRaises(UncertainDispatch) as caught:
                ssh.rpc('very-long-rpc-source', {}, mutation=True)
        self.assertIn('SSH exit 255', str(caught.exception))
        self.assertIn('Connection closed', str(caught.exception))
        self.assertNotIn('very-long-rpc-source', str(caught.exception))
        self.assertEqual(run.call_count, 1)

    def test_only_existing_exact_approved_key_is_selected(self):
        key = base64.b64encode(b"fixture-key").decode()
        with patch("scripts.controller.transport.subprocess.run", return_value=SimpleNamespace(stdout="host ssh-ed25519 " + key)):
            self.assertEqual(pinned_key(Path("fixture"), "host", fingerprint(key)), key)
            with self.assertRaises(ApprovalError): pinned_key(Path("fixture"), "host", "SHA256:" + "X" * 43)


class RemotePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name) / "nested/project"; self.base.mkdir(parents=True)
        a, r, _ = fixture(); a["approved_workspace"] = str(self.base)
        a["approved_remote_identity"] = {"uid": os.getuid(), "gid": os.getgid(), "hostname": socket.gethostname()}
        a["valid_until_utc"] = "2999-01-01T00:00:00Z"
        r["remote_output"] = str(self.base / "agent_work/runs/fixture-run")
        r["approved_workspace"] = str(self.base)
        self.data = b'"fixture only"\n'
        name = "runtime/kv_controller/__init__.py"
        self.ticket = {"approval": a, "request": r, "request_sha256": digest(r), "reservation_id": "a" * 32,
                       "deadline_utc": "2998-01-01T00:00:00Z", "assets_sha256": {name: hashlib.sha256(self.data).hexdigest()},
                       "ancestor_identity": {str(p): [p.stat().st_uid, p.stat().st_gid, stat.S_IMODE(p.stat().st_mode)] for p in (self.base, *self.base.parents)}}
        self.payload = {"operation": "prepare", "ticket": self.ticket, "assets": {name: {"sha256": hashlib.sha256(self.data).hexdigest(), "base64": base64.b64encode(self.data).decode()}}}

    def tearDown(self): self.temp.cleanup()

    def test_preparation_is_exclusive_and_preserves_existing_result(self):
        remote_rpc.dispatch(self.payload)
        root = Path(self.ticket["request"]["remote_output"])
        self.assertEqual((root / "runtime/kv_controller/__init__.py").read_bytes(), self.data)
        with self.assertRaises(ValueError): remote_rpc.dispatch(self.payload)
        self.assertEqual(json.loads((root / "ticket.json").read_text()), self.ticket)

    def test_corrupt_assets_do_not_create_remote_subtree(self):
        next(iter(self.payload["assets"].values()))["base64"] = base64.b64encode(b"changed").decode()
        with self.assertRaises(ValueError): remote_rpc.dispatch(self.payload)
        self.assertFalse((self.base / "agent_work").exists())

    def test_symlinked_existing_subtree_is_refused_without_touching_target(self):
        foreign = Path(self.temp.name) / "foreign"; foreign.mkdir()
        (self.base / "agent_work").symlink_to(foreign, target_is_directory=True)
        with self.assertRaises(ValueError): remote_rpc.dispatch(self.payload)
        self.assertEqual(list(foreign.iterdir()), [])

    def test_ownership_and_ancestry_change_are_refused(self):
        self.ticket["ancestor_identity"][str(self.base)][2] ^= 0o010
        with self.assertRaises(ValueError): remote_rpc.dispatch(self.payload)
        self.assertFalse((self.base / "agent_work").exists())

    def test_start_intent_prevents_second_launch_even_with_lost_reply(self):
        remote_rpc.dispatch(self.payload)
        with patch("scripts.controller.remote_rpc.subprocess.Popen", return_value=SimpleNamespace(pid=os.getpid())) as launch:
            remote_rpc.dispatch({"operation": "start", "ticket": self.ticket})
            with self.assertRaises(FileExistsError): remote_rpc.dispatch({"operation": "start", "ticket": self.ticket})
            self.assertEqual(launch.call_count, 1)

    def test_status_absent_never_creates_or_launches_anything(self):
        value = remote_rpc.dispatch({"operation": "status", "ticket": self.ticket})
        self.assertEqual(value["status"], "ABSENT"); self.assertFalse(value["relaunch_allowed"])
        self.assertFalse((self.base / "agent_work").exists())


if __name__ == "__main__": unittest.main()
