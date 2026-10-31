"""Local durable prepare/start/reconcile orchestration. No automatic retry of writes."""

from __future__ import annotations

import base64
import datetime
import json
from pathlib import Path

from .approval import ApprovalError, TRUSTED_APPROVAL, load_approval_snapshot
from .durable import digest, write_new_json
from .policy import validate_execution_authority, validate_request
from .quota import QuotaLedger


def build_assets(files: dict[str, Path]) -> tuple[dict, dict]:
    import hashlib
    hashes, encoded, total = {}, {}, 0
    for name, path in files.items():
        if path.is_symlink() or not path.is_file():
            raise ApprovalError("Asset must be an existing regular file")
        total += path.stat().st_size
        if path.stat().st_size > 32*2**20 or total > 64*2**20:
            raise ApprovalError("Local assets exceed the unchanged remote single-file/total transfer bounds")
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        encoded[name] = {"sha256": hashes[name], "base64": base64.b64encode(data).decode()}
    return hashes, encoded


class Dispatcher:
    def __init__(self, ledger: QuotaLedger, transport: object, rpc_source: str,
                 now: object = lambda: datetime.datetime.now(datetime.timezone.utc),
                 authority_loader: object = lambda: load_approval_snapshot(TRUSTED_APPROVAL)) -> None:
        self.ledger, self.transport, self.rpc_source = ledger, transport, rpc_source
        self.now, self.authority_loader = now, authority_loader

    def _check_authority(self, ticket: dict) -> None:
        current, sha = self.authority_loader()
        if current != ticket["approval"] or sha != ticket["approval_sha256"]:
            raise ApprovalError("User-issued approval changed; reconcile rather than dispatch")
        validate_execution_authority(current, self.now())
        validate_request(ticket["request"], current)
        deadline = datetime.datetime.fromisoformat(ticket["deadline_utc"].replace("Z", "+00:00"))
        if deadline <= self.now() + datetime.timedelta(seconds=180):
            raise ApprovalError("Insufficient time remains for two idle windows and cleanup")

    def launch(self, ticket: dict, assets: dict) -> dict:
        run_id = ticket["request"]["run_id"]
        row = self.ledger.get(run_id)
        if row["state"] != "RESERVED" or row["reservation_id"] != ticket["reservation_id"] or row["request_sha256"] != digest(ticket["request"]):
            raise ApprovalError("No matching undispatched durable reservation")
        self._check_authority(ticket)
        # Both preparation and process startup are mutations. Intent is durable before either SSH call.
        self.ledger.transition(run_id, "DISPATCHING", {"ticket_sha256": digest(ticket), "phase": "prepare"})
        try:
            # The original24MB trace needs over600s on the observed SSH link.
            # This is a bounded CPU transfer window, not a larger GPU budget.
            self.transport.rpc(self.rpc_source, {"operation": "prepare", "ticket": ticket, "assets": assets}, mutation=True, timeout=1200)
            self._check_authority(ticket)
            response = self.transport.rpc(self.rpc_source, {"operation": "start", "ticket": ticket}, mutation=True, timeout=90)
        except Exception as error:
            self.ledger.transition(run_id, "UNKNOWN", {"error": str(error), "relaunch_allowed": False})
            raise
        self.ledger.transition(run_id, "RUNNING", response)
        return response

    def reconcile(self, ticket: dict) -> dict:
        run_id = ticket["request"]["run_id"]
        row = self.ledger.get(run_id)
        if row["request_sha256"] != ticket["request_sha256"] or row["reservation_id"] != ticket["reservation_id"]:
            raise ApprovalError("Reconcile ticket differs from durable reservation")
        response = self.transport.rpc(self.rpc_source, {"operation": "status", "ticket": ticket})
        if row["state"] in ("SETTLED", "CANCELLED", "RESERVED"):
            return response
        result_file = response.get("files", {}).get("result.json")
        if result_file:
            evidence = result_file["value"]
            if digest(evidence) != result_file["sha256"]:
                raise ApprovalError("Remote result digest differs from transmitted canonical bytes")
            # A durable result and a fresh Docker query must both show release.
            if evidence.get("gpu_release_verified") is True and response.get("live_owned_containers") == []:
                self.ledger.settle(run_id, evidence)
                return response
        state = "RUNNING" if response.get("supervisor_alive") is True else "UNKNOWN"
        self.ledger.transition(run_id, state, {"readonly_status": response, "relaunch_allowed": False})
        return response

    def stop(self, ticket: dict) -> dict:
        row = self.ledger.get(ticket["request"]["run_id"])
        if row["state"] not in ("DISPATCHING", "RUNNING", "UNKNOWN") or row["reservation_id"] != ticket["reservation_id"]:
            raise ApprovalError("No owned active reservation to stop")
        # First inspect old state read-only; request a stop only for this exact immutable run.
        self.reconcile(ticket)
        if self.ledger.get(ticket["request"]["run_id"])["state"] == "SETTLED":
            return {"status": "ALREADY_SETTLED"}
        try:
            return self.transport.rpc(self.rpc_source, {"operation": "stop", "ticket": ticket}, mutation=True)
        except Exception as error:
            self.ledger.transition(ticket["request"]["run_id"], "UNKNOWN", {"stop_error": str(error)})
            raise
