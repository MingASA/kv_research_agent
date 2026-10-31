"""Guarded controller CLI. Reads authority only from the fixed user-owned source.

No polling scheduler is installed. Each launch requires a complete frozen local
package; reconcile/status never issue a start or repeat a lost mutation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

from .approval import ApprovalError, TRUSTED_APPROVAL, load_approval_snapshot
from .dispatch import Dispatcher, build_assets
from .durable import digest, secure_directory, write_new_json
from .policy import SOURCE_FILES, validate_execution_authority, validate_request
from .quota import ACTIVE, QuotaLedger
from .transport import PinnedSSH


ROOT = Path(__file__).resolve().parents[2]
LEDGER_DIRECTORY = ROOT / "state/controller"


def _source(name: str) -> str:
    return (Path(__file__).parent / name).read_text()


def _local_package(path: Path) -> tuple[dict, dict, dict]:
    if (path.parent / 'INVALID.json').exists():
        raise ApprovalError("Local proposal was explicitly invalidated; use a fresh verified package")
    package = json.loads(path.read_text())
    if package.get("schema") != "kv-frozen-local-package-v1" or package.get("classification") != "proposal":
        raise ApprovalError("Expected a frozen proposal package")
    files = {}
    for remote, entry in package["assets"].items():
        relative = PurePosixPath(remote)
        local = Path(entry["local_path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) != remote:
            raise ApprovalError("Invalid package asset name")
        if local.resolve(strict=True) != local or not any(local.is_relative_to(root) for root in
                (ROOT, Path("/home/mingasa/projects/xllm_quant_kvcache-agent-native-quality-20261026"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-xqa-20261026"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-bf16-xqa-20261027"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-prefill-workspace-20261027"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-distribution-20261027"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-bf16q-decode-20261027"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-bf16q-quality-20261027"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-vscale-coverage-20261027"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-components-20261027"),
                 Path("/home/mingasa/projects/xllm_quant_kvcache-agent-nextgen-native-kv-20261027"))):
            raise ApprovalError("Package asset escaped the discovered local scientific source roots")
        files[remote] = local
    hashes, assets = build_assets(files)
    if hashes != {name: row["sha256"] for name, row in package["assets"].items()}:
        raise ApprovalError("Frozen package asset changed; create a new package/version")
    for name,reference in package.get('remote_asset_reuse',{}).items():
        if (name not in assets or not name.startswith(('input/','runtime/'))
                or set(reference)!={'sha256','reuse_from_run_id','reuse_parent_result_sha256'}
                or reference['sha256']!=hashes[name]):
            raise ApprovalError('Unbound or non-input remote asset reuse')
        assets[name]=reference
    # Exercise the exact stdlib remote decoder locally before any SSH mutation.
    # A policy-approved new trace kind must also pass transport type/size/SHA gates.
    from .remote_rpc import decode_dataset_asset
    archive = package['request'].get('dataset_archive')
    if archive is not None:
        selected = [(PurePosixPath(name), path.read_bytes()) for name, path in files.items()
                    if name in (archive.get('relative_path'), 'input/payload_trace.jsonl')]
        decode_dataset_asset({'request': package['request']}, selected)
    return package["request"], hashes, assets


def validate_frozen_policy(package_path: Path, request: dict, approval: dict) -> None:
    """Run the actual frozen CPU policy before any SSH write or reservation."""
    package=json.loads(package_path.read_text())
    entry=package['assets'].get('runtime/kv_controller/policy.py')
    if entry is None:
        raise ApprovalError('Missing frozen runtime policy')
    policy=Path(entry['local_path']);runtime=policy.parent.parent
    source=('import json,sys\nfrom pathlib import Path\nimport kv_controller.policy as policy\n'
            'p=json.load(sys.stdin)\n'
            'assert Path(policy.__file__).resolve()==Path(p["policy"])\n'
            'policy.validate_request(p["request"],p["approval"])\n'
            'print(json.dumps({"frozen_policy_request_valid":True}))\n')
    env={'PATH':os.environ['PATH'],'PYTHONPATH':str(runtime),'PYTHONDONTWRITEBYTECODE':'1',
         'CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    result=subprocess.run([sys.executable,'-B','-S','-c',source],
        input=json.dumps({'policy':str(policy.resolve()),'request':request,'approval':approval}),
        text=True,capture_output=True,env=env,cwd=runtime,timeout=30)
    if result.returncode or json.loads(result.stdout).get('frozen_policy_request_valid') is not True:
        raise ApprovalError('Frozen runtime policy refused actual manifest: '+result.stderr[-2048:])


def _ticket(run_id: str) -> dict:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id) is None:
        raise ApprovalError("Invalid run ID")
    return json.loads((LEDGER_DIRECTORY / "tickets" / (run_id + ".json")).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for operation in ("preflight", "launch"):
        command = sub.add_parser(operation)
        command.add_argument("--package", type=Path, required=True)
    for operation in ("reconcile", "stop"):
        command = sub.add_parser(operation)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        if args.operation in ("reconcile", "stop"):
            ticket = _ticket(args.run_id)
            # Preserve old machine pin for recovery even after authority expires/revokes.
            transport = PinnedSSH(ticket["approval"], LEDGER_DIRECTORY / "pins" / ticket["reservation_id"])
            with QuotaLedger(LEDGER_DIRECTORY) as ledger:
                dispatch = Dispatcher(ledger, transport, _source("remote_rpc.py"))
                result = dispatch.reconcile(ticket) if args.operation == "reconcile" else dispatch.stop(ticket)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        approval, approval_sha = load_approval_snapshot(TRUSTED_APPROVAL)
        now = datetime.datetime.now(datetime.timezone.utc)
        validate_execution_authority(approval, now)
        request, hashes, assets = _local_package(args.package)
        validate_request(request, approval)
        validate_frozen_policy(args.package,request,approval)
        config_name = "NEXTGEN_CONFIG.local.json" if request["schema"] == "kv-nextgen-run-v1" else "CONFIG.local.json"
        config = json.loads((ROOT / "state" / config_name).read_text())
        for field, expected in (("ssh_target", config["ssh_target"]), ("approved_host_fingerprint", config["known_host_ed25519_fingerprint"]),
                                ("approved_workspace", config["remote_base"]), ("server_binary", config["remote_new_binary_path"]),
                                ("server_source_commit", config["source_commit"]), ("binary_sha256", config["binary_sha256"])):
            if request.get(field) != expected:
                raise ApprovalError("v1 request differs from reconciled identity/provenance: " + field)
        # Preflight may create only local pin evidence and quota files; it never dispatches remote mutations.
        transport = PinnedSSH(approval, LEDGER_DIRECTORY / "pins" / "preflight")
        binary_cpp_sources = [name for name in request['backend_source_sha256'] if name.endswith(('.cpp', '.cc', '.cu'))]
        observed = transport.rpc(_source("discovery.py"), {**config, "image_id": request["image_id"],
                                 "hash_paths": [request["server_binary"]] + [request["source_repo"] + "/" + name for name in sorted(set(SOURCE_FILES) | set(binary_cpp_sources))]})
        if observed["source_sha256"][request["server_binary"]] != request["binary_sha256"] or observed["model_config_sha256"] != request["model_config_sha256"]:
            raise ApprovalError("Live binary/model provenance differs")
        from .nextgen import PAGES_KIND
        standalone_pages = (request.get('schema') == 'kv-nextgen-run-v1'
                            and request.get('workload', {}).get('kind') == PAGES_KIND
                            and request.get('server_binary_execution') == 'none')
        if (not standalone_pages and observed["source_sha256"][request["source_repo"] + "/xllm/core/runtime/worker_impl.cpp"]
                != request["backend_source_sha256"]["xllm/core/runtime/worker_impl.cpp"]):
            raise ApprovalError("Python-only runtime update changed C++ source; a separately built binary would be required")
        if any(observed['source_sha256'][request['source_repo'] + '/' + name] != request['backend_source_sha256'][name]
               for name in binary_cpp_sources):
            raise ApprovalError('Pinned API/attention C++ source differs from the existing binary source checkout')
        if observed["free_bytes"] < int(request["max_output_gib"] * 2**30) + 5 * 2**30:
            raise ApprovalError("Insufficient live shared disk headroom")
        with QuotaLedger(LEDGER_DIRECTORY) as ledger:
            for row in ledger.connection.execute("SELECT run_id FROM runs WHERE state IN ('DISPATCHING','RUNNING','UNKNOWN')").fetchall():
                prior = _ticket(row["run_id"])
                old_transport = PinnedSSH(prior["approval"], LEDGER_DIRECTORY / "pins" / prior["reservation_id"])
                Dispatcher(ledger, old_transport, _source("remote_rpc.py")).reconcile(prior)
            known = set(ledger.usage(approval)["known_run_ids"])
            if any(item["name"] not in known or item["symlink"] or item["uid"] != config["remote_identity"]["uid"]
                   for item in observed["run_inventory"]):
                raise ApprovalError("Unknown remote run subtree; reconcile ownership before writing")
            # Register archived runs so none can be relaunched under a historical run_id.
            import csv
            with (ROOT / "results/INDEX.csv").open() as stream:
                for row in csv.DictReader(stream):
                    name = row.get("run_id")
                    if name and not ledger.connection.execute("SELECT 1 FROM runs WHERE run_id=?", (name,)).fetchone():
                        ledger.register_history({name: digest({key: row.get(key) for key in
                                ("run_id", "manifest_path", "git_sha", "binary_sha256", "dataset_sha256")})})
            from .approval import validate_scope
            static = validate_scope(approval, request, ledger.usage(approval), now)
            if args.operation == "preflight":
                print(json.dumps({"can_launch": False, "static_scope_valid": True, "remote_writes": 0,
                                  "os_resource_sandbox": False, "observations": observed,
                                  "remaining_gates": ["durable reservation and unique preparation", "120s stable idle/reservations",
                                                      "container/driver visibility and independent watchdog admission"]}, ensure_ascii=False, indent=2))
                return
            row = ledger.reserve(approval, request, approval_sha, now)
            ticket = {"schema": "kv-controller-ticket-v1", "request": request, "request_sha256": digest(request),
                      "approval": approval, "approval_sha256": approval_sha, "reservation_id": row["reservation_id"],
                      "deadline_utc": (now + datetime.timedelta(seconds=request["max_wall_seconds"])).isoformat(),
                      "ancestor_identity": observed["ancestor_identity"], "assets_sha256": hashes,
                      "controller_git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()}
            tickets = LEDGER_DIRECTORY / "tickets"
            secure_directory(tickets)
            write_new_json(tickets / (request["run_id"] + ".json"), ticket)
            result = Dispatcher(ledger, transport, _source("remote_rpc.py")).launch(ticket, assets)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ApprovalError, OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"can_launch": False, "error": type(error).__name__ + ": " + str(error)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
