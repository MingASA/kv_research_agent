#!/usr/bin/env python3
"""Freeze one concrete fixed-decode package locally; never launch or write remotely."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from controller.durable import digest, secure_directory, write_new_json
from controller.policy import SOURCE_FILES


ROOT = Path(__file__).resolve().parents[1]


def prepare(isl: int, run_id: str, out: Path) -> dict:
    config = json.loads((ROOT / "state/CONFIG.local.json").read_text())
    base = config["remote_base"]
    remote = base + "/agent_work/runs/" + run_id
    local_source = Path("/home/mingasa/projects/xllm_quant_kvcache-agent-native-quality-20261026")
    input_root = ROOT / f"results/prepared/fixed-decode-{isl // 1024}k512-20261026T075846Z"
    parent = ROOT / "results/mirrors/takeover-20261026T040426Z-attempt02/files/support/inputs/aiperf"
    files = {"runtime/kv_controller/" + name: ROOT / "scripts/controller" / name for name in
             ("__init__.py", "approval.py", "durable.py", "policy.py", "processes.py", "supervisor.py", "watchdog.py", "worker.py")}
    files["runtime/kv_controller/evidence.py"] = ROOT / "scripts/audit_native_results.py"
    for name in ("aiperf_h800_v4_2_native.py", "aiperf_h800_v2_1.py", "workload_prepare_v3.py"):
        files["runtime_support/" + name] = parent / name
    files["runtime_support/tokenizer_compat/sitecustomize.py"] = parent / "aiperf_v3_tokenizer_compat/sitecustomize.py"
    files["input/canonical_inputs.json"] = input_root / "canonical_inputs.json"
    files["input/canonical_inputs.json.manifest.json"] = input_root / "canonical_inputs.json.manifest.json"
    model = ROOT / "results/mirrors/quality-protocols-20261026T075846Z/files/model_metadata/config.json"
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    request = {"schema": "kv-native-run-v1", "run_id": run_id,
               "ssh_target": config["ssh_target"], "approved_host_fingerprint": config["known_host_ed25519_fingerprint"],
               "approved_workspace": base, "remote_output": remote,
               "source_repo": str(Path(config["remote_new_binary_path"]).parents[3]),
               "server_binary": config["remote_new_binary_path"], "server_source_commit": config["source_commit"],
               "binary_sha256": config["binary_sha256"], "model_path": config["remote_model_path"],
               "model_config_sha256": sha(model), "dataset_path": remote + "/input/canonical_inputs.json",
               "dataset_sha256": sha(input_root / "canonical_inputs.json"),
               "backend_source_sha256": {name: sha(local_source / name) for name in SOURCE_FILES},
               "image_id": "sha256:a77472b8bcd25797bad288712f7abd676beae6264b353de301ec43a08bd39dc2",
               "gpu_uuids": ["GPU-0c628a64-5f31-3501-fa3c-723f5b69254a"], "arms": ["bf16", "fp8_e4m3"],
               "max_wall_seconds": 3600, "max_output_gib": 5, "graph_mode": "cudagraphs", "prefix_cache": False,
               "fp8_scales": {"key": .125, "value": .0625}, "concurrency": 4 if isl == 16384 else 2,
               "workload": {"kind": "fixed_decode", "requests": 128, "warmup": 8, "isl": isl,
                            "output_tokens": 512, "ignore_eos": True},
               "research_reason": "New fixed 512-token decode control removes the unequal natural stopping lengths in prior ceff paired services; tests H2 decode pressure at matched input and output lengths.",
               "replicate": 0}
    # Scientific identity omits run directory, UUID and sequence order; repeat requires a new explicit replicate.
    request["experiment_key"] = digest({key: request[key] for key in
              ("server_source_commit", "binary_sha256", "dataset_sha256", "backend_source_sha256", "image_id", "graph_mode",
               "prefix_cache", "fp8_scales", "concurrency", "workload")})
    package = {"schema": "kv-frozen-local-package-v1", "classification": "proposal", "request": request,
               "gpu_job_launched": False,
               "assets": {name: {"local_path": str(path), "sha256": sha(path)} for name, path in files.items()}}
    if out.exists():
        raise FileExistsError("New package directory required")
    secure_directory(out)
    write_new_json(out / "package.json", package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isl", type=int, choices=(16384, 24576), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.isl, args.run_id, args.out)
    print(json.dumps({"run_id": args.run_id, "package": str(args.out / "package.json"),
                      "experiment_key": result["request"]["experiment_key"], "gpu_job_launched": False}))


if __name__ == "__main__":
    main()
