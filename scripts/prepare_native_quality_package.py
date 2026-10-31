#!/usr/bin/env python3
"""Freeze the native 80-record quality pilot and reviewed data audit, CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from controller.durable import digest, secure_directory, write_new_json
from controller.protocols import TASKS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/mingasa/projects/xllm_quant_kvcache-agent-native-quality-20261026")


def prepare(base_package: Path, run_id: str, out: Path) -> dict:
    original = json.loads(base_package.read_text())
    request = original["request"]
    base = request["approved_workspace"]
    remote = base + "/agent_work/runs/" + run_id
    pilot = ROOT / "results/prepared/native-quality-pilot-20261026T075846Z"
    source_data = ROOT / "results/mirrors/quality-protocols-20261026T075846Z/files/support/inputs"
    scorer = source_data / "longbench-scorer"
    shots = source_data / "quality/eval-data/gsm8k-3101c7d5072418e28b9008a6636bde82a006892c/prepared/five_train_examples.json"
    sys.path.insert(0, str(SOURCE))
    from tools import run_qwen25_joint_kv_quality as runner
    args = runner._parser().parse_args(["--model", str(ROOT / "results/mirrors/quality-protocols-20261026T075846Z/files/model_metadata"),
              "--gsm8k", str(pilot / "gsm8k.jsonl"), "--gsm8k-shots", str(shots), "--longbench-data", str(pilot),
              "--longbench-repo", str(scorer), "--attention-backend", "native-flashinfer", "--graph-mode", "cudagraphs",
              "--cohort-scope", "pilot", "--output", str(out / "unused-dry-output")])
    audit = runner._data_audit(args, {task: pilot / (task + ".jsonl") for task in TASKS})
    for task, item in audit["datasets"].items():
        item["path"] = remote + "/input/pilot/" + task + ".jsonl"
    if out.exists():
        raise FileExistsError("New quality package directory required")
    secure_directory(out)
    write_new_json(out / "approved_data_manifest.json", {"data_audit": audit})
    files = {name: Path(entry["local_path"]) for name, entry in original["assets"].items()
             if name.startswith("runtime/")}
    for name in ("run_qwen25_joint_kv_quality.py", "eval_kv_cache_quality.py", "qwen25_experiment_model.py", "native_kv_quality_contract.py"):
        files["tools_source/tools/" + name] = SOURCE / "tools" / name
    files["tools_source/scripts/logger.py"] = SOURCE / "scripts/logger.py"
    for task in TASKS:
        files["input/pilot/" + task + ".jsonl"] = pilot / (task + ".jsonl")
    files["input/pilot/pilot_manifest.json"] = pilot / "pilot_manifest.json"
    files["input/approved_data_manifest.json"] = out / "approved_data_manifest.json"
    files["input/five_train_examples.json"] = shots
    for name in ("metrics.py", "config/dataset2prompt.json", "config/dataset2maxlen.json"):
        files["input/scorer/LongBench/" + name] = scorer / "LongBench" / name
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    request.update(run_id=run_id, remote_output=remote, concurrency=1, arms=["bf16", "fp8_e4m3"],
                   dataset_path=remote + "/input/pilot/pilot_manifest.json", dataset_sha256=sha(pilot / "pilot_manifest.json"),
                   research_reason="First matched native FlashInfer Graph fixed-scale BF16/FP8 quality pilot: frozen 80 source records with natural EOS, separate from legacy Triton/eager scores.",
                   workload={"kind": "native_quality", "cohort": "pilot", "samples_per_arm": 80, "ignore_eos": False,
                             "runner_path": remote + "/tools_source/tools/run_qwen25_joint_kv_quality.py",
                             "runner_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
                             "longbench_repo": remote + "/input/scorer", "gsm8k_shots": remote + "/input/five_train_examples.json",
                             "approved_data_manifest": remote + "/input/approved_data_manifest.json"})
    request["experiment_key"] = digest({key: request[key] for key in
              ("server_source_commit", "binary_sha256", "dataset_sha256", "backend_source_sha256", "image_id", "graph_mode",
               "prefix_cache", "fp8_scales", "concurrency")} | {"protocol": "native_fixed_scale_quality_pilot_v1",
                  "runner_source_commit": request["workload"]["runner_source_commit"], "scorer_sha256": sha(scorer / "LongBench/metrics.py")})
    package = {"schema": "kv-frozen-local-package-v1", "classification": "proposal", "request": request,
               "gpu_job_launched": False,
               "assets": {name: {"local_path": str(path), "sha256": sha(path)} for name, path in files.items()}}
    write_new_json(out / "package.json", package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package = prepare(args.base_package, args.run_id, args.out.absolute())
    print(json.dumps({"package": str(args.out / "package.json"), "samples_per_arm": 80,
                      "experiment_key": package["request"]["experiment_key"], "gpu_job_launched": False}))


if __name__ == "__main__":
    main()
