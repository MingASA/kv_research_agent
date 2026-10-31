#!/usr/bin/env python3
"""Freeze a real ShareGPT/modern-FA3 service candidate; no remote writes or GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from controller.durable import digest, secure_directory, write_new_json, write_new_bytes
from controller.policy import PRODUCTION_SOURCES, XQA_SOURCES, XQA_OPERATORS, BF16_XQA_OPERATORS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/mingasa/projects/xllm_quant_kvcache-agent-native-quality-20261026")
XQA_SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-xqa-20261026')
BF16_XQA_SOURCE = Path('/home/mingasa/projects/xllm_quant_kvcache-agent-bf16-xqa-20261027')


def prepare(base_package: Path, run_id: str, out: Path, bundle: Path,
            profile: str = 'fa3', gpu_uuid: str | None = None,
            replicate: int = 0, research_note: str = '', max_wall_seconds: int = 7200) -> dict:
    if profile not in ('fa3','xqa','xqa-bf16-reference'):
        raise ValueError('Unknown production serving candidate')
    if type(replicate) is not int or replicate < 0:
        raise ValueError('Explicit nonnegative scientific replicate required')
    if type(max_wall_seconds) is not int or not 600 <= max_wall_seconds <= 7200:
        raise ValueError('Explicit bounded whole-job wall time required')
    xqa = profile in ('xqa', 'xqa-bf16-reference')
    candidate_source = BF16_XQA_SOURCE if profile == 'xqa-bf16-reference' else XQA_SOURCE if xqa else SOURCE
    base = json.loads(base_package.read_text())
    request = base["request"]
    remote = request["approved_workspace"] + "/agent_work/runs/" + run_id
    data = ROOT / "results/prepared/sharegpt-natural-production-20261026-v2"
    manifest = json.loads((data / "manifest.json").read_text())
    if (data / "INVALID.json").exists() or manifest["prompt_tokens_cpu"]["min"] < 3 or manifest["ignore_eos"] is not False:
        raise ValueError("Production input has not passed natural-stop/context preparation")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=candidate_source, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=candidate_source, text=True)
    if dirty:
        raise ValueError("Production source must be committed before freezing a bundle")
    files = {"runtime/kv_controller/" + name: ROOT / "scripts/controller" / name for name in
             ("__init__.py", "approval.py", "durable.py", "policy.py", "processes.py", "supervisor.py", "watchdog.py", "worker.py", "production.py")}
    files["runtime/kv_controller/evidence.py"] = ROOT / "scripts/audit_native_results.py"
    files["runtime_support/tokenizer_compat/sitecustomize.py"] = Path(base["assets"]["runtime_support/tokenizer_compat/sitecustomize.py"]["local_path"])
    files["input/canonical_inputs.json"] = data / "canonical_inputs.json"
    files["input/production_manifest.json"] = data / "manifest.json"
    files["input/source.bundle"] = bundle.absolute()
    request.update(schema="kv-production-run-v1", run_id=run_id, remote_output=remote, concurrency=128, replicate=replicate,
                   max_wall_seconds=max_wall_seconds, max_output_gib=10, prefix_cache=True,
                   python_source_repo=remote + "/source_repo", python_source_commit=commit,
                   dataset_path=remote + "/input/canonical_inputs.json", dataset_sha256=manifest["canonical_sha256"],
                   backend_source_sha256={name: sha(candidate_source / name) for name in (XQA_SOURCES if xqa else PRODUCTION_SOURCES)},
                   service={"max_seqs_per_batch": 128, "max_tokens_per_batch": 10240, "max_concurrent_requests": 200, "memory_utilization": .8},
                   operators={"bf16": {"planner": "fa3", "q": "bf16"},
                              "fp8_e4m3": {"planner": "fa3", "q": "fp8_e4m3", "q_scale": 1.0}},
                   workload={"kind": "sharegpt_natural", "requests": 512, "warmup": 16, "ignore_eos": False,
                             "source_content_sha256": manifest["source_sha256"],
                             "content_scope": manifest["production_scope"], "arrival_mode": "closed_loop_concurrency",
                             "eligible_source_requests": manifest["eligible_sessions"]},
                   research_reason="Production support decision: genuine ShareGPT conversation history and natural EOS, prefix caching and production scheduler, modern FA3 BF16 vs FP8-Q/FP8-KV under realistic sustained client concurrency; quality remains a separate mandatory gate.")
    if xqa:
        request['operators'] = XQA_OPERATORS
        request['workload']['kind'] = 'sharegpt_natural_xqa'
        request['research_reason'] = ('Production natural semantic serving with the exact XQA quality candidate: '
            'genuine frozen ShareGPT context, natural EOS/cap1024, resident FP8, FA2 BF16-Q prefill plus modern '
            'SM90 XQA decode versus FA3 BF16, prefix/Graph and real scheduler. Changed deployable operator config '
            'requires its own clean measurement; no reuse of old FA3 speed or quality as certification.')
    if profile == 'xqa-bf16-reference':
        request['max_output_gib'] = 3
        request['operators'] = BF16_XQA_OPERATORS
        request['workload']['kind'] = 'sharegpt_natural_xqa_bf16_reference'
        request['research_reason'] = ('Production baseline fairness: BF16 FA3 prefill plus SM90 XQA BF16 decode '
            'versus FP8 FA2 BF16-Q prefill plus XQA internal-FP8-Q decode. Same GPU, genuine frozen ShareGPT '
            'natural EOS/cap1024, prefix/Graph and real scheduler. New source only enables the BF16 profile '
            'and matching numerical/partial-page/padded-Graph operator admission; FP8 dispatch is unchanged. '
            'A fresh simultaneous-condition pair avoids joining old GPU/kernel/scheduling measurements; '
            'quality remains separately measured and has not certified this new BF16 reference.')
    if gpu_uuid is not None:
        request['gpu_uuids'] = [gpu_uuid]
    if research_note:
        request['research_reason'] += ' ' + research_note
    request["experiment_key"] = digest({key: request[key] for key in
          ("python_source_commit", "binary_sha256", "dataset_sha256", "backend_source_sha256", "image_id", "graph_mode",
           "prefix_cache", "fp8_scales", "operators", "service", "concurrency", "workload")})
    package = {"schema": "kv-frozen-local-package-v1", "classification": "proposal", "request": request,
               "gpu_job_launched": False, "production_decision_standard": "ops/PRODUCTION_DECISION_STANDARD.md",
               "assets": {name: {"local_path": str(path), "sha256": sha(path)} for name, path in files.items()}}
    if out.exists():
        raise FileExistsError("New immutable production package directory required")
    secure_directory(out)
    for name,path in list(files.items()):
        if name.startswith(('runtime/','runtime_support/')):
            frozen=out/'frozen_assets'/name;secure_directory(frozen.parent)
            write_new_bytes(frozen,path.read_bytes());files[name]=frozen
    package['assets']={name:{'local_path':str(path.absolute()),'sha256':sha(path)} for name,path in files.items()}
    write_new_json(out / "package.json", package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument('--profile',choices=('fa3','xqa','xqa-bf16-reference'),default='fa3')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--replicate',type=int,default=0)
    parser.add_argument('--research-note',default='')
    parser.add_argument('--max-wall-seconds',type=int,default=7200)
    args = parser.parse_args()
    result = prepare(args.base_package, args.run_id, args.out.absolute(), args.bundle, args.profile, args.gpu_uuid,
                     args.replicate,args.research_note,args.max_wall_seconds)
    print(json.dumps({"package": str(args.out / "package.json"), "experiment_key": result["request"]["experiment_key"],
                      "gpu_job_launched": False, "model_quality_validated": False}))


if __name__ == "__main__":
    main()
