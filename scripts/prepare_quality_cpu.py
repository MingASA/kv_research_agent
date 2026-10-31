#!/usr/bin/env python3
"""Tokenize/validate a frozen native-quality pilot locally, without a model run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

from controller.durable import write_new_json
from controller.protocols import TASKS, freeze_pilot, restore_quality_records, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Output must be new")
    sys.path.insert(0, str(args.source_repo.resolve()))
    from tools import eval_kv_cache_quality as evaluator
    from transformers import AutoTokenizer

    records, _ = restore_quality_records(args.batch)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    prompt_formats, limits, metrics, _ = evaluator._longbench_assets(args.scorer)
    shots = json.loads(args.shots.read_text())
    if len(shots) != 5:
        raise ValueError("Expected frozen five-shot GSM8K examples")
    # Exercise real official scorer functions; importing dependencies alone is insufficient.
    probes = {"qa_f1": metrics.qa_f1_score("blue", "blue"), "code_sim": metrics.code_sim_score("value = 1", "value = 1")}
    counts = {}
    caps = {}
    for task in TASKS:
        prompts = [evaluator._gsm_prompt(row["question"], shots) if task == "gsm8k" else
                   prompt_formats[task].format(context=row["context"], input=row["input"]) for row in records[task]]
        chat = task not in ("lcc", "repobench-p")
        counts[task] = [evaluator._token_count(tokenizer, prompt, chat) for prompt in prompts]
        caps[task] = 512 if task == "gsm8k" else int(limits[task])
        if any(number + caps[task] > evaluator.MAX_CONTEXT_TOKENS for number in counts[task]):
            raise ValueError("Frozen task exceeds declared native context budget: " + task)
    pilot = freeze_pilot(args.batch, counts, args.out)
    versions = {name: importlib.metadata.version(name) for name in
                ("transformers", "tokenizers", "jinja2", "jieba", "fuzzywuzzy", "rouge", "nltk")}
    report = {"schema": "kv-native-quality-cpu-preflight-v1", "classification": "confirmed_cpu_only",
              "model_inference_performed": False, "gpu_used": False, "full_cohort_counts": {task: len(rows) for task, rows in records.items()},
              "pilot_counts": {task: len(pilot["selected_source_indices"][task]) for task in TASKS},
              "prompt_tokens": {task: {"min": min(counts[task]), "max": max(counts[task]),
                                        "generation_cap": caps[task], "selected": pilot["selected_prompt_tokens"][task]} for task in TASKS},
              "official_scorer_probes": probes, "versions": versions,
              "scorer_provenance": {name: sha(args.scorer / "LongBench" / name) for name in
                                    ("metrics.py", "config/dataset2prompt.json", "config/dataset2maxlen.json")},
              "tokenizer_metadata_sha256": {path.name: sha(path) for path in args.tokenizer.iterdir() if path.is_file()},
              "shots_sha256": sha(args.shots), "weights_loaded_or_hashed": False,
              "pilot_manifest_sha256": sha(args.out / "pilot_manifest.json"),
              "limitation": "CPU tokenizer/scorer validation on WSL; production H800 container import and actual native generation remain unverified"}
    write_new_json(args.out / "cpu_preflight.json", report)
    print(json.dumps({"pilot_samples_per_arm": 80, "tasks": len(TASKS), "gpu_used": False,
                      "model_inference_performed": False, "out": str(args.out)}))


if __name__ == "__main__":
    main()
