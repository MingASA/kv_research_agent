"""CPU-only immutable input preparation; never synthesize model predictions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from .durable import canonical_bytes, digest, secure_directory, write_new_bytes, write_new_json


TASKS = ("gsm8k", "narrativeqa", "qasper", "multifieldqa_en", "2wikimqa", "hotpotqa", "musique", "lcc", "repobench-p")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive_fixed_decode(source: Path, expected_sha256: str, out: Path) -> dict:
    if sha(source) != expected_sha256:
        raise ValueError("Frozen parent input SHA mismatch")
    parent = json.loads(source.read_text())
    source_manifest = json.loads(source.with_suffix(".json.manifest.json").read_text())
    if (source_manifest.get("canonical_sha256") != expected_sha256 or source_manifest.get("kind") != "synthetic"
            or source_manifest.get("isl") not in (16384, 24576) or source_manifest.get("osl") != 512):
        raise ValueError("Expected the frozen 16K/24K + 512 synthetic parent")
    sessions = parent.get("data", [])
    if len(sessions) != 136 or len({row.get("session_id") for row in sessions}) != 136:
        raise ValueError("Expected 136 unique frozen sessions")
    derived = copy.deepcopy(parent)
    payload_hashes = []
    parent_payload_hashes = []
    for original, selected in zip(sessions, derived["data"]):
        if len(original.get("payloads", [])) != 1:
            raise ValueError("Fixed-decode protocol does not split multi-turn sessions")
        payload = selected["payloads"][0]
        if payload.get("model") != "Qwen2.5-14B-Instruct" or payload.get("stop") or payload.get("stop_token_ids"):
            raise ValueError("Unexpected model or explicit stopping rule in frozen source")
        parent_payload_hashes.append(digest(original["payloads"][0]))
        payload.update(max_tokens=512, temperature=0, top_p=1, ignore_eos=True,
                       stream=True, stream_options={"include_usage": True})
        if payload["messages"] != original["payloads"][0]["messages"]:
            raise ValueError("Source prompt changed")
        payload_hashes.append(digest(payload))
    data = canonical_bytes(derived)
    manifest = {**source_manifest, "schema": "kv-derived-fixed-decode-v1", "kind": "synthetic",
                "canonical_sha256": hashlib.sha256(data).hexdigest(),
                "parent_input_sha256": expected_sha256, "parent_manifest_sha256": sha(source.with_suffix(".json.manifest.json")),
                "preparation_script_sha256": sha(Path(__file__)), "preparation_uses_gpu": False,
                "model_predictions_generated": False, "requests": 128, "warmup": 8,
                "control": {"max_tokens": 512, "ignore_eos": True, "temperature": 0, "top_p": 1},
                "parent_payload_sha256": parent_payload_hashes, "payload_sha256": payload_hashes}
    if out.exists():
        raise FileExistsError("Prepared input output must be a new directory")
    secure_directory(out)
    write_new_bytes(out / "canonical_inputs.json", data)
    write_new_json(out / "canonical_inputs.json.manifest.json", manifest)
    if sha(source) != expected_sha256:
        raise ValueError("Frozen source changed during preparation")
    return manifest


def restore_quality_records(batch: Path) -> tuple[dict, dict]:
    manifest = json.loads((batch / "shard_manifest.json").read_text())
    recovered = {task: {} for task in TASKS}
    for group, assignment in manifest["assignment"].items():
        if group not in {"gpu-" + str(index) for index in range(8)}:
            raise ValueError("Unknown historical shard")
        for task in TASKS:
            path = batch / "shards" / group / (task + ".jsonl")
            if sha(path) != manifest["shard_sha256"][group][task]:
                raise ValueError("Frozen shard SHA mismatch")
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            indices = assignment[task]
            if len(rows) != len(indices):
                raise ValueError("Frozen assignment length differs")
            for index, row in zip(indices, rows):
                if type(index) is not int or index < 0 or index in recovered[task]:
                    raise ValueError("Duplicate/invalid global source index")
                if task == "gsm8k" and row.get("_source_index") != index:
                    raise ValueError("GSM8K source index changed")
                recovered[task][index] = row
    result = {}
    for task in TASKS:
        count = 1319 if task == "gsm8k" else 50
        if set(recovered[task]) != set(range(count)) or manifest["global_counts"][task] != count:
            raise ValueError("Historical quality cohort is missing/duplicating source records")
        result[task] = [recovered[task][index] for index in range(count)]
        if len({digest(row) for row in result[task]}) != count:
            raise ValueError("Duplicate complete dataset records")
    return result, manifest


def pilot_indices(counts: dict[str, list[int]], seed: int = 20261026) -> dict[str, list[int]]:
    selected = {}
    for task in TASKS:
        lengths = counts[task]
        if len(lengths) != (1319 if task == "gsm8k" else 50) or any(type(value) is not int or value < 1 for value in lengths):
            raise ValueError("Pilot selection needs actual full-cohort CPU token counts")
        ordered = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
        if task == "gsm8k":
            chosen = []
            for quartile in range(4):
                block = ordered[len(ordered) * quartile // 4:len(ordered) * (quartile + 1) // 4]
                chosen.extend(sorted(block, key=lambda index: digest([seed, task, index]))[:16])
            selected[task] = sorted(chosen)
        else:
            selected[task] = sorted([ordered[0], ordered[-1]])
    return selected


def freeze_pilot(batch: Path, counts: dict[str, list[int]], out: Path) -> dict:
    records, source_manifest = restore_quality_records(batch)
    indices = pilot_indices(counts)
    manifest = {"schema": "kv-native-quality-pilot-v1", "classification": "proposal",
                "scope": "protocol_pilot_no_full_quality_claim", "preparation_uses_gpu": False,
                "model_predictions_generated": False, "samples_per_arm": 80,
                "parent_shard_manifest_sha256": sha(batch / "shard_manifest.json"),
                "source_task_sha256": {task: source_manifest["sources"][task]["sha256"] for task in TASKS},
                "selection_rule": "GSM8K:16 deterministic hash-ranked records per prompt-token quartile; LongBench:shortest+longest prompt per task",
                "seed": 20261026, "selected_source_indices": indices, "data_sha256": {},
                "selected_prompt_tokens": {task: [counts[task][index] for index in indices[task]] for task in TASKS},
                "preparation_script_sha256": sha(Path(__file__))}
    if out.exists():
        raise FileExistsError("Pilot output must be a new directory")
    secure_directory(out)
    for task in TASKS:
        data = b"".join(canonical_bytes(records[task][index]) for index in indices[task])
        path = out / (task + ".jsonl")
        write_new_bytes(path, data)
        manifest["data_sha256"][task] = hashlib.sha256(data).hexdigest()
    write_new_json(out / "pilot_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fixed = sub.add_parser("fixed-decode")
    fixed.add_argument("--source", type=Path, required=True)
    fixed.add_argument("--expected-sha256", required=True)
    fixed.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = derive_fixed_decode(args.source, args.expected_sha256, args.out)
    print(json.dumps({"canonical_sha256": result["canonical_sha256"], "preparation_uses_gpu": False,
                      "model_predictions_generated": False, "out": str(args.out)}))


if __name__ == "__main__":
    main()
