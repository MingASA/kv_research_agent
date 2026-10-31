#!/usr/bin/env python3
"""Freeze unmodified real ShareGPT requests with natural stopping, using CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from controller.durable import canonical_bytes, digest, secure_directory, write_new_bytes, write_new_json


def prepare(source: Path, tokenizer_path: Path, output: Path) -> dict:
    from transformers import AutoTokenizer
    frozen = json.loads(source.read_text())
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    sessions, lineage, exclusions = [], [], []
    candidates = list(frozen["records"])
    random.Random(frozen["seed"]).shuffle(candidates)
    for item in candidates:
        record = item["record"]
        conversation = record.get("conversations", [])
        messages = []
        for turn in conversation:
            role = {"human": "user", "user": "user", "system": "system", "gpt": "assistant",
                    "assistant": "assistant"}.get(turn.get("from"))
            if role not in ("user", "system", "assistant") or not isinstance(turn.get("value"), str):
                messages = []
                break
            messages.append({"role": role, "content": turn["value"]})
        user_positions = [index for index, message in enumerate(messages) if message["role"] == "user"]
        if not user_positions:
            exclusions.append({"source_index": item["source_index"], "reason": "no_valid_user_request"})
            continue
        # Replay an actual final user request with its unchanged available conversation history.
        messages = messages[:user_positions[-1] + 1]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if count + 1024 > 32768:
            exclusions.append({"source_index": item["source_index"], "reason": "outside_deployed_32k_context_envelope",
                               "prompt_tokens": count})
            continue
        payload = {"model": "Qwen2.5-14B-Instruct", "messages": messages, "temperature": 0,
                   "top_p": 1, "max_tokens": 1024, "stream": True, "stream_options": {"include_usage": True}}
        identity = "sharegpt-source-" + str(item["source_index"])
        sessions.append({"session_id": identity, "payloads": [payload]})
        lineage.append({"session_id": identity, "source_index": item["source_index"], "source_record_id": record.get("id"),
                        "source_record_sha256": digest(record), "source_user_messages_sha256": digest(messages),
                        "payload_sha256": digest(payload), "prompt_tokens_cpu": count,
                        "original_conversation_turns": len(conversation)})
    if len(sessions) < 528:
        raise ValueError("Insufficient unique real requests for 512 profiling + 16 warmup without repeats")
    if output.exists():
        raise FileExistsError("Output must be a new directory")
    secure_directory(output)
    data = canonical_bytes({"data": sessions})
    write_new_bytes(output / "canonical_inputs.json", data)
    counts = sorted(row["prompt_tokens_cpu"] for row in lineage)
    manifest = {"schema": "kv-sharegpt-production-input-v1", "classification": "confirmed_input_preparation_only",
                "production_scope": "real_final_user_requests_with_source_conversation_history_natural_stop", "model_inference_performed": False,
                "gpu_used": False, "source_path": frozen["source_path"], "source_sha256": frozen["source_sha256"],
                "source_records": frozen["source_records"], "frozen_selection_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "seed": frozen["seed"], "eligible_sessions": len(sessions), "excluded_records": exclusions,
                "request_order": "deterministically shuffled eligible source conversations; source indices retained",
                "canonical_sha256": hashlib.sha256(data).hexdigest(), "max_tokens": 1024, "ignore_eos": False,
                "prompt_tokens_cpu": {"min": counts[0], "p50": counts[len(counts)//2], "p95": counts[int(len(counts)*.95)],
                                      "max": counts[-1], "total": sum(counts)},
                "session_lineage": lineage,
                "tokenizer_metadata_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                              for path in tokenizer_path.iterdir() if path.is_file()},
                "limitation": "One genuine final user request per source conversation with reference history; no original arrival timestamps or turn-by-turn live session replay; no population-wide production coverage claimed",
                "preparation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_new_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.tokenizer, args.out)
    print(json.dumps({key: result[key] for key in ("eligible_sessions", "prompt_tokens_cpu", "canonical_sha256", "ignore_eos", "gpu_used")}))


if __name__ == "__main__":
    main()
