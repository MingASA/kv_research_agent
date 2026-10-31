#!/usr/bin/env python3
"""Freeze anonymous trace payloads and timing separately; CPU-only, no inference.

Original source timestamps/hash IDs feed the installed AIPerf loader unchanged.
Only after prompt generation are relative timestamps written to a new payload
trace, so later clock scaling cannot silently regenerate different prefixes.
"""

from __future__ import annotations

import argparse
import array
import collections
import hashlib
import json
from pathlib import Path
import sys

from controller.durable import canonical_bytes, secure_directory, write_new_bytes, write_new_json


def freeze(base: Path, model: Path, out: Path) -> dict:
    from aiperf.common import random_generator
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.dataset.generator.prompt import PromptGenerator
    from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
    from aiperf.dataset.loader.burst_gpt import BurstGPTTraceDatasetLoader
    from transformers import AutoTokenizer

    if sys.byteorder != "little":
        raise ValueError("Prefix audit explicitly encodes uint32 little-endian token IDs")
    if out.exists():
        raise FileExistsError("Trace preparation requires a new immutable output directory")
    secure_directory(out)
    random_generator.init(20261026)
    tokenizer = Tokenizer.from_pretrained(str(model), resolve_alias=False)
    hf = AutoTokenizer.from_pretrained(model, local_files_only=True)
    files = [(name, base / "v3_sources" / (name + "_trace.jsonl"))
             for name in ("conversation", "toolagent", "synthetic")]
    files.append(("burstgpt", base / "traces/frozen/burstgpt_peak10m.csv"))
    report = {"schema": "kv-production-trace-payloads-v1", "classification": "cpu_prepared_no_gpu_result",
              "gpu_used": False, "model_inference_performed": False, "seed": 20261026,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "traces": {}}
    for family, path in files:
        source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if family != "burstgpt":
            prior = base / "v3_frozen" / (path.stem + "_264_seed20261020.jsonl.manifest.json")
            if source_sha != json.loads(prior.read_text())["source_sha256"]:
                raise ValueError("Original Mooncake source changed from the verified archive")
        generator = PromptGenerator(prompts=None, prefix_prompts=None, tokenizer=tokenizer)
        cls = BurstGPTTraceDatasetLoader if family == "burstgpt" else MooncakeTraceDatasetLoader
        loader = cls(filename=path, prompt_generator=generator, default_block_size=512)
        sessions = loader.load_dataset()
        items = list(sessions.items())
        if any(len(traces) != 1 for _, traces in items):
            raise ValueError("Multi-turn sessions require a separate schedule-preserving protocol")
        origin = items[0][1][0].timestamp
        eligible = [i for i, (_, rows) in enumerate(items) if rows[0].input_length > 0
                    and rows[0].output_length > 0 and rows[0].input_length + rows[0].output_length <= 31744]
        if family == "burstgpt":
            old = json.loads((path.parent / "manifest.json").read_text())
            width = old["burst_window_seconds"] * 1000
            start = (5 + old["burst_peak_window_index"] * old["burst_window_seconds"]) * 1000
            if hashlib.sha256(path.read_bytes()).hexdigest() != old["files_sha256"][path.name]:
                raise ValueError("Previously verified BurstGPT peak changed")
        else:
            width = 300000
            last = max(rows[0].timestamp for _, rows in items)
            bins = collections.Counter(int((items[i][1][0].timestamp - origin) // width) for i in eligible
                                       if origin + (int((items[i][1][0].timestamp - origin) // width) + 1) * width <= last)
            chosen = min(bins, key=lambda value: (-bins[value], value))
            start = origin + chosen * width
        indices = [i for i in eligible if start <= items[i][1][0].timestamp < start + width]
        selected = {items[i][0]: items[i][1] for i in indices}
        conversations = loader.convert_to_conversations(selected)
        if len(conversations) != len(indices):
            raise ValueError("Actual loader changed the number/order of selected sessions")
        directory = out / family
        secure_directory(directory)
        rows, mapping, prefixes, corrections = [], [], {}, []
        prefix_checks = 0
        first = items[indices[0]][1][0].timestamp
        for index, conversation in zip(indices, conversations, strict=True):
            trace = items[index][1][0]
            turn = conversation.turns[0]
            prompt = turn.texts[0].contents[0]
            ids = tokenizer.encode(prompt)
            if len(ids) != trace.input_length:
                before = len(ids)
                prompt = generator._decode_to_exact_len(ids, trace.input_length)
                ids = tokenizer.encode(prompt)
                corrections.append({"source_index": index, "before": before, "after": len(ids),
                                    "target": trace.input_length})
            if len(ids) != trace.input_length or turn.timestamp != trace.timestamp or turn.max_tokens != trace.output_length:
                failure = {"family": family, "source_index": index, "source_input": trace.input_length,
                           "reencoded_input": len(ids), "source_timestamp": trace.timestamp,
                           "turn_timestamp": turn.timestamp, "source_output": trace.output_length,
                           "turn_max_tokens": turn.max_tokens}
                write_new_json(directory / "PREPARATION_FAILED.json", failure)
                raise ValueError("Actual loader trace mismatch: " + json.dumps(failure))
            messages = [{"role": "user", "content": prompt}]
            rendered = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chat_count = len(hf.encode(rendered, add_special_tokens=False))
            if chat_count + trace.output_length > 32768:
                raise ValueError("Rendered chat exceeds the declared model context")
            cumulative = hashlib.sha256()
            hashes = getattr(trace, "hash_ids", None) or []
            for block in range(min(len(hashes), len(ids) // 512)):
                cumulative.update(array.array("I", ids[block * 512:(block + 1) * 512]).tobytes())
                key, observed = tuple(hashes[:block + 1]), cumulative.hexdigest()
                if key in prefixes:
                    prefix_checks += 1
                    if prefixes[key] != observed:
                        raise ValueError("Source shared prefix changed after actual model tokenization")
                else:
                    prefixes[key] = observed
            replay_id = family + "-source-" + str(index)
            payload = {"model": "Qwen2.5-14B-Instruct", "messages": messages, "temperature": 0,
                       "top_p": 1, "max_tokens": trace.output_length, "ignore_eos": True,
                       "stream": True, "stream_options": {"include_usage": True}}
            rows.append({"session_id": replay_id, "timestamp": trace.timestamp - first,
                         "output_length": trace.output_length, "payload": payload})
            mapping.append({"replay_id": replay_id, "source_data_row_index_zero_based": index,
                            "source_timestamp_ms": trace.timestamp, "source_input_tokens": trace.input_length,
                            "rendered_chat_tokens_cpu": chat_count, "source_output_tokens": trace.output_length,
                            "payload_sha256": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
                            "logical_phase": "warm_cache" if trace.timestamp < start + 60000 else "measurement"})
        encoded = b"".join(canonical_bytes(row) for row in rows)
        write_new_bytes(directory / "payload_trace.jsonl", encoded)
        info = {"source": str(path), "source_sha256": source_sha,
                "loaded_source_requests": len(items), "eligible_source_requests": len(eligible),
                "excluded_source_requests": len(items) - len(eligible), "window_begin_ms": start,
                "window_end_ms": start + width, "first_retained_timestamp_ms": first,
                "selected_requests": len(indices), "input_context_filter_plus_output": 31744,
                "warm_cache_source_seconds": 60, "clock_origin": "first_retained_arrival",
                "arrival_scaling": 1.0, "prefix_block_source_tokens": 512,
                "shared_prefix_token_sha_checks": prefix_checks, "reencoded_lengths_all_exact": True,
                "parallel_decode_length_corrections": corrections,
                "length_correction_method": "installed AIPerf _decode_to_exact_len; all shared full-prefix token SHAs still required",
                "payload_trace_sha256": hashlib.sha256(encoded).hexdigest(), "records": mapping,
                "logical_phase_counts": dict(collections.Counter(row["logical_phase"] for row in mapping)),
                "source_input_tokens_total": sum(row["source_input_tokens"] for row in mapping),
                "source_output_tokens_total": sum(row["source_output_tokens"] for row in mapping)}
        write_new_json(directory / "manifest.json", info)
        report["traces"][family] = {key: value for key, value in info.items() if key != "records"}
    report["limitations"] = ["Anonymous lengths/arrivals/prefixes, not original semantic conversations",
        "Recorded per-request token work is emulated with ignore_eos, not natural chat quality",
        "Logical first60s warm-cache requests share one continuous schedule; do not drain/reset an AIPerf warmup phase",
        "CPU tokenizer/loader checks do not prove actual H800 usage, arrival timing or performance",
        "Mooncake peak300s window selected by eligible request count; BurstGPT retains its independently verified peak600s",
        "Any future clock scaling must preserve these frozen payload bytes; no synthesis hash-ID rewriting"]
    write_new_json(out / "manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "tokenizer", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.base.absolute(), args.tokenizer.absolute(), args.out.absolute())
    print(json.dumps({"out": str(args.out), "gpu_used": False,
                      "requests": {name: row["selected_requests"] for name, row in result["traces"].items()}}))


if __name__ == "__main__":
    main()
