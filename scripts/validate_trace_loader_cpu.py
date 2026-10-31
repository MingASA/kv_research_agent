#!/usr/bin/env python3
"""Exercise actual mirrored AIPerf trace loaders/tokenization, without inference."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def validate(base: Path, model: Path) -> dict:
    from aiperf.common import random_generator
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.dataset.generator.prompt import PromptGenerator
    from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
    from aiperf.dataset.loader.burst_gpt import BurstGPTTraceDatasetLoader

    random_generator.init(20261026)
    tokenizer = Tokenizer.from_pretrained(str(model), resolve_alias=False)
    report = {'classification': 'confirmed_cpu_loader_only', 'gpu_used': False, 'model_inference_performed': False,
              'tokenizer': str(model), 'cases': {}, 'scheduling_or_server_verified': False}
    paths = list(sorted((base / 'v3_sources').glob('*.jsonl'))) + [base / 'traces/frozen/burstgpt_peak10m.csv']
    for path in paths:
        generator = PromptGenerator(prompts=None, prefix_prompts=None, tokenizer=tokenizer)
        loader_class = BurstGPTTraceDatasetLoader if path.suffix == '.csv' else MooncakeTraceDatasetLoader
        loader = loader_class(filename=path, prompt_generator=generator, default_block_size=512)
        all_sessions = loader.load_dataset()
        selected = dict(itertools.islice(all_sessions.items(), 8))
        conversations = loader.convert_to_conversations(selected)
        originals = [traces[0] for traces in selected.values()]
        tokens, lengths = [], []
        for conversation, trace in zip(conversations, originals, strict=True):
            turn = conversation.turns[0]
            if turn.timestamp != trace.timestamp or turn.max_tokens != trace.output_length:
                raise ValueError('Actual trace loader changed timestamp or per-request generation cap')
            prompt = turn.texts[0].contents[0]
            encoded = tokenizer.encode(prompt)
            tokens.append(encoded)
            lengths.append({'source_input': trace.input_length, 'reencoded_input': len(encoded),
                            'source_output': trace.output_length, 'loader_timestamp_ms': turn.timestamp})
        prefix_checks = []
        for i, j in itertools.combinations(range(len(tokens)), 2):
            left, right = getattr(originals[i], 'hash_ids', []) or [], getattr(originals[j], 'hash_ids', []) or []
            shared_blocks = sum(1 for _ in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(left, right)))
            expected = min(shared_blocks, originals[i].input_length // 512, originals[j].input_length // 512) * 512
            if expected:
                observed = sum(1 for _ in itertools.takewhile(lambda pair: pair[0] == pair[1], zip(tokens[i], tokens[j])))
                prefix_checks.append({'pair': [i, j], 'expected_full_source_prefix_tokens': expected,
                                      'observed_reencoded_prefix_tokens': observed, 'deficit': max(0, expected - observed)})
        report['cases'][path.name] = {
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'loaded_requests': len(all_sessions),
            'contiguous_first_requests_exercised': len(conversations), 'block_size': loader._block_size,
            'lengths': lengths, 'prefix_checks': prefix_checks,
            'exact_input_lengths': all(row['source_input'] == row['reencoded_input'] for row in lengths),
            'full_source_prefixes_preserved': all(row['deficit'] == 0 for row in prefix_checks)}
    report['limitations'] = [
        'Actual loader/prompt/decoder exercised on WSL with tokenizer metadata; H800 production CLI/wire still required',
        'Eight contiguous requests per file are interface probes, not statistical samples or a throughput benchmark',
        'Timestamp fields do not prove actual open-loop send timing; chat-template overhead also needs wire/server verification',
        'Trace output length becomes a generation cap; natural EOS may stop earlier, so actual original OSL replay is unverified']
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.base, args.tokenizer)
    with args.out.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'evidence': str(args.out), 'cases': len(report['cases']), 'gpu_used': False,
                     'exact_length_cases': [name for name, case in report['cases'].items() if case['exact_input_lengths']],
                     'prefix_preserved_cases': [name for name, case in report['cases'].items() if case['full_source_prefixes_preserved']]}))


if __name__ == '__main__':
    main()
