#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Freeze real BFCL native-tool requests using the unchanged official adapter.

CPU input preparation only. No answers, model generation or external tool calls.
"""

from __future__ import annotations

import argparse
import ast
import copy
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.controller.durable import canonical_bytes, digest, secure_directory, write_new_json


SOURCE_COMMIT = '6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'
PREFIX = 'berkeley-function-call-leaderboard/bfcl_eval/'


def _load_sources(root: Path) -> dict[str, bytes]:
    files = {}
    for manifest_name in ('source_manifest.json', 'evaluator_source_manifest.json', 'adapter_constants_source_manifest.json'):
        manifest = json.loads((root / manifest_name).read_text())
        if manifest['commit'] != SOURCE_COMMIT or manifest.get('ground_truth_answers_fetched', False):
            raise ValueError('Require the pinned public source snapshot before reading evaluation answers')
        for entry in manifest['files']:
            path = root / entry['path']
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Public source file escaped its frozen local tree')
            data = path.read_bytes()
            if len(data) != entry['size'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('Pinned BFCL source bytes changed')
            files[entry['path']] = data
    return files


def _official_adapter(files: dict[str, bytes]) -> tuple[object, dict]:
    source = files[PREFIX + 'model_handler/utils.py']
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_cast_to_openai_type', 'convert_to_tool')]
    enum_nodes = [node for node in ast.parse(files[PREFIX + 'constants/enums.py']).body
                  if isinstance(node, ast.ClassDef) and node.name == 'ModelStyle']
    mappings = [node for node in ast.parse(files[PREFIX + 'constants/type_mappings.py']).body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == 'GORILLA_TO_OPENAPI' for target in node.targets)]
    if len(functions) != 2 or len(enum_nodes) != 1 or len(mappings) != 1:
        raise ValueError('Official adapter dependency contract changed')
    # Execute exactly these inspected, pinned definitions. This does not import
    # the BFCL provider SDKs or pretend the full upstream package was installed.
    module = ast.Module(body=enum_nodes + mappings + functions, type_ignores=[])
    namespace = {'Enum': Enum, 'copy': copy, 'json': json, 're': re}
    exec(compile(module, 'pinned_BFCL_official_conversion_definitions', 'exec'), namespace)
    def convert(functions: list[dict]) -> list[dict]:
        return namespace['convert_to_tool'](functions, namespace['GORILLA_TO_OPENAPI'],
                                             namespace['ModelStyle'].OPENAI_COMPLETIONS)
    return convert, {'mode': 'unchanged_official_AST_definitions_not_full_provider_package_import',
        'model_style': 'OPENAI_COMPLETIONS',
        'definition_AST_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
        'source_sha256': hashlib.sha256(source).hexdigest()}


def prepare(source: Path, tokenizer_path: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError('New immutable tool input directory required')
    from transformers import AutoTokenizer
    files = _load_sources(source)
    convert, adapter = _official_adapter(files)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    sessions, records, source_ids = [], [], set()
    for phase, category in (('warmup', 'live_relevance'), ('measurement', 'live_simple')):
        source_name = PREFIX + 'data/BFCL_v4_' + category + '.json'
        lines = files[source_name].splitlines()
        for index, line in enumerate(lines):
            row = json.loads(line)
            if row['id'] in source_ids or len(row['question']) != 1 or row['question'][0][-1]['role'] != 'user':
                raise ValueError('Source identity or single-turn conversation contract changed')
            source_ids.add(row['id'])
            tools = convert(row['function'])
            if (not tools or any(tool.get('type') != 'function' or
                    re.fullmatch(r'[A-Za-z0-9_-]{1,64}', tool['function']['name']) is None for tool in tools)
                    or len({tool['function']['name'] for tool in tools}) != len(tools)):
                raise ValueError('Official native tool conversion produced an incompatible or ambiguous function name')
            payload = {'model': 'Qwen2.5-14B-Instruct', 'messages': copy.deepcopy(row['question'][0]),
                'tools': tools, 'tool_choice': 'auto', 'temperature': 0, 'top_p': 1,
                'max_tokens': 1024, 'ignore_eos': False, 'stream': True,
                'stream_options': {'include_usage': True}}
            text = tokenizer.apply_chat_template(payload['messages'], tools=tools, tokenize=False, add_generation_prompt=True)
            count = len(tokenizer.encode(text, add_special_tokens=False))
            if not 0 < count <= 32768 - 1024:
                raise ValueError('Source tool context exceeds the actual model budget; do not silently truncate')
            session = {'session_id': 'bfclv4-' + row['id'], 'payloads': [payload]}
            sessions.append(session)
            records.append({'source_id': row['id'], 'session_id': session['session_id'], 'phase': phase,
                'source_file': source_name, 'source_line_zero_based': index,
                'source_line_sha256_without_LF': hashlib.sha256(line).hexdigest(),
                'payload_sha256': digest(payload), 'rendered_chat_tokens_cpu': count,
                'function_name_mapping': {original['name']: converted['function']['name']
                                          for original, converted in zip(row['function'], tools)}})
    if (len([row for row in records if row['phase'] == 'measurement']) != 258
            or len([row for row in records if row['phase'] == 'warmup']) != 16):
        raise ValueError('Expected all258 public live_simple and16 independent live_relevance warm requests')
    measured = {row['payload_sha256'] for row in records if row['phase'] == 'measurement'}
    if any(row['payload_sha256'] in measured for row in records if row['phase'] == 'warmup'):
        raise ValueError('Warm payload duplicates a measured request')
    secure_directory(out)
    write_new_json(out / 'tool_inputs.json', {'data': sessions})
    manifest = {'classification': 'confirmed_CPU_public_input_preparation_not_GPU_or_task_success',
        'source_commit': SOURCE_COMMIT, 'source_repo': 'https://github.com/ShishirPatil/gorilla',
        'license': 'Apache-2.0 as declared by pinned source dataset card and repository LICENSE',
        'requests': 258, 'warmup': 16, 'arrival_mode': 'closed_loop_no_original_production_timestamps',
        'record_order': 'all16 live_relevance sourceorder warm then all258 live_simple sourceorder',
        'native_tool_parser_required': 'qwen25', 'natural_EOS': True, 'max_tokens': 1024,
        'official_adapter': adapter, 'records': records,
        'source_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        'dataset_sha256': hashlib.sha256((out / 'tool_inputs.json').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'ground_truth_used_for_selection_or_prompt': False, 'model_generation_or_GPU_started': False,
        'limits': ['Single-turn tool-name/arguments task proxy, not full multi-step agent success',
                   'No external tool invocation or code execution; independent evaluator/runtime validation pending',
                   'CPU tokenizer lengths still require actual server usage verification']}
    write_new_json(out / 'manifest.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'tokenizer', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.source, args.tokenizer, args.out)
    print(json.dumps({key: value[key] for key in ('classification', 'source_commit', 'requests', 'warmup', 'dataset_sha256')}))


if __name__ == '__main__':
    main()
