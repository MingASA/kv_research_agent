#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Exercise frozen native tool inputs and the official checker without a model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.bfcl_evaluator import load_official_checker, score_native_tool_calls
from scripts.controller.durable import digest, write_new_json
from scripts.prepare_tool_calls import PREFIX, SOURCE_COMMIT


def _first_value(choices: list) -> object:
    if not choices:
        # Preserve the empty candidate list as an explicit upstream-reference
        # limitation. This placeholder is not claimed to be a valid answer.
        return []
    return copy.deepcopy(next((value for value in choices if value != ''), choices[0]))


def _dictionary_value(value: dict) -> dict:
    return {key: _first_value(choices) for key, choices in value.items()}


def _representative_arguments(description: dict, answers: dict) -> dict:
    parameters = {}
    for name, choices in answers.items():
        if name not in description['parameters']['required'] and '' in choices:
            continue
        value = _first_value(choices)
        schema = description['parameters']['properties'][name]
        if schema['type'] == 'dict' and isinstance(value, dict):
            value = _dictionary_value(value)
        elif schema['type'] == 'array' and schema.get('items', {}).get('type') == 'dict':
            value = [_dictionary_value(item) for item in value]
        parameters[name] = value
    return parameters


def validate(source: Path, prepared: Path) -> dict:
    frozen = json.loads((prepared / 'manifest.json').read_text())
    dataset = (prepared / 'tool_inputs.json').read_bytes()
    if hashlib.sha256(dataset).hexdigest() != frozen['dataset_sha256']:
        raise ValueError('Frozen request cohort changed after answer download')
    answers_manifest = json.loads((source / 'post_freeze_answer_manifest.json').read_text())
    if (answers_manifest['commit'] != SOURCE_COMMIT
            or answers_manifest['pre_gold_dataset_sha256'] != frozen.get('parent_source_dataset_sha256', frozen['dataset_sha256'])
            or answers_manifest['answers_fetched_only_after_prompt_freeze'] is not True):
        raise ValueError('Answers do not bind the already frozen input')
    answer_entry = answers_manifest['files'][0]
    answer_bytes = (source / answer_entry['path']).read_bytes()
    if (hashlib.sha256(answer_bytes).hexdigest() != answer_entry['sha256']
            or len(answer_bytes) != answer_entry['size']):
        raise ValueError('Pinned reference answers changed')
    answers = {row['id']: row for row in map(json.loads, answer_bytes.splitlines())}
    source_bytes = (source / (PREFIX + 'data/BFCL_v4_live_simple.json')).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != frozen['source_sha256'][PREFIX + 'data/BFCL_v4_live_simple.json']:
        raise ValueError('Pinned original task records changed')
    tasks = {row['id']: row for row in map(json.loads, source_bytes.splitlines())}
    records = [row for row in frozen['records'] if row['phase'] == 'measurement']
    if len(records) != 258 or {row['source_id'] for row in records} != set(tasks) or set(tasks) != set(answers):
        raise ValueError('Original task/answer/full frozen request IDs differ')
    checker, provenance = load_official_checker(source)
    canonical, negatives, empty_candidates = [], [], []
    for record in records:
        task, gold = tasks[record['source_id']], answers[record['source_id']]['ground_truth']
        if len(task['function']) != 1 or len(gold) != 1:
            raise ValueError('Only original single-function tasks are admitted')
        description = task['function'][0]
        function_name, values = next(iter(gold[0].items()))
        if function_name != description['name']:
            raise ValueError('Reference function does not bind original tool schema')
        for name, choices in values.items():
            if choices == []:
                empty_candidates.append({'source_id': record['source_id'], 'parameter': name,
                    'required': name in description['parameters']['required'],
                    'schema_type': description['parameters']['properties'][name]['type']})
        arguments = _representative_arguments(description, values)
        calls = [{'id': 'CPU_fixture_only', 'type': 'function', 'function': {
            'name': record['function_name_mapping'][function_name],
            'arguments': json.dumps(arguments, ensure_ascii=False, allow_nan=False)}}]
        result = score_native_tool_calls(checker, task['function'], calls, gold)
        canonical.append({'source_id': record['source_id'], 'valid': result['task_correct'],
                          'result': result['official_result']})
        wrong = copy.deepcopy(calls)
        wrong[0]['function']['name'] = 'CPU_unexpected_function'
        rejected = score_native_tool_calls(checker, task['function'], wrong, gold)
        empty = score_native_tool_calls(checker, task['function'], [], gold)
        negatives.append({'source_id': record['source_id'], 'wrong_function_rejected': not rejected['task_correct'],
                          'no_call_rejected': not empty['task_correct']})
    first_task = tasks[records[0]['source_id']]
    first_gold = answers[records[0]['source_id']]['ground_truth']
    malformed = {}
    for name, arguments in {'duplicate_keys': '{"x":1,"x":2}', 'NaN': '{"x":NaN}',
                            'overflow': '{"x":1e999}', 'array_root': '[1]',
                            'python_code': '__import__("os").system("false")'}.items():
        calls = [{'id': 'CPU_fixture_only', 'type': 'function',
                  'function': {'name': 'get_user_info', 'arguments': arguments}}]
        malformed[name] = not score_native_tool_calls(checker, first_task['function'], calls, first_gold)['native_contract_valid']
    # A production response can be structurally valid but semantically wrong;
    # conversely the official checker intentionally tolerates string formatting.
    fixture = [{'name': 'calendar.create_event', 'parameters': {'type': 'dict', 'required': ['title', 'days'],
        'properties': {'title': {'type': 'string'}, 'days': {'type': 'integer'}}}}]
    gold = [{'calendar.create_event': {'title': ['Planning Meeting'], 'days': [2]}}]
    calls = [{'id': 'CPU_dot_fixture', 'type': 'function', 'function': {'name': 'calendar_create_event',
        'arguments': '{"title":"planning-meeting","days":2}'}}]
    dot_pass = score_native_tool_calls(checker, fixture, calls, gold)['task_correct']
    calls[0]['function']['arguments'] = '{"title":"planning-meeting","days":3}'
    wrong_parameter = score_native_tool_calls(checker, fixture, calls, gold)
    limited_ids = {row['source_id'] for row in empty_candidates}
    checks = {'all_other_canonical_gold_examples_accepted':
                all(row['valid'] for row in canonical if row['source_id'] not in limited_ids),
        'empty_candidate_reference_examples_explicitly_rejected_by_unchanged_checker':
                all(not row['valid'] for row in canonical if row['source_id'] in limited_ids),
        'all_wrong_functions_rejected': all(row['wrong_function_rejected'] for row in negatives),
        'all_missing_calls_rejected': all(row['no_call_rejected'] for row in negatives),
        'all_non_JSON_or_ambiguous_arguments_rejected': all(malformed.values()),
        'official_dot_mapping_and_string_normalization': dot_pass,
        'valid_native_contract_with_wrong_argument_is_task_failure':
            wrong_parameter['native_contract_valid'] and not wrong_parameter['task_correct']}
    return {'classification': 'confirmed_CPU_evaluator_contract_only_not_model_quality_or_GPU',
        'checks': checks, 'passed': all(checks.values()), 'requests': len(records),
        'official_checker': provenance, 'reference_answer_sha256': answer_entry['sha256'],
        'dataset_sha256': frozen['dataset_sha256'], 'manifest_sha256': digest(frozen),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'adapter_script_sha256': hashlib.sha256((Path(__file__).parent / 'bfcl_evaluator.py').read_bytes()).hexdigest(),
        'canonical_fixtures': canonical, 'negative_fixtures': negatives, 'malformed_arguments': malformed,
        'reference_empty_candidate_lists': empty_candidates,
        'reference_limited_source_ids': sorted(limited_ids),
        'all258_original_requests_and_official_denominator_retained': True,
        'no_model_generation': True, 'no_external_tool_invocation': True,
        'limits': ['Python live_simple native tool-name/argument proxy only',
                   'Official checker normalization and optional-argument semantics remain unchanged',
                   'Full BFCL provider package is not imported; actual server API/usage still unverified']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'prepared', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.source, args.prepared)
    write_new_json(args.out, result)
    print(json.dumps({key: result[key] for key in ('classification', 'passed', 'checks', 'requests')}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
