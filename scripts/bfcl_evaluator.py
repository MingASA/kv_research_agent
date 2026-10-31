#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Pinned BFCL Python argument checker for native API tool-call responses.

Only inspected upstream definitions are executed. Model text is parsed as JSON;
no generated Python, external tools or provider SDKs are executed.
"""

from __future__ import annotations

import ast
from enum import Enum
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Callable

from scripts.prepare_tool_calls import PREFIX, SOURCE_COMMIT, _load_sources
from scripts.controller.native_tools import parse_native_tool_calls


MODEL_PROFILE = 'xllm-local-qwen25'


def _unsupported_language(*args: object, **kwargs: object) -> object:
    raise ValueError('Only the frozen Python live_simple protocol is supported')


def load_official_checker(source: Path) -> tuple[Callable, dict]:
    files = _load_sources(source)
    checker_source = files[PREFIX + 'eval_checker/ast_eval/ast_checker.py']
    definitions = [node for node in ast.parse(checker_source).body
                   if not isinstance(node, (ast.Import, ast.ImportFrom))]
    if any(not isinstance(node, (ast.FunctionDef, ast.Assign)) for node in definitions):
        raise ValueError('Pinned checker definition contract changed')
    language_nodes = [node for node in ast.parse(files[PREFIX + 'constants/enums.py']).body
                      if isinstance(node, ast.ClassDef) and node.name == 'Language']
    if len(language_nodes) != 1:
        raise ValueError('Pinned upstream Language enum changed')
    module = ast.Module(body=language_nodes + definitions, type_ignores=[])
    namespace = {'Enum': Enum, 're': re,
                 'MODEL_CONFIG_MAPPING': {MODEL_PROFILE: SimpleNamespace(underscore_to_dot=True)},
                 'JAVA_TYPE_CONVERSION': {}, 'JS_TYPE_CONVERSION': {},
                 'java_type_converter': _unsupported_language, 'js_type_converter': _unsupported_language}
    # Preserve every upstream checker function body and constant AST. The static
    # profile declares exactly the same dot conversion used by the frozen OAI
    # adapter. Java/provider imports are outside this Python-only protocol.
    exec(compile(module, 'pinned_BFCL_official_Python_checker_definitions', 'exec'), namespace)

    def check(functions: list[dict], output: list[dict], answers: list[dict]) -> dict:
        return namespace['ast_checker'](functions, output, answers,
                                        namespace['Language'].PYTHON, 'live_simple', MODEL_PROFILE)

    return check, {'source_commit': SOURCE_COMMIT,
        'source_sha256': hashlib.sha256(checker_source).hexdigest(),
        'definition_AST_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
        'mode': 'unchanged_official_function_and_constant_AST_not_full_provider_package',
        'model_profile': MODEL_PROFILE, 'underscore_to_dot': True,
        'language': 'python', 'test_category': 'live_simple'}


def score_native_tool_calls(checker: Callable, functions: list[dict], calls: list[dict],
                            answers: list[dict]) -> dict:
    try:
        output = parse_native_tool_calls(calls)
    except (ValueError, TypeError, RecursionError) as error:
        return {'native_contract_valid': False, 'task_correct': False,
                'error_type': 'native_tool_contract', 'error': str(error)}
    result = checker(functions, output, answers)
    return {'native_contract_valid': True, 'task_correct': result['valid'],
            'official_result': result, 'parsed_calls': output}
