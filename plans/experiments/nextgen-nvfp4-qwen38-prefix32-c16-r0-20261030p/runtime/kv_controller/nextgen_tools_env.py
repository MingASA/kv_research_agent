"""Pinned BFCL in-memory tools with explicit JSON dispatch, never model eval.

This loads inspected upstream definitions, not a generic Python sandbox. Runtime
tools belong in a separate bounded CPU child; nothing here contacts real APIs,
opens model-selected host files or imports the upstream dynamic dispatcher.
"""

from __future__ import annotations

import ast
import copy
import datetime
import decimal
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import typing


SOURCE_COMMIT='6ea57973c7a6097fd7c5915698c54c17c5b1b6c8'
SCHEMAS={'GorillaFileSystem':'gorilla_file_system','MathAPI':'math_api','MessageAPI':'message_api',
         'TwitterAPI':'posting_api','TicketAPI':'ticket_api','TradingBot':'trading_bot',
         'TravelAPI':'travel_booking','VehicleControlAPI':'vehicle_control'}
_PREFIX='eval_checker/multi_turn_eval/func_source_code/'


def verified_upstream_files(root: Path) -> tuple[dict[str,bytes],dict]:
    manifest=json.loads((root/'source_manifest.json').read_text())
    if manifest.get('source_commit')!=SOURCE_COMMIT:raise ValueError('Wrong official multi-turn source commit')
    files={}
    for row in manifest['files']:
        path=root/'files'/row['path']
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to((root/'files').resolve(strict=True)):
            raise ValueError('Source path escaped the fixed local snapshot')
        raw=path.read_bytes()
        if (len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']
                or hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=row['git_blob_sha1']):
            raise ValueError('Official source/schema differs from its pinned Git blob')
        files[row['path']]=raw
    return files,manifest


def _typed(value: object, schema: dict) -> None:
    kind=schema.get('type');kind={'float':'number','dict':'object','int':'integer'}.get(kind,kind)
    if kind is None:return
    checks={'string':isinstance(value,str),'boolean':type(value) is bool,'integer':type(value) is int,
            'number':type(value) in (int,float) and (not isinstance(value,float) or math.isfinite(value)),
            'array':isinstance(value,(list,tuple)),'object':isinstance(value,dict),'null':value is None}
    if kind not in checks or not checks[kind]:raise ValueError('Argument type differs from declared '+str(kind))
    if kind=='array':
        for row in value:_typed(row,schema.get('items',{}))
    if kind=='object':
        if any(key not in value for key in schema.get('required',[])):raise ValueError('Missing required tool argument')
        for key,item in value.items():
            if key in schema.get('properties',{}):_typed(item,schema['properties'][key])


def _tool_result(value: object) -> str:
    # Match the official executor's string/dict/other serialization exactly.
    if type(value) is str:return value
    if type(value) is dict:
        try:return json.dumps(value)
        except (TypeError,ValueError):return str(value)
    return str(value)


class ToolFactory:
    """One reviewed class pool so upstream instance equality stays unchanged."""

    def __init__(self, source: Path) -> None:
        files,manifest=verified_upstream_files(source)
        self.source_manifest=manifest;self.classes={}
        constants={}
        for node in ast.parse(files[_PREFIX+'long_context.py']).body:
            if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):
                raise ValueError('Upstream long-context constants are no longer literals')
            constants[node.targets[0].id]=ast.literal_eval(node.value)
        import mpmath
        modules={'random':random,'copy':copy,'datetime':datetime,'decimal':decimal,
                 'typing':typing,'math':math,'mpmath':mpmath}
        self.module_AST_sha256={}
        for class_name,stem in SCHEMAS.items():
            raw=files[_PREFIX+stem+'.py'];tree=ast.parse(raw)
            forbidden={'eval','exec','open','compile','__import__','globals','locals','input','exit','quit','breakpoint',
                       '__builtins__','subprocess'}
            if any(isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id in forbidden
                   for node in ast.walk(tree)):
                raise ValueError('Unreviewed external/code-execution operation in tool source')
            namespace={'__name__':'pinned_BFCL_'+stem}
            body=[]
            for node in tree.body:
                if isinstance(node,ast.Import):
                    for alias in node.names:
                        # This import is unused in the pinned in-memory FS.
                        if alias.name=='subprocess':continue
                        if alias.name not in modules:raise ValueError('Unreviewed tool module import')
                        namespace[alias.asname or alias.name]=modules[alias.name]
                elif isinstance(node,ast.ImportFrom):
                    module=node.module or ''
                    if module=='bfcl_eval.eval_checker.multi_turn_eval.func_source_code.long_context':
                        for alias in node.names:namespace[alias.asname or alias.name]=copy.deepcopy(constants[alias.name])
                    else:
                        if module not in modules:raise ValueError('Unreviewed tool dependency')
                        for alias in node.names:
                            if alias.name.startswith('_'):raise ValueError('Private imported tool dependency')
                            namespace[alias.asname or alias.name]=getattr(modules[module],alias.name)
                else:body.append(node)
            # Only pinned, inspected source definitions enter exec. Model calls
            # below are data and never enter compile/eval/exec or dynamic import.
            selected=ast.Module(body=body,type_ignores=[])
            self.module_AST_sha256[stem]=hashlib.sha256(ast.dump(selected,include_attributes=False).encode()).hexdigest()
            exec(compile(selected,'pinned_BFCL_in_memory_'+stem,'exec'),namespace)
            self.classes[class_name]=namespace[class_name]
        checker=ast.parse(files['eval_checker/multi_turn_eval/multi_turn_checker.py'])
        names={'state_checker','response_checker','_compare_instances','_is_subsequence_unordered'}
        definitions=[node for node in checker.body if isinstance(node,ast.FunctionDef) and node.name in names]
        if {node.name for node in definitions}!=names:raise ValueError('Official state/response checker changed')
        selected=ast.Module(body=definitions,type_ignores=[]);namespace={}
        exec(compile(selected,'pinned_BFCL_official_state_response_definitions','exec'),namespace)
        self.state_checker=namespace['state_checker'];self.response_checker=namespace['response_checker']
        self.checker_AST_sha256=hashlib.sha256(ast.dump(selected,include_attributes=False).encode()).hexdigest()

    def environment(self, task: dict):
        return ToolEnvironment(self,task)


class ToolEnvironment:
    def __init__(self, factory: ToolFactory, task: dict) -> None:
        self.instances={};self.methods={};self.schemas={}
        original=json.loads(task['original_source_record_JSON'])
        if (hashlib.sha256(task['original_source_record_JSON'].encode()).hexdigest()!=task['source_line_sha256_without_LF']
                or original['id']!=task['id'] or original['initial_config']!=task['initial_config']
                or original['involved_classes']!=task['involved_classes']):
            raise ValueError('Task environment differs from its frozen original source record')
        for name in task['involved_classes']:
            if name not in SCHEMAS:raise ValueError('Unreviewed runtime tool class')
            instance=factory.classes[name]()
            if name!='MathAPI':instance._load_scenario(copy.deepcopy(original['initial_config'].get(name,{})),long_context=False)
            self.instances[name]=instance
        for tool in task['tools']:
            name=tool['function']['name'];owner=task['function_classes'].get(name)
            if (name.startswith('_') or owner not in self.instances or name in self.methods
                    or not hasattr(self.instances[owner],name)):
                raise ValueError('Tool schema does not name one unambiguous declared in-memory method')
            method=getattr(self.instances[owner],name)
            if not inspect.ismethod(method):raise ValueError('Declared tool is not a bound source method')
            self.methods[name]=method;self.schemas[name]=tool['function']['parameters']

    def call(self, name: str, arguments: dict) -> dict:
        """Only a declared method with data arguments; failures are tool feedback."""
        try:
            if name not in self.methods:raise ValueError('Function is not declared in this task tool schema')
            if not isinstance(arguments,dict):raise ValueError('Tool arguments must be a JSON object')
            _typed(arguments,self.schemas[name]);inspect.signature(self.methods[name]).bind(**arguments)
            value=self.methods[name](**arguments)
            result=_tool_result(value)
            if len(result.encode())>2**20:raise ValueError('Tool response exceeded bounded1MiB feedback')
            return {'result':result,'dispatch_valid':True,'tool_reported_error':isinstance(value,dict) and bool(value.get('error'))}
        except Exception as error:
            return {'result':'Error during execution: '+str(error),'dispatch_valid':False,'tool_reported_error':True}

    def reference_call(self, expression: str) -> dict:
        """Reference calls also use a literal-only AST, never Python eval."""
        node=ast.parse(expression,mode='eval').body
        if (not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name)
                or node.func.id not in self.methods or any(keyword.arg is None for keyword in node.keywords)):
            raise ValueError('Reference is not a declared literal function call')
        args=[ast.literal_eval(value) for value in node.args]
        kwargs={keyword.arg:ast.literal_eval(keyword.value) for keyword in node.keywords}
        if len(kwargs)!=len(node.keywords):raise ValueError('Duplicate reference keyword')
        bound=inspect.signature(self.methods[node.func.id]).bind(*args,**kwargs)
        return self.call(node.func.id,dict(bound.arguments))
