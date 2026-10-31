"""Audit only a known read-only runtime metadata addition across model commits."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess


FILE='xllm/python/attention/flashinfer.py'


def normalized_runtime_module(source: str) -> str:
    tree=ast.parse(source)
    for cls in tree.body:
        if not isinstance(cls,ast.ClassDef) or cls.name!='FlashInferBackend':continue
        for function in cls.body:
            if not isinstance(function,ast.FunctionDef) or function.name!='native_kv_runtime_info':continue
            for node in ast.walk(function):
                if isinstance(node,ast.Dict):
                    keep=[]
                    for key,value in zip(node.keys,node.values):
                        if isinstance(key,ast.Constant) and key.value=='decode_probability_data_type':
                            if any(isinstance(child,(ast.Call,ast.NamedExpr,ast.Await,ast.Yield)) for child in ast.walk(value)):
                                raise ValueError('Runtime metadata expression may have side effects')
                            continue
                        keep.append((key,value))
                    node.keys=[key for key,_ in keep];node.values=[value for _,value in keep]
    return ast.dump(tree,include_attributes=False)


def verify_model_source_equivalence(repo: Path, old: str, new: str) -> dict:
    if any(re.fullmatch('[0-9a-f]{40}',value) is None for value in (old,new)):
        raise ValueError('Source comparison requires exact immutable commits')
    changed=subprocess.check_output(['git','-C',str(repo),'diff','--name-only',old,new,'--','xllm'],text=True,timeout=30).splitlines()
    if any(name!=FILE for name in changed):raise ValueError('Actual model/runtime/C++ implementation changed since quality evaluation')
    before=subprocess.check_output(['git','-C',str(repo),'show',old+':'+FILE],text=True,timeout=30)
    after=subprocess.check_output(['git','-C',str(repo),'show',new+':'+FILE],text=True,timeout=30)
    if normalized_runtime_module(before)!=normalized_runtime_module(after):
        raise ValueError('Operational model AST differs; quality cannot qualify this deployment')
    return {'old_commit':old,'new_commit':new,'changed_xllm_files':changed,
            'only_excluded_change':'read-only decode_probability_data_type key in FlashInferBackend.native_kv_runtime_info',
            'all_operational_model_code_and_Cpp_unchanged':True,'quality_scores_reused_as_measurement':False}
