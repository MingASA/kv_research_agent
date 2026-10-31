"""Bind real service evaluations to an independently scored model deployment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .durable import write_new_json


def verify_quality_reference(request: dict,root: Path,uid: int,relative: str,expected_sha: str) -> dict:
    path=root/relative
    if (path.resolve(strict=True)!=path or path.stat().st_uid!=uid or not path.is_file()
            or path.stat().st_size>2**20):raise ValueError('Unowned/oversized deployment quality prerequisite')
    raw=path.read_bytes();quality=json.loads(raw)
    if (hashlib.sha256(raw).hexdigest()!=expected_sha
            or quality.get('all_raw_API_and_usage_contracts_independently_verified') is not True
            or quality.get('prediction_count')!=1720 or quality.get('model_id')!=request['model_id']
            or quality.get('arm')!=request['arms'][0] or quality.get('binary_sha256')!=request['binary_sha256']
            or quality.get('server_source_commit')!=request['server_source_commit']
            or quality.get('operators')!=request['operators'] or quality.get('service')!=request['service']
            or quality.get('kv_scales')!=request.get('kv_scales')):
        raise ValueError('Actual deployment differs from the independently evaluated quality configuration')
    if quality.get('python_source_commit')!=request['python_source_commit']:
        from .model_source_equivalence import verify_model_source_equivalence
        proof=verify_model_source_equivalence(Path(request['python_source_repo']),quality['python_source_commit'],
                                             request['python_source_commit'])
        path=root/'quality_model_source_equivalence.json'
        if path.exists():
            if json.loads(path.read_text())!=proof:raise ValueError('Source-equivalence evidence changed')
        else:write_new_json(path,proof)
    return quality  # This is provenance qualification, never automatic quality acceptance.
