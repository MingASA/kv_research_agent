"""Static compiled-source compatibility for an explicitly reviewed Python overlay.

This does not certify dynamic extension ABI, runtime loading, numerical output,
quality or performance. Those remain separate deployment/qualification gates.
All source identities come from immutable Git objects, never the worktree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from .approval import ApprovalError


SCHEMA = 'reviewed_python_overlay_binary_compatibility_v1'
REVIEW_SCHEMA = 'exact_python_overlay_change_review_v1'
SCOPE = 'unchanged_compiled_source_tree_with_reviewed_python_overlay'


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode()).hexdigest()


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is not None


def _git(repo: Path, *arguments: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(repo), *arguments], timeout=30)


def _tree(repo: Path, commit: str) -> dict:
    if not _hex(commit, 40) or _git(repo, 'rev-parse', '--verify', commit+'^{commit}').strip().decode() != commit:
        raise ValueError('Exact immutable Git commit required')
    result = {}
    for record in _git(repo, 'ls-tree', '-r', '-z', '--full-tree', commit).split(b'\0'):
        if not record:
            continue
        header, raw_path = record.split(b'\t', 1)
        mode, kind, oid = header.decode('ascii').split(' ')
        path = raw_path.decode('utf-8')
        if (not path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                or str(PurePosixPath(path)) != path or any(ord(c) < 32 for c in path)
                or path in result or not _hex(oid, 40)):
            raise ValueError('Unsafe/duplicate Git tree path')
        result[path] = {'mode': mode, 'type': kind, 'git_oid': oid}
    return result


def _role(path: str) -> str:
    if (not isinstance(path, str) or not path or '\\' in path
            or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
            or str(PurePosixPath(path)) != path or any(ord(c) < 32 for c in path)):
        raise ValueError('Noncanonical reviewed path')
    p = PurePosixPath(path)
    # Never allow an explicit review to authorize compiled/build/codegen inputs.
    if (any(part.lower() in ('cmake', 'build', 'codegen', 'third_party') for part in p.parts)
            or any(word in p.name.lower() for word in ('cmake', 'codegen', 'generate', 'setup', 'build'))
            or p.suffix.lower() in ('.h', '.hpp', '.cuh', '.c', '.cc', '.cpp', '.cu', '.so', '.a')):
        raise ValueError('Compiled/build/codegen input changed: '+path)
    if path.startswith('xllm/python/') and p.suffix == '.py':
        return 'runtime_python'
    if p.suffix == '.md' and (len(p.parts) == 1 or p.parts[0] == 'docs'):
        return 'documentation'
    if p.parts[0] in ('tests', 'test') and p.suffix == '.py':
        return 'test'
    if p.parts[0] == 'tools' and p.suffix == '.py':
        return 'tool'
    raise ValueError('Path outside the Python/docs/tests/tools overlay: '+path)


def _entry(repo: Path, item: dict | None) -> dict | None:
    if item is None:
        return None
    if item['type'] != 'blob' or item['mode'] not in ('100644', '100755'):
        raise ValueError('Changed gitlink, symlink or nonregular entry is forbidden')
    return {**item, 'sha256': hashlib.sha256(_git(repo, 'cat-file', 'blob', item['git_oid'])).hexdigest()}


def inventory_overlay_changes(repo: Path, old: str, new: str) -> dict:
    """Produce a review candidate; this inventory is not review authorization."""
    before, after = _tree(repo, old), _tree(repo, new)
    changes = []
    for path in sorted(before.keys() | after.keys()):
        left, right = before.get(path), after.get(path)
        if left == right:
            continue
        role = _role(path)
        if left is not None and right is not None and left['mode'] != right['mode']:
            raise ValueError('File mode change is forbidden: '+path)
        changes.append({'path': path, 'role': role, 'old': _entry(repo, left), 'new': _entry(repo, right)})
    def tree_summary(tree: dict, commit: str) -> dict:
        return {'commit': commit, 'tree_oid': _git(repo, 'rev-parse', commit+'^{tree}').strip().decode(),
                'entries': len(tree), 'gitlinks': sum(v['type'] == 'commit' for v in tree.values()),
                'path_mode_object_sha256': _digest([{'path': p, **tree[p]} for p in sorted(tree)])}
    return {'old': tree_summary(before, old), 'new': tree_summary(after, new), 'changes': changes}


def verify_overlay_git(repo: Path, old: str, new: str, review: dict) -> dict:
    """Compare the entire immutable trees against an exact, externally reviewed list."""
    comparison = inventory_overlay_changes(repo, old, new)
    if (not isinstance(review, dict) or set(review) != {'schema', 'review_id', 'old_commit', 'new_commit', 'changes'}
            or review.get('schema') != REVIEW_SCHEMA
            or not isinstance(review.get('review_id'), str)
            or re.fullmatch('[A-Za-z0-9_-]{1,128}', review['review_id']) is None
            or review.get('old_commit') != old or review.get('new_commit') != new
            or review.get('changes') != comparison['changes']):
        raise ValueError('Entire Git delta must match the exact reviewed old/new SHA allowlist')
    return comparison


def make_overlay_proof(repo: Path, old: str, new: str, review: dict, binary_identity: dict) -> dict:
    comparison = verify_overlay_git(repo, old, new, review)
    return {'schema': SCHEMA, 'scope': SCOPE, 'review': review, 'comparison': comparison,
            'binary_identity': binary_identity, 'runtime_load_verified': False,
            'runtime_extension_ABI_verified': False, 'model_output_equivalence': False,
            'quality_reuse': False}


def validate_overlay_request(request: dict, repo: Path | None = None) -> None:
    """Validate a declared proof; pass repo at dispatch/worker to recompute Git evidence."""
    proof = request.get('python_overlay_binary_compatibility')
    if proof is None:
        return
    try:
        if (not isinstance(proof, dict) or set(proof) != {'schema', 'scope', 'review', 'comparison', 'binary_identity',
                'runtime_load_verified', 'runtime_extension_ABI_verified', 'model_output_equivalence', 'quality_reuse'}
                or proof['schema'] != SCHEMA or proof['scope'] != SCOPE
                or any(proof[k] is not False for k in ('runtime_load_verified', 'runtime_extension_ABI_verified',
                                                       'model_output_equivalence', 'quality_reuse'))):
            raise ValueError('Unknown proof scope or unsupported runtime/quality claim')
        old, new = request['server_source_commit'], request['python_source_commit']
        if not _hex(old, 40) or not _hex(new, 40) or old == new:
            raise ValueError('Overlay needs distinct exact compiled and Python commits')
        comparison, review, binary = proof['comparison'], proof['review'], proof['binary_identity']
        if (set(review) != {'schema', 'review_id', 'old_commit', 'new_commit', 'changes'}
                or review['schema'] != REVIEW_SCHEMA or re.fullmatch('[A-Za-z0-9_-]{1,128}', review['review_id']) is None
                or review['old_commit'] != old or review['new_commit'] != new
                or set(comparison) != {'old', 'new', 'changes'} or comparison['changes'] != review['changes']):
            raise ValueError('Review does not bind this exact deployment')
        for key, commit in (('old', old), ('new', new)):
            summary = comparison[key]
            if (set(summary) != {'commit', 'tree_oid', 'entries', 'gitlinks', 'path_mode_object_sha256'}
                    or summary['commit'] != commit or not _hex(summary['tree_oid'], 40)
                    or not _hex(summary['path_mode_object_sha256'], 64)
                    or type(summary['entries']) is not int or summary['entries'] < 1
                    or type(summary['gitlinks']) is not int or not 0 <= summary['gitlinks'] <= summary['entries']):
                raise ValueError('Missing complete tree identity')
        seen = set()
        for change in review['changes']:
            if (set(change) != {'path', 'role', 'old', 'new'} or change['path'] in seen
                    or change['role'] != _role(change['path'])):
                raise ValueError('Unsafe reviewed path')
            seen.add(change['path'])
            left, right = change['old'], change['new']
            if left == right or left is None and right is None:
                raise ValueError('Review contains a nonexistent delta')
            for entry in (left, right):
                if entry is not None and (set(entry) != {'mode', 'type', 'git_oid', 'sha256'}
                        or entry['mode'] not in ('100644', '100755') or entry['type'] != 'blob'
                        or not _hex(entry['git_oid'], 40) or not _hex(entry['sha256'], 64)):
                    raise ValueError('Only regular reviewed blobs may change')
            if left and right and left['mode'] != right['mode']:
                raise ValueError('Mode change forbidden')
        required = {'source_commit', 'binary_sha256', 'binary_bytes', 'image_id', 'build_metadata_sha256',
                    'build_manifest_sha256', 'private_generated_header_sha256', 'zeromq_library_sha256'}
        if (set(binary) != required or binary['source_commit'] != old
                or binary['binary_sha256'] != request['binary_sha256'] or binary['image_id'] != request['image_id']
                or type(binary['binary_bytes']) is not int or binary['binary_bytes'] <= 0
                or any(not _hex(binary[k], 64) for k in ('binary_sha256', 'build_metadata_sha256',
                                                       'build_manifest_sha256', 'zeromq_library_sha256'))
                or binary['zeromq_library_sha256'] != request['zeromq_library_sha256']
                or not isinstance(binary['private_generated_header_sha256'], dict)
                or any(not isinstance(k, str) or not _hex(v, 64)
                       for k, v in binary['private_generated_header_sha256'].items())):
            raise ValueError('Exact original compiled artifact identity required')
        hashes = request['backend_source_sha256']
        if not isinstance(hashes, dict) or not hashes or any(not _hex(h, 64) for h in hashes.values()):
            raise ValueError('Complete backend SHA map required')
        compiled = request.get('compiled_backend_source_sha256')
        if compiled != {p: h for p, h in hashes.items() if not p.endswith('.py')}:
            raise ValueError('Overlay must retain exact compiled C++/header hashes')
        for change in review['changes']:
            if change['path'] in hashes and (change['new'] is None or hashes[change['path']] != change['new']['sha256']):
                raise ValueError('Request backend SHA differs from reviewed new Git blob')
        if repo is not None:
            if verify_overlay_git(repo, old, new, review) != comparison:
                raise ValueError('Immutable Git tree proof changed')
            tree = _tree(repo, new)
            for path, sha in hashes.items():
                entry = _entry(repo, tree.get(path))
                if entry is None or entry['sha256'] != sha:
                    raise ValueError('Backend identity differs from immutable new Git blob: '+path)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ApprovalError('Invalid Python overlay binary compatibility: '+str(error)) from error
