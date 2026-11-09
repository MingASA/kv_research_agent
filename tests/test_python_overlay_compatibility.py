"""Real Git fixtures exercise the complete-tree overlay boundary; no GPU imports."""

import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.controller.python_overlay_compatibility import (
    REVIEW_SCHEMA, inventory_overlay_changes, make_overlay_proof,
    validate_overlay_request, verify_overlay_git,
)


class OverlayGitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.git('init', '-q')
        self.write('xllm/python/attention/runtime.py', 'value = 1\n')
        self.write('xllm/core/runtime/worker.cpp', '// frozen compiled input\n')
        self.write('CMakeLists.txt', '# frozen build input\n')
        self.write('README.md', '# baseline\n')
        self.old = self.commit()

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.PIPE).decode().strip()

    def write(self, name, value):
        path = self.repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def commit(self):
        self.git('add', '-A')
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'fixture')
        return self.git('rev-parse', 'HEAD')

    def overlay(self):
        self.write('xllm/python/attention/runtime.py', 'value = 2\n')
        self.write('xllm/python/attention/new_runtime.py', 'value = 3\n')
        new = self.commit()
        comparison = inventory_overlay_changes(self.repo, self.old, new)
        review = {'schema': REVIEW_SCHEMA, 'review_id': 'fixture-root-review',
                  'old_commit': self.old, 'new_commit': new, 'changes': comparison['changes']}
        return new, review

    def request(self):
        new, review = self.overlay()
        binary = {'source_commit': self.old, 'binary_sha256': 'a'*64, 'binary_bytes': 99,
                  'image_id': 'sha256:'+'b'*64, 'build_metadata_sha256': 'c'*64,
                  'build_manifest_sha256': 'd'*64, 'private_generated_header_sha256': {'brpc/config.h': 'e'*64},
                  'zeromq_library_sha256': 'f'*64}
        hashes = {'xllm/core/runtime/worker.cpp': hashlib.sha256((self.repo/'xllm/core/runtime/worker.cpp').read_bytes()).hexdigest()}
        hashes.update({c['path']: c['new']['sha256'] for c in review['changes']})
        return {'server_source_commit': self.old, 'python_source_commit': new,
                'binary_sha256': binary['binary_sha256'], 'image_id': binary['image_id'],
                'zeromq_library_sha256': binary['zeromq_library_sha256'], 'backend_source_sha256': hashes,
                'compiled_backend_source_sha256': {k: v for k, v in hashes.items() if not k.endswith('.py')},
                'python_overlay_binary_compatibility': make_overlay_proof(self.repo, self.old, new, review, binary)}

    def test_complete_git_tree_and_exact_review(self):
        new, review = self.overlay()
        proof = verify_overlay_git(self.repo, self.old, new, review)
        self.assertEqual(proof['old']['entries'], 4)
        self.assertEqual(proof['new']['entries'], 5)
        self.assertEqual(len(proof['changes']), 2)
        self.assertEqual(proof['changes'][0]['new']['sha256'], hashlib.sha256(b'value = 3\n').hexdigest())

    def test_untracked_and_dirty_worktree_do_not_change_immutable_evidence(self):
        new, review = self.overlay()
        before = verify_overlay_git(self.repo, self.old, new, review)
        self.write('CMakeLists.txt', 'dirty, not the frozen tree\n')
        self.write('untracked.cpp', 'untracked\n')
        self.assertEqual(before, verify_overlay_git(self.repo, self.old, new, review))

    def test_stale_or_incomplete_review_is_rejected(self):
        new, review = self.overlay()
        for changes in (review['changes'][:1], []):
            with self.assertRaisesRegex(ValueError, 'exact reviewed'):
                verify_overlay_git(self.repo, self.old, new, dict(review, changes=changes))
        changed = copy.deepcopy(review)
        changed['changes'][0]['new']['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            verify_overlay_git(self.repo, self.old, new, changed)

    def test_compiled_changes_anywhere_in_tree_are_rejected(self):
        for path in ('xllm/core/runtime/worker.cpp', 'CMakeLists.txt', 'tools/setup.py', 'hidden/config.h'):
            self.git('reset', '--hard', self.old)
            self.write(path, 'changed\n')
            new = self.commit()
            with self.subTest(path=path), self.assertRaises(ValueError):
                inventory_overlay_changes(self.repo, self.old, new)

    def test_symlink_changes_are_rejected(self):
        (self.repo/'xllm/python/attention/link.py').symlink_to('runtime.py')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            inventory_overlay_changes(self.repo, self.old, self.commit())

    def test_mode_changes_are_rejected(self):
        (self.repo/'xllm/python/attention/runtime.py').chmod(0o755)
        with self.assertRaisesRegex(ValueError, 'mode'):
            inventory_overlay_changes(self.repo, self.old, self.commit())

    def test_gitlink_changes_are_rejected_and_unchanged_gitlinks_counted(self):
        self.git('update-index', '--add', '--cacheinfo', '160000,'+self.old+',third_party/fixture')
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'gitlink')
        linked = self.git('rev-parse', 'HEAD')
        self.write('README.md', '# reviewed documentation\n')
        self.git('add', 'README.md')
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@invalid', 'commit', '-qm', 'docs')
        current = self.git('rev-parse', 'HEAD')
        self.assertEqual(inventory_overlay_changes(self.repo, linked, current)['old']['gitlinks'], 1)
        with self.assertRaises(ValueError):
            inventory_overlay_changes(self.repo, self.old, linked)

    def test_request_rebinds_binary_source_and_backend_hashes(self):
        request = self.request()
        validate_overlay_request(request, self.repo)
        for key, value in (('binary_sha256', '0'*64), ('python_source_commit', self.old),
                           ('compiled_backend_source_sha256', {})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_overlay_request(dict(request, **{key: value}), self.repo)

    def test_quality_or_extension_claims_are_rejected(self):
        request = self.request()
        for key in ('quality_reuse', 'model_output_equivalence', 'runtime_extension_ABI_verified', 'runtime_load_verified'):
            changed = copy.deepcopy(request)
            changed['python_overlay_binary_compatibility'][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_overlay_request(changed, self.repo)

    def test_no_overlay_is_a_noop_for_existing_contracts(self):
        validate_overlay_request({})


if __name__ == '__main__':
    unittest.main()
