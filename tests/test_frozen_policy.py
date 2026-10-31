# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""A current local policy must never mask an incompatible shipped policy."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.cli import validate_frozen_policy


class FrozenPolicyTests(unittest.TestCase):
    def _package(self,root: Path,source: str) -> Path:
        package=root/'runtime/kv_controller';package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        policy=package/'policy.py';policy.write_text(source)
        path=root/'package.json';path.write_text(json.dumps({'assets':{
            'runtime/kv_controller/policy.py':{'local_path':str(policy)}}}))
        return path

    def test_old_shipped_policy_rejects_new_xz_before_any_remote_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            p=self._package(Path(temp),'def validate_request(request,approval):\n'
                ' if request["codec"]=="xz":raise ValueError("old gzip-only policy")\n')
            with self.assertRaisesRegex(ApprovalError,'old gzip-only policy'):
                validate_frozen_policy(p,{'codec':'xz'},{})

    def test_matching_shipped_policy_actually_executes_without_GPU(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            p=self._package(Path(temp),'import os\ndef validate_request(request,approval):\n'
                ' assert os.environ["CUDA_VISIBLE_DEVICES"]==""\n'
                ' assert request["codec"]=="xz"\n')
            validate_frozen_policy(p,{'codec':'xz'},{})


if __name__=='__main__':
    unittest.main()
