"""Actual CMake target layouts must be explicit and unambiguous."""

import tempfile
import unittest
from pathlib import Path

from scripts.native_cpu_build_driver import _find_binary


class BinaryArtifactTests(unittest.TestCase):
    def test_current_and_explicit_bin_layouts_are_found(self):
        for name in ('xllm/xllm', 'bin/xllm'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path = root/name
                path.parent.mkdir()
                path.write_bytes(b'ELF-fixture')
                path.chmod(0o700)
                self.assertEqual(_find_binary(root), path)

    def test_missing_stale_ambiguous_or_symlink_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):_find_binary(root)
            for name in ('xllm/xllm', 'bin/xllm'):
                path = root/name;path.parent.mkdir();path.write_bytes(b'ELF-fixture');path.chmod(0o700)
            with self.assertRaises(ValueError):_find_binary(root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);(root/'bin').mkdir();(root/'real').write_bytes(b'fixture')
            (root/'real').chmod(0o700);(root/'bin/xllm').symlink_to(root/'real')
            with self.assertRaises(ValueError):_find_binary(root)


if __name__ == '__main__':
    unittest.main()
