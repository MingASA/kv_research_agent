"""Load-point replay must preserve a complete runtime and current ownership guards."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_trace_package import ROOT, trace_runtime_files


class TraceRuntimeReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        paths = {'runtime/kv_controller/'+p.name:p for p in (ROOT/'scripts/controller').glob('*.py')}
        paths['runtime/kv_controller/evidence.py'] = ROOT/'scripts/audit_native_results.py'
        self.package = {'assets':{name:{'local_path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                                   for name,p in paths.items()}}

    def test_identical_complete_runtime_retained(self) -> None:
        paths = trace_runtime_files(self.package, True)
        self.assertEqual(set(paths), set(self.package['assets']))

    def test_missing_import_dependency_rejected(self) -> None:
        self.package['assets'].pop('runtime/kv_controller/scale_coverage.py')
        with self.assertRaisesRegex(ValueError, 'module closure'):
            trace_runtime_files(self.package, True)

    def test_rehashed_old_or_changed_ownership_guard_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old_supervisor.py'
            path.write_text('raise RuntimeError("CPU unsafe old-runtime fixture only")\n')
            self.package['assets']['runtime/kv_controller/supervisor.py'] = {
                'local_path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, 'safety/ownership'):
                trace_runtime_files(self.package, True)


if __name__ == '__main__':
    unittest.main()
