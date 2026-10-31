"""Atomic progress replacement must not abort a healthy owned experiment."""

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.controller.supervisor import tree_bytes


class OutputTreeAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def test_atomic_progress_rename_retries_a_full_scan_and_counts_final_file(self):
        temporary=self.root/'.quality_progress.json.example';temporary.write_bytes(b'progress')
        final=self.root/'quality_progress.json'
        def directory_listing(*args,**kwargs):
            if temporary.exists():
                names=[temporary.name];temporary.rename(final)
            else:names=[final.name]
            yield str(self.root),[],names
        with patch('scripts.controller.supervisor.os.walk',side_effect=directory_listing) as scans:
            self.assertEqual(tree_bytes(self.root),len(b'progress'))
        self.assertEqual(scans.call_count,2)

    def test_continuous_disappearance_stops_after_three_scans_instead_of_zero_usage(self):
        from scripts.controller.supervisor import UnsafeState
        with patch('scripts.controller.supervisor.os.walk',return_value=[(str(self.root),[],['missing'])]) as scans:
            with self.assertRaises(UnsafeState):tree_bytes(self.root)
        self.assertEqual(scans.call_count,3)

    def test_files_and_symlink_metadata_are_counted_without_following_outside_root(self):
        (self.root/'data').write_bytes(b'12345');outside=self.root.parent/'not-opened-by-accounting'
        link=self.root/'link';link.symlink_to(outside)
        self.assertEqual(tree_bytes(self.root),5+link.lstat().st_size)

    def test_permission_or_other_accounting_error_is_never_silenced(self):
        with patch('scripts.controller.supervisor.os.walk',side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):tree_bytes(self.root)


if __name__=='__main__':unittest.main()
