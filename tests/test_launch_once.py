"""Repeated local calls cannot truncate receipts or spawn another CLI."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_guarded_once as once
from scripts.controller.durable import secure_directory
from scripts.controller.quota import QuotaLedger


class LaunchOnceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.package=self.root/'package.json';self.package.write_text(json.dumps({'request':{'run_id':'example'}}))
        self.receipt=self.root/'receipt.json';self.errors=self.root/'stderr.log'
        self.context=patch.object(once,'ROOT',self.root);self.context.start();self.addCleanup(self.context.stop)
    def test_existing_intent_refuses_before_spawn_and_preserves_receipt(self):
        path=self.root/'state/controller/local_launch_once';secure_directory(path)
        (path/'example.json').write_text('original intent');self.receipt.write_text('original ACK')
        with patch.object(once.subprocess,'Popen') as spawn:
            with self.assertRaises(ValueError):once.launch(self.package,self.receipt,self.errors)
            spawn.assert_not_called()
        self.assertEqual(self.receipt.read_text(),'original ACK')
    def test_existing_receipt_refuses_without_truncating(self):
        self.receipt.write_text('original ACK')
        with patch.object(once.subprocess,'Popen') as spawn:
            with self.assertRaises(FileExistsError):once.launch(self.package,self.receipt,self.errors)
            spawn.assert_not_called()
        self.assertEqual(self.receipt.read_text(),'original ACK')
    def test_registered_history_never_relaunches(self):
        with QuotaLedger(self.root/'state/controller') as ledger:ledger.register_history({'example':'a'*64})
        with patch.object(once.subprocess,'Popen') as spawn:
            with self.assertRaises(ValueError):once.launch(self.package,self.receipt,self.errors)
            spawn.assert_not_called()
        self.assertFalse(self.receipt.exists())


if __name__=='__main__':unittest.main()
