import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import observe_nextgen_long_metrics as observer


class OriginalAckTests(unittest.TestCase):
    def test_missing_or_failed_ack_never_contacts_remote_or_creates_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run_id=observer.ALLOWED[-2];out=root/'observation'
            directory=root/'state/controller/local_launch_once';directory.mkdir(parents=True)
            with patch.object(observer,'ROOT',root),patch.object(observer,'_ticket') as ticket, \
                    patch.object(observer,'PinnedSSH') as ssh, \
                    patch('sys.argv',['observer','--run-id',run_id,'--out',str(out)]):
                with self.assertRaises(FileNotFoundError):observer.main()
                (directory/(run_id+'.finished.json')).write_text(json.dumps(
                    {'run_id':run_id,'local_CLI_exit_code':2}))
                (directory/(run_id+'.json')).write_text(json.dumps({'run_id':run_id}))
                with self.assertRaises(ValueError):observer.main()
                self.assertFalse(out.exists());ticket.assert_not_called();ssh.assert_not_called()

    def test_success_requires_same_original_dispatch_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run_id=observer.ALLOWED[-2]
            directory=root/'state/controller/local_launch_once';directory.mkdir(parents=True)
            receipt=root/'receipt.json';receipt.write_text(json.dumps({'ok':True,'status':'DISPATCHED'}))
            (directory/(run_id+'.json')).write_text(json.dumps({'run_id':run_id,'receipt':str(receipt)}))
            finished=directory/(run_id+'.finished.json')
            finished.write_text(json.dumps({'run_id':run_id,'local_CLI_exit_code':0,'receipt':'receipt.json'}))
            with patch.object(observer,'ROOT',root):
                observer.require_original_ack(run_id)
                receipt.write_text(json.dumps({'ok':True,'status':'UNKNOWN'}))
                with self.assertRaises(ValueError):observer.require_original_ack(run_id)
