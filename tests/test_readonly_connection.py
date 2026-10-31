"""The actual OpenSSH parser must keep recovery multiplexing noninteractive."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.controller.readonly_connection import ReadOnlyConnection


class ReadOnlyConnectionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ssh'), 'OpenSSH client is unavailable')
    def test_master_options_do_not_request_session_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            connection=object.__new__(ReadOnlyConnection)
            connection.directory=directory
            connection.socket=directory/'mux'
            connection.transport=SimpleNamespace(argv=[
                shutil.which('ssh'),'-T','-o','ControlMaster=no','-o','ControlPath=none',
                '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','example.invalid'])
            # Capture the exact intended spawn without contacting a server.
            with patch('scripts.controller.readonly_connection.subprocess.Popen',
                       side_effect=RuntimeError('captured before execution')) as spawn:
                with self.assertRaisesRegex(RuntimeError,'captured before execution'):
                    connection.start()
            command=spawn.call_args.args[0]
            parsed=subprocess.run([command[3],'-G',*command[4:]],
                                  capture_output=True,text=True,check=True,timeout=10)
            options=dict(line.split(' ',1) for line in parsed.stdout.splitlines() if ' ' in line)
            self.assertEqual(options['controlmaster'],'true')
            self.assertEqual(options['batchmode'],'yes')
            self.assertEqual(options['stricthostkeychecking'],'true')
            self.assertEqual(options['controlpath'],str(connection.socket))

    def test_recovery_connection_rejects_mutations_before_rpc(self):
        connection=object.__new__(ReadOnlyConnection)
        with self.assertRaisesRegex(ValueError,'read-only'):
            connection.rpc('unused',{},mutation=True)


if __name__=='__main__':
    unittest.main()
