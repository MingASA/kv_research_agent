"""Only proven pre-session refusal reconnects; ambiguous writes never replay."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.transport import PinnedSSH,UncertainDispatch,preauthentication_banner_refused


def refusal(stdout=b''):
    return subprocess.CalledProcessError(255,['ssh'],output=stdout,
        stderr=b'kex_exchange_identification: Connection closed by remote host\r\nConnection closed by 47.94.207.53 port 45291\r\n')


class PreauthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.ssh=PinnedSSH.__new__(PinnedSSH);self.ssh.argv=['ssh-fixture']
        self.ssh.local_directory=Path(self.temporary.name);self.ssh.preauth_reconnect_allowed=True

    def test_two_banner_refusals_then_one_exec_success_is_bounded_and_audited(self):
        with patch('scripts.controller.transport.subprocess.run',side_effect=[refusal(),refusal(),SimpleNamespace(stdout=b'{"ok":true}')]) as calls,\
                patch('scripts.controller.transport.time.sleep') as sleep:
            self.assertTrue(self.ssh.rpc('pass',{},mutation=True)['ok'])
        self.assertEqual(calls.call_count,3);self.assertEqual(sleep.call_count,2)
        self.assertEqual(len(list(self.ssh.local_directory.glob('SSH_PREAUTH_REFUSAL_*.json'))),2)

    def test_three_pre_auth_refusals_stop_without_unbounded_retries(self):
        with patch('scripts.controller.transport.subprocess.run',side_effect=refusal()) as calls,\
                patch('scripts.controller.transport.time.sleep') as sleep:
            with self.assertRaises(UncertainDispatch):self.ssh.rpc('pass',{},mutation=True)
        self.assertEqual(calls.call_count,3);self.assertEqual(sleep.call_count,2)

    def test_timeout_post_session_close_remote_error_and_partial_response_never_retry(self):
        cases=[subprocess.TimeoutExpired('ssh',90),
               subprocess.CalledProcessError(255,['ssh'],stderr=b'Connection closed by remote host'),
               refusal(stdout=b'{"remote_started":true}'),
               subprocess.CalledProcessError(1,['ssh'],stderr=b'Traceback: remote mutation failed')]
        for error in cases:
            with self.subTest(error=str(error)),patch('scripts.controller.transport.subprocess.run',side_effect=error) as calls,\
                    patch('scripts.controller.transport.time.sleep') as sleep:
                with self.assertRaises(UncertainDispatch):self.ssh.rpc('pass',{},mutation=True)
                self.assertEqual(calls.call_count,1);sleep.assert_not_called()

    def test_unverified_client_or_proxy_cannot_use_preauth_exception(self):
        self.ssh.preauth_reconnect_allowed=False
        with patch('scripts.controller.transport.subprocess.run',side_effect=refusal()) as calls:
            with self.assertRaises(UncertainDispatch):self.ssh.rpc('pass',{},mutation=True)
        self.assertEqual(calls.call_count,1)

    def test_kex_then_ambiguous_reply_loss_never_sends_a_third_request(self):
        with patch('scripts.controller.transport.subprocess.run',
                   side_effect=[refusal(),subprocess.TimeoutExpired('ssh',90)]) as calls,\
                patch('scripts.controller.transport.time.sleep') as sleep:
            with self.assertRaises(UncertainDispatch):self.ssh.rpc('pass',{},mutation=True)
        self.assertEqual(calls.call_count,2);self.assertEqual(sleep.call_count,1)

    def test_extra_banner_or_authenticated_stderr_is_not_whitelisted(self):
        self.assertTrue(preauthentication_banner_refused(refusal()))
        error=refusal();error.stderr=b'Authenticated to host\n'+error.stderr
        self.assertFalse(preauthentication_banner_refused(error))


if __name__=='__main__':unittest.main()
