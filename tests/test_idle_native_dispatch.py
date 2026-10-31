"""Busy, disconnected and already-owned runs must never dispatch GPU work."""

import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from scripts import idle_native_dispatch as dispatcher
from scripts.controller.quota import QuotaLedger
import test_quota_ledger as fixtures


class IdleAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.directory=self.root/'queue';self.directory.mkdir(mode=0o700)
        self.package=self.root/'package.json';self.package.write_text('{}')
        self.approval,self.request,_=fixtures._fixture()
        self.request.update(workload={'kind':'native_hybrid_core_quality'},quality_completion={'anchor':'original'})
        self.manifest={'run_id':self.request['run_id'],'package':str(self.package),
            'package_sha256':hashlib.sha256(b'{}').hexdigest(),'expected_boot_id':'boot',
            'runner_sha256':hashlib.sha256(Path(dispatcher.__file__).read_bytes()).hexdigest(),
            'not_after_utc':(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=1)).isoformat()}
        self.state={'sample_index':0}
        self.patches=[patch.object(dispatcher,'ROOT',self.root),
            patch.object(dispatcher,'_local_package',return_value=(self.request,{},{})),
            patch.object(dispatcher,'load_approval_snapshot',return_value=(self.approval,'sha')),
            patch.object(dispatcher,'validate_execution_authority'),patch.object(dispatcher,'validate_request'),
            patch.object(dispatcher,'validate_scope'),patch.object(dispatcher,'validate_frozen_policy')]
        for item in self.patches:item.start();self.addCleanup(item.stop)

    def test_foreign_busy_snapshot_creates_neither_intent_nor_reservation(self):
        snapshot={'ok':True,'remote_writes':False,'boot_id':'boot',
            'gpus':{self.request['gpu_uuids'][0]:{'memory_mib':71000,'util_pct':0}},'apps':[]}
        with patch.object(dispatcher,'PinnedSSH') as ssh,patch.object(dispatcher.subprocess,'Popen') as launch:
            ssh.return_value.rpc.return_value=snapshot
            self.assertFalse(dispatcher.cycle(self.manifest,self.directory,self.state));launch.assert_not_called()
        self.assertEqual(self.state['status'],'WAIT_FOREIGN_GPU_OR_RESERVATION')
        self.assertFalse((self.directory/'ADMISSION_INTENT.json').exists())
        with QuotaLedger(self.root/'state/controller') as ledger:self.assertEqual(ledger.usage(self.approval)['known_run_ids'],[])

    def test_network_unknown_never_launches_or_reserves(self):
        with patch.object(dispatcher,'PinnedSSH') as ssh,patch.object(dispatcher.subprocess,'Popen') as launch:
            ssh.return_value.rpc.side_effect=RuntimeError('Readonly SSH exhausted')
            self.assertFalse(dispatcher.cycle(self.manifest,self.directory,self.state));launch.assert_not_called()
        self.assertEqual(self.state['status'],'WAIT_READONLY_NETWORK_RECOVERY_COOLDOWN')
        self.assertFalse((self.directory/'ADMISSION_INTENT.json').exists())

    def test_existing_original_local_intent_halts_before_any_SSH_or_second_launch(self):
        path=self.root/'state/controller/local_launch_once';path.mkdir(parents=True)
        (path/(self.request['run_id']+'.json')).write_text('{}')
        with patch.object(dispatcher,'PinnedSSH') as ssh,patch.object(dispatcher.subprocess,'Popen') as launch:
            self.assertTrue(dispatcher.cycle(self.manifest,self.directory,self.state))
            ssh.assert_not_called();launch.assert_not_called()
        self.assertEqual(self.state['status'],'HALTED_EXISTING_ORIGINAL_INTENT_RECONCILE_ONLY')

    def test_second_runner_does_not_modify_original_state(self):
        state=self.directory/'STATE.json';state.write_text('{"original":true}')
        with (self.directory/'runner.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):dispatcher.run(self.root/'absent.json',self.directory)
        self.assertEqual(state.read_text(),'{"original":true}')

    def test_same_run_in_another_state_directory_shares_the_global_lock(self):
        manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps({**self.manifest,'schema':'native-one-shot-idle-admission-v1','poll_seconds':300}))
        locks=self.root/'state/controller/idle_dispatch_locks';locks.mkdir(parents=True)
        with (locks/(self.request['run_id']+'.lock')).open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):dispatcher.run(manifest,self.root/'another-state-dir')
        self.assertFalse((self.root/'another-state-dir/LOCAL_STARTED.json').exists())
        self.assertFalse((self.root/'another-state-dir/STATE.json').exists())

    def test_timed_out_original_child_is_retained_and_global_claim_prevents_reentry(self):
        from unittest.mock import Mock
        snapshot={'ok':True,'remote_writes':False,'boot_id':'boot',
            'gpus':{self.request['gpu_uuids'][0]:{'memory_mib':0,'util_pct':0}},'apps':[]}
        child=Mock(pid=123);child.wait.side_effect=subprocess.TimeoutExpired('original',2100)
        with patch.object(dispatcher,'PinnedSSH') as ssh,patch.object(dispatcher.subprocess,'Popen',return_value=child),\
                patch.object(dispatcher,'_process_identity',return_value={'pid':123,'uid':0}):
            ssh.return_value.rpc.return_value=snapshot
            self.assertTrue(dispatcher.cycle(self.manifest,self.directory,self.state))
        self.assertEqual(self.state['status'],'HALTED_WAIT_ORIGINAL_LOCAL_LAUNCH_PROCESS')
        child.kill.assert_not_called();child.terminate.assert_not_called()
        with patch.object(dispatcher,'PinnedSSH') as ssh,patch.object(dispatcher.subprocess,'Popen') as launch:
            self.assertTrue(dispatcher.cycle(self.manifest,self.directory,self.state))
            ssh.assert_not_called();launch.assert_not_called()


if __name__=='__main__':unittest.main()
