"""Fault injection for the sequential queue; no SSH or GPU operations."""

import copy
import datetime
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import guarded_native_queue as queue
from scripts.controller.durable import digest, write_new_json
from scripts.controller.quota import QuotaLedger
from test_quota_ledger import _fixture


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.directory=self.root/'queue';self.directory.mkdir(mode=0o700)
        (self.root/'state/controller/tickets').mkdir(parents=True,mode=0o700)
        self.approval,self.request,self.now=_fixture()
        self.manifest={'jobs':[{'run_id':self.request['run_id'],'package':'unused'}],
            'not_after_utc':(self.now+datetime.timedelta(hours=12)).isoformat(),
            'expected_boot_id':'boot','poll_seconds':300}
        self.state={'cursor':0,'sample':0,'current_run_id':None}
        self.patches=[patch.object(queue,'ROOT',self.root),patch.object(queue,'_publish'),
            patch.object(queue,'load_approval_snapshot',return_value=(self.approval,'approval-sha')),
            patch.object(queue,'_now',return_value=self.now)]
        for item in self.patches:item.start()

    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.temp.cleanup()

    def reserve(self,status='RUNNING'):
        with QuotaLedger(self.root/'state/controller') as ledger:
            row=ledger.reserve(self.approval,self.request,digest(self.approval),self.now)
            if status in ('DISPATCHING','RUNNING','UNKNOWN'):
                ledger.transition(self.request['run_id'],'DISPATCHING',{})
            if status in ('RUNNING','UNKNOWN'):ledger.transition(self.request['run_id'],status,{})
        write_new_json(self.root/'state/controller/tickets'/(self.request['run_id']+'.json'),
                       {'approval':self.approval,'reservation_id':row['reservation_id']})
        return row

    def test_root_vllm_with_zero_utilization_still_blocks(self):
        snapshot={'ok':True,'remote_writes':False,'boot_id':'boot',
            'gpus':{'selected':{'memory_mib':0,'util_pct':0}},
            'apps':[{'gpu_uuid':'another','uid':0,'comm':'VLLM::Worker_TP','pid':123}]}
        self.assertFalse(queue.coarse_gpu_gate(snapshot,['selected'],'boot')[0])
        snapshot['apps']=[];snapshot['gpus']['selected']['memory_mib']=75091
        self.assertFalse(queue.coarse_gpu_gate(snapshot,['selected'],'boot')[0])
        snapshot['gpus']['selected']['memory_mib']=0
        snapshot['apps']=[{'gpu_uuid':'selected','uid':None,'pid':123}]
        self.assertFalse(queue.coarse_gpu_gate(snapshot,['selected'],'boot')[0])
        snapshot['apps']=[]
        self.assertTrue(queue.coarse_gpu_gate(snapshot,['selected'],'boot')[0])
        snapshot['boot_id']='new-boot'
        with self.assertRaises(ValueError):queue.coarse_gpu_gate(snapshot,['selected'],'boot')

    def test_durable_intent_without_ticket_never_relaunches(self):
        self.state['current_run_id']=self.request['run_id']
        with patch.object(queue,'_cli') as cli:
            self.assertTrue(queue._cycle(self.manifest,self.directory,self.state));cli.assert_not_called()
        self.assertIn('HALTED_UNCERTAIN',self.state['status'])

    def test_unknown_or_undispatched_reservation_never_relaunches(self):
        self.reserve('UNKNOWN')
        with patch.object(queue,'_cli') as cli:
            self.assertTrue(queue._cycle(self.manifest,self.directory,self.state));cli.assert_not_called()
        self.assertEqual(self.state['status'],'HALTED_UNKNOWN_NO_RELAUNCH')
        self.assertTrue(queue.action_for_existing('RESERVED').startswith('HALT'))

    def test_recovered_running_job_only_reconciles_same_identity(self):
        self.reserve()
        with patch.object(queue,'_cli',return_value=(0,{'status':'RUNNING','supervisor_alive':True})) as cli:
            self.assertFalse(queue._cycle(self.manifest,self.directory,self.state))
        self.assertEqual(cli.call_args.args[2],['reconcile','--run-id',self.request['run_id']])
        self.assertEqual(self.state['cursor'],0)

    def test_failed_experiment_stops_before_next_job(self):
        self.reserve()
        with patch.object(queue,'_cli',return_value=(0,{'status':'FAILED','supervisor_alive':False})),\
                patch.object(queue,'_postprocess') as post:
            self.assertTrue(queue._cycle(self.manifest,self.directory,self.state));post.assert_not_called()
        self.assertEqual(self.state['cursor'],0)

    def test_completion_needs_fresh_release_and_independent_postprocessing(self):
        self.reserve()
        status={'status':'COMPLETE','supervisor_alive':False,'live_owned_containers':[],
                'files':{'result.json':{'value':{'gpu_release_verified':True}}}}
        with patch.object(queue,'_cli',return_value=(0,status)),patch.object(queue,'PinnedSSH'),\
                patch.object(queue,'_postprocess',return_value={'independent_scores':'verified'}) as post:
            self.assertFalse(queue._cycle(self.manifest,self.directory,self.state));post.assert_called_once()
        self.assertEqual(self.state['cursor'],1);self.assertIsNone(self.state['current_run_id'])
        self.assertEqual(len(self.state['completed']),1)

    def test_surviving_container_cannot_advance(self):
        self.reserve()
        status={'status':'COMPLETE','supervisor_alive':False,'live_owned_containers':['still-live'],
                'files':{'result.json':{'value':{'gpu_release_verified':True}}}}
        with patch.object(queue,'_cli',return_value=(0,status)),patch.object(queue,'_postprocess') as post:
            self.assertTrue(queue._cycle(self.manifest,self.directory,self.state));post.assert_not_called()
        self.assertEqual(self.state['cursor'],0)

    def test_live_previous_cli_is_waited_without_concurrent_reconciliation(self):
        with patch.object(queue,'_child_alive',return_value=True),patch.object(queue,'_cli') as cli:
            self.assertFalse(queue._cycle(self.manifest,self.directory,self.state));cli.assert_not_called()
        self.assertEqual(self.state['status'],'WAIT_ORIGINAL_LOCAL_CLI_PROCESS')

    def test_expired_admission_stops_without_ssh(self):
        self.manifest['not_after_utc']=(self.now-datetime.timedelta(seconds=1)).isoformat()
        with patch.object(queue,'PinnedSSH') as ssh:
            self.assertTrue(queue._cycle(self.manifest,self.directory,self.state));ssh.assert_not_called()

    def test_busy_cycle_never_calls_cli_or_reserves(self):
        snapshot={'ok':True,'remote_writes':False,'boot_id':'boot',
            'gpus':{},'apps':[{'uid':0,'comm':'VLLM::Worker_TP','gpu_uuid':'busy'}]}
        with patch.object(queue,'validate_execution_authority'),patch.object(queue,'validate_queue_manifest'),\
                patch.object(queue,'validate_scope'),patch.object(queue,'_local_package',return_value=(self.request,{},{})),\
                patch.object(queue,'PinnedSSH') as ssh,patch.object(queue,'_cli') as cli:
            ssh.return_value.rpc.return_value=snapshot
            self.assertFalse(queue._cycle(self.manifest,self.directory,self.state));cli.assert_not_called()
        with QuotaLedger(self.root/'state/controller') as ledger:
            self.assertEqual(ledger.usage(self.approval)['known_run_ids'],[])

    def test_second_runner_cannot_overwrite_state_or_read_manifest(self):
        saved=self.directory/'STATE.json';saved.write_text('{"original":"unchanged"}')
        with (self.directory/'runner.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(queue.QueueAlreadyRunning):
                queue.run(self.root/'does-not-exist',self.directory)
        self.assertEqual(saved.read_text(),'{"original":"unchanged"}')

    def test_naive_deadline_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'UTC'):queue._utc('2026-10-26T10:00:00')

    def test_changed_package_fails_before_request_or_ssh(self):
        package=self.root/'package.json';package.write_text('{}')
        manifest={'schema':'kv-guarded-sequential-queue-v1','poll_seconds':300,
            'not_after_utc':self.manifest['not_after_utc'],'expected_boot_id':'boot',
            'jobs':[{'run_id':'one','package':str(package),'package_sha256':'0'*64},
                    {'run_id':'two','package':str(package),'package_sha256':'0'*64}]}
        with patch.object(queue,'validate_execution_authority'),patch.object(queue,'_local_package') as local:
            with self.assertRaisesRegex(ValueError,'package changed'):queue.validate_queue_manifest(manifest)
            local.assert_not_called()

    def test_queue_launch_wrapper_exclusively_owns_original_receipts(self):
        output=self.directory/'launch-0.json';errors=self.directory/'launch-0.stderr.log'
        def start(argv,**kwargs):
            self.assertIn('scripts.launch_guarded_once',argv)
            self.assertFalse(output.exists());self.assertFalse(errors.exists())
            self.assertEqual(Path(argv[argv.index('--receipt')+1]),output)
            output.write_text('{"status":"DISPATCHED"}');errors.write_text('')
            from unittest.mock import Mock
            child=Mock(pid=100);child.wait.return_value=0
            return child
        with patch.object(queue.subprocess,'Popen',side_effect=start),\
                patch.object(queue,'_process_identity',return_value={'pid':100}):
            code,ack=queue._cli(self.directory,'launch-0',['launch','--package','frozen-package'],self.state)
        self.assertEqual((code,ack['status']),(0,'DISPATCHED'))
        self.assertTrue((self.directory/'launch-0.wrapper.json').exists())
        saved=output.read_bytes()
        with patch.object(queue.subprocess,'Popen') as child,self.assertRaises(ValueError):
            queue._cli(self.directory,'launch-0',['launch','--package','frozen-package'],self.state)
        child.assert_not_called();self.assertEqual(output.read_bytes(),saved)

    def test_wrapper_refusal_cannot_be_treated_as_dispatched_or_retried(self):
        from unittest.mock import Mock
        child=Mock(pid=100);child.wait.return_value=2
        with patch.object(queue.subprocess,'Popen',return_value=child),\
                patch.object(queue,'_process_identity',return_value={'pid':100}):
            code,ack=queue._cli(self.directory,'launch-0',['launch','--package','frozen-package'],self.state)
        self.assertEqual(code,2);self.assertNotIn('status',ack)
        self.assertFalse((self.directory/'launch-0.json').exists())
        with patch.object(queue.subprocess,'Popen') as child,self.assertRaises(ValueError):
            queue._cli(self.directory,'launch-0',['launch','--package','frozen-package'],self.state)
        child.assert_not_called()


if __name__=='__main__':unittest.main()
