"""Bounded stream, stale data, retry and byte-preserving index fault tests."""

import datetime as dt
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import fast_idle_research_queue as queue


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.at=dt.datetime(2026,11,6,tzinfo=dt.timezone.utc)
        self.sample={'ok':True,'remote_writes':False,'boot_id':'boot','utc':self.at.isoformat(),
                     'gpus':{'GPU-a':{'memory_mib':0,'util_pct':0,'pids':[]},
                             'GPU-b':{'memory_mib':49000,'util_pct':0,'pids':[42]}}}

    def test_idle_other_card_is_not_blocked_by_foreign_job_elsewhere(self):
        self.assertEqual(queue.idle_candidates(self.sample,['GPU-a','GPU-b'],'boot',self.at),['GPU-a'])

    def test_zero_utilization_is_not_idle_memory_or_process_proof(self):
        for update in ({'memory_mib':1},{'pids':[99]},{'util_pct':6},{'memory_mib':False}):
            sample=json.loads(json.dumps(self.sample));sample['gpus']['GPU-a'].update(update)
            self.assertEqual(queue.idle_candidates(sample,['GPU-a'],'boot',self.at),[])

    def test_stale_future_unknown_identity_and_query_error_do_not_admit(self):
        for update in ({'utc':(self.at-dt.timedelta(seconds=11)).isoformat()},
                       {'utc':(self.at+dt.timedelta(seconds=6)).isoformat()},
                       {'utc':'malformed'},{'boot_id':'other'},{'ok':False},{'remote_writes':True}):
            sample=dict(self.sample,**update)
            self.assertEqual(queue.idle_candidates(sample,['GPU-a'],'boot',self.at),[])

    def reader(self,raw):
        probe=queue.Probe.__new__(queue.Probe)
        probe.latest={};probe.received=0;probe.error='';probe.lock=threading.Lock()
        probe.process=SimpleNamespace(stdout=io.BytesIO(raw));probe._read();return probe

    def test_stream_keeps_only_latest_frame(self):
        p=self.reader(b'{"seq":1}\n{"seq":2}\n')
        self.assertEqual(p.sample()[0],{'seq':2})

    def test_stream_rejects_truncated_or_oversized_json(self):
        for raw in (b'{"seq":1}',b'x'*16385+b'\n',b'[]\n'):
            self.assertIn('Error',self.reader(raw).error)

    def test_index_update_preserves_all_historical_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'results').mkdir();p=root/'results/INDEX.csv'
            historical=b'run_id,status,model,backend,git_sha,binary_sha256,dataset_sha256,gpu_uuid,manifest_path,validity,notes\r\nold,COMPLETE,oldmodel,b,s,b,d,G,p,v,keep-this-byte-for-byte\r\n'
            p.write_bytes(historical)
            r={'run_id':'new','model_id':'model','operators':{},'python_source_commit':'a'*40,
               'binary_sha256':'b'*64,'dataset_sha256':'c'*64,'gpu_uuids':['GPU-a'],'remote_output':'/owned/new'}
            with patch.object(queue,'ROOT',root):
                queue.index_run(r,'RUNNING');queue.index_run(r,'CONFLICT')
            self.assertTrue(p.read_bytes().startswith(historical))
            self.assertEqual(p.read_bytes().count(b'new,CONFLICT'),1)

    def test_changed_template_is_rejected_before_any_attempt_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'package.json';p.write_text('{}')
            with self.assertRaises(ValueError):queue.prepare_attempt({}, {'package':str(p),'package_sha256':'0'*64},'GPU-a',1)

    def terminal_case(self,status):
        context=tempfile.TemporaryDirectory();root=Path(context.name)
        (root/'state/controller/tickets').mkdir(parents=True);(root/'state/controller/local_launch_once').mkdir()
        run='original';(root/f'state/controller/tickets/{run}.json').write_text(json.dumps({'request':{'run_id':run,'gpu_uuids':['GPU-a']},'approval':{}}))
        (root/f'state/controller/local_launch_once/{run}.finished.json').write_text('{}')
        runner=queue.Runner.__new__(queue.Runner);runner.state={'active_run_id':run,'run_directory':str(root/'run'),'cursor':0}
        (root/'run').mkdir();runner.launcher=None;runner.handles=[];runner.stop=False;runner.next_status=0
        runner.manifest={'stages':[{'id':'stage'}]};events=[]
        runner.event=lambda state,**kw:events.append(state)
        result={'status':status,'gpu_release_verified':True,'live_owned_containers':[]}
        ledger=SimpleNamespace(get=lambda _: {'state':'SETTLED','evidence_json':json.dumps(result)})
        manager=SimpleNamespace(__enter__=lambda _:ledger,__exit__=lambda *_:None)
        class LedgerContext:
            def __enter__(self):return ledger
            def __exit__(self,*_):pass
        class Dispatch:
            def __init__(self,*_):pass
            def reconcile(self,_):return {'live_owned_containers':[]}
        (root/'scripts/controller').mkdir(parents=True);(root/'scripts/controller/remote_rpc.py').write_text('# fixture')
        with patch.object(queue,'ROOT',root),patch.object(queue,'QuotaLedger',return_value=LedgerContext()),\
             patch.object(queue,'PinnedSSH'),patch.object(queue,'Dispatcher',Dispatch),patch.object(queue,'index_run'):
            runner.active()
        context.cleanup();return runner,events

    def test_resource_conflict_retries_fresh_id_but_keeps_stage(self):
        runner,events=self.terminal_case('CONFLICT')
        self.assertIsNone(runner.state['active_run_id']);self.assertEqual(runner.state['cursor'],0)
        self.assertEqual(runner.state['last_conflict_gpu'],'GPU-a')
        self.assertFalse(runner.stop);self.assertIn('ABANDONED_RESOURCE_ATTEMPT_RETURN_TO_WATCH',events)

    def test_numerical_or_execution_failure_is_not_retried_as_a_conflict(self):
        runner,events=self.terminal_case('FAILED')
        self.assertTrue(runner.stop);self.assertEqual(runner.state['cursor'],0)
        self.assertIn('HALTED_EXPERIMENT_FAILURE_NEEDS_ATTRIBUTION',events)

    def test_completed_prerequisite_advances_once_without_acceptance_claim(self):
        runner,events=self.terminal_case('COMPLETE')
        self.assertEqual(runner.state['cursor'],1);self.assertIsNone(runner.state['active_run_id'])
        self.assertIn('STAGE_COMPLETE_RAW_HANDOFF',events)

    def test_manual_stop_or_identity_refusal_is_never_automatically_retried(self):
        result={'status':'NOT_LAUNCHED','gpu_release_verified':True,'live_owned_containers':[],
                'allocation_intervals_ns':[],'arm_results':[]}
        for error in ('TimeoutError: Stop requested while waiting; no worker will be admitted',
                      'UnsafeState: Project owner changed'):
            self.assertFalse(queue.resource_retryable(dict(result,error=error)))
        self.assertTrue(queue.resource_retryable(dict(result,error='UnsafeState: GPU state changed between stable idle and worker create')))

    def test_resumed_exit_preserves_original_receipt_and_active_handoff(self):
        with tempfile.TemporaryDirectory() as folder:
            runner=queue.Runner.__new__(queue.Runner)
            runner.directory=Path(folder);runner.state={'active_run_id':'original-running'}
            runner.probe=None;runner.stop=True;runner.event=lambda *_:None
            runner.run()
            original=(runner.directory/'EXIT.json').read_bytes()
            runner.run()
            self.assertEqual((runner.directory/'EXIT.json').read_bytes(),original)
            exits=list(runner.directory.glob('EXIT*.json'))
            self.assertEqual(len(exits),2)
            for path in exits:
                data=json.loads(path.read_bytes())
                self.assertEqual(data['active_run_id'],'original-running')
                self.assertTrue(data['original_run_must_be_reconciled'])
                self.assertFalse(data['GPU_mutations_replayed'])


if __name__=='__main__':unittest.main()
