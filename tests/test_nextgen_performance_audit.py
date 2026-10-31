"""Raw failures and optimistic summaries must not become performance evidence."""

import json
import unittest

from scripts.audit_nextgen_performance import audit_phase
from scripts.controller.nextgen_performance import aiperf_command
from scripts.controller.production import audit_requests
import test_production_serving as fixtures


class IndependentPerformancePhaseTests(unittest.TestCase):
    def setUp(self):
        fixtures.ProductionRawAuditTests.setUp(self)
        for row in self.rows:row['metadata']['benchmark_phase']='profiling'
        for row in self.raws:row['metadata']['benchmark_phase']='profiling'
        summary=json.loads((self.artifacts/'profile_export_aiperf.json').read_text())
        summary['request_throughput']['avg']=3;summary['output_token_throughput']['avg']=5
        (self.artifacts/'profile_export_aiperf.json').write_text(json.dumps(summary))
        for filename,rows in (('profile_export.jsonl',self.rows),('profile_export_raw.jsonl',self.raws)):
            (self.artifacts/filename).write_text('\n'.join(map(json.dumps,rows))+'\n')
        self.command=aiperf_command('model',self.source,self.artifacts,requests=3,concurrency=2)
        (self.root/'command.json').write_text(json.dumps(self.command))
        (self.root/'aiperf_cli_CPU_acceptance.json').write_text(json.dumps({
            'actual_Cyclopts_CLIConfig_accepted':True,'aiperf_version':'0.13.0','cli_config_sha256':'a'*64,
            'request_count':3,'concurrency':2,'warmup_request_count':None,'benchmark_started':False,'GPU_used':False}))
        result=audit_requests(self.artifacts,self.source,3,0,allow_multi_turn=True)
        result.update(phase='measured',worker_observation_started_monotonic_ns=1000,
                      worker_observation_ended_monotonic_ns=1_000_001000)
        (self.root/'independent_worker_audit.json').write_text(json.dumps(result))

    def tearDown(self):self.temp.cleanup()

    def audit(self):
        return audit_phase(self.root,self.source,expected_requests=3,phase='measured',command=self.command,
                           cli_config_sha256='a'*64)

    def test_complete_raw_phase_recomputes_actual_counts_rates_and_one_token_semantics(self):
        result=self.audit()
        self.assertEqual(result['profiling_records'],3);self.assertEqual(result['warmup_records'],0)
        self.assertEqual(result['actual_output_tokens'],5);self.assertEqual(result['request_rps'],3)
        self.assertEqual(result['one_token_answers'],1);self.assertEqual(result['itl_p50_ms'],1)

    def test_cached_success_cannot_hide_actual_429_or_changed_payload(self):
        for mutation in ('status','payload'):
            original=(self.artifacts/'profile_export_raw.jsonl').read_text()
            rows=[json.loads(line) for line in original.splitlines()]
            if mutation=='status':rows[0]['status']=429
            else:rows[0]['payload']['messages'][0]['content']='different request'
            (self.artifacts/'profile_export_raw.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.audit()
            (self.artifacts/'profile_export_raw.jsonl').write_text(original)

    def test_inflated_cached_throughput_is_rejected_even_when_all_requests_succeeded(self):
        path=self.root/'independent_worker_audit.json';data=json.loads(path.read_text());data['output_tps']*=2
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'recomputed actual raw'):self.audit()

    def test_changed_cli_or_embedded_warmup_is_refused(self):
        path=self.root/'command.json';path.write_text(json.dumps(self.command+['--ignore-eos']))
        with self.assertRaisesRegex(ValueError,'argv'):self.audit()
        path.write_text(json.dumps(self.command))
        path=self.root/'aiperf_cli_CPU_acceptance.json';data=json.loads(path.read_text());data['warmup_request_count']=1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'warmup admission'):self.audit()

    def test_parser_accepting_wrong_session_count_cannot_pass_audit(self):
        self.command=aiperf_command('model',self.source,self.artifacts,requests=3,concurrency=2,sessions=3)
        (self.root/'command.json').write_text(json.dumps(self.command))
        path=self.root/'aiperf_cli_CPU_acceptance.json';data=json.loads(path.read_text())
        data['conversation_num']=2;path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'parser/config/warmup admission'):self.audit()
        data['conversation_num']=3;path.write_text(json.dumps(data))
        self.assertEqual(self.audit()['profiling_records'],3)


if __name__=='__main__':unittest.main()
