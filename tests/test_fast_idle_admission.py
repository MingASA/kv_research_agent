"""Real supervisor fault injection for user-requested opportunistic admission."""

import copy
import json
import unittest

from scripts.controller.approval import ApprovalError
from scripts.controller.idle_admission import FAST_ADMISSION,idle_parameters
from tests import test_supervisor as fixtures


class FastAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SupervisorTests();self.fixture.setUp()
        self.fixture.request.update(arms=['bf16'],
            workload={'kind':'upstream_FP8_independent_numeric_calibration'},
            resource_admission=copy.deepcopy(FAST_ADMISSION))

    def tearDown(self):self.fixture.tearDown()

    def test_legacy_gate_unchanged(self):
        self.assertEqual(idle_parameters({}),(120,30))

    def test_fast_idle_still_runs_precreate_checks(self):
        result=self.fixture.run_engine()
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(self.fixture.backend.created[0][2],self.fixture.clock.origin)

    def test_busy_gpu_is_polled_every5_seconds(self):
        self.fixture.backend.busy_until=self.fixture.clock.origin+16_000_000_000
        result=self.fixture.run_engine()
        self.assertEqual(result['status'],'COMPLETE')
        self.assertEqual(self.fixture.backend.created[0][2]-self.fixture.clock.origin,20_000_000_000)

    def test_race_after_idle_cannot_create_container(self):
        def occupy():self.fixture.backend.busy_until=self.fixture.clock.now()+99_000_000_000
        self.fixture.backend.verify_preflight=occupy
        result=self.fixture.run_engine()
        self.assertEqual(result['status'],'NOT_LAUNCHED')
        self.assertEqual(self.fixture.backend.created,[])

    def test_foreign_activity_stops_only_owned_allocation(self):
        self.fixture.backend.foreign=True
        result=self.fixture.run_engine()
        self.assertEqual(result['status'],'CONFLICT')
        self.assertEqual(len(self.fixture.backend.stopped),1)

    def test_unknown_policy_or_legacy_workload_rejected(self):
        for change in ({'resource_admission':dict(FAST_ADMISSION,poll_seconds=1)},
                       {'workload':{'kind':'upstream_official_A_model_qualification'}}):
            request=copy.deepcopy(self.fixture.request);request.update(change)
            with self.assertRaises(ApprovalError):idle_parameters(request)

    def test_busy_logs_bounded_by_changes_and20minute_heartbeat(self):
        fixture=self.fixture
        fixture.backend.busy_until=fixture.clock.now()+3000_000_000_000
        engine=fixtures.Supervisor(fixture.ticket,fixture.root,fixture.backend,
                                  fixture.clock.now,fixture.clock.sleep,fixture.clock.wall)
        with self.assertRaises(TimeoutError):engine._idle(fixture.clock.now()+2000_000_000_000)
        self.assertEqual(len((fixture.root/'events.jsonl').read_text().splitlines()),2)


if __name__=='__main__':unittest.main()
