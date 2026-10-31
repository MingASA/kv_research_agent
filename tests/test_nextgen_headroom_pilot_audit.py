import unittest

from scripts.audit_nextgen_headroom_pilot import select_samples


class HeadroomEvidenceTests(unittest.TestCase):
    def test_requires_server_residency_in_one_sample_during_real_long_client_overlap(self):
        requests=[{'request_started_monotonic_ns':10,'request_ended_monotonic_ns':50},
                  {'request_started_monotonic_ns':20,'request_ended_monotonic_ns':40}]
        def row(clock,running,pages):
            return {'monotonic_ns':clock,'errors':{},'values':{'num_running_requests':running,'num_used_blocks':pages}}
        rows=[row(25,2,20),row(30,1,40),row(45,2,40)]
        eligible,selected=select_samples(rows,requests,30)
        self.assertEqual(len(eligible),2);self.assertIsNone(selected)
        rows.append(row(35,2,40));_,selected=select_samples(rows,requests,30)
        self.assertEqual(selected['monotonic_ns'],35)
