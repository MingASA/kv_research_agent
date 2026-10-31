"""Protect the failed-load denominator; CPU-only, no files or model calls."""
import copy
import unittest
from audit_failed_mixed_pressure import validate_inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.source={(f'doc{i}',turn):{} for i in range(16) for turn in (0,1)}
        self.rows=[{'metadata':{'conversation_id':session,'turn_index':turn,'x_request_id':str(n),
            'benchmark_phase':'profiling','was_cancelled':False,'context_overflow_skip':False}}
            for n,(session,turn) in enumerate(self.source)]
        self.rows[4]['error']={'type':'TimeoutError'}

    def test_one_timeout_keeps_all_original32(self):
        self.assertIs(validate_inventory(self.rows,self.source),self.rows[4])

    def test_filtering_failed_request_cannot_create_a_successful_cohort(self):
        with self.assertRaisesRegex(ValueError,'All32'):
            validate_inventory([row for row in self.rows if not row.get('error')],self.source)

    def test_duplicate_unknown_or_skipped_requests_are_rejected(self):
        for name in ['duplicate','unknown','skipped']:
            rows=copy.deepcopy(self.rows)
            if name=='duplicate':rows[0]=copy.deepcopy(rows[1])
            elif name=='unknown':rows[0]['metadata']['conversation_id']='another'
            else:rows[0]['metadata']['context_overflow_skip']=True
            with self.subTest(name=name),self.assertRaises(ValueError):validate_inventory(rows,self.source)

    def test_other_error_type_or_no_error_cannot_use_this_failure_certificate(self):
        for error in (None,{'type':'HTTP500'}):
            rows=copy.deepcopy(self.rows);rows[4]['error']=error
            with self.subTest(error=error),self.assertRaises(ValueError):validate_inventory(rows,self.source)


if __name__=='__main__':unittest.main(verbosity=2)
