import copy
import unittest

from scripts.compare_nextgen_tools import paired_scores


class ToolPairTests(unittest.TestCase):
    def score(self):
        return {'all_actual_histories_SSE_usage_and_tool_replays_verified':True,
            'all_original_tasks_retained_in_denominator':True,'tasks':32,'correct':32,'task_success_percent':100,
            'source_tasks_sha256':'a','reference_sha256':'b','scores':[
                {'task_id':str(i),'task_correct':True,'budget_or_incomplete_conversation':False,
                 'official_checker_AST_sha256':'c','original_user_turns':3} for i in range(32)]}

    def test_incomplete_task_stays_in_original_denominator(self):
        left=self.score();right=copy.deepcopy(left);right['scores'][2].update(task_correct=False,budget_or_incomplete_conversation=True)
        right.update(correct=31,task_success_percent=96.875)
        result=paired_scores(left,right)
        self.assertEqual(result['change_percentage_points'],-3.125)
        self.assertEqual(result['paired_outcome_ids']['BF16_correct_candidate_failed'],['2'])
        right['scores'][2]['task_correct']=True;right.update(correct=32,task_success_percent=100)
        with self.assertRaises(ValueError):paired_scores(left,right)

    def test_changed_cohort_or_duplicate_task_is_refused(self):
        left=self.score();right=self.score();right['scores'][1]['task_id']='0'
        with self.assertRaises(ValueError):paired_scores(left,right)
        right=self.score();right['reference_sha256']='changed'
        with self.assertRaises(ValueError):paired_scores(left,right)
