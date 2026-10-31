"""Repeated questions about one book are not independent bootstrap units."""

import unittest

import numpy as np

from scripts.compare_nextgen_quality import _cluster_bootstrap


class ContextClusterQualityTests(unittest.TestCase):
    def test_all_questions_in_a_cluster_move_together(self):
        values=np.array([1.,1.,-1.])
        boot=_cluster_bootstrap(values,['book-a','book-a','book-b'],np.random.default_rng(1),400)
        self.assertEqual(set(np.unique(boot)),{-1.,1/3,1.})

    def test_tied_paired_scores_produce_zero_difference_interval(self):
        boot=_cluster_bootstrap(np.zeros(5),['a','a','b','c','c'],np.random.default_rng(3),100)
        np.testing.assert_array_equal(boot,np.zeros(100))


if __name__=='__main__':unittest.main()
