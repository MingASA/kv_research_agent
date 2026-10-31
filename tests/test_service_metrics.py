# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""CPU bvar semantics checks; no HTTP, model, Docker or GPU operations."""

import unittest
from scripts.controller.service_metrics import parse_metric


class ServiceMetricsTests(unittest.TestCase):
    def test_real_own_server_html_shape_and_ratio_units(self):
        self.assertEqual(parse_metric('num_free_blocks', '<p class="variable">num_free_blocks : <span id="value-num_free_blocks">6</span></p>'), 6)
        self.assertEqual(parse_metric('kv_cache_utilization_perc', '<span id="value-kv_cache_utilization_perc">0.8</span>'), .8)
        self.assertEqual(parse_metric('kv_cache_utilization_perc', '<span id="value-kv_cache_utilization_perc">1.00068</span>'), 1.00068)
        with self.assertRaises(ValueError):
            parse_metric('kv_cache_utilization_perc', '<span id="value-kv_cache_utilization_perc">80</span>')

    def test_missing_duplicate_nonfinite_or_wrong_count_cannot_become_valid_data(self):
        name = 'num_running_requests'
        for text in ('', '<span id="value-other">2</span>',
                     '<span id="value-num_running_requests">nan</span>',
                     '<span id="value-num_running_requests">1.5</span>',
                     '<span id="value-num_running_requests">-1</span>',
                     '<span id="value-num_running_requests">1</span><span id="value-num_running_requests">2</span>'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_metric(name, text)


if __name__ == '__main__':
    unittest.main()
