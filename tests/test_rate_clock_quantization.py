# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""AIPerf datetime export precision must not hide real throughput mismatches."""

import datetime
import unittest

from scripts.controller.production import _verify_observation_rates


class RateClockQuantizationTests(unittest.TestCase):
    def test_nanosecond_rate_window_survives_microsecond_datetime_export(self) -> None:
        start_ns=1793100000123456789;duration_ns=27340000617;tokens=71945;requests=128
        summary={'start_time':datetime.datetime.fromtimestamp(start_ns/1e9).isoformat(),
            'end_time':datetime.datetime.fromtimestamp((start_ns+duration_ns)/1e9).isoformat(),
            'output_token_throughput':{'avg':tokens/(duration_ns/1e9)},
            'request_throughput':{'avg':requests/(duration_ns/1e9)}}
        result=_verify_observation_rates(summary,tokens,requests)
        self.assertGreater(abs(result['difference_from_ISO_ns']['output_token_throughput']),280)
        self.assertLessEqual(abs(result['difference_from_ISO_ns']['output_token_throughput']),result['export_conversion_bound_ns'])
        self.assertLess(result['export_conversion_bound_ns'],2000)
        with self.assertRaises(ValueError):_verify_observation_rates(summary,tokens+1,requests)
        with self.assertRaises(ValueError):_verify_observation_rates(summary,tokens,requests-1)

    def test_real_rate_or_window_changes_still_fail(self) -> None:
        summary={'start_time':'2026-10-27T19:00:00.000000','end_time':'2026-10-27T19:00:30.000000',
            'output_token_throughput':{'avg':1000.01},'request_throughput':{'avg':128/30}}
        with self.assertRaises(ValueError):_verify_observation_rates(summary,30000,128)
        summary['output_token_throughput']['avg']=1000
        summary['end_time']='2026-10-27T19:00:30.000010'
        with self.assertRaises(ValueError):_verify_observation_rates(summary,30000,128)


if __name__=='__main__':unittest.main()
