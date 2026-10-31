import json
import unittest

from scripts.audit_nextgen_api_cache import cache_row,totals


class NativeCacheAuditTests(unittest.TestCase):
    def raw(self,cached):
        return {'metadata':{'benchmark_phase':'profiling','turn_index':1,
                'x_request_id':'r','conversation_id':'s'},'responses':[{'packets':[
                    {'name':'data','value':json.dumps({'usage':{'prompt_tokens':100,
                    'prompt_tokens_details':{'cached_tokens':cached}}})},
                    {'name':'data','value':'[DONE]'}]}]}

    def test_rejects_invalid_native_cache_counts_and_duplicate_usage(self):
        for cached in (-1,101,True,10.5):
            with self.subTest(cached=cached),self.assertRaises(ValueError):cache_row(self.raw(cached))
        raw=self.raw(64);raw['responses'][0]['packets'].insert(0,raw['responses'][0]['packets'][0])
        with self.assertRaises(ValueError):cache_row(raw)

    def test_full_denominator_includes_zero_hit_requests(self):
        result=totals([cache_row(self.raw(0)),cache_row(self.raw(64))])
        self.assertEqual(result,{'requests':2,'prompt_tokens':200,'cached_tokens':64,
            'requests_with_nonzero_cached_tokens':1,'cached_input_fraction':.32})
