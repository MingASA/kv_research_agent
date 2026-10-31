"""Read-only metadata can qualify a source anchor; numerical edits cannot."""

import unittest
from scripts.controller.model_source_equivalence import normalized_runtime_module


class SourceEquivalenceTests(unittest.TestCase):
    def test_only_metadata_key_addition_normalizes_equal(self):
        old='class FlashInferBackend:\n def native_kv_runtime_info(self):\n  return {"old":1}\n def forward(self,x):\n  return x*2\n'
        new=old.replace('{"old":1}','{"old":1,"decode_probability_data_type":None}')
        self.assertEqual(normalized_runtime_module(old),normalized_runtime_module(new))
        self.assertNotEqual(normalized_runtime_module(old),normalized_runtime_module(new.replace('x*2','x*3')))
    def test_other_metadata_or_side_effect_cannot_hide_in_exclusion(self):
        source='class FlashInferBackend:\n def native_kv_runtime_info(self):\n  return {"decode_probability_data_type":modify_state()}\n'
        with self.assertRaises(ValueError):normalized_runtime_module(source)


if __name__=='__main__':unittest.main()
