# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Real Unicode separators must survive AIPerf JSONL framing without raw repair."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.controller.production import _jsonl_rows


class JsonlUnicodeTests(unittest.TestCase):
    def test_valid_payload_and_response_unicode_separators_remain_in_the_same_record(self) -> None:
        records=[{'payload':{'messages':[{'role':'user','content':'before\u2028middle\u2029after\u0085end'}]},
                  'responses':[{'text':'visible\u2028text'}]}, {'payload':{'content':'line1\nline2'}}]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile_export_raw.jsonl'
            path.write_bytes(b''.join((json.dumps(row,ensure_ascii=False)+'\n').encode() for row in records))
            self.assertGreater(len(path.read_text().splitlines()),len(records))
            self.assertEqual(list(_jsonl_rows(path)),records)

    def test_genuinely_truncated_raw_record_is_rejected_without_concatenation_or_repair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'raw.jsonl';path.write_bytes(b'{"valid":true}\n{"text":"unfinished\n')
            rows=_jsonl_rows(path);self.assertEqual(next(rows),{'valid':True})
            with self.assertRaises(json.JSONDecodeError):next(rows)


if __name__=='__main__':unittest.main()
