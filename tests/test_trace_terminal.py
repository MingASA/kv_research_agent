# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Preserve terminal conflicts and reject recovery during measured requests."""

import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_trace_run import _terminal_context, _conflict_after_responses
from scripts.controller.durable import digest


class TraceTerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        request = {'run_id': 'fixture-trace'}
        self.ticket = {'request': request, 'request_sha256': digest(request)}
        self.result = {'run_id': request['run_id'], 'request_sha256': digest(request), 'status': 'CONFLICT',
                       'error': 'fixture-only stop', 'gpu_release_verified': True, 'live_owned_containers': []}
        self.stop = {'action': 'STOP_OR_FAILURE', 'status': 'CONFLICT', 'error': 'fixture-only stop',
                     'utc': '2026-10-27T00:00:10+00:00'}
        (self.root/'events.jsonl').write_text(json.dumps(self.stop)+'\n')

    def tearDown(self): self.temp.cleanup()

    def test_conflict_context_retains_stop_and_requires_all_responses_before_it(self):
        when = _terminal_context(self.root, self.ticket, self.result)
        epoch = round(when.timestamp()*1e9)
        self.assertEqual(_conflict_after_responses(when, {'per_request': [{'request_end_ns': epoch-2_000_000_000}]}), 2)
        for end in (epoch, epoch+1):
            with self.assertRaises(ValueError): _conflict_after_responses(when, {'per_request': [{'request_end_ns': end}]})
        self.assertFalse((self.root/'COMPLETE').exists())

    def test_unreleased_or_wrong_run_or_fabricated_complete_cannot_be_recovered(self):
        for key, wrong in [('gpu_release_verified', False), ('live_owned_containers', ['foreign']),
                           ('run_id', 'different'), ('status', 'FAILED')]:
            changed = dict(self.result); changed[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                _terminal_context(self.root, self.ticket, changed)
        (self.root/'COMPLETE').write_text('{}')
        with self.assertRaises(ValueError): _terminal_context(self.root, self.ticket, self.result)

    def test_missing_ambiguous_or_wrong_stop_event_blocks_recovery(self):
        for events in ([], [self.stop, self.stop], [dict(self.stop, status='FAILED')],
                       [dict(self.stop, error='wrong')], [dict(self.stop, utc='2026-10-27T00:00:10')]):
            (self.root/'events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in events))
            with self.assertRaises(ValueError): _terminal_context(self.root, self.ticket, self.result)

    def test_clean_complete_must_still_seal_exact_result_bytes(self):
        self.result['status'] = 'COMPLETE'
        raw = (json.dumps(self.result)+'\n').encode(); (self.root/'result.json').write_bytes(raw)
        (self.root/'COMPLETE').write_text(json.dumps({'result_sha256': hashlib.sha256(raw).hexdigest()}))
        self.assertIsNone(_terminal_context(self.root, self.ticket, self.result))
        (self.root/'result.json').write_bytes(raw+b' ')
        with self.assertRaises(ValueError): _terminal_context(self.root, self.ticket, self.result)
