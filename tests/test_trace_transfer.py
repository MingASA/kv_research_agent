# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Compressed input stays byte-pinned and cannot exceed its declared bound."""

import copy
import gzip
import hashlib
import lzma
from pathlib import PurePosixPath
import unittest

from scripts.controller.remote_rpc import decode_dataset_asset


class TraceTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = b'{"fixture":"CPU only"}\n'*100
        self.request = {'workload': {'kind': 'anonymous_trace_xqa'}, 'remote_output': '/owned/new-run',
            'dataset_path': '/owned/new-run/input/payload_trace.jsonl', 'dataset_sha256': hashlib.sha256(self.raw).hexdigest(),
            'dataset_archive': {'relative_path': 'input/payload_trace.jsonl.gz', 'uncompressed_size_bytes': len(self.raw)}}
        self.assets = [(PurePosixPath('input/payload_trace.jsonl.gz'), gzip.compress(self.raw))]

    def test_exact_payload_is_recovered_without_mutating_any_files(self) -> None:
        self.assertEqual(decode_dataset_asset({'request': self.request}, self.assets), self.raw)

    def test_size_sha_and_destination_changes_are_rejected(self) -> None:
        for change in ('size', 'sha', 'path', 'kind'):
            request = copy.deepcopy(self.request)
            if change == 'size': request['dataset_archive']['uncompressed_size_bytes'] -= 1
            elif change == 'sha': request['dataset_sha256'] = '0'*64
            elif change == 'path': request['dataset_path'] = '/other/payload_trace.jsonl'
            else: request['workload']['kind'] = 'sharegpt_natural'
            with self.subTest(change=change), self.assertRaises(ValueError):
                decode_dataset_asset({'request': request}, self.assets)

    def test_missing_archive_or_ambiguous_raw_target_is_rejected(self) -> None:
        for assets in ([], self.assets+[(PurePosixPath('input/payload_trace.jsonl'), self.raw)]):
            with self.assertRaises(ValueError): decode_dataset_asset({'request': self.request}, assets)

    def test_xz_recovers_identical_payload_with_explicit_codec(self) -> None:
        request=copy.deepcopy(self.request)
        request['dataset_archive'].update(codec='xz',relative_path='input/payload_trace.jsonl.xz')
        assets=[(PurePosixPath('input/payload_trace.jsonl.xz'),lzma.compress(self.raw,preset=6))]
        self.assertEqual(decode_dataset_asset({'request':request},assets),self.raw)

    def test_xz_truncation_trailing_stream_and_output_overflow_rejected(self) -> None:
        request=copy.deepcopy(self.request)
        request['dataset_archive'].update(codec='xz',relative_path='input/payload_trace.jsonl.xz')
        encoded=lzma.compress(self.raw,preset=6)
        for body in (encoded[:-8],encoded+b'trailing',encoded+encoded):
            with self.subTest(size=len(body)),self.assertRaises((ValueError,lzma.LZMAError)):
                decode_dataset_asset({'request':request},[(PurePosixPath('input/payload_trace.jsonl.xz'),body)])
        request['dataset_archive']['uncompressed_size_bytes']-=1
        with self.assertRaises(ValueError):
            decode_dataset_asset({'request':request},[(PurePosixPath('input/payload_trace.jsonl.xz'),encoded)])

    def test_unknown_codec_rejected_before_any_execution(self) -> None:
        request=copy.deepcopy(self.request);request['dataset_archive']['codec']='unknown'
        with self.assertRaises(ValueError):decode_dataset_asset({'request':request},self.assets)

    def test_advanced_bf16_trace_keeps_the_same_bounded_decoder_contract(self) -> None:
        request = copy.deepcopy(self.request)
        request['workload']['kind'] = 'anonymous_trace_xqa_bf16_reference'
        self.assertEqual(decode_dataset_asset({'request': request}, self.assets), self.raw)
        request['dataset_archive']['uncompressed_size_bytes'] += 1
        with self.assertRaises(ValueError): decode_dataset_asset({'request': request}, self.assets)

    def test_component_diagnostic_uses_exact_bounded_decoder_and_rejects_corruption(self) -> None:
        request = copy.deepcopy(self.request)
        request['workload']['kind'] = 'anonymous_trace_component_diagnostic'
        request['dataset_archive'].update(codec='xz', relative_path='input/payload_trace.jsonl.xz')
        assets = [(PurePosixPath('input/payload_trace.jsonl.xz'), lzma.compress(self.raw))]
        self.assertEqual(decode_dataset_asset({'request': request}, assets), self.raw)
        request['dataset_sha256'] = '0' * 64
        with self.assertRaises(ValueError): decode_dataset_asset({'request': request}, assets)


if __name__ == '__main__':
    unittest.main()
