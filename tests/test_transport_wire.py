# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Real local subprocess verifies the exact bounded stdin envelope; no SSH/GPU."""

import gzip
import hashlib
import json
import shlex
import subprocess
import unittest
from unittest.mock import patch

from scripts.controller.approval import ApprovalError
from scripts.controller.durable import canonical_bytes
from scripts.controller.transport import wire_request


class WireEnvelopeTests(unittest.TestCase):
    def test_large_payload_and_future_import_roundtrip_unchanged(self) -> None:
        source = ('from __future__ import annotations\nimport hashlib,json,sys\n'
                  'p=json.load(sys.stdin)\n'
                  'print(json.dumps({"ok":True,"sha256":hashlib.sha256(p["body"].encode()).hexdigest()}))\n')
        payload = {'body':'数据withquotes"and$()and`literal`'*30000}
        command,wire=wire_request(source,payload)
        self.assertEqual(gzip.decompress(wire),canonical_bytes(payload))
        self.assertIn('Allowlisted stdin RPC, executed by pinned SSH',command)
        result=subprocess.run(shlex.split(command),input=wire,capture_output=True,check=True)
        self.assertEqual(json.loads(result.stdout)['sha256'],hashlib.sha256(payload['body'].encode()).hexdigest())

    def test_small_readonly_payload_keeps_plain_protocol(self) -> None:
        command,wire=wire_request('pass',{'operation':'status'})
        self.assertEqual(shlex.split(command),['python3','-B','-c','pass'])
        self.assertEqual(wire,canonical_bytes({'operation':'status'}))

    def test_oversized_json_rejected_before_subprocess(self) -> None:
        with patch('scripts.controller.transport.MAX_RPC_JSON_BYTES',100):
            with self.assertRaises(ApprovalError):wire_request('pass',{'body':'x'*101})

    def test_remote_decompression_limit_precedes_original_source(self) -> None:
        source='raise RuntimeError("ORIGINAL_SOURCE_EXECUTED")'
        command,_=wire_request(source,{'body':'x'*300000})
        wrapper=shlex.split(command)[3].replace('100663297','101').replace('100663296','100')
        result=subprocess.run(['python3','-B','-c',wrapper],input=gzip.compress(b'x'*101),capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'Oversized RPC JSON',result.stderr)
        self.assertNotIn(b'RuntimeError: ORIGINAL_SOURCE_EXECUTED',result.stderr)

    def test_truncated_transport_never_executes_original_program(self) -> None:
        command,wire=wire_request('print("ROOT_MUTATION")',{'body':'x'*300000})
        result=subprocess.run(shlex.split(command),input=wire[:-8],capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(result.stdout,b'')
        self.assertIn(b'EOFError',result.stderr)


if __name__ == '__main__':
    unittest.main()
