"""Actual own CPU-process fault injection for blocked/partial tool IPC."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.controller.nextgen_tool_child import ToolChild


STUB='''
import json,sys,time
first=json.loads(sys.stdin.buffer.readline())
ready={'sequence':1,'task_id':first['task']['id'],'status':'READY',
       'source_manifest_sha256':sys.argv[1],'CPU_only_imports':True,
       'CUDA_visibility_hidden':True,'CUDA_runtime_initialized':False,
       'address_space_limit_bytes':512*2**20,'CPU_seconds_limit':30}
print(json.dumps(ready),flush=True)
if sys.argv[2]=='partial':
    sys.stdin.buffer.readline()
    sys.stdout.write('{"sequence":2');sys.stdout.flush()
time.sleep(10)
'''


class BoundedToolChildTests(unittest.TestCase):
    def test_partial_response_timeout_reaps_only_original_CPU_child(self):
        self.run_fault('partial',{'text':'small'},'exceeded2s')

    def test_nonreading_child_cannot_block_a_large_command_write_forever(self):
        self.run_fault('blocked',{'text':'x'*(300*1024)},'write deadline')

    def run_fault(self,mode,arguments,reason):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'source_manifest.json').write_text('{}')
            sha=hashlib.sha256(b'{}').hexdigest();original_Popen=subprocess.Popen;processes=[]
            def start(command,**kwargs):
                self.assertIn('scripts.controller.nextgen_tool_child',command)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
                process=original_Popen([sys.executable,'-B','-c',STUB,sha,mode],**kwargs)
                processes.append(process);return process
            with patch('scripts.controller.nextgen_tool_child.subprocess.Popen',side_effect=start):
                child=ToolChild(root,{'id':'CPU_fault_fixture'},root/'stderr.log')
                started=time.monotonic()
                try:
                    with self.assertRaisesRegex(TimeoutError,reason):child.call('echo',arguments)
                    self.assertLess(time.monotonic()-started,3)
                    self.assertIsNotNone(child.process.returncode)
                    with self.assertRaises((ValueError,BrokenPipeError,OSError)):child.call('echo',{'text':'never retry'})
                    self.assertEqual(len(processes),1)
                finally:child.close()


if __name__=='__main__':unittest.main()
