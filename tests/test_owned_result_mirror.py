"""Terminal output mirroring keeps partial evidence and rejects corruption."""

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.controller.owned_result_mirror import SOURCE,mirror_terminal


class Transport:
    def __init__(self,files):
        self.files=files;self.corrupt=False;self.calls=[]
    def rpc(self,source,payload,**kwargs):
        assert kwargs['mutation'] is False
        self.calls.append(payload['operation'])
        if payload['operation']=='inventory':
            return {'ok':True,'remote_writes':False,'run_id':'owned-run','terminal':'COMPLETE',
                'result_sha256':'result-sha','files':[{'relative':name,'size':len(raw),'inode':1,'mtime_ns':2,
                'sha256':hashlib.sha256(raw).hexdigest()} for name,raw in self.files.items()]}
        raw=self.files[payload['file']['relative']][payload['offset']:payload['offset']+2**20]
        return {'ok':True,'remote_writes':False,'offset':payload['offset'],'bytes':len(raw),
                'base64':base64.b64encode(raw).decode(),
                'sha256':'wrong' if self.corrupt else hashlib.sha256(raw).hexdigest()}


class MirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'mirror'
        self.ticket={'request':{'run_id':'owned-run'}}
    def tearDown(self):self.temp.cleanup()

    def test_chunked_empty_and_existing_files_verify_without_overwrite(self):
        transport=Transport({'output/large.log':b'x'*(2**20+5),'empty.log':b''})
        manifest=mirror_terminal(transport,self.ticket,'rpc',self.root)
        self.assertTrue(manifest.is_file());self.assertEqual(transport.calls.count('chunk'),2)
        before=manifest.read_bytes();transport.calls=[]
        self.assertEqual(mirror_terminal(transport,self.ticket,'rpc',self.root).read_bytes(),before)
        self.assertEqual(transport.calls,[])

    def test_corrupt_chunk_never_publishes_manifest(self):
        transport=Transport({'out.log':b'hello'});transport.corrupt=True
        with self.assertRaisesRegex(ValueError,'Corrupt'):mirror_terminal(transport,self.ticket,'rpc',self.root)
        self.assertFalse((self.root/'SHA_MANIFEST.json').exists())
        self.assertEqual((self.root/'files/out.log.partial').read_bytes(),b'')
        transport.corrupt=False
        self.assertTrue(mirror_terminal(transport,self.ticket,'rpc',self.root).exists())

    def test_existing_corruption_is_preserved_and_rejected(self):
        transport=Transport({'out.log':b'hello'})
        mirror_terminal(transport,self.ticket,'rpc',self.root)
        (self.root/'files/out.log').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'never overwrite'):mirror_terminal(transport,self.ticket,'rpc',self.root)
        self.assertEqual((self.root/'files/out.log').read_bytes(),b'wrong')

    def test_path_traversal_or_symlink_is_rejected(self):
        transport=Transport({'../outside':b'bad'})
        with self.assertRaises(ValueError):mirror_terminal(transport,self.ticket,'rpc',self.root)
        self.assertFalse((self.root/'outside').exists())
        other=Path(self.temp.name)/'other';other.mkdir()
        root2=Path(self.temp.name)/'mirror2';root2.mkdir(mode=0o700)
        (root2/'files').symlink_to(other,target_is_directory=True)
        with self.assertRaises(ValueError):mirror_terminal(Transport({'out.log':b'x'}),self.ticket,'rpc',root2)
        self.assertEqual(list(other.iterdir()),[])

    def remote_inventory(self,result,worker_output=False):
        owned=Path(self.temp.name)/'owned';owned.mkdir(exist_ok=True)
        (owned/'result.json').write_text(json.dumps(result))
        if worker_output:
            (owned/'output').mkdir(exist_ok=True);(owned/'output/worker.log').write_text('unexpected')
        library=(
            "from pathlib import Path\n"
            "def require(value,message):\n"
            " if not value:raise ValueError(message)\n"
            "def check_ticket(ticket,mutation):return Path(ticket['request']['remote_output'])\n"
            "def existing(root,ticket):pass\n"
            "def regular(path):return path.read_bytes()\n")
        return subprocess.run([sys.executable,'-B','-c',SOURCE],input=json.dumps({'operation':'inventory',
            'rpc_source':library,'ticket':{'request':{'run_id':'owned-run','remote_output':str(owned)}}}),
            capture_output=True,text=True,timeout=10)

    def zero_result(self):
        return {'status':'NOT_LAUNCHED','gpu_release_verified':True,'live_owned_containers':[],
            'error':'TimeoutError: Budget expired while waiting for 120s genuinely idle GPUs',
            'allocation_started_monotonic_ns':None,'allocation_ended_monotonic_ns':None,
            'allocation_intervals_ns':[],'arm_results':[]}

    def test_real_readonly_program_can_inventory_completed_zero_GPU_wait(self):
        result=self.remote_inventory(self.zero_result())
        self.assertEqual(result.returncode,0,result.stderr);body=json.loads(result.stdout)
        self.assertEqual(body['terminal'],'NOT_LAUNCHED');self.assertFalse(body['remote_writes'])
        self.assertEqual([row['relative'] for row in body['files']],['result.json'])

    def test_mislabelled_allocation_or_worker_output_is_not_a_zero_wait_mirror(self):
        for changes in ({'allocation_started_monotonic_ns':1},{'error':'another failure'},
                        {'allocation_intervals_ns':[{}]},{'arm_results':[{}]}):
            result=self.remote_inventory(self.zero_result()|changes)
            with self.subTest(changes=changes):
                self.assertNotEqual(result.returncode,0);self.assertIn('zero-GPU idle-wait result',result.stderr)
        result=self.remote_inventory(self.zero_result(),worker_output=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('unexpected worker output',result.stderr)


if __name__=='__main__':unittest.main()
