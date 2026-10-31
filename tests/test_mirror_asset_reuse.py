"""Cache reuse must retain real output transfers and exact fresh remote hashes."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.controller.durable import write_new_json
from scripts.reuse_verified_mirror_assets import assemble, destination, sha


class MirrorAssetReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory();self.root = Path(self.tmp.name)
        self.network = self.root/'network';self.network.mkdir()
        prefix = 'support/agent_work/runs/fixture/'
        blobs = {'result.json':b'{"status":"COMPLETE"}\n',
                 'ticket.json':b'{"request_sha256":"request"}\n',
                 'output/bf16/artifacts/profile_export_raw.jsonl':b'{"original_raw":1}\n'}
        network = []
        for name, data in blobs.items():
            path = self.network/(prefix+name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            network.append({'source':'/approved/'+prefix.removeprefix('support/')+name,
                            'relative_path':prefix+name,'size':len(data),'sha256':sha(path)})
        self.raw = self.network/(prefix+'output/bf16/artifacts/profile_export_raw.jsonl')
        self.cache = self.root/'input.json';self.cache.write_bytes(b'{"frozen_input":1}\n')
        cached = {'source':'/approved/agent_work/runs/fixture/input/input.json',
            'relative_path':prefix+'input/input.json','size':self.cache.stat().st_size,'sha256':sha(self.cache)}
        proof = {'run_id':'fixture','terminal_sha256':network[0]['sha256'],
                 'request_sha256':'request','observed_utc':'fixture','files':[cached]}
        self.proof = self.root/'proof.json';write_new_json(self.proof,proof)
        write_new_json(self.network/'SNAPSHOT.json',{'base':'/approved','uid':1005,'files':network})
        self.plan = self.root/'plan.json'
        write_new_json(self.plan,{'run_id':'fixture','base':'/approved','request_sha256':'request',
            'terminal_sha256':network[0]['sha256'],'remote_proof':str(self.proof),
            'remote_proof_sha256':sha(self.proof),
            'original_selection':{'files':[{'base_rel':r['relative_path'].removeprefix('support/'),
                                          'size':r['size']} for r in network+[cached]]},
            'cached':[{'base_rel':cached['relative_path'].removeprefix('support/'),
                       'local_path':str(self.cache),'sha256':cached['sha256']}]})
        self.out = self.root/'assembled/files'

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_all_real_outputs_transfer_and_input_cache_reuse_declared(self) -> None:
        assemble(self.plan,self.network,self.out)
        result = json.loads((self.out/'SNAPSHOT.json').read_text())
        self.assertEqual(len(result['files']),4)
        self.assertEqual(result['cached_asset_reuse']['raw_files_reused'],0)
        self.assertEqual(sha(self.out/'support/agent_work/runs/fixture/output/bf16/artifacts/profile_export_raw.jsonl'),sha(self.raw))

    def test_modified_cached_bytes_cannot_pass_remote_attestation(self) -> None:
        self.cache.write_bytes(b'{"frozen_input":2}\n')
        with self.assertRaisesRegex(ValueError,'remote SHA'):
            assemble(self.plan,self.network,self.out)
        self.assertFalse((self.out/'SNAPSHOT.json').exists())

    def test_independent_network_parts_join_without_omitting_raw(self) -> None:
        first = json.loads((self.network/'SNAPSHOT.json').read_text())
        raw = next(r for r in first['files'] if r['relative_path'].endswith('profile_export_raw.jsonl'))
        second = self.root/'network2';second.mkdir()
        path = second/raw['relative_path'];path.parent.mkdir(parents=True);path.write_bytes(self.raw.read_bytes())
        write_new_json(second/'SNAPSHOT.json',{**first,'files':[raw]})
        (self.network/'SNAPSHOT.json').write_text(json.dumps({**first,'files':[r for r in first['files'] if r!=raw]}))
        assemble(self.plan,[self.network,second],self.out)
        result = json.loads((self.out/'SNAPSHOT.json').read_text())
        self.assertEqual(len(result['files']),4)
        self.assertEqual(len(result['cached_asset_reuse']['network_snapshot_sha256']),2)

    def test_modified_network_raw_cannot_be_hidden_by_cache(self) -> None:
        self.raw.write_bytes(b'{"original_raw":2}\n')
        with self.assertRaisesRegex(ValueError,'remote SHA'):
            assemble(self.plan,self.network,self.out)
        self.assertFalse((self.out/'SNAPSHOT.json').exists())

    def test_traversal_cannot_escape_new_assembly(self) -> None:
        with self.assertRaises(ValueError):
            destination(self.out,'../../outside')


if __name__ == '__main__':
    unittest.main()
