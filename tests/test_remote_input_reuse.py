"""Remote input reuse saves transfer, never fabricates output or weakens hashes."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.controller.cli import _local_package
from scripts.controller.durable import canonical_bytes,digest
from scripts.controller.remote_rpc import derived_quality_input,reused_input_asset


class RemoteInputReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.parent=self.base/'agent_work/runs/parent'
        (self.parent/'input').mkdir(parents=True);self.data=b'original frozen request bytes'
        self.name='input/requests.json';(self.parent/self.name).write_bytes(self.data)
        self.hash=hashlib.sha256(self.data).hexdigest();prior={'run_id':'parent','model_id':'model'}
        self.old={'request':prior,'request_sha256':digest(prior),'reservation_id':'nonce','assets_sha256':{self.name:self.hash}}
        (self.parent/'ticket.json').write_bytes(canonical_bytes(self.old))
        self.result={'run_id':'parent','request_sha256':digest(prior),'reservation_id':'nonce','status':'COMPLETE',
                     'gpu_release_verified':True,'live_owned_containers':[]}
        self.ticket={'approval':{'approved_workspace':str(self.base)},'request':{'run_id':'new','model_id':'model'},
                     'assets_sha256':{self.name:self.hash}}
        self.freeze()

    def freeze(self):
        raw=canonical_bytes(self.result);(self.parent/'result.json').write_bytes(raw)
        sha=hashlib.sha256(raw).hexdigest();(self.parent/'COMPLETE').write_bytes(canonical_bytes({'result_sha256':sha}))
        self.reference={'sha256':self.hash,'reuse_from_run_id':'parent','reuse_parent_result_sha256':sha}

    def test_completed_owned_input_is_exact_and_parent_is_not_modified(self):
        before=(self.parent/self.name).stat()
        self.assertEqual(reused_input_asset(self.ticket,self.name,self.reference),self.data)
        after=(self.parent/self.name).stat();self.assertEqual((before.st_size,before.st_mtime_ns),(after.st_size,after.st_mtime_ns))

    def test_any_output_or_escaping_asset_is_rejected(self):
        for name in ('output/measurement.json','input/../output/result','/input/file'):
            with self.subTest(name=name),self.assertRaises(ValueError):reused_input_asset(self.ticket,name,self.reference)

    def test_changed_input_or_terminal_hash_is_rejected(self):
        (self.parent/self.name).write_bytes(b'changed')
        with self.assertRaises(ValueError):reused_input_asset(self.ticket,self.name,self.reference)
        (self.parent/self.name).write_bytes(self.data);self.reference['reuse_parent_result_sha256']='0'*64
        with self.assertRaises(ValueError):reused_input_asset(self.ticket,self.name,self.reference)

    def test_incomplete_unreleased_or_live_parent_cannot_supply_reused_inputs(self):
        original=dict(self.result)
        for name,value in [('status','FAILED'),('gpu_release_verified',False),('live_owned_containers',['live'])]:
            self.result=original|{name:value};self.freeze()
            with self.subTest(name=name),self.assertRaises(ValueError):reused_input_asset(self.ticket,self.name,self.reference)

    def test_self_parent_or_symlink_or_other_model_is_rejected(self):
        self.reference['reuse_from_run_id']='new'
        with self.assertRaises(ValueError):reused_input_asset(self.ticket,self.name,self.reference)
        self.reference['reuse_from_run_id']='parent';self.ticket['request']['model_id']='other'
        with self.assertRaises(ValueError):reused_input_asset(self.ticket,self.name,self.reference)
        self.ticket['request']['model_id']='model';path=self.parent/self.name;path.unlink()
        target=self.base/'other';target.write_bytes(self.data);path.symlink_to(target)
        with self.assertRaises(ValueError):reused_input_asset(self.ticket,self.name,self.reference)

    def test_extra_encoded_body_cannot_override_a_reuse_reference(self):
        with self.assertRaises(ValueError):reused_input_asset(self.ticket,self.name,self.reference|{'base64':'ignored'})

    def test_local_package_preserves_sha_and_replaces_only_input_wire_bytes(self):
        input_file=self.base/'requests.json';input_file.write_bytes(self.data)
        package={'schema':'kv-frozen-local-package-v1','classification':'proposal','request':{},
            'assets':{self.name:{'local_path':str(input_file),'sha256':self.hash}},'remote_asset_reuse':{self.name:self.reference}}
        path=self.base/'package.json';path.write_text(json.dumps(package))
        with patch('scripts.controller.cli.ROOT',self.base):
            _,hashes,wire=_local_package(path)
            self.assertEqual(hashes[self.name],self.hash);self.assertNotIn('base64',wire[self.name])
            self.assertEqual(wire[self.name],self.reference)

    def test_disjoint_missing_input_derives_exact_source_rows_never_outputs(self):
        name='input/quality_requests.jsonl';rows=[{'id':'first','request':{'text':'original'}},
                                                {'id':'second','request':{'text':'also original'}}]
        full=b''.join(canonical_bytes(row) for row in rows);(self.parent/name).write_bytes(full)
        source_sha=hashlib.sha256(full).hexdigest();self.old['assets_sha256'][name]=source_sha
        (self.parent/'ticket.json').write_bytes(canonical_bytes(self.old))
        selected=canonical_bytes(rows[1]);target_sha=hashlib.sha256(selected).hexdigest()
        self.ticket['assets_sha256'][name]=target_sha;self.ticket['request'].update(
            quality_completion={'declared':True},quality_evaluation={'requests':1})
        ref={'sha256':target_sha,'derive_requests_from_run_id':'parent','source_requests_sha256':source_sha,
             'source_parent_result_sha256':self.reference['reuse_parent_result_sha256'],'selected_request_ids':['second']}
        self.assertEqual(derived_quality_input(self.ticket,name,ref),selected)
        with self.assertRaises(ValueError):derived_quality_input(self.ticket,'output/fake_predictions.jsonl',ref)
        with self.assertRaises(ValueError):derived_quality_input(self.ticket,name,ref|{'selected_request_ids':['missing']})


if __name__=='__main__':unittest.main()
