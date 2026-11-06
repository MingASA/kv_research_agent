"""Actual file-copy and predecessor contracts, without loading GPU code."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.controller.native_jit_reuse import CACHE_RELATIVE,CACHE_BASE,MODULE,install,validate


class JITReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.prior=self.root/'agent_work/runs/qualified';self.prior.mkdir(parents=True)
        self.source=self.prior/'cache/mxfp4'/CACHE_RELATIVE;self.source.parent.mkdir(parents=True)
        self.raw=b'qualified binary fixture; never loaded';self.source.write_bytes(self.raw)
        self.result=self.prior/'result.json'
        self.result.write_text(json.dumps({'status':'COMPLETE','gpu_release_verified':True,'live_owned_containers':[]}))
        reference={'run_id':'qualified','result_sha256':hashlib.sha256(self.result.read_bytes()).hexdigest()}
        self.request={'approved_workspace':str(self.root),'python_source_commit':'a'*40,'image_id':'image',
            'arms':['mxfp4'],'next_stage_profile':'ocp_mxfp4_20261105','operator_qualification_reference':reference}
        artifact={'schema':'qualified_MXFP4_JIT_reuse_v1','module':MODULE,'cache_relative':CACHE_RELATIVE,
            'sha256':hashlib.sha256(self.raw).hexdigest(),'bytes':len(self.raw),'source_commit':'a'*40,
            'image_id':'image','qualification_reference':reference}
        self.request['native_jit_artifact']=artifact
        (self.prior/'ticket.json').write_text(json.dumps({'request':self.request}))
        output=self.prior/'output/mxfp4/operators';output.mkdir(parents=True)
        (output/'acceptance.json').write_text(json.dumps({'schema':'MXFP4_SM90_operator_gate_v1',
            'writer_CPU_bitwise':True,'whole_page_copy':True,'results':{'prefill':{'pass':True},
                'decode':{'pass':True},'Graph':{'changed_inputs':True,'bitwise_equal_direct':True}}}))
        self.cache=self.root/'new-cache';self.cache.mkdir();self.output=self.root/'new-output';self.output.mkdir()

    def tearDown(self):self.tmp.cleanup()

    def test_qualified_predecessor_is_copied_without_modification(self):
        install(self.request,self.output,self.cache)
        self.assertEqual((self.cache/CACHE_RELATIVE).read_bytes(),self.raw)
        self.assertEqual(self.source.read_bytes(),self.raw)
        self.assertTrue(json.loads((self.output/'native_JIT_reuse.json').read_bytes())['source_and_destination_SHA_verified'])

    def test_changed_source_is_rejected_before_destination_copy(self):
        self.source.write_bytes(b'X'*len(self.raw))
        with self.assertRaisesRegex(ValueError,'SHA'):install(self.request,self.output,self.cache)
        self.assertFalse((self.cache/CACHE_RELATIVE).exists())

    def test_unknown_destination_is_preserved(self):
        target=self.cache/CACHE_RELATIVE;target.parent.mkdir(parents=True);target.write_bytes(b'other')
        with self.assertRaisesRegex(ValueError,'overwrite'):install(self.request,self.output,self.cache)
        self.assertEqual(target.read_bytes(),b'other')

    def test_unqualified_or_unreleased_predecessor_is_not_used(self):
        self.result.write_text(json.dumps({'status':'FAILED','gpu_release_verified':False,'live_owned_containers':['live']}))
        with self.assertRaisesRegex(ValueError,'completed and released'):install(self.request,self.output,self.cache)
        self.assertFalse((self.cache/CACHE_RELATIVE).exists())

    def test_wrong_scope_source_image_path_and_unbounded_bytes_rejected(self):
        for update in ({'source_commit':'b'*40},{'image_id':'other'},{'cache_relative':'../outside'},
                       {'bytes':128*2**20+1},{'qualification_reference':{'run_id':'other'}}):
            request=dict(self.request,native_jit_artifact=dict(self.request['native_jit_artifact'],**update))
            with self.assertRaises(ValueError):validate(request)
        with self.assertRaises(ValueError):validate(dict(self.request,arms=['fp8_e4m3']))

    def cpu_request(self):
        build_id='upstream-ac-ba5eaec0-20261105e'
        prior=self.root/'agent_work/builds'/build_id;prior.mkdir(parents=True)
        source=prior/'cache'/CACHE_RELATIVE;source.parent.mkdir(parents=True);source.write_bytes(self.raw)
        result={'schema':'MXFP4_SM90a_CPU_JIT_build_v1','success':True,'CPU_compile_and_load':True,
            'GPU_job':False,'CUDA_initialized':False,'FLASHINFER_CUDA_ARCH_LIST':'9.0a',
            'source_commit':'a'*40,'library_sha256':hashlib.sha256(self.raw).hexdigest(),'library_bytes':len(self.raw)}
        paths={'build_result.json':result,'CPU_ONLY_ADMISSION.json':{'GPU_device_nodes':[],'source_commit':'a'*40},
               'manifest.json':{'image_id':'image'}}
        for name,data in paths.items():(prior/name).write_text(json.dumps(data))
        cpu={'schema':'MXFP4_SM90a_CPU_JIT_seed_v1','build_id':build_id,
            'build_result_sha256':hashlib.sha256((prior/'build_result.json').read_bytes()).hexdigest(),
            'CPU_admission_sha256':hashlib.sha256((prior/'CPU_ONLY_ADMISSION.json').read_bytes()).hexdigest(),
            'source_commit':'a'*40,'image_id':'image','sha256':hashlib.sha256(self.raw).hexdigest(),'bytes':len(self.raw)}
        request={k:v for k,v in self.request.items() if k not in ('native_jit_artifact','next_stage_profile','operator_qualification_reference')}
        request.update(cpu_native_jit_artifact=cpu,workload={'kind':'upstream_next_stage_native_operator'})
        return request,prior

    def test_cpu_compiled_seed_is_labeled_prerequisite_without_GPU_claim(self):
        request,_=self.cpu_request();install(request,self.output,self.cache)
        self.assertEqual((self.cache/CACHE_RELATIVE).read_bytes(),self.raw)
        proof=json.loads((self.output/'native_JIT_reuse.json').read_bytes())
        self.assertTrue(proof['CPU_compile_only_before_GPU_qualification'])

    def test_cpu_seed_cannot_replace_model_GPU_qualification(self):
        request,_=self.cpu_request();request['workload']={'kind':'upstream_next_stage_model_qualification'}
        with self.assertRaises(ValueError):validate(request)

    def test_cpu_seed_changed_build_evidence_rejected_without_copy(self):
        request,prior=self.cpu_request();(prior/'build_result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'proof SHA'):install(request,self.output,self.cache)
        self.assertFalse((self.cache/CACHE_RELATIVE).exists())

    def tree_request(self):
        header=self.prior/'cache/mxfp4'/CACHE_BASE/'generated'/MODULE/'mxfp4_prefill.cuh'
        header.parent.mkdir(parents=True);header.write_bytes(b'qualified private header')
        files=[]
        for path in (self.source,header):
            info=path.stat();files.append({'relative':str(path.relative_to(self.prior/'cache/mxfp4')),
                'size':info.st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mtime_ns':info.st_mtime_ns})
        request=dict(self.request,native_jit_artifact=dict(self.request['native_jit_artifact'],cache_tree=files))
        return request,header

    def test_full_tree_copy_preserves_verified_dependency_mtime(self):
        request,header=self.tree_request();install(request,self.output,self.cache)
        for item in request['native_jit_artifact']['cache_tree']:
            target=self.cache/item['relative']
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(),item['sha256'])
            self.assertEqual(target.stat().st_mtime_ns,item['mtime_ns'])
        proof=json.loads((self.output/'native_JIT_reuse.json').read_bytes())
        self.assertEqual(proof['copied_cache_tree_files'],2)
        self.assertTrue(proof['ninja_noop_not_yet_proven'])

    def test_changed_original_dependency_is_not_silently_rebuilt(self):
        request,header=self.tree_request();header.write_bytes(b'different private header')
        with self.assertRaises(ValueError):install(request,self.output,self.cache)

    def test_cache_tree_cannot_escape_the_two_qualified_module_directories(self):
        request,_=self.tree_request()
        request['native_jit_artifact']['cache_tree'][0]['relative']='../outside'
        with self.assertRaises(ValueError):validate(request)


if __name__=='__main__':unittest.main()
