"""P1 source identity and CPU-cache admission use real files without GPU imports."""

import copy
import hashlib
import json
from pathlib import Path
import unittest

from scripts.controller.native_jit_reuse import (
    P1_MODULE, P1_CACHE_RELATIVE, artifact_identity, install, source_manifest_identity, validate,
    JIT_PYTHON_SOURCES,
)
from tests import test_native_jit_reuse as legacy_fixtures


MANIFEST=json.loads((Path(__file__).parent/'fixtures/kv_p1_mxfp4_graph_manifest.json').read_bytes())


class P1JITContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture=legacy_fixtures.JITReuseTests();self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def cpu_request(self):
        f=self.fixture
        source='a'*40
        module,relative,manifest_sha=source_manifest_identity(MANIFEST)
        build_id='upstream-ac-'+source[:8]+'-20261109a'
        prior=f.root/'agent_work/builds'/build_id;prior.mkdir(parents=True)
        library=prior/'cache'/relative;library.parent.mkdir(parents=True);library.write_bytes(f.raw)
        header=prior/'cache'/Path(relative).parents[2]/'generated'/module/'mxfp4_scheduler.cuh'
        header.parent.mkdir(parents=True);header.write_bytes(b'exact generated dependency fixture')
        tree=[{'relative':str(p.relative_to(prior/'cache')),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
               'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (library,header)]
        build={'schema':'MXFP4_SM90a_CPU_JIT_build_v2','success':True,'CPU_compile_and_load':True,
               'GPU_job':False,'CUDA_initialized':False,'FLASHINFER_CUDA_ARCH_LIST':'9.0a',
               'source_commit':source,'library_sha256':hashlib.sha256(f.raw).hexdigest(),'library_bytes':len(f.raw),
               'module':module,'source_manifest_sha256':manifest_sha}
        files={'build_result.json':build,'CPU_ONLY_ADMISSION.json':{'GPU_device_nodes':[],'source_commit':source},
               'manifest.json':{'source_commit':source,'image_id':'image','native_jit_source_manifest':MANIFEST}}
        for name,data in files.items():(prior/name).write_text(json.dumps(data))
        cpu={'schema':'MXFP4_SM90a_CPU_JIT_seed_v2','build_id':build_id,
             'build_result_sha256':hashlib.sha256((prior/'build_result.json').read_bytes()).hexdigest(),
             'CPU_admission_sha256':hashlib.sha256((prior/'CPU_ONLY_ADMISSION.json').read_bytes()).hexdigest(),
             'build_manifest_sha256':hashlib.sha256((prior/'manifest.json').read_bytes()).hexdigest(),
             'source_commit':source,'image_id':'image','sha256':hashlib.sha256(f.raw).hexdigest(),'bytes':len(f.raw),
             'module':module,'cache_relative':relative,'source_manifest_sha256':manifest_sha,'cache_tree':tree}
        request={'approved_workspace':str(f.root),'python_source_commit':source,'image_id':'image','arms':['mxfp4'],
                 'p1_extension':{'schema':'kv_p0_p1_deployment_v1','profile':'kv_p0_p1_20261109'},
                 'workload':{'kind':'upstream_KV_P0_P1_operator'},'native_jit_source_manifest':MANIFEST,
                 'cpu_native_jit_artifact':cpu}
        return request,prior,header

    def test_complete_pinned_manifest_has_a_fresh_derived_namespace(self):
        module,relative,sha=source_manifest_identity(MANIFEST)
        self.assertEqual(module,P1_MODULE)
        self.assertEqual(relative,P1_CACHE_RELATIVE)
        self.assertEqual(sha,'0828b8e7cadf4d0d3b5d4a82f172d3dd4553828466e3c61f17acbf20a09f7a8a')

    def test_omitted_scheduler_binding_template_or_changed_parameters_rejected(self):
        for name in ('include/flashinfer/attention/scheduler.cuh','csrc/batch_prefill_jit_binding.cu',
                     'csrc/batch_prefill_customize_config.jinja'):
            changed=copy.deepcopy(MANIFEST);del changed['inputs'][name]
            with self.subTest(name=name),self.assertRaises(ValueError):source_manifest_identity(changed)
        changed=copy.deepcopy(MANIFEST);changed['parameters']['head_dim_qk']=128
        with self.assertRaises(ValueError):source_manifest_identity(changed)

    def test_p1_cannot_fall_back_to_legacy_namespace(self):
        request,_,_=self.cpu_request();del request['native_jit_source_manifest']
        with self.assertRaises(ValueError):artifact_identity(request)
        request['native_jit_source_manifest']=MANIFEST
        request['cpu_native_jit_artifact']['module']='xllm_mxfp4_bf16_d256_b06c014c4f939393'
        with self.assertRaises(ValueError):validate(request)

    def test_exact_v2_CPU_cache_copied_with_mtime_and_no_GPU_claim(self):
        request,_,_=self.cpu_request();f=self.fixture
        install(request,f.output,f.cache)
        for item in request['cpu_native_jit_artifact']['cache_tree']:
            path=f.cache/item['relative']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item['sha256'])
            self.assertEqual(path.stat().st_mtime_ns,item['mtime_ns'])
        receipt=json.loads((f.output/'native_JIT_reuse.json').read_bytes())
        self.assertEqual(receipt['schema'],'MXFP4_JIT_reuse_v2')
        self.assertTrue(receipt['CPU_compile_only_before_GPU_qualification'])
        self.assertTrue(receipt['ninja_noop_not_yet_proven'])

    def test_changed_generated_dependency_is_rejected_without_overwrite(self):
        request,_,header=self.cpu_request();header.write_bytes(b'changed')
        with self.assertRaises(ValueError):install(request,self.fixture.output,self.fixture.cache)

    def test_original_build_manifest_is_hash_bound(self):
        request,prior,_=self.cpu_request();(prior/'manifest.json').write_text('{}')
        with self.assertRaises(ValueError):install(request,self.fixture.output,self.fixture.cache)

    def test_seed_requires_full_tree_new_date_and_matching_source_id(self):
        request,_,_=self.cpu_request()
        for change in ({'cache_tree':[]},{'build_id':'upstream-ac-aaaaaaaa-20261105a'},
                       {'source_commit':'b'*40},{'source_manifest_sha256':'0'*64}):
            changed=copy.deepcopy(request);changed['cpu_native_jit_artifact'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):validate(changed)

    def test_CPU_seed_cannot_qualify_a_model(self):
        request,_,_=self.cpu_request();request['workload']={'kind':'upstream_next_stage_model_qualification'}
        with self.assertRaises(ValueError):validate(request)

    def qualified_request(self):
        import shutil
        from tests.test_kv_p1_operator import P1AcceptanceTest
        request,build,_=self.cpu_request();f=self.fixture
        seed=request.pop('cpu_native_jit_artifact')
        shutil.copytree(build/'cache',f.prior/'cache/mxfp4',dirs_exist_ok=True)
        artifact={k:seed[k] for k in ('module','cache_relative','sha256','bytes',
                  'source_commit','image_id','source_manifest_sha256','cache_tree')}
        artifact.update(schema='qualified_MXFP4_JIT_reuse_v2',
                        qualification_reference=f.request['operator_qualification_reference'])
        request.update(native_jit_artifact=artifact,next_stage_profile='kv_p0_p1_20261109',
                       backend_source_sha256={name:'7'*64 for name in JIT_PYTHON_SOURCES},
                       probe_driver_sha256='2'*64)
        (f.prior/'ticket.json').write_text(json.dumps({'request':request}))
        acceptance=P1AcceptanceTest().acceptance(mx=True)
        acceptance.update(phase='pilot',runtime_source_commit=request['python_source_commit'])
        (f.prior/'output/mxfp4/operators/acceptance.json').write_text(json.dumps(acceptance))
        request['python_source_commit']='b'*40
        return request

    def test_qualified_same_native_program_reuses_cache_across_outer_python_commit(self):
        request=self.qualified_request();f=self.fixture
        install(request,f.output,f.cache)
        receipt=json.loads((f.output/'native_JIT_reuse.json').read_bytes())
        self.assertTrue(receipt['cross_commit_cache_only'])
        self.assertEqual(receipt['cache_source_commit'],'a'*40)
        self.assertEqual(receipt['target_python_source_commit'],'b'*40)
        self.assertEqual((f.cache/P1_CACHE_RELATIVE).read_bytes(),f.raw)
        self.assertNotIn('model_qualified',receipt)

    def test_cross_commit_changed_adapter_or_generator_never_copies_cache(self):
        request=self.qualified_request();f=self.fixture
        for name in JIT_PYTHON_SOURCES:
            bad=copy.deepcopy(request);bad['backend_source_sha256'][name]='8'*64
            with self.subTest(name=name),self.assertRaises(ValueError):install(bad,f.output,f.cache)
        self.assertFalse((f.cache/P1_CACHE_RELATIVE).exists())

    def test_cross_commit_wrong_image_or_unreleased_origin_is_rejected(self):
        request=self.qualified_request();f=self.fixture
        bad=copy.deepcopy(request);bad['image_id']='different'
        with self.assertRaises(ValueError):validate(bad)
        result=json.loads(f.result.read_bytes());result['gpu_release_verified']=False
        f.result.write_text(json.dumps(result))
        request['native_jit_artifact']['qualification_reference']['result_sha256']=hashlib.sha256(f.result.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):install(request,f.output,f.cache)


if __name__=='__main__':unittest.main()
