"""Untrusted calls stay data and cannot reach undeclared methods or other sessions."""

import copy
import hashlib
import json
from types import SimpleNamespace
import unittest

from scripts.controller.nextgen_tools_env import ToolEnvironment


class FixtureMemoryAPI:
    def _load_scenario(self,scenario,long_context=False):self.values=copy.deepcopy(scenario)
    def echo(self,text):return {'echo':text}
    def put(self,key,value):self.values[key]=value;return {'stored':True}


class ToolDataDispatchTests(unittest.TestCase):
    def setUp(self):
        original={'id':'CPU_only','involved_classes':['GorillaFileSystem'],
                  'initial_config':{'GorillaFileSystem':{'original':'unchanged'}}}
        raw=json.dumps(original)
        self.task={**copy.deepcopy(original),'original_source_record_JSON':raw,
                   'source_line_sha256_without_LF':hashlib.sha256(raw.encode()).hexdigest(),
                   'function_classes':{'echo':'GorillaFileSystem','put':'GorillaFileSystem'},'tools':[]}
        for name,properties in (('echo',{'text':{'type':'string'}}),
                                ('put',{'key':{'type':'string'},'value':{'type':'integer'}})):
            self.task['tools'].append({'type':'function','function':{'name':name,'parameters':
                {'type':'object','properties':properties,'required':list(properties)}}})
        self.factory=SimpleNamespace(classes={'GorillaFileSystem':FixtureMemoryAPI})

    def test_code_shaped_string_is_literal_tool_data(self):
        env=ToolEnvironment(self.factory,self.task);text="__import__('os').system('false')"
        row=env.call('echo',{'text':text});self.assertTrue(row['dispatch_valid'])
        self.assertEqual(json.loads(row['result']),{'echo':text})
        for expression in ("echo(text=__import__('os'))","os.system('false')","echo(**{'text':'x'})"):
            with self.subTest(expression=expression),self.assertRaises((ValueError,TypeError,SyntaxError)):
                env.reference_call(expression)

    def test_state_mutation_isolated_and_literal_reference_uses_same_method(self):
        a,b=ToolEnvironment(self.factory,self.task),ToolEnvironment(self.factory,self.task)
        self.assertTrue(a.reference_call("put(key='probe',value=7)")['dispatch_valid'])
        self.assertEqual(a.instances['GorillaFileSystem'].values['probe'],7)
        self.assertNotIn('probe',b.instances['GorillaFileSystem'].values)
        self.assertNotIn('probe',self.task['initial_config']['GorillaFileSystem'])

    def test_unknown_methods_missing_keys_and_wrong_types_are_feedback_without_execution(self):
        env=ToolEnvironment(self.factory,self.task)
        for name,arguments in (('__class__',{}),('getattr',{}),('put',{'key':'x'}),
                               ('put',{'key':'x','value':True}),('put',{'key':'x','value':'7'})):
            with self.subTest(name=name,arguments=arguments):self.assertFalse(env.call(name,arguments)['dispatch_valid'])
        self.assertEqual(env.instances['GorillaFileSystem'].values,{'original':'unchanged'})

    def test_source_identity_or_private_schema_mutation_prevents_environment_creation(self):
        for mutation in ('source_hash','private_name'):
            task=copy.deepcopy(self.task)
            if mutation=='source_hash':task['source_line_sha256_without_LF']='0'*64
            else:task['tools'][0]['function']['name']='_load_scenario'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):ToolEnvironment(self.factory,task)


if __name__=='__main__':unittest.main()
