"""Count the actual native schema rendering while preserving the wire request."""

import copy
from types import SimpleNamespace
import unittest

from scripts.controller.official_tokenization import OfficialJSONTokenizer,_native_tools_for_render


class OfficialNativeTokenizationTests(unittest.TestCase):
    def test_only_schema_object_maps_are_sorted_in_a_render_copy(self):
        tools=[{'type':'function','function':{'name':'move','parameters':{
            'type':'object','required':['source','destination'],'properties':{
                'source':{'type':'string','description':'src'},
                'destination':{'type':'string','description':'dst'}}}}}]
        original=copy.deepcopy(tools)
        actual=_native_tools_for_render(tools)
        self.assertEqual(tools,original)
        schema=actual[0]['function']['parameters']
        self.assertEqual(list(schema),['properties','required','type'])
        self.assertEqual(list(schema['properties']),['destination','source'])
        self.assertEqual(schema['required'],['source','destination'])
        self.assertEqual(list(schema['properties']['source']),['description','type'])

    def test_protobuf_drops_extra_tool_fields_and_encodes_numbers_as_double(self):
        tools=[{'type':'function','extra':'ignored','function':{'name':'tail','response':{'type':'string'},
            'parameters':{'default':10,'enum':[1,False,None,2.5],'type':'integer'}}}]
        before=copy.deepcopy(tools);actual=_native_tools_for_render(tools)
        self.assertEqual(tools,before)
        self.assertEqual(list(actual[0]),['type','function'])
        self.assertEqual(list(actual[0]['function']),['name','description','parameters'])
        self.assertEqual(actual[0]['function']['description'],'')
        schema=actual[0]['function']['parameters']
        self.assertIs(type(schema['default']),float)
        self.assertIs(type(schema['enum'][0]),float)
        self.assertIs(schema['enum'][1],False)
        self.assertIsNone(schema['enum'][2])
        self.assertNotIn('response',actual[0]['function'])

    def test_wire_tool_arguments_stay_strings_and_message_order_is_preserved(self):
        tokenizer=object.__new__(OfficialJSONTokenizer);captured={}
        def render(messages,**options):captured.update(messages=messages,options=options);return 'rendered'
        tokenizer.renderer=SimpleNamespace(apply_chat_template=render)
        tokenizer.encoder=SimpleNamespace(encode=lambda *args,**kwargs:SimpleNamespace(ids=[1,2,3]))
        request={'messages':[{'role':'assistant','tool_calls':[{'function':{
            'name':'move','arguments':'{"source":"a","destination":"b"}'}}]},
            {'role':'tool','content':'ok'}],
            'tools':[{'function':{'name':'move','parameters':{'type':'object','properties':{'z':{},'a':{}}}}}]}
        before=copy.deepcopy(request);self.assertEqual(tokenizer.count(request),3)
        self.assertEqual(request,before)
        self.assertIsInstance(request['messages'][0]['tool_calls'][0]['function']['arguments'],str)
        self.assertIsInstance(captured['messages'][0]['tool_calls'][0]['function']['arguments'],dict)
        self.assertEqual([m['role'] for m in captured['messages']],['assistant','tool'])


if __name__=='__main__':unittest.main()
