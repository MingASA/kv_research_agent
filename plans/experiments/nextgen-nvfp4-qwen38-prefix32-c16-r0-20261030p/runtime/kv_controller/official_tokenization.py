"""Official JSON BPE counting and Jinja rendering for actual API histories."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path


NATIVE_TOOLS_RENDER_PROTOCOL='protobuf_JsonTool_parameters_double_map_v2'


def _native_tools_for_render(tools: list) -> list:
    """Reproduce protobuf Struct and the server's explicit JsonTool envelope."""
    def proto_value(value):
        if isinstance(value,dict):return {key:proto_value(value[key]) for key in sorted(value)}
        if isinstance(value,list):return [proto_value(item) for item in value]
        if value is None or isinstance(value,(str,bool)):return value
        if isinstance(value,(int,float)):
            number=float(value)
            if not math.isfinite(number):raise ValueError('Nonfinite protobuf schema number')
            return number
        raise ValueError('Tool schema is not protobuf JSON data')
    rendered=[]
    for tool in tools:
        function=tool['function']
        # Unknown function metadata, such as the BFCL response schema, is not
        # a proto::Function field and never reaches the production template.
        rendered.append({'type':tool.get('type',''),'function':{
            'name':function.get('name',''),'description':function.get('description',''),
            'parameters':proto_value(function.get('parameters',{}))}})
    return rendered


class OfficialJSONTokenizer:
    def __init__(self,directory: Path) -> None:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
        self.renderer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        self.encoder=Tokenizer.from_file(str(directory/'tokenizer.json'))

    def count(self,request: dict) -> int:
        if 'messages' in request:
            options=dict(request.get('chat_template_kwargs',{'enable_thinking':False}))
            if request.get('tools') is not None:options['tools']=_native_tools_for_render(request['tools'])
            # OpenAI wire arguments remain strings; only the render copy is
            # normalized to the mapping required by the official Qwen template.
            messages=copy.deepcopy(request['messages'])
            for message in messages:
                for call in message.get('tool_calls',[]) or []:
                    function=call.get('function',call);arguments=function.get('arguments')
                    if isinstance(arguments,str):
                        value=json.loads(arguments)
                        if not isinstance(value,dict):raise ValueError('Tool history arguments must encode a JSON object')
                        function['arguments']=value
            text=self.renderer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**options)
        else:text=request['prompt']
        return len(self.encoder.encode(text,add_special_tokens=True).ids)
