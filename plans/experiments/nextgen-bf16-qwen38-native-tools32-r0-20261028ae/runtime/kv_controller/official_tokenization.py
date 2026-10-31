"""Official JSON BPE counting and Jinja rendering for actual API histories."""

from __future__ import annotations

import copy
import json
from pathlib import Path


class OfficialJSONTokenizer:
    def __init__(self,directory: Path) -> None:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
        self.renderer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        self.encoder=Tokenizer.from_file(str(directory/'tokenizer.json'))

    def count(self,request: dict) -> int:
        if 'messages' in request:
            options=dict(request.get('chat_template_kwargs',{'enable_thinking':False}))
            if request.get('tools') is not None:options['tools']=request['tools']
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
