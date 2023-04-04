import os, platform
from transformers import AutoTokenizer, AutoModel
from models import LLMModel
import jittor as jt
import torch

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

def build_prompt(history):
    prompt = ''
    for query, response in history:
        prompt += f"\n用户输入：{query}"
        prompt += f"\nChatGLM-6B：{response}"
    return prompt

class ChatGLMMdoel(LLMModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(os.path.dirname(__file__), trust_remote_code=True)
        if jt.has_cuda:
            self.model.half().cuda()
        else:
            self.model.float32()
            torch.half = torch.float
            torch.Tensor.half = torch.Tensor.float
        self.model.eval()

    def chat(self) -> str:
        global stop_stream
        history = []
        while True:
            text = input("用户输入:")
            for response, history in self.model.stream_chat(self.tokenizer, text, history=history):
                print(response, end='\r')
            print(flush=True)
    
    def run_web_demo(self, input_text, history=[]):
        while True:
            yield self.run(input_text, history=history)

    def run(self, text, history=[]):
        return self.model.chat(self.tokenizer, text, history=history)

def get_model(args):
    return ChatGLMMdoel(args)