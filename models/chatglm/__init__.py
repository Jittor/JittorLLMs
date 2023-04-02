import os, platform
from transformers import AutoTokenizer, AutoModel
from models import LLMModel

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
        self.model = AutoModel.from_pretrained(os.path.dirname(__file__), trust_remote_code=True).half().cuda()
        self.model.eval()
        #self.model = self.model.eval()

    def chat(self) -> str:
        global stop_stream
        history = []
        while True:
            text = input("用户输入:")
            for response, history in self.model.stream_chat(self.tokenizer, text, history=history):
                os.system(clear_command)
                print(build_prompt(history), flush=True)
            os.system(clear_command)
            print(build_prompt(history), flush=True)
    
    def run_web_demo(self, input_text, history=[]):
        while True:
            yield self.run(input_text, history=history)

def get_model(args):
    return ChatGLMMdoel(args)