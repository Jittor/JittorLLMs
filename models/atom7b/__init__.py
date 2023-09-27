import os, platform
from transformers import AutoTokenizer, AutoModel
from models import LLMModel
import jittor as jt
import torch

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False

# jtorch patch for transformers==4.33.0
import sys
sys.modules["torch.utils._pytree"] = torch
torch.utils._pytree = torch
torch._register_pytree_node = lambda *args, **kw: None
torch._dict_flatten = lambda *args, **kw: None
torch.get_default_dtype = lambda : "float32"
torch.Module.named_buffers = lambda *args, **kw: []
torch.Var.storage = lambda x: x
torch.Var.data_ptr = lambda x: x.id
class FakeStorage:
    def __init__(self, var):
        self.var = var
    def nbytes(self):
        return self.nbytes
torch.Var.untyped_storage = lambda x: FakeStorage(x)
# jtorch patch end

def build_prompt(history):
    prompt = ''
    for query, response in history:
        prompt += f"\n用户输入：{query}"
        prompt += f"\nAtom-7B：{response}"
    return prompt

class Atom7BMdoel(LLMModel):
    def __init__(self, args) -> None:
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__), trust_remote_code=True)
        # self.model = AutoModel.from_pretrained(os.path.dirname(__file__), trust_remote_code=True)
        from . import modeling_llama
        # hook modeling_llama
        sys.modules["transformers.models.llama.modeling_llama"] = modeling_llama
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        assert transformers.__version__ == "4.33.0", f'transformers.__version__{transformers.__version__} != "4.33.0"'
        # self.model = AutoModelForCausalLM.from_pretrained("/mnt/nfs0/cjld/atom-7b")
        # self.tokenizer = AutoTokenizer.from_pretrained("/mnt/nfs0/cjld/atom-7b")
        self.model = AutoModelForCausalLM.from_pretrained("FlagAlpha/Atom-7B")
        self.tokenizer = AutoTokenizer.from_pretrained("FlagAlpha/Atom-7B")
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
    return Atom7BMdoel(args)