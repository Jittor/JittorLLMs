import jittor as jt
jt.flags.use_cuda = 1

from models import LLMModel



class ChatRWKVMdoel(LLMModel):
    def __init__(self, args) -> None:
        super().__init__()

        ckpt_dir = getattr(args, "ckpt_dir", "data/llama/7B")
        tokenizer_path = getattr(args, "tokenizer_path", "data/llama/tokenizer.model")

        self.generator = load(
            ckpt_dir, tokenizer_path, max_seq_len=512, max_batch_size=32
        )
        jt.gc()

    def run(self, input_text: str) -> str:
        output = self.generator.generate([input_text], max_gen_len=256, temperature=0.8, top_p=0.95)
        return output


def get_model(args):
    return ChatRWKVMdoel(args)
