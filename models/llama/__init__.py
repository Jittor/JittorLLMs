import time
import json
from pathlib import Path

import jittor as jt
jt.flags.use_cuda = 1

from models import LLMModel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = str(checkpoints[0])
    print("Loading")
    checkpoint = jt.load(ckpt_path)
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    model = Transformer(model_args)
    model.load_state_dict(checkpoint)
    model.half()

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


class LLaMAModel(LLMModel):
    def __init__(self, args) -> None:
        super().__init__()

        ckpt_dir = getattr(args, "ckpt_dir", "data/llama/7BJ")
        tokenizer_path = getattr(args, "tokenizer_path", "data/llama/tokenizer.model")

        self.generator = load(
            ckpt_dir, tokenizer_path, max_seq_len=512, max_batch_size=32
        )
        jt.gc()

    def run(self, input_text: str) -> str:
        output = self.generator.generate([input_text], max_gen_len=256, temperature=0.8, top_p=0.95)
        return output


def get_model(args):
    return LLaMAModel(args)
