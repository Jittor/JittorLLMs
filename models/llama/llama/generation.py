# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
from typing import List

import jittor as jt

from llama.tokenizer import Tokenizer
from llama.model import Transformer

spell = '''A dialog, where User interacts with AI. AI is helpful, kind, obedient, honest, and knows its own limits. AI replies the User in one line.
User: Hello, AI.
AI: Hello! How can I assist you today?
User: '''


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        prompts = [spell + prompt + '\n' for prompt in prompts]

        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = jt.full((bsz, total_len), self.tokenizer.pad_id).int32()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = jt.array(t).int32()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = jt.nn.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = jt.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = jt.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos
            jt.sync_all()

            if cur_pos <= start_pos:
                continue
            for i, t in enumerate(tokens.tolist()):
                # cut to max gen len
                t = t[start_pos: cur_pos]

                if cur_pos >= len(prompt_tokens[i]) + max_gen_len:
                    return
                # cut to eos tok if any
                try:
                    t = t[start_pos: t.index(self.tokenizer.eos_id)]
                except ValueError:
                    pass
                new_decode = self.tokenizer.decode(t)
                if new_decode.startswith("AI: "):
                    new_decode = new_decode[4:]
                if '\n' in new_decode:
                    return new_decode
                yield new_decode


def sample_top_p(probs, p):
    probs_idx, probs_sort = jt.argsort(probs, dim=-1, descending=True)
    probs_sum = jt.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdims=True)
    next_token = jt.multinomial(probs_sort, num_samples=1)
    next_token = jt.gather(probs_idx, -1, next_token)
    return next_token
