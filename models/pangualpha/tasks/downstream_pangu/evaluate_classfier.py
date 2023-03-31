# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import sys
import numpy as np
import torch
import time
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.text_generation_utils import pad_batch, get_batch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive

# from megatron.model.transformer import LayerNorm
from megatron import mpu


from load_iflytek import iflytek_dataset
def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=False)

    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    return parser

def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    # Set up model and load checkpoint.
    model = get_model(model_provider)
    model.eval()

    args = get_args()
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    task = 'few_shot'
    config = [
        ('task', [task]), #'zero_shot','one_shot','few_shot'
        ('max_len', [50]), #None,200,100
        ('tag_new_example', [True]), #True, False
        ('few_shot_num_sample', [3]), #2,3,4
        ('np_seed', [233]), #233,235,237,239
        ('new_mask', [False]), #True, False
        ('input_str_format', [
            # "{label}：{sentence}",
            "这是关于{label}的应用程序：{sentence}",
        ])
    ]
    para_config_list = [{y0[0]:y1 for y0,y1 in zip(config,x)} for x in itertools.product(*[x[1] for x in config])]
    para_config = para_config_list[0]

    dataset = iflytek_dataset()
    samples = dataset.getSamples(para_config)

    acc = []
    time_recorder = time.time()
    for ind,sample in enumerate(samples):
        loss_ = []
        for class_sample,loss_mask in zip(sample['input_ids_list'],sample['loss_mask_list']):
            context_tokens_tensor = torch.cuda.LongTensor([class_sample])
            labels = context_tokens_tensor[:, 1:].contiguous().cuda()
            tokens = context_tokens_tensor[:, :-1].contiguous()
            tokens, attention_mask, position_ids = get_batch(tokens)
            type_ids = None
            with torch.no_grad():
                logits = model(tokens,
                               position_ids,
                               attention_mask,
                               tokentype_ids=type_ids,
                               forward_method_parallel_output=False)
            losses = mpu.vocab_parallel_cross_entropy(
                logits.contiguous().float(), labels.contiguous())
            loss_mask = torch.cuda.LongTensor(loss_mask)
            loss = torch.sum(
                losses.view(-1) * loss_mask.contiguous().view(-1).float())
            loss_.append(loss)

        predict = np.argmin(loss_)
        acc.append(predict==sample['label'])
        if (ind%100 == 0):
            time_cost = time.time() - time_recorder
            print(f'[{ind}] iflytek-{task}: acc={sum(acc)}/{len(acc)}={(sum(acc)/len(acc)):.5f} time={time_cost:.2f}')

    accuracy = sum(acc)/len(acc)
    print(f'{para_config} iflytek-{task}: acc={sum(acc)}/{len(acc)}={(sum(acc)/len(acc)):.5f} time={time_cost:.2f}')
    pass


if __name__ == "__main__":

    main()
