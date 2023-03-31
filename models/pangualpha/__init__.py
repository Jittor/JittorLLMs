import os, sys
import numpy as np
import torch
import jittor as jt

from megatron.text_generation_utils import pad_batch, get_batch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model
from megatron.training import get_model as megatron_get_model

from models import LLMModel


def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=False)

    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=5,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                       'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')
    return parser


def generate(model, context_tokens, args, tokenizer, max_num=50, begin=0):

    valid_length = len(context_tokens)
    context_tokens_, context_lengths = pad_batch([context_tokens],
                                                 tokenizer.pad_id, args)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens_)
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
    type_ids = None
    bs,_  = tokens.shape
    cnt = 0
    while valid_length < args.seq_length:
        with torch.no_grad():
            logits = model(tokens,
                           position_ids,
                           attention_mask,
                           tokentype_ids=type_ids,
                           forward_method_parallel_output=False)
        logits = logits[:,:,:tokenizer.vocab_size].cpu()
        logits = logits.numpy()
        logits = logits.reshape(bs, args.seq_length, -1)
        probs = logits[0, valid_length-1, :]
        p_args = probs.argsort()[::-1][:args.top_k]

        p = probs[p_args]
        p = p / sum(p)
        for i in range(1000):
            target_index = np.random.choice(len(p), p=p)
            if p_args[target_index] != tokenizer.unk:
                break

        if p_args[target_index] == tokenizer.eod or \
                valid_length == args.seq_length-1 or cnt>=max_num:
            outputs = tokens.cpu().numpy()
            return

        if (begin > 0) and (valid_length >= begin):
            #print(p_args[target_index])
            character = tokenizer.convert_ids_to_tokens([int(p_args[target_index])])
            if character == '问' or character == '答':
                return
            print(character, end='')
            sys.stdout.flush()

        tokens[0][valid_length] = p_args[target_index]
        valid_length += 1
        cnt += 1

    length = np.sum(outputs != tokenizer.pad_id)
    outputs = outputs[0][:length]
    return outputs


def main():
    """Main program."""


    print('loading checkpoint ...')

    samples = ['上联：瑞风播福泽，事业具昌盛千家乐',
               '四川的省会是?',
               '上联：春雨润人间，社会和谐万象新',
               '''书生：羌笛何须怨杨柳，春风不度玉门关。
                    飞云：（这诗怎么这么耳熟？且过去跟他聊聊如何。）
                    书生：小兄弟，要不要一起喝一杯？
                    飞云：你请我呀？你若是请我，我便和你喝一杯；你若不请我，我便一个人去喝。
                    书生：小兄弟，看你年纪轻轻，不至于这么势利吧？
                    飞云：''',
               '张无忌拿出屠龙宝刀，手起刀落，周芷若掉了一颗门牙，身旁的赵敏喜极而泣，',
               '人工智能成为国际竞争的新焦点。人工智能是引领未来的战略性技术，世界主要发达国家把发展人工智能作为提升国家竞争力、维护国家安全的重大战略，加紧出台规划和政策，围绕核心技术、顶尖人才、标准规范等强化部署，力图在新一轮国际科技竞争中掌握主导权。当前，',
               '中国和美国和日本和法国和加拿大和澳大利亚的首都分别是哪里？']
    for sample in samples:
        raw_text = sample
        tokenizer = get_tokenizer()
        context_tokens = tokenizer.tokenize(raw_text)

        import time
        start = time.time()
        output_ids = generate(model, context_tokens, args, tokenizer)
        end = time.time()
        output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())

        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('per token costs:', (end-start)/(len(output_ids)-len(tokenizer.tokenize(raw_text))))
        print('Input is:', sample)
        print('Output is:', output_samples[len(sample):], flush=True)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


class PanGuAlphaModel(LLMModel):
    def __init__(self) -> None:
        super().__init__()

        import sys
        sys.argv = [sys.argv[0], 
                    "--model-parallel-size", "1", 
                    "--num-layers", "31", 
                    "--hidden-size" , "2560", 
                    "--load" , f"{os.path.join(jt.compiler.ck_path, 'pangu', 'Pangu-alpha_2.6B_fp16_mgt')}", 
                    "--num-attention-heads", "32", 
                    "--max-position-embeddings", "1024", 
                    "--tokenizer-type", "GPT2BPETokenizer", 
                    # "--fp16", 
                    "--batch-size", "1", 
                    "--seq-length", "1024", 
                    "--out-seq-length", "50",
                    "--temperature", "1.0",
                    "--vocab-file", "models/pangualpha/megatron/tokenizer/bpe_4w_pcl/vocab", 
                    "--num-samples", "0",
                    "--top_k", "2",
                    "--finetune"]

        initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

        # Set up model and load checkpoint.
        self.model = megatron_get_model(model_provider)
        self.model.eval()

        self.args = get_args()
        if self.args.load is not None:
            _ = load_checkpoint(self.model, None, None)

    def run(self, input_text: str) -> str:
        tokenizer = get_tokenizer()
        input_text = "问：" + input_text + "？答："
        context_tokens = tokenizer.tokenize(input_text)
        output_ids = generate(self.model, context_tokens, self.args, tokenizer, 100, len(input_text))
        #output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
        #return output_samples[len(input_text):]
        return ""


def get_model(args):
    return PanGuAlphaModel()
