import os, copy, types, gc, sys
import numpy as np
from prompt_toolkit import prompt

from models import LLMModel
import jittor as jt
import os
jt.flags.use_cuda = 1


from src.model_run import RWKV_RNN
from src.utils import TOKENIZER


class ChatRWKVModel(LLMModel):
    def __init__(self, model_path, tokenizer_path) -> None:
        super().__init__()

        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        args = types.SimpleNamespace()

        args.RUN_DEVICE = "cuda"  # cuda // cpu
        os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE

        args.FLOAT_MODE = "fp32"
        CHAT_LANG = 'Chinese' # English // Chinese // more to come
        QA_PROMPT = False # True: Q & A prompt // False: User & Bot prompt
        args.MODEL_PATH = model_path

        args.ctx_len = 1024
        self.CHAT_LEN_SHORT = 40
        self.CHAT_LEN_LONG = 150
        self.FREE_GEN_LEN = 200
        self.GEN_TEMP = 1.0
        self.GEN_TOP_P = 0.85

        AVOID_REPEAT = '，。：？！'
        # print(f'\nLoading ChatRWKV - {CHAT_LANG} - {args.RUN_DEVICE} - {args.FLOAT_MODE} - QA_PROMPT {QA_PROMPT}')


        self.tokenizer = TOKENIZER(tokenizer_path)

        args.vocab_size = 50277
        args.head_qk = 0
        args.pre_ffn = 0
        args.grad_cp = 0
        args.my_pos_emb = 0
        self.args = args
        MODEL_PATH = args.MODEL_PATH


        if CHAT_LANG == 'English':
            self.interface = interface = ":"

            if QA_PROMPT:
                self.user = user = "User"
                self.bot = bot = "Bot" # Or: 'The following is a verbose and detailed Q & A conversation of factual information.'
                init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
            else:
                self.user = user = "Bob"
                self.bot = bot = "Alice"
                init_prompt = f'''
The following is a verbose detailed conversation between {user} and a young girl {bot}. {bot} is intelligent, friendly and cute. {bot} is unlikely to disagree with {user}.

{user}{interface} Hello {bot}, how are you doing?

{bot}{interface} Hi {user}! Thanks, I'm fine. What about you?

{user}{interface} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{bot}{interface} Not at all! I'm listening.

'''
        elif CHAT_LANG == 'Chinese':
            self.interface = interface = ":"
            if QA_PROMPT:
                self.user = user = "Q"
                self.bot = bot = "A"
                init_prompt = f'''
Expert Questions & Helpful Answers

Ask Research Experts

'''
            else:
                self.user = user = "User"
                self.bot = bot = "Bot"
                init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

{user}{interface} 企鹅会飞吗

{bot}{interface} 企鹅是不会飞的。它们的翅膀主要用于游泳和平衡，而不是飞行。

'''

        print(f'Loading model - {MODEL_PATH}')
        self.model = RWKV_RNN(args)

        self.model_tokens = []
        self.model_state = None

        self.AVOID_REPEAT_TOKENS = []
        for i in AVOID_REPEAT:
            dd = self.tokenizer.encode(i)
            assert len(dd) == 1
            self.AVOID_REPEAT_TOKENS += dd

        self.all_state = {}

        out = self.run_rnn(self.tokenizer.encode(init_prompt))
        self.save_all_stat('', 'chat_init', out)
        gc.collect()
        jt.gc()

        srv_list = ['dummy_server']
        for s in srv_list:
            self.save_all_stat(s, 'chat', out)

        # print(f'{self.tokenizer.decode(self.model_tokens)}'.replace(f'\n\n{bot}',f'\n{bot}'), end='')
        self.history = []

    def run_rnn(self, tokens, newline_adj = 0):
        tokens = [int(x) for x in tokens]
        self.model_tokens += tokens
        out, self.model_state = self.model.forward(tokens, self.model_state)

        # print(f'### model ###\n{tokens}\n[{tokenizer.decode(model_tokens)}]')

        out[0] = -999999999  # disable <|endoftext|>
        out[187] += newline_adj # adjust \n probability
        # if newline_adj > 0:
        #     out[15] += newline_adj / 2 # '.'
        if self.model_tokens[-1] in self.AVOID_REPEAT_TOKENS:
            out[self.model_tokens[-1]] = -999999999
        return out

    def save_all_stat(self, srv, name, last_out):
        n = f'{name}_{srv}'
        self.all_state[n] = {}
        self.all_state[n]['out'] = last_out
        self.all_state[n]['rnn'] = copy.deepcopy(self.model_state)
        self.all_state[n]['token'] = copy.deepcopy(self.model_tokens)

    def load_all_stat(self, srv, name):
        n = f'{name}_{srv}'
        self.model_state = copy.deepcopy(self.all_state[n]['rnn'])
        self.model_tokens = copy.deepcopy(self.all_state[n]['token'])
        return self.all_state[n]['out']


    def run(self, message: str, is_web=False) -> str:
        srv = 'dummy_server'

        msg = message.replace('\\n','\n').strip()

        x_temp = self.GEN_TEMP
        x_top_p = self.GEN_TOP_P
        if ("-temp=" in msg):
            x_temp = float(msg.split("-temp=")[1].split(" ")[0])
            msg = msg.replace("-temp="+f'{x_temp:g}', "")
            # print(f"temp: {x_temp}")
        if ("-top_p=" in msg):
            x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
            msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
            # print(f"top_p: {x_top_p}")
        if x_temp <= 0.2:
            x_temp = 0.2
        if x_temp >= 5:
            x_temp = 5
        if x_top_p <= 0:
            x_top_p = 0
        
        if msg == '+reset':
            out = self.load_all_stat('', 'chat_init')
            self.save_all_stat(srv, 'chat', out)
            print("Chat reset.\n")

        elif msg[:5].lower() == '+gen ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

            if msg[:5].lower() == '+gen ':
                new = '\n' + msg[5:].strip()
                # print(f'### prompt ###\n[{new}]')
                self.model_state = None
                self.model_tokens = []
                out = self.run_rnn(self.tokenizer.encode(new))
                self.save_all_stat(srv, 'gen_0', out)

            elif msg[:4].lower() == '+qq ':
                new = '\nQ: ' + msg[4:].strip() + '\nA:'
                # print(f'### prompt ###\n[{new}]')
                self.model_state = None
                self.model_tokens = []
                out = self.run_rnn(self.tokenizer.encode(new))
                self.save_all_stat(srv, 'gen_0', out)

            elif msg[:4].lower() == '+qa ':
                out = self.load_all_stat('', 'chat_init')

                real_msg = msg[4:].strip()
                new = f"{self.user}{self.interface} {real_msg}\n\n{self.bot}{self.interface}"
                # print(f'### qa ###\n[{new}]')
                
                out = self.run_rnn(self.tokenizer.encode(new))
                self.save_all_stat(srv, 'gen_0', out)

            elif msg.lower() == '+++':
                try:
                    out = self.load_all_stat(srv, 'gen_1')
                    self.save_all_stat(srv, 'gen_0', out)
                except:
                    return

            elif msg.lower() == '++':
                try:
                    out = self.load_all_stat(srv, 'gen_0')
                except:
                    return

            begin = len(self.model_tokens)
            out_last = begin

            for i in range(self.FREE_GEN_LEN+100):
                token = self.tokenizer.sample_logits(
                    out,
                    self.model_tokens,
                    self.args.ctx_len,
                    temperature=x_temp,
                    top_p=x_top_p,
                )
                if msg[:4].lower() == '+qa ':# or msg[:4].lower() == '+qq ':
                    out = self.run_rnn([token], newline_adj=-2)
                else:
                    out = self.run_rnn([token])
                
                xxx = self.tokenizer.decode(self.model_tokens[out_last:])
                if '\ufffd' not in xxx: # avoid utf-8 display issues
                    print(xxx, end='', flush=True)
                    out_last = begin + i + 1
                    if i >= self.FREE_GEN_LEN:
                        break
            print("\n")
            # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
            # print(f'### send ###\n[{send_msg}]')
            # reply_msg(send_msg)
            self.save_all_stat(srv, 'gen_1', out)
        else:
            if msg.lower() == '+':
                try:
                    out = self.load_all_stat(srv, 'chat_pre')
                except:
                    return
            else:
                out = self.load_all_stat(srv, 'chat')
                new = f"{self.user}{self.interface} {msg}\n\n{self.bot}{self.interface}"
                # print(f'### add ###\n[{new}]')
                out = self.run_rnn(self.tokenizer.encode(new), newline_adj=-999999999)
                self.save_all_stat(srv, 'chat_pre', out)

            begin = len(self.model_tokens)
            out_last = begin
            text_out = ""
            for i in range(999):
                if i <= 0:
                    newline_adj = -999999999
                elif i <= self.CHAT_LEN_SHORT:
                    newline_adj = (i - self.CHAT_LEN_SHORT) / 10
                elif i <= self.CHAT_LEN_LONG:
                    newline_adj = 0
                else:
                    newline_adj = (i - self.CHAT_LEN_LONG) * 0.25 # MUST END THE GENERATION
                token = self.tokenizer.sample_logits(
                    out,
                    self.model_tokens,
                    self.args.ctx_len,
                    temperature=x_temp,
                    top_p=x_top_p,
                )
                out = self.run_rnn([token], newline_adj=newline_adj)

                xxx = self.tokenizer.decode(self.model_tokens[out_last:])
                if '\ufffd' not in xxx: # avoid utf-8 display issues
                    text_out += xxx.replace("\n", "")
                    if not is_web:
                        print(xxx, end='', flush=True)
                    out_last = begin + i + 1
                send_msg = self.tokenizer.decode(self.model_tokens[begin:])
                if '\n\n' in send_msg:
                    send_msg = send_msg.strip()
                    break

            self.save_all_stat(srv, 'chat', out)
        return text_out

    def chat(self):
        while True:
            text = input("用户输入:")
            print("ChatRWKV: ", end="")
            self.run(text)

    def run_web_demo(self, input_text, history=[]):
        self.history = []
        response = self.run(input_text, is_web=True)
        history.append([input_text, response])
        yield response, history

def get_model(args):
    return ChatRWKVModel(os.path.join(jt.compiler.ck_path, "ChatRWKV", "RWKV-4-Pile-3B-EngChn-test4-20230115-fp32.pth"),
                         "models/chatrwkv/20B_tokenizer.json")
