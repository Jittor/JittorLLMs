########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types, math, os, gc
import jittor as jt

MyModule = jt.Module
def __nop(ob):
    return ob
MyFunction = __nop

RWKV_RESCALE_LAYER = 6 # set x = x/2 every X layer (to avoid FP16 overflow)

############################################################################################################

class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        if args.FLOAT_MODE == 'fp32':
            self.FLOAT_MODE = jt.float
        elif args.FLOAT_MODE == 'fp16':
            self.FLOAT_MODE = jt.float16
        self.RUN_DEVICE = args.RUN_DEVICE

        with jt.no_grad():
            w = jt.load(args.MODEL_PATH)
            print("Loaded")
            args.n_embd = w['emb.weight'].shape[1]
            args.n_layer = 0
            keys = list(w.keys()) # refine weights and send to correct device
            print_need_newline = False
            for x in keys:
                w[x].requires_grad = False
                if x == 'emb.weight' or 'ln0' in x:
                    continue
                
                block_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                args.n_layer = max(args.n_layer, block_id+1)
                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x:
                    w[x] = w[x].t()
                
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -jt.exp(w[x])
                else:
                    w[x] = w[x].float()

                if args.FLOAT_MODE == 'fp16':
                    if 'att.output.weight' in x:
                        w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))
                    if 'ffn.value.weight' in x:
                        w[x] = w[x] / (2 ** int(block_id // RWKV_RESCALE_LAYER))

                shape = w[x].shape
                shape = [i for i in shape if i != 1]
                if len(shape) > 1:
                    shape = f"  {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}"
                else:
                    shape = f"  {str(shape[0]).rjust(5)}      "
                if block_id == 0:
                    if print_need_newline:
                        print('\n', end = '')
                        print_need_newline = False
                    # print(x.ljust(32), str(w[x].dtype).ljust(10), shape)
                else:
                    print_need_newline = True
                    # print('.', end = '', flush = True)
        # print(f'\nn_layer {args.n_layer} n_embd {args.n_embd} ctx_len {args.ctx_len}')

        keys = list(w.keys()) # store weights in self.w
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        with jt.no_grad(): # precompute embedding
            try:
                x = self.LN(self.w.emb.weight, self.w.blocks[0].ln0)
            except:
                x = jt.nn.layer_norm(self.w.emb.weight.float(), (self.args.n_embd,), weight=self.w.blocks[0].ln0.weight.float(), bias=self.w.blocks[0].ln0.bias.float())
            self.w.emb.weight = x

        self.eval()
        gc.collect()
        jt.gc()

    def LN(self, x, w):
        return jt.nn.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    # state[] 0=ffn_xx 1=att_xx 2=att_aa 3=att_bb 4=att_pp

    @MyFunction
    def FF_one(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = state[5*i+0]
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x.float()

        r = jt.sigmoid(xr @ rw)
        k = jt.nn.relu(xk @ kw) ** 2
        kv = k @ vw
        return r * kv

    @MyFunction
    def FF_seq(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        xx = jt.cat((state[5*i+0].unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+0] = x[-1,:].float()

        r = jt.sigmoid(xr @ rw)
        k = jt.nn.relu(xk @ kw) ** 2
        kv = k @ vw
        return r * kv

    @MyFunction
    def SA_one(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = state[5*i+1]
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x.float()

        r = jt.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        ww = time_first + k
        p = jt.maximum(pp, ww)
        e1 = jt.exp(pp - p)
        e2 = jt.exp(ww - p)
        a = e1 * aa + e2 * v
        b = e1 * bb + e2
        ww = pp + time_decay
        p = jt.maximum(ww, k)
        e1 = jt.exp(ww - p)
        e2 = jt.exp(k - p)
        state[5*i+2] = e1 * aa + e2 * v
        state[5*i+3] = e1 * bb + e2
        state[5*i+4] = p
        wkv = (a / b)
        return (r * wkv) @ ow

    @MyFunction
    def SA_seq(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        xx = jt.cat((state[5*i+1].unsqueeze(0), x[:-1,:]))
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        state[5*i+1] = x[-1,:].float()

        r = jt.sigmoid(xr @ rw)
        k = (xk @ kw).float()
        v = (xv @ vw).float()

        aa = state[5*i+2]
        bb = state[5*i+3]
        pp = state[5*i+4]
        T = x.shape[0]
        for t in range(T):
            ww = time_first + k[t]
            p = jt.maximum(pp, ww)
            e1 = jt.exp(pp - p)
            e2 = jt.exp(ww - p)
            a = e1 * aa + e2 * v[t]
            b = e1 * bb + e2
            ww = pp + time_decay
            p = jt.maximum(ww, k[t])
            e1 = jt.exp(ww - p)
            e2 = jt.exp(k[t] - p)
            if t != T - 1:
                aa = e1 * aa + e2 * v[t]
                bb = e1 * bb + e2
                pp = p
            else:
                state[5*i+2] = e1 * aa + e2 * v[t]
                state[5*i+3] = e1 * bb + e2
                state[5*i+4] = p
            xx[t] = (a / b)
        return (r * xx) @ ow

    def forward(self, tokens, state, preprocess_only = False):
        with jt.no_grad():
            w = self.w
            args = self.args

            seq_mode = len(tokens) > 1

            x = w.emb.weight[tokens] if seq_mode else w.emb.weight[tokens[0]]

            if state is None:
                state = jt.zeros(args.n_layer * 5, args.n_embd)
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30

            SA = self.SA_seq if seq_mode else self.SA_one
            FF = self.FF_seq if seq_mode else self.FF_one

            for i in range(args.n_layer):
                ww = w.blocks[i].att
                x = x + SA(self.LN(x, w.blocks[i].ln1), state, i, 
                    ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks[i].ffn
                x = x + FF(self.LN(x, w.blocks[i].ln2), state, i, 
                    ww.time_mix_k, ww.time_mix_r, 
                    ww.key.weight, ww.value.weight, ww.receptance.weight)
                
                if args.FLOAT_MODE == 'fp16':
                    if (i+1) % RWKV_RESCALE_LAYER == 0:
                        x = x / 2

            if preprocess_only:
                return state

            x = self.LN(x[-1,:], w.ln_out) if seq_mode else self.LN(x, w.ln_out)
            x = w.head.weight @ x

            return x.float(), state
