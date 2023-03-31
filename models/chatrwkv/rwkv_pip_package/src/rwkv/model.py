########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import types, gc, os, time
import jittor as jt
import jittor.nn as nn
jt.flags.use_cuda = 1
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True

current_path = os.path.dirname(os.path.abspath(__file__))

MyModule = jt.nn.Module
MyFunction = lambda x: x

# if os.environ.get('RWKV_CUDA_ON') == '1':
#     from torch.utils.cpp_extension import load
#     wkv_cuda = load(name=f"wkv_cuda", sources=[f"{current_path}/cuda/wkv_op.cpp", f"{current_path}/cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"])

#     class WKV(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, T, C, w, u, k, v, aa, bb, pp):
#             assert 1 * C % min(C, 32) == 0
#             dtype = k.dtype
#             w = w.float().contiguous()
#             u = u.float().contiguous()
#             k = k.float().contiguous()
#             v = v.float().contiguous()
#             y = jt.empty((T, C), dtype=jt.float32)
#             wkv_cuda.forward(1, T, C, w, u, k, v, y, aa, bb, pp)
#             return y.to(dtype=dtype), aa, bb, pp
#     def RUN_CUDA(T, C, w, u, k, v, aa, bb, pp):
#         return WKV.apply(T, C, w, u, k, v, aa, bb, pp)
# else:
#     os.environ["RWKV_CUDA_ON"] = '0'


def RUN_CUDA(T, C, w, u, k, v, aa, bb, pp):
    assert 1 * C % min(C, 32) == 0
    dtype = k.dtype
    w = w.float32()
    u = u.float32()
    k = k.float32()
    v = v.float32()
    y = jt.code((T, C), jt.float32, [w, u, k, v, aa, bb, pp],
        cuda_src='''
            @alias(w, in0)
            @alias(u, in1)
            @alias(k, in2)
            @alias(v, in3)
            @alias(aa, in4)
            @alias(bb, in5)
            @alias(pp, in6)
            @alias(y, out)

            const int B = 1;
            const int T = out_shape0;
            const int C = out_shape1;

            dim3 threadsPerBlock( min(C, 32) );
            // assert(B * C % threadsPerBlock.x == 0);
            dim3 numBlocks(B * C / threadsPerBlock.x);
            kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w_p, u_p, k_p, v_p, y_p, aa_p, bb_p, pp_p);
        ''',
        cuda_header='''
            #define F out_type
            namespace jittor {
            __global__ void kernel_forward(const int B, const int T, const int C,
                                           const F *__restrict__ const _w, 
                                           const F *__restrict__ const _u,
                                           const F *__restrict__ const _k,
                                           const F *__restrict__ const _v,
                                           F *__restrict__ const _y,
                                           F *__restrict__ const _aa,
                                           F *__restrict__ const _bb,
                                           F *__restrict__ const _pp) {

                const int idx = blockIdx.x * blockDim.x + threadIdx.x;
                const int _b = idx / C;
                const int _c = idx % C;
                const int _offset = _b * T * C + _c;
                const int _state_offset = _b * C + _c;

                F u = _u[_c];
                F w = _w[_c];
                const F *__restrict__ const k = _k + _offset;
                const F *__restrict__ const v = _v + _offset;
                F *__restrict__ const y = _y + _offset;

                F aa = _aa[_state_offset];
                F bb = _bb[_state_offset];
                F pp = _pp[_state_offset];
                for (int i = 0; i < T; i++) {
                    const int ii = i * C;
                    F kk = k[ii];
                    F vv = v[ii];
                    F ww = u + kk;
                    F p = max(pp, ww);
                    F e1 = exp(pp - p);
                    F e2 = exp(ww - p);
                    y[ii] = (e1 * aa + e2 * vv) / (e1 * bb + e2);
                    ww = w + pp;
                    p = max(ww, kk);
                    e1 = exp(ww - p);
                    e2 = exp(kk - p);
                    aa = e1 * aa + e2 * vv;
                    bb = e1 * bb + e2;
                    pp = p;
                }
                _aa[_state_offset] = aa;
                _bb[_state_offset] = bb;
                _pp[_state_offset] = pp;
            }
            
            }
        ''')
    return y.astype(dtype), aa, bb, pp

########################################################################################################

class RWKV(MyModule):
    def __init__(self, model, strategy):
        super().__init__()
        self.args = types.SimpleNamespace()
        args = self.args
        args.MODEL_NAME = model

        # Rescale for fp16 mode: set x = x/2 every X layer (to avoid overflow)
        self.RESCALE_LAYER = 6 if 'fp16' in strategy else 0
        print(f'RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]} RWKV_CUDA_ON {os.environ["RWKV_CUDA_ON"]} RESCALE_LAYER {self.RESCALE_LAYER}\n')

        # We will load model to CPU first
        args.MODEL_NAME = args.MODEL_NAME.strip()
        if not args.MODEL_NAME.endswith('.pth'):
            args.MODEL_NAME += '.pth'
        print(f'Loading {args.MODEL_NAME} ...')
        with jt.no_grad():
            self.w = jt.load(args.MODEL_NAME)
            w = self.w
            args.n_embd = w['emb.weight'].shape[1]
            try: # precompute embedding
                w['emb.weight'] = jt.nn.layer_norm(w['emb.weight'], (args.n_embd,), weight=w['blocks.0.ln0.weight'], bias=w['blocks.0.ln0.bias'])
            except:
                w['emb.weight'] = jt.nn.layer_norm(w['emb.weight'].float(), (args.n_embd,), weight=w['blocks.0.ln0.weight'].float(), bias=w['blocks.0.ln0.bias'].float())
            del w['blocks.0.ln0.weight']
            del w['blocks.0.ln0.bias']

            keys = list(w.keys())
            args.n_layer = 0
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                args.n_layer = max(args.n_layer, layer_id+1)

            # Compute strategy
            s = [x.strip().split(' ') for x in strategy.split('->')]
            plan = [0] * len(s)
            stream_i = -1
            stream_count = 0
            to_allocate = args.n_layer + 1
            allocated = 0
            free_slots = 0
            for i in range(len(s)):
                si = s[i]
                if si[1] == 'fp32': si[1] = jt.float32
                elif si[1] == 'fp16': si[1] = jt.float16
                elif si[1] == 'bf16': si[1] = jt.bfloat16
                if len(si) > 2:
                    ss = si[2]
                    assert ss.startswith('*')
                    if ss.endswith('+'):
                        plan[i] = int(ss[1:-1])
                        stream_i = i
                    else:
                        plan[i] = int(ss[1:])
                    allocated += plan[i]
                    if allocated >= to_allocate:
                        plan[i] += to_allocate - allocated
                        break
                else:
                    free_slots += 1
            if stream_i < 0:
                if free_slots > 0 and to_allocate > allocated:
                    for i in range(len(s)):
                        if plan[i] == 0:
                            plan[i] = (to_allocate - allocated) // free_slots
                            allocated += plan[i]
                            free_slots -= 1
                if to_allocate > allocated:
                    plan[len(s)-1] += to_allocate - allocated
            else:
                if to_allocate > allocated:
                    stream_count = to_allocate - allocated
                    plan[stream_i] += stream_count
            print(f'Strategy: (total {args.n_layer}+1={args.n_layer+1} layers)')
            for i in range(len(s)):
                ss = s[i]
                if i != stream_i:
                    print(f'* {ss[0]} {ss[1]}, store {plan[i]} layers')
                else:
                    print(f'* {ss[0]} {ss[1]}, store {plan[i]-stream_count} layers, stream {stream_count} layers')
                plan[i] += (0 if i == 0 else plan[i-1])
            self.strategy = [None] * (args.n_layer + 1)
            strategy = self.strategy
            for n in range(args.n_layer + 1):
                for i in range(len(s)):
                    if n < plan[i]:
                        strategy[n] = types.SimpleNamespace()
                        strategy[n].device = s[i][0]
                        strategy[n].dtype = s[i][1]
                        strategy[n].stream = False
                        if i == stream_i and n >= (plan[i] - stream_count):
                            strategy[n].stream = True
                        break
                print(f"{n}-{strategy[n].device}-{str(strategy[n].dtype).replace('torch.','')}{'-stream' if strategy[n].stream else ''}",end=' ')
            print()

            # Load weights
            print_need_newline = False
            for x in keys:
                w[x].requires_grad = False
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                if ('ln_out.' in x) or ('head.' in x):
                    layer_id = args.n_layer
                dd = strategy[layer_id]
                DEVICE = dd.device
                DTYPE = dd.dtype
                
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x or 'head.weight' in x:
                    w[x] = w[x].t()
                
                if '.time_decay' in x: # need fp32 for this
                    w[x] = -jt.exp(w[x].float())
                elif '.time_first' in x: # need fp32 for this
                    w[x] = w[x].float()
                else:
                    w[x] = w[x].astype(DTYPE)

                if self.RESCALE_LAYER > 0:
                    if 'att.output.weight' in x:
                        w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))
                    if 'ffn.value.weight' in x:
                        w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))
                
                if 'emb.' in x:
                    pass
                elif (dd.stream) and (('key.weight' in x) or ('value.weight' in x) or ('receptance.weight' in x) or ('output.weight' in x)):
                    try:
                        w[x] = w[x].pin_memory() # if you see "CUDA error: out of memory" here, that's out of CPU RAM, not VRAM. Get more RAM :)
                    except:
                        print('Note: You are running out of RAM. Get more CPU RAM. Now this will run much slower.')
                elif DEVICE != 'cpu':
                    w[x] = w[x].to(device=DEVICE)

                shape = [i for i in w[x].shape if i != 1]
                if len(shape) > 1:
                    shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}"
                else:
                    shape = f" {str(shape[0]).rjust(5)}      "
                if layer_id == 0 or layer_id >= args.n_layer-1:
                    if print_need_newline:
                        print('\n', end = '')
                        print_need_newline = False
                    dt = str(w[x].dtype).replace('torch.', '')
                    dt = dt.replace('float32', 'fp32').replace('bfloat16', 'bf16').replace('float16', 'fp16')
                    print(x.ljust(32), dt, shape)
                else:
                    print_need_newline = True
                    print('.', end = '', flush = True)
            assert len(keys) == 4 + (4+9+5) * args.n_layer, 'Error: not a RWKV-4 model (4a and 4b models are not supported as of now)'
            gc.collect()
        
    @MyFunction
    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw):
        xx = jt.nn.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = jt.sigmoid(rx @ rw)
        k = (nn.relu(kx @ kw)) ** 2
        out = r * (k @ vw)
        return x + out, xx
    
    @MyFunction
    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw):
        xx = jt.nn.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = jt.concat((sx.unsqueeze(0), xx[:-1,:]))
        xk = xx * k_mix + sx * (1 - k_mix)
        xr = xx * r_mix + sx * (1 - r_mix)

        r = jt.sigmoid(xr @ rw)
        k = (nn.relu(xk @ kw)) ** 2
        out = r * (k @ vw)
        return x + out, xx[-1,:]

    @MyFunction
    def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow):
        xx = jt.nn.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = jt.sigmoid(rx @ rw)
        k = (kx @ kw).float()
        v = (vx @ vw).float()

        ww = t_first + k
        p = jt.maximum(pp, ww)
        e1 = jt.exp(pp - p)
        e2 = jt.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=r.dtype)
        ww = t_decay + pp
        p = jt.maximum(ww, k)
        e1 = jt.exp(ww - p)
        e2 = jt.exp(k - p)

        out = (r * wkv) @ ow
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p

    @MyFunction
    def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow):
        xx = jt.nn.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        sx = jt.concat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        r = jt.sigmoid(rx @ rw)
        k = (kx @ kw).float()
        v = (vx @ vw).float()

        T = x.shape[0]
        for t in range(T):
            kk = k[t]
            vv = v[t]
            ww = t_first + kk
            p = jt.maximum(pp, ww)
            e1 = jt.exp(pp - p)
            e2 = jt.exp(ww - p)
            sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=r.dtype)
            ww = t_decay + pp
            p = jt.maximum(ww, kk)
            e1 = jt.exp(ww - p)
            e2 = jt.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        out = (r * sx) @ ow
        return x + out, xx[-1,:], aa, bb, pp
    
    @MyFunction
    def cuda_att_pre(self, x, sx, ln_w, ln_b, k_mix, v_mix, r_mix, kw, vw, rw):
        T, C = x.size()
        xx = jt.nn.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
        sx = jt.concat((sx.unsqueeze(0), xx[:-1,:]))
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        r = jt.sigmoid(rx @ rw)
        k = kx @ kw
        v = vx @ vw
        return xx, r, k, v
    def cuda_att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow):
        T, C = x.size()
        xx, r, k, v = self.cuda_att_pre(x, sx, ln_w, ln_b, k_mix, v_mix, r_mix, kw, vw, rw)
        y, aa, bb, pp = RUN_CUDA(T, C, t_decay, t_first, k, v, aa, bb, pp)
        out = (r * y) @ ow
        return x + out, xx[-1,:], aa, bb, pp

    def forward(self, tokens, state, full_output=False):
        with jt.no_grad():
            w = self.w
            args = self.args

            if state == None:
                state = [None] * args.n_layer * 5
                for i in range(args.n_layer): # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
                    dd = self.strategy[i]
                    dtype = dd.dtype
                    state[i*5+0] = jt.zeros(args.n_embd, dtype=dtype)
                    state[i*5+1] = jt.zeros(args.n_embd, dtype=jt.float)
                    state[i*5+2] = jt.zeros(args.n_embd, dtype=jt.float)
                    state[i*5+3] = jt.zeros(args.n_embd, dtype=jt.float) - 1e30
                    state[i*5+4] = jt.zeros(args.n_embd, dtype=dtype)

            seq_mode = len(tokens) > 1

            x = w['emb.weight'][tokens if seq_mode else tokens[0]]

            for i in range(args.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'
                dd = self.strategy[i]
                dtype = dd.dtype
                if seq_mode:
                    ATT = self.cuda_att_seq
                    # if 'cuda' in str(dev) and os.environ["RWKV_CUDA_ON"] == '1':
                    #     ATT = self.cuda_att_seq
                    # else:
                    #     ATT = self.att_seq
                    FFN = self.ffn_seq
                else:
                    ATT = self.att_one
                    FFN = self.ffn_one

                x = x.to(dtype=dtype)
                if dd.stream:
                    kw = w[f'{att}key.weight']
                    vw = w[f'{att}value.weight']
                    rw = w[f'{att}receptance.weight']
                    ow = w[f'{att}output.weight']
                    x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = ATT(
                        x, sx=state[i*5+0], aa=state[i*5+1], bb=state[i*5+2], pp=state[i*5+3],
                        ln_w=w[f'{bbb}ln1.weight'], ln_b=w[f'{bbb}ln1.bias'],
                        k_mix=w[f'{att}time_mix_k'], v_mix=w[f'{att}time_mix_v'], r_mix=w[f'{att}time_mix_r'],
                        t_decay = w[f'{att}time_decay'], t_first = w[f'{att}time_first'],
                        kw=kw, vw=vw, rw=rw, ow=ow)
                    del kw
                    del vw
                    del rw
                    del ow
                else:
                    x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = ATT(
                        x, sx=state[i*5+0], aa=state[i*5+1], bb=state[i*5+2], pp=state[i*5+3],
                        ln_w=w[f'{bbb}ln1.weight'], ln_b=w[f'{bbb}ln1.bias'],
                        k_mix=w[f'{att}time_mix_k'], v_mix=w[f'{att}time_mix_v'], r_mix=w[f'{att}time_mix_r'],
                        t_decay = w[f'{att}time_decay'], t_first = w[f'{att}time_first'],
                        kw=w[f'{att}key.weight'],
                        vw=w[f'{att}value.weight'],
                        rw=w[f'{att}receptance.weight'],
                        ow=w[f'{att}output.weight'])
                if dd.stream:
                    kw = w[f'{ffn}key.weight']
                    vw = w[f'{ffn}value.weight']
                    rw = w[f'{ffn}receptance.weight']
                    x, state[i*5+4] = FFN(
                        x, sx=state[i*5+4],
                        ln_w=w[f'{bbb}ln2.weight'], ln_b=w[f'{bbb}ln2.bias'],
                        k_mix=w[f'{ffn}time_mix_k'], r_mix=w[f'{ffn}time_mix_r'],
                        kw=kw, vw=vw, rw=rw)
                    del kw
                    del vw
                    del rw
                else:
                    x, state[i*5+4] = FFN(
                        x, sx=state[i*5+4],
                        ln_w=w[f'{bbb}ln2.weight'], ln_b=w[f'{bbb}ln2.bias'],
                        k_mix=w[f'{ffn}time_mix_k'], r_mix=w[f'{ffn}time_mix_r'],
                        kw=w[f'{ffn}key.weight'],
                        vw=w[f'{ffn}value.weight'],
                        rw=w[f'{ffn}receptance.weight'])

                if self.RESCALE_LAYER > 0:
                    if (i+1) % self.RESCALE_LAYER == 0:
                        x = x / 2
            
            x = x[-1,:] if (seq_mode and (not full_output)) else x
            # x = x.to(dtype=self.strategy[args.n_layer].dtype, device=self.strategy[args.n_layer].device)
            x = x.astype(self.strategy[args.n_layer].dtype)
            x = jt.nn.layer_norm(x, (args.n_embd,), weight=w['ln_out.weight'], bias=w['ln_out.bias'])
            x = x @ w['head.weight']

            return x.float(), state
