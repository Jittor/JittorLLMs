import jittor as jt
from time import time

from deepseek_v2_lite_jt.tokenization_llama_fast import LlamaTokenizerFast
from deepseek_v2_lite_jt.modeling_deepseek import DeepseekV2ForCausalLM, DeepseekV2Config



# 检查 GPU 是否可用
if jt.has_cuda:
    jt.flags.use_cuda = True
    print(f"Using device: CUDA")
else:
    jt.flags.use_cuda = False
    print(f"Using device: CPU")


tokenizer = LlamaTokenizerFast(tokenizer_file='./tokenizer.json')

model = DeepseekV2ForCausalLM(
    config=DeepseekV2Config()
)
model.load_state_dict(
    jt.load('DeepSeek-V2-Lite-Chat-float32.pth')
)

print(f'model size: {sum(p.numel() for name, p in model.named_parameters()) / 1_000_000_000} B')



prompt = 'Give me a short introduction to large language model.',

print('Input:')
print(prompt, '\n')
print('Output:')
start_time = time()


text = '<｜begin▁of▁sentence｜>User: {}\n\nAssistant:'.format(prompt)
input_tensor = jt.array(tokenizer._batch_encode_plus([text])['input_ids'])

outputs = model.generate(input_tensor, max_new_tokens=1000)

result = tokenizer._decode(outputs[0][input_tensor.shape[1]:].detach().cpu().tolist(), skip_special_tokens=True)
print(result)


end_time = time()
print('-----------------')

print('Time (response): {:.3f}s'.format(end_time - start_time))
print('Time (token):    {:.3f}s'.format((end_time - start_time) / outputs[0][input_tensor.shape[1]:].shape[0]))
print('=========================')
