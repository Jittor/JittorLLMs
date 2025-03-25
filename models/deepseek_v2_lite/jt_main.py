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

qa_pairs = [
    (
        'Give me a short introduction to large language model.',
        ' Large language models are a type of artificial intelligence model that have been trained on vast amounts of text data, such as books, articles, and web pages. These models are designed to understand and generate human-like text, and they are often used in natural language processing (NLP) tasks, such as language translation, text summarization, and chatbots.\n\nLarge language models are typically built using deep learning techniques, such as neural networks, and they can be trained on hundreds of millions or even billions of words of text. This allows them to learn complex patterns and structures in language, and to generate text that is coherent, contextually relevant, and grammatically correct.\n\nOne of the most well-known large language models is GPT-3 (Generative Pre-trained Transformer 3), which has been trained on a dataset of over 570 gigabytes of text. GPT-3 can generate text on a wide range of topics, from writing poetry to generating code, and it has been used in a variety of applications, such as content creation, language translation, and even art creation.\n\nOverall, large language models have the potential to revolutionize the field of NLP and to enable new applications and services that were previously not possible.'
    )
]
for prompt, std_response in qa_pairs:
    
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

    if result == std_response:
        print('SAME as standard response (from model "deepseek-ai/DeepSeek-V2-Lite-Chat", for certain seeds, on CPU)')
    else:
        print('DIFFERENT from standard response (from model "deepseek-ai/DeepSeek-V2-Lite-Chat", for certain seeds, on CPU)')

    print('Time (response): {:.3f}s'.format(end_time - start_time))
    print('Time (token):    {:.3f}s'.format((end_time - start_time) / outputs[0][input_tensor.shape[1]:].shape[0]))
    print('=========================')
