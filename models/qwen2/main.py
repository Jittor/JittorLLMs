import jittor as jt
from time import time

from qwen2_jt.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
from qwen2_jt.tokenization_qwen2_fast import Qwen2TokenizerFast

jt.flags.use_cuda = True


def main():
    ## model
    # Qwen2.5-0.5B-Instruct
    model = Qwen2ForCausalLM()
    model.load_state_dict(jt.load("./Qwen2.5-0.5B-Instruct.pth")) 

    # # Qwen2.5-7B-Instruct
    # model = Qwen2ForCausalLM(
    #     config=Qwen2Config(
    #         vocab_size=152064, 
    #         hidden_size=3584,
    #         intermediate_size=18944,
    #         num_hidden_layers=28,
    #         num_attention_heads=28,
    #         num_key_value_heads=4,
    #     )
    # )
    # model.load_state_dict(jt.load("./Qwen2.5-7B-Instruct.pth")) 

    print(f'model size: {sum(p.numel() for p in model.parameters()) / 1_000_000_000} B')

    ## tokenizer
    tokenizer = Qwen2TokenizerFast(
        tokenizer_file = 'Qwen-tokenizer.json',
    )

    prompt = "Give me a short introduction to large language model."
    text_template = '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'


    start_time = time()

    print('Input:')
    print(prompt, '\n')
    print('Output:')

    model_inputs = tokenizer._batch_encode_plus([text_template.format(prompt)])
    for k, v in model_inputs.items():
        model_inputs[k] = jt.Var(v)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs['input_ids'], generated_ids)
    ]

    response = tokenizer._decode(generated_ids[0].detach().cpu().tolist(), skip_special_tokens=True)
    print(response)

    print('-----------------')

    end_time = time()
    print('Time (sequence): {:.3f}s'.format(end_time - start_time))
    print('Time (token):    {:.3f}s'.format((end_time - start_time) / generated_ids[0].shape[0]))
    print('=========================')


if __name__ == '__main__':
    main()
