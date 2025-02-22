import torch
from transformers import Qwen2ForCausalLM


model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# model_name = "Qwen/Qwen2.5-7B-Instruct"

## model
model = Qwen2ForCausalLM.from_pretrained(model_name)

torch.save(model.state_dict(), "Qwen2.5-0.5B-Instruct.pth")
