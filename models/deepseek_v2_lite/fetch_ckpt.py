import torch
from utils.modeling_deepseek import DeepseekV2ForCausalLM

model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"

model = DeepseekV2ForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cpu()

torch.save(model.state_dict(), 'DeepSeek-V2-Lite-Chat-float32.pth')

