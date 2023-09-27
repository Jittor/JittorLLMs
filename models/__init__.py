import importlib
import os
import sys
from pathlib import Path
import traceback

sys.path.append(str((Path(__file__).parent / 'llama' ).absolute()))
sys.path.append(str((Path(__file__).parent / 'pangualpha').absolute()))
sys.path.append(str((Path(__file__).parent / 'chatrwkv').absolute()))
sys.path.append(str((Path(__file__).parent / 'chatrwkv' / 'rwkv_pip_package' / 'src').absolute()))

from .util import *

__all__ = [
    "available_models",
    "get_model",
    "LLMModel"
]

availabel_models = ["chatglm", "pangualpha", "llama", "chatrwkv", "llama2", "atom7b"]


class LLMModel:
    def __init__(self) -> None:
        pass

    def run(self, input_text: str) -> str:
        pass


def get_model(args) -> LLMModel:
    model_name = args.model
    assert model_name in availabel_models

    if model_name == "pangualpha":
        os.environ["log_silent"] = "1"

    globals()[f"get_{model_name}"]()

    try:
        if model_name == "chatglm":
            import transformers
            if transformers.__version__ != '4.26.1':
                raise RuntimeError(f"transformers 版本不匹配 {transformers.__version__} != 4.26.1, 请运行 'python -m pip install -r models/{model_name}/requirements.txt -i https://pypi.jittor.org/simple' ")
        
        if model_name == "llama2":
            model_name = "llama"
        module = importlib.import_module(f"models.{model_name}")
        return module.get_model(args)
    except ModuleNotFoundError:
        traceback.print_exc()
        print(f"Import Error, maybe the dependencies are not installed, please try 'python -m pip install -r models/{model_name}/requirements.txt -i https://pypi.jittor.org/simple'")
        print(f"导入错误，可能没有安装此模型需要的依赖，请尝试运行 'python -m pip install -r models/{model_name}/requirements.txt -i https://pypi.jittor.org/simple'")
        exit()
