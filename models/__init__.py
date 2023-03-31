import importlib
import sys
from pathlib import Path

sys.path.append(str((Path(__file__).parent / 'llama' ).absolute()))
sys.path.append(str((Path(__file__).parent / 'pangualpha').absolute()))
sys.path.append(str((Path(__file__).parent / 'chatrwkv').absolute()))
from .util import *

__all__ = [
    "available_models",
    "get_model",
    "LLMModel"
]

availabel_models = ["chatglm", "pangualpha", "llama", "chatrwkv"]


class LLMModel:
    def __init__(self) -> None:
        pass

    def run(self, input_text: str) -> str:
        pass


def get_model(model_name, args) -> LLMModel:
    assert model_name in availabel_models
    globals()[f"get_{model_name}"]()
    try:
        module = importlib.import_module(f"models.{model_name}")
        return module.get_model(args)
    except ImportError as e:
        print(f"Import Error, maybe the dependencies are not installed, please try 'python3 -m pip install -r models/{model_name}/requirements.txt'")
        print(f"导入错误，可能没有安装此模型需要的依赖，请尝试运行 'python3 -m pip install -r models/{model_name}/requirements.txt'")
        raise e
