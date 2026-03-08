import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

def default_load_weights(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认的加载方式：直接将读取到的权重 copy 进我们的 Parameter 中"""
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    # 尝试获取模型自定义的映射表（后面我们在写 Qwen3 模型时会定义这个 mapping）
    packed_module_mapping = getattr(model, "packed_module_mapping", {})

    # 遍历目录下所有的 .safetensors 权重文件
    for file in glob(os.path.join(path, '*.safetensors')):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查当前读取的权重是否需要做映射/打包 (例如 q_proj -> qkv_proj)
                for k in packed_module_mapping:
                    if k in weight_name:
                        v, shard_id = packed_module_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)

                        # 调用参数自带的 weight_loader 进行加载（用于张量并行或特殊打包）
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 如果不需要特殊映射，直接走默认的普通加载
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_load_weights)
                    weight_loader(param, f.get_tensor(weight_name))
                    