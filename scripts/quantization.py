import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro
from transformers import TrainingArguments
import gr00t
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.utils.peft import get_lora_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def quantize_tensor(tensor):
    # 注意：这里量化成8bit的整数，但后面前向需要反量化回 float
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scale = (tensor_max - tensor_min) / 255.0
    # 计算零点（round后保证为整数）
    zero_point = (-tensor_min / scale).round()
    quantized = ((tensor / scale).round() + zero_point).clamp(0, 255).byte()
    return quantized, scale, zero_point

# 封装 Linear 模块的包装类
class QuantizedLinear(nn.Module):
    def __init__(self, orig_linear, quantized_weight, scale, zero_point):
        super().__init__()
        # 将原来的 bias 保留下来
        self.bias = orig_linear.bias  
        # 保存其他属性，比如输入输出特征数
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        # 保存量化后的 weight 以及相关量化信息为 buffer（不会作为参数更新）
        self.register_buffer("quantized_weight", quantized_weight)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        # 前向时先反量化后再做线性计算
        dequant_weight = (self.quantized_weight.float() - self.zero_point) * self.scale
        return F.linear(x, dequant_weight, self.bias)

# 封装 Conv2d 模块的包装类
class QuantizedConv2d(nn.Module):
    def __init__(self, orig_conv, quantized_weight, scale, zero_point):
        super().__init__()
        self.bias = orig_conv.bias
        self.stride = orig_conv.stride
        self.padding = orig_conv.padding
        self.dilation = orig_conv.dilation
        self.groups = orig_conv.groups
        self.kernel_size = orig_conv.kernel_size

        self.register_buffer("quantized_weight", quantized_weight)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, x):
        dequant_weight = (self.quantized_weight.float() - self.zero_point) * self.scale
        return F.conv2d(x, dequant_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# 辅助函数：根据模块名（字符串路径）在模型中替换对应的模块
def set_module_by_name(model, module_name, new_module):
    components = module_name.split('.')
    submodule = model
    for comp in components[:-1]:
        submodule = getattr(submodule, comp)
    setattr(submodule, components[-1], new_module)

# 针对模型中希望量化的模块（例如 Linear 和 Conv2d）进行遍历替换
def replace_with_quantized_modules(model):
    # 这里用 state_dict keys 来定位权重，因为模块的层次可能比较嵌套
    for name, module in model.named_modules():
        # 针对 Linear 层
        if isinstance(module, nn.Linear):
            # 取出 weight 并量化
            quant_weight, scale, zero_point = quantize_tensor(module.weight.data.cpu())
            quant_module = QuantizedLinear(module, quant_weight, scale, zero_point)
            # 替换掉该层：注意 name 是类似 "backbone.model.language_model.model.embed_tokens"
            set_module_by_name(model, name, quant_module)
            print(f"Replaced Linear layer: {name}")
        # 针对 Conv2d 层
        elif isinstance(module, nn.Conv2d):
            quant_weight, scale, zero_point = quantize_tensor(module.weight.data.cpu())
            quant_module = QuantizedConv2d(module, quant_weight, scale, zero_point)
            set_module_by_name(model, name, quant_module)
            print(f"Replaced Conv2d layer: {name}")

# 示例：对 GR00T_N1 模型中的 vision_model、language_model 和 DiT_block 替换权重
if __name__ == "__main__":
    # 假设已经加载模型 GR00T_N1
    MODEL_PATH = "nvidia/GR00T-N1-2B"
    model = GR00T_N1.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
    # 若需要半精度可调用：
    model = model.half()
    
    # 分别在 vision_model, language_model 以及 action_head 部分的 DiT_block 内进行替换
    print("Replacing quantized modules in vision_model ...")
    replace_with_quantized_modules(model.backbone.model.vision_model)
    
    print("Replacing quantized modules in language_model ...")
    replace_with_quantized_modules(model.backbone.model.language_model)
    
    print("Replacing quantized modules in DiT_block ...")
    replace_with_quantized_modules(model.action_head.model.transformer_blocks)
    
    # 最后可以验证模型结构
    print(model.dtype)
    print(model.backbone.model.vision_model.dtype)
    # 假设你要检查 vision_model 中第一个被替换的 Linear 层
    for name, module in model.backbone.model.vision_model.named_modules():
        if isinstance(module, QuantizedLinear):
            print(f"{name}: quantized_weight dtype = {module.quantized_weight.dtype}")
            break

