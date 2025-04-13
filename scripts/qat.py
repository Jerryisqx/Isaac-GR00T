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

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1
from gr00t.utils.peft import get_lora_model
from gr00t.model.policy import Gr00tPolicy

# ---------------------------------------------------------------------
# 量化函数和包装类定义
# ---------------------------------------------------------------------
def quantize_tensor(tensor):
    # 计算最小值、最大值、量化比例及零点（8bit量化）
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    scale = (tensor_max - tensor_min) / 255.0
    zero_point = (-tensor_min / scale).round()
    quantized = ((tensor / scale).round() + zero_point).clamp(0, 255).byte()  # 得到 torch.uint8
    return quantized, scale, zero_point

# 封装 Linear 模块，增加可学习的 scale 和 zero_point，并保存浮点权重作为参考
class QuantizedLinear(nn.Module):
    def __init__(self, orig_linear, quantized_weight, scale, zero_point):
        super().__init__()
        self.bias = orig_linear.bias
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        # 量化权重保存为 buffer（不可训练），原始浮点权重也保存下来用于校准参考
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("fp_weight", orig_linear.weight.data.clone())
        # 将 scale 和 zero_point 设为可学习参数（初始化为量化时计算的值）
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.zero_point = nn.Parameter(torch.tensor(zero_point, dtype=torch.float32))

    def forward(self, x):
        # 用当前可学习参数还原出 dequantized weight
        dequant_weight = (self.quantized_weight.float() - self.zero_point) * self.scale
        return F.linear(x, dequant_weight, self.bias)

# 封装 Conv2d 模块，做法同上
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
        self.register_buffer("fp_weight", orig_conv.weight.data.clone())
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.zero_point = nn.Parameter(torch.tensor(zero_point, dtype=torch.float32))

    def forward(self, x):
        dequant_weight = (self.quantized_weight.float() - self.zero_point) * self.scale
        return F.conv2d(x, dequant_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

# 辅助函数，用于根据模块在模型中的路径替换子模块
def set_module_by_name(model, module_name, new_module):
    components = module_name.split('.')
    submodule = model
    for comp in components[:-1]:
        submodule = getattr(submodule, comp)
    setattr(submodule, components[-1], new_module)

def replace_with_quantized_modules(model):
    """
    遍历模型，将 Linear 与 Conv2d 层替换为量化包装类
    """
    for name, module in model.named_modules():
        # 替换 Linear 层
        if isinstance(module, nn.Linear):
            quant_weight, scale, zero_point = quantize_tensor(module.weight.data.cpu())
            quant_module = QuantizedLinear(module, quant_weight, scale, zero_point)
            set_module_by_name(model, name, quant_module)
            print(f"Replaced Linear layer: {name}")
        # 替换 Conv2d 层
        elif isinstance(module, nn.Conv2d):
            quant_weight, scale, zero_point = quantize_tensor(module.weight.data.cpu())
            quant_module = QuantizedConv2d(module, quant_weight, scale, zero_point)
            set_module_by_name(model, name, quant_module)
            print(f"Replaced Conv2d layer: {name}")

# ---------------------------------------------------------------------
# 校准流程：对每个量化模块（包装类）进行校准，利用随机（或实际）输入微调 scale 和 zero_point
# ---------------------------------------------------------------------
def calibrate_module(module, num_steps=100, lr=1e-3, device='cuda'):
    # 如果 bias 存在，则使用其 dtype，否则默认 torch.float32
    target_dtype = module.bias.dtype if module.bias is not None else torch.float32

    if isinstance(module, QuantizedLinear):
        # 对于 Linear 层，输入维度为 in_features
        input_shape = (32, module.in_features)
        optimizer = torch.optim.Adam([module.scale, module.zero_point], lr=lr)
        for step in range(num_steps):
            # 构造随机输入，确保 dtype 与 bias 相同
            x = torch.randn(*input_shape, device=device, dtype=target_dtype)
            # 计算反量化后的权重，并转换为目标数据类型
            dequant_weight = ((module.quantized_weight.float() - module.zero_point) * module.scale).to(target_dtype)
            ref_weight = module.fp_weight.to(target_dtype)
            # 分别使用当前量化参数和原始 fp 权重计算输出
            q_out = F.linear(x, dequant_weight, module.bias)
            ref_out = F.linear(x, ref_weight, module.bias)
            loss = F.mse_loss(q_out, ref_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"[Linear] {module}: step {step}, loss={loss.item():.6f}")
    elif isinstance(module, QuantizedConv2d):
        # 对于 Conv2d 层，假定输入通道数为 quantized_weight.shape[1]
        in_channels = module.quantized_weight.shape[1]
        # 这里假设输入尺寸为 224x224，根据实际情况调整
        input_shape = (32, in_channels, 224, 224)
        optimizer = torch.optim.Adam([module.scale, module.zero_point], lr=lr)
        for step in range(num_steps):
            # 构造随机输入，确保 dtype 与 bias 相同
            x = torch.randn(*input_shape, device=device, dtype=target_dtype)
            dequant_weight = ((module.quantized_weight.float() - module.zero_point) * module.scale).to(target_dtype)
            ref_weight = module.fp_weight.to(target_dtype)
            q_out = F.conv2d(x, dequant_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
            ref_out = F.conv2d(x, ref_weight, module.bias, module.stride, module.padding, module.dilation, module.groups)
            loss = F.mse_loss(q_out, ref_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"[Conv2d] {module}: step {step}, loss={loss.item():.6f}")


def calibrate_quantized_model(model, num_steps=100, lr=1e-3, device='cuda'):
    """
    遍历模型中所有量化包装的模块，执行校准过程
    """
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            print(f"Calibrating module: {name}")
            calibrate_module(module, num_steps=num_steps, lr=lr, device=device)

# ---------------------------------------------------------------------
# finetune 主程序（基于你提供的代码）并在训练结束后进行校准
# ---------------------------------------------------------------------

@dataclass
class Config:
    """Configuration for GR00T model fine-tuning."""
    # 数据集参数
    dataset_path: str = None
    output_dir: str = "/tmp/gr00t"
    data_config: str = "gr1_arms_only"
    # 训练参数
    batch_size: int = 16
    max_steps: int = 10000
    num_gpus: int = 1
    save_steps: int = 500
    # 模型参数
    base_model_path: str = "nvidia/GR00T-N1-2B"
    tune_llm: bool = False
    tune_visual: bool = True
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    resume: bool = False
    # 高级训练参数
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    dataloader_num_workers: int = 8
    report_to: str = "wandb"
    # 数据加载参数
    embodiment_tag: str = "new_embodiment"
    video_backend: str = "decord"
    split_num: int = 10
    num_nodes: int = 1
    quantization_bit: int = None  # 当使用后训练量化时，可传入对应的量化位数

def main(config: Config):
    """Main training function."""
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    train_dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        split=config.split_num
    )

    model = GR00T_N1.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
        quantization_bit=config.quantization_bit
    )
    model = model.half()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 用我们的包装类替换模型中所有的 Linear 与 Conv2d 层，
    # 这样包装类会存储量化权重（uint8）以及可微调的 scale 和 zero_point
    replace_with_quantized_modules(model)

    # 将模型移动到设备上（如有需要）
    model.to(device)

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        evaluation_strategy="no",
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()
    # 使用部分校准数据（这里简单地使用随机输入；更推荐用真实校准数据）对量化模块进行校准
    # print("开始校准量化模块...")
    # calibrate_quantized_model(model, num_steps=100, lr=1e-3, device=device)
    # print("校准完成！")
    # print(model)
    
    

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    config = tyro.cli(Config)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    assert config.num_gpus <= available_gpus, f"Requested GPUs ({config.num_gpus}) > available ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")
    if config.num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
            cmd = [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                f"--nnodes={config.num_nodes}",
                str(script_path),
            ]
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    cmd.append(f"--{key.replace('_', '-')}")
                    cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
