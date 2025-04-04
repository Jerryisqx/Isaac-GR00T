import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
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

# change the following paths
MODEL_PATH = "nvidia/GR00T-N1-2B"
REPO_PATH = os.path.dirname(os.path.dirname(gr00t.__file__))
DATASET_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim.PickNPlace")
EMBODIMENT_TAG = "gr1"
QUANTBIT_TAG = 8
DATA_CONFIG: str = "gr1_arms_only"
LORA_RANK = 128
LORA_ALPHA = 
LORA_DROPOUT = 

data_config = DATA_CONFIG_MAP[DATA_CONFIG]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

model = GR00T_N1.from_pretrained(
        pretrained_model_name_or_path=MODEL_PATH,
        # tune_llm=config.tune_llm,  # backbone's LLM
        # tune_visual=config.tune_visual,  # backbone's vision tower
        # tune_projector=config.tune_projector,  # action head's projector
        # tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
        quantization_bit=QUANTBIT_TAG
    )

model.compute_dtype = "fp8"
model.config.compute_dtype = "fp8"

if LORA_RANK > 0:
    model = get_lora_model(
        model,
        rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )

