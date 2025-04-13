# example.py
import imageio
import torch
import gymnasium as gym
import numpy as np
import gym_aloha
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.dataset import ModalityConfig

data_config = DATA_CONFIG_MAP["bimanual_panda_gripper"]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1-2B",
    embodiment_tag="gr1",
    modality_config=modality_config,
    modality_transform=modality_transform,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()
frames = []

for _ in range(1000):
    action = policy.get_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)