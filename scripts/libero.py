from gr00t.utils.misc import any_describe
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
print("可用配置：", list(DATA_CONFIG_MAP.keys()))
dataset_path = "/scratch_net/biwidl313/gr00t_dataset/libero_spatial_no_noops_lerobot"   # change this to your dataset path

data_config = DATA_CONFIG_MAP["libero"]

dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=data_config.modality_config(),
    embodiment_tag="new_embodiment",
    video_backend="torchvision_av",
)

resp = dataset[7]
any_describe(resp)