import os

from stable_baselines3.a2c import A2C
from stable_baselines3.common.utils import get_system_info
from stable_baselines3.ddpg import DDPG
from stable_baselines3.dqn import DQN
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.ppo import PPO
# from stable_baselines3.influence_ppo import INFLUENCE_PPO
from stable_baselines3.influence_ppo import INFLUENCE_PPO_HARVEST_V2, INFLUENCE_PPO_HARVEST_V0, INFLUENCE_PPO_HARVEST_V2, \
    INFLUENCE_PPO_CP_RPD, INFLUENCE_PPO_CP_RPD_V2, INFLUENCE_PPO_CLEANUP_V0, INFLUENCE_PPO_SIMPLE_CLEANUP_V0, \
    INFLUENCE_PPO_HARVEST_VECTOR_V0, INFLUENCE_PPO_HARVEST_VECTOR_V1_TRUE_PARTNER, \
    INFLUENCE_PPO_HARVEST_VECTOR_V1_TRUE_PARTNER_PRETRAINED_DESIRED
from stable_baselines3.sac import SAC
from stable_baselines3.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )
