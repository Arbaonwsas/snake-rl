"""面向贪吃蛇 100% 满盘通关的 Maskable PPO + GPU 原生向量化引擎模块。"""

from snake_rl.ppo.network import SEResNetActorCritic
from snake_rl.ppo.vec_env_gpu import OBS_CHANNELS, TorchSnakeVecEnv, extract_obs_and_mask_from_core

__all__ = [
    "OBS_CHANNELS",
    "SEResNetActorCritic",
    "TorchSnakeVecEnv",
    "extract_obs_and_mask_from_core",
]
