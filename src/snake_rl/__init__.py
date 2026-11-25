"""Snake RL 软件包入口与 Gymnasium 环境自动注册。"""

from gymnasium.envs.registration import register

from snake_rl.config import RenderConfig, RewardConfig, SnakeConfig
from snake_rl.core import Direction, SnakeCore
from snake_rl.env import FEATURE_DIM, SnakeEnv

__all__ = [
    "Direction",
    "FEATURE_DIM",
    "RenderConfig",
    "RewardConfig",
    "SnakeConfig",
    "SnakeCore",
    "SnakeEnv",
]

register(
    id="SnakeRL-v0",
    entry_point="snake_rl.env:SnakeEnv",
)
