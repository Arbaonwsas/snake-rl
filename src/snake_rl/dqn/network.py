"""Dueling Double DQN 神经网络结构定义。"""

from __future__ import annotations

import torch
from torch import nn


class DuelingDQNNet(nn.Module):
    """带 LayerNorm 与 SiLU 激活的双流竞争 Q 网络 (Dueling Architecture)。

    将 Q(s, a) 解耦为状态价值流 V(s) 与动作优势流 A(s, a)：
        Q(s, a) = V(s) + (A(s, a) - mean_a'(A(s, a')))
    """

    def __init__(
        self,
        obs_dim: int = 24,
        action_dim: int = 3,
        hidden_dim: int = 256,
        stream_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 共享特征提取主干：LayerNorm 显著提升强化学习非平稳目标下的稳定性
        self.feature_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 状态价值流 V(s): 评估当前盘面本身的生存潜力与得分前景
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, 1),
        )

        # 动作优势流 A(s, a): 评估直行/左转/右转各动作的相对优劣
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, stream_dim),
            nn.SiLU(),
            nn.Linear(stream_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """采用正交初始化提升早期探索阶段的梯度传导效率。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """前向计算各离散动作的 Q 值，输入形状 (B, obs_dim)，输出形状 (B, action_dim)。"""
        feat = self.feature_trunk(obs)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))
