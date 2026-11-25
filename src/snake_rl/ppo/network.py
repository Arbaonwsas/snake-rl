"""带 Squeeze-and-Excitation (SE) 通道注意力与非法动作掩码的残差卷积 Actor-Critic 网络。

专为贪吃蛇满盘全局拓扑规划设计：
- 4 层 3x3 残差块覆盖整个棋盘全局感受野，不丢失任何角落与蛇尾的空间坐标细节（无池化降采样）
- SE 全局通道注意力使局部卷积随时感知全图整体占满度与通道间语义关联
- 支持 Invalid Action Masking（将必死动作 Logit 屏蔽为极小负数），聚焦 100% 算力于长远规划
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical, Distribution

from snake_rl.ppo.vec_env_gpu import OBS_CHANNELS

# 关闭每次构造分类分布时的主机端 NaN 同步校验以释放 GPU 全速流水线
Distribution.set_default_validate_args(False)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 全局通道注意力模块。"""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SEResidualBlock(nn.Module):
    """IMPALA 风格高吞吐无池化 3x3 残差卷积块 + SE 全局通道注意力（零 GroupNorm 显存带宽瓶颈）。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.act = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(x)
        out = self.act(self.conv1(out))
        out = self.conv2(out)
        out = self.se(out)
        return x + out


class SEResNetActorCritic(nn.Module):
    """面向网格全局路径规划的高吞吐 Maskable Actor-Critic 深度残差网络。"""

    def __init__(
        self,
        grid_size: int = 10,
        in_channels: int = OBS_CHANNELS,
        channels: int = 64,
        num_blocks: int = 3,
        action_dim: int = 3,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.in_channels = in_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.action_dim = action_dim

        # 输入主干（无步长降采样，精确保留每一格坐标拓扑）
        self.stem = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[SEResidualBlock(channels) for _ in range(num_blocks)])
        self.post_act = nn.SiLU(inplace=True)

        # 1x1 卷积压缩空间通道后展平保留精确网格坐标，拼接全局池化特征
        head_ch = 16
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(channels, head_ch, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        flat_dim = head_ch * grid_size * grid_size + channels

        self.shared_fc = nn.Sequential(
            nn.Linear(flat_dim, 384),
            nn.LayerNorm(384),
            nn.SiLU(inplace=True),
        )

        # Actor 策略头与 Critic 价值头独立分支
        self.actor_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, action_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """PPO 标准正交初始化：策略输出层使用 0.01 小增益以保证初始探索均匀。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 残差分支末尾卷积使用小增益缩放以稳定深层前向方差
        for blk in self.blocks:
            nn.init.orthogonal_(blk.conv2.weight, gain=0.3)

        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """提取融合精确网格坐标与全局池化上下文的隐层表征。"""
        feat_map = self.post_act(self.blocks(self.stem(obs)))
        global_ctx = feat_map.mean(dim=(-2, -1))
        spatial_flat = self.spatial_proj(feat_map).flatten(start_dim=1)
        combined = torch.cat([spatial_flat, global_ctx], dim=-1)
        return self.shared_fc(combined)

    def forward_logits_and_value(
        self,
        obs: torch.Tensor,
        action_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向计算屏蔽非法动作后的策略 Logits (B, 3) 与状态价值 V(s) (B,)。"""
        hidden = self.encode(obs)
        logits = self.actor_head(hidden).float()
        value = self.critic_head(hidden).squeeze(-1).float()

        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, -1e8)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_masks: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样或评估给定动作，返回 (action, log_prob, entropy, value)。"""
        logits, value = self.forward_logits_and_value(obs, action_masks)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
