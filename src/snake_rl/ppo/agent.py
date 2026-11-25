"""Maskable PPO 智能体封装（支持 AMP Tensor Core 混合精度更新、动作掩码推理与检查点管理）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

from snake_rl.core import SnakeCore
from snake_rl.ppo.lookahead import select_action_with_lookahead
from snake_rl.ppo.network import SEResNetActorCritic
from snake_rl.ppo.vec_env_gpu import OBS_CHANNELS, extract_obs_and_mask_from_core


@dataclass(slots=True)
class PPOConfig:
    """面向 100% 满盘通关的 Maskable PPO 超参数配置。"""

    grid_size: int = 10
    in_channels: int = OBS_CHANNELS
    channels: int = 48
    num_blocks: int = 3
    action_dim: int = 3

    # GAE 与 PPO 核心超参数（高折扣因子 gamma=0.995 支持百步级哈密顿长远规划）
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef_start: float = 0.02
    ent_coef_end: float = 0.001
    lr_start: float = 6e-4
    lr_end: float = 4e-5
    max_grad_norm: float = 0.8

    num_envs: int = 1024
    rollout_steps: int = 128
    update_epochs: int = 3
    minibatch_size: int = 8192
    use_amp: bool = True


class MaskablePPOAgent:
    """结合 SE-ResNet 与非法动作硬掩码的 PPO 智能体。"""

    def __init__(
        self,
        config: PPOConfig | None = None,
        device: str | torch.device = "auto",
        seed: int = 42,
    ) -> None:
        self.config = config or PPOConfig()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.amp_enabled = bool(self.config.use_amp and self.device.type == "cuda")
        self.amp_dtype = torch.bfloat16 if (self.amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

        self.net = SEResNetActorCritic(
            grid_size=self.config.grid_size,
            in_channels=self.config.in_channels,
            channels=self.config.channels,
            num_blocks=self.config.num_blocks,
            action_dim=self.config.action_dim,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.lr_start,
            eps=1e-5,
            weight_decay=1e-4,
        )

    def set_learning_rate(self, lr: float) -> None:
        """动态更新优化器学习率。"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.no_grad()
    def predict_from_core(
        self,
        core: SnakeCore,
        deterministic: bool = True,
        use_lookahead: bool = True,
    ) -> tuple[int, np.ndarray, float, np.ndarray]:
        """对单局 CPU `SnakeCore` 执行推理（默认启用神经拓扑前瞻护盾），返回 (动作, 概率分布, V(s), 合法掩码)。"""
        obs_np, mask_np = extract_obs_and_mask_from_core(core)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device).unsqueeze(0)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            logits, val = self.net.forward_logits_and_value(obs_t, mask_t)

        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        value = float(val.item())

        if use_lookahead and deterministic:
            action, post_probs = select_action_with_lookahead(self, core, probs, mask_np)
            return action, post_probs, value, mask_np

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))
        return action, probs, value, mask_np

    def update_rollout(
        self,
        b_obs: torch.Tensor,
        b_masks: torch.Tensor,
        b_actions: torch.Tensor,
        b_logprobs: torch.Tensor,
        b_advantages: torch.Tensor,
        b_returns: torch.Tensor,
        b_values: torch.Tensor,
        ent_coef: float,
    ) -> dict[str, float]:
        """在收集完一整轮 Rollout 数据后执行多轮 Minibatch Clipped PPO 更新。"""
        cfg = self.config
        batch_size = b_obs.shape[0]
        dev = b_obs.device

        pg_loss_sum = torch.zeros((), device=dev)
        v_loss_sum = torch.zeros((), device=dev)
        ent_sum = torch.zeros((), device=dev)
        kl_sum = torch.zeros((), device=dev)
        clip_sum = torch.zeros((), device=dev)
        num_mbs = 0

        self.net.train()
        for _ in range(cfg.update_epochs):
            inds = torch.randperm(batch_size, device=dev)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb_inds = inds[start : start + cfg.minibatch_size]

                mb_obs = b_obs[mb_inds]
                mb_masks = b_masks[mb_inds]
                mb_actions = b_actions[mb_inds]
                mb_old_logprobs = b_logprobs[mb_inds]
                mb_adv = b_advantages[mb_inds]
                mb_ret = b_returns[mb_inds]
                mb_old_val = b_values[mb_inds]

                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                with torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    _, new_logprob, entropy, new_value = self.net.get_action_and_value(
                        mb_obs, mb_masks, mb_actions
                    )

                    logratio = new_logprob - mb_old_logprobs
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clip_frac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_clipped = mb_old_val + torch.clamp(new_value - mb_old_val, -cfg.clip_coef, cfg.clip_coef)
                    v_loss_unclipped = (new_value - mb_ret) ** 2
                    v_loss_clipped = (v_clipped - mb_ret) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss + cfg.vf_coef * v_loss - ent_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                pg_loss_sum += pg_loss.detach()
                v_loss_sum += v_loss.detach()
                ent_sum += entropy_loss.detach()
                kl_sum += approx_kl.detach()
                clip_sum += clip_frac.detach()
                num_mbs += 1

        self.net.eval()
        denom = float(max(1, num_mbs))
        return {
            "pg_loss": float(pg_loss_sum.item() / denom),
            "v_loss": float(v_loss_sum.item() / denom),
            "entropy": float(ent_sum.item() / denom),
            "approx_kl": float(kl_sum.item() / denom),
            "clip_frac": float(clip_sum.item() / denom),
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """保存网络权重、超参配置及评估记录至检查点文件。"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metadata": metadata or {},
            },
            save_path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "auto") -> tuple[MaskablePPOAgent, dict[str, Any]]:
        """从检查点文件恢复 MaskablePPOAgent。"""
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = PPOConfig(**ckpt["config"])
        agent = cls(config=cfg, device=device)
        agent.net.load_state_dict(ckpt["model_state_dict"])
        agent.net.eval()
        return agent, ckpt.get("metadata", {})
