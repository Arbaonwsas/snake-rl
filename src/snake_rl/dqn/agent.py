"""Dueling Double DQN 智能体策略封装（动作推断、TD 更新、模型持久化加载）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from snake_rl.dqn.network import DuelingDQNNet
from snake_rl.dqn.replay_buffer import TransitionBatch


@dataclass(slots=True)
class DQNConfig:
    """Dueling Double DQN 训练与模型超参数配置。"""

    obs_dim: int = 24
    action_dim: int = 3
    hidden_dim: int = 256
    stream_dim: int = 128

    gamma: float = 0.99
    n_step: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 256
    buffer_capacity: int = 120_000
    warmup_steps: int = 4_000

    # 目标网络软更新系数 (Polyak Averaging: theta_target <- tau * theta + (1 - tau) * theta_target)
    tau: float = 0.005
    max_grad_norm: float = 10.0

    # Epsilon-Greedy 探索率线性+指数平滑退火参数
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 120_000


class DQNAgent:
    """管理在线 Q 网络、目标 Q 网络、Double DQN 优化步骤及推理接口的智能体。"""

    def __init__(
        self,
        config: DQNConfig | None = None,
        device: str | torch.device = "auto",
        seed: int = 42,
    ) -> None:
        self.config = config or DQNConfig()
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.online_net = DuelingDQNNet(
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            stream_dim=self.config.stream_dim,
        ).to(self.device)

        self.target_net = DuelingDQNNet(
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            stream_dim=self.config.stream_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.online_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def epsilon_by_step(self, step: int) -> float:
        """根据当前全局环境步数计算探索率 epsilon。"""
        cfg = self.config
        if step >= cfg.eps_decay_steps:
            return cfg.eps_end
        ratio = float(step) / float(max(1, cfg.eps_decay_steps))
        return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * ratio

    @torch.no_grad()
    def select_actions_batch(self, obs_batch: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """对一批环境观测 (N, obs_dim) 并行执行 epsilon-greedy 动作选择。"""
        num_envs = obs_batch.shape[0]
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        q_values = self.online_net(obs_t)
        greedy_actions = q_values.argmax(dim=-1).cpu().numpy()

        if epsilon <= 0.0:
            return greedy_actions

        rand_mask = self.rng.random(num_envs) < epsilon
        if np.any(rand_mask):
            random_actions = self.rng.integers(0, self.config.action_dim, size=int(rand_mask.sum()))
            greedy_actions[rand_mask] = random_actions
        return greedy_actions

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> tuple[int, np.ndarray]:
        """单样本确定性推理（供评估与可视化界面调用），返回 (最优动作, 3维Q值数组)。"""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_vals = self.online_net(obs_t).squeeze(0).cpu().numpy()
        return int(np.argmax(q_vals)), q_vals

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        """执行一步 Double DQN + Huber Loss 参数更新与目标网络软更新。"""
        # 当前动作对应的预测 Q(s_t, a_t)
        q_all = self.online_net(batch.obs)
        q_pred = q_all.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN：用 online_net 挑选下一状态最优动作，用 target_net 评估该动作的价值
            next_best_actions = self.online_net(batch.next_obs).argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(batch.next_obs).gather(1, next_best_actions).squeeze(1)
            td_target = batch.rewards + batch.gammas * (1.0 - batch.terminated) * next_q_target

        loss = F.smooth_l1_loss(q_pred, td_target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Polyak 软更新目标网络参数
        tau = self.config.tau
        with torch.no_grad():
            for p_online, p_target in zip(self.online_net.parameters(), self.target_net.parameters()):
                p_target.data.mul_(1.0 - tau).add_(p_online.data, alpha=tau)

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_pred.mean().item()),
            "max_q": float(q_pred.max().item()),
            "grad_norm": float(grad_norm.item()),
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """保存网络权重、优化器状态、超参数配置及训练元数据至磁盘。"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "model_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, save_path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "auto") -> tuple[DQNAgent, dict[str, Any]]:
        """从检查点文件加载训练好的 DQNAgent 及附加元数据。"""
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = DQNConfig(**ckpt["config"])
        agent = cls(config=cfg, device=device)
        agent.online_net.load_state_dict(ckpt["model_state_dict"])
        if "target_state_dict" in ckpt:
            agent.target_net.load_state_dict(ckpt["target_state_dict"])
        else:
            agent.target_net.load_state_dict(ckpt["model_state_dict"])
        agent.online_net.eval()
        agent.target_net.eval()
        return agent, ckpt.get("metadata", {})
