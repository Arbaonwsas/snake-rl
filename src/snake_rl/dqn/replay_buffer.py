"""支持多步回报（N-Step Return）预聚合的高性能纯 NumPy 环形经验回放池。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(slots=True)
class TransitionBatch:
    """加载至目标计算设备（GPU/CPU）的训练微批次张量。"""

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    gammas: torch.Tensor


class NStepAccumulator:
    """单个环境实例的 N 步滑动窗口累加器。

    自动处理多步折扣累计奖励 R_t^(n) = sum_{k=0}^{m-1} gamma^k r_{t+k}，
    并在遇到回合结束（terminated 或 truncated）时将窗口内剩余残留步全部冲刷输出，
    避免跨回合数据泄漏。
    """

    def __init__(self, n_step: int = 3, gamma: float = 0.99) -> None:
        self.n_step = max(1, n_step)
        self.gamma = gamma
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()

    def append(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> list[tuple[np.ndarray, int, float, np.ndarray, bool, float]]:
        """推入单步转移，返回所有已就绪的多步转移元组：(s_t, a_t, R^(m), s_{t+m}, term, gamma^m)。"""
        self.buffer.append((obs, action, reward, next_obs, terminated))
        ready: list[tuple[np.ndarray, int, float, np.ndarray, bool, float]] = []

        if terminated or truncated:
            # 回合终止或截断时，依次冲刷队列中所有未输出的起始步
            while self.buffer:
                ready.append(self._pack_current())
                self.buffer.popleft()
        elif len(self.buffer) >= self.n_step:
            ready.append(self._pack_current())
            self.buffer.popleft()

        return ready

    def _pack_current(self) -> tuple[np.ndarray, int, float, np.ndarray, bool, float]:
        """将当前窗口头部起算的最多 n_step 步打包为一条多步经验。"""
        s0, a0, _, _, _ = self.buffer[0]
        cum_reward = 0.0
        discount = 1.0
        last_next_obs = s0
        last_term = False

        for _, _, r_k, next_s_k, term_k in self.buffer:
            cum_reward += discount * r_k
            discount *= self.gamma
            last_next_obs = next_s_k
            last_term = term_k
            if term_k:
                break

        return s0, a0, cum_reward, last_next_obs, last_term, discount


class ReplayBuffer:
    """连续内存布局的环形经验回放池，零 Python 对象碎片开销。"""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 42) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros(capacity, dtype=np.int64)
        self.rewards_buf = np.zeros(capacity, dtype=np.float32)
        self.terminated_buf = np.zeros(capacity, dtype=np.float32)
        self.gammas_buf = np.zeros(capacity, dtype=np.float32)

        self.ptr: int = 0
        self.size: int = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        gamma_n: float,
    ) -> None:
        """写入一条已计算的（单步或多步）转移样本。"""
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.actions_buf[idx] = action
        self.rewards_buf[idx] = reward
        self.next_obs_buf[idx] = next_obs
        self.terminated_buf[idx] = 1.0 if terminated else 0.0
        self.gammas_buf[idx] = gamma_n

        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        """均匀随机采样 batch_size 条经验并传输至指定 PyTorch 设备。"""
        indices = self.rng.integers(0, self.size, size=batch_size)
        return TransitionBatch(
            obs=torch.as_tensor(self.obs_buf[indices], device=device),
            actions=torch.as_tensor(self.actions_buf[indices], device=device),
            rewards=torch.as_tensor(self.rewards_buf[indices], device=device),
            next_obs=torch.as_tensor(self.next_obs_buf[indices], device=device),
            terminated=torch.as_tensor(self.terminated_buf[indices], device=device),
            gammas=torch.as_tensor(self.gammas_buf[indices], device=device),
        )
