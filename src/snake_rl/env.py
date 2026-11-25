"""兼容 Gymnasium 1.0+ 标准接口的贪吃蛇强化学习环境。

支持两种观测表示：
- "features"（默认）：24 维归一化紧凑特征向量，值域 [-1.0, 1.0]，适合 MLP (DQN / PPO) 快速收敛
- "grid"：(4, H, W) 多通道空间张量，值域 [0.0, 1.0]，含蛇身梯度编码，适合 CNN 网络架构
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from snake_rl.config import RenderConfig, RewardConfig, SnakeConfig
from snake_rl.core import SnakeCore, StepResult


FEATURE_DIM = 24


class SnakeEnv(gym.Env[np.ndarray, int]):
    """面向强化学习训练与实时可视化分析的贪吃蛇 Gymnasium 环境。"""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 15,
    }

    def __init__(
        self,
        grid_size: int | tuple[int, int] = 12,
        obs_mode: Literal["features", "grid"] = "features",
        action_mode: Literal["relative", "absolute"] = "relative",
        render_mode: Literal["human", "rgb_array"] | None = None,
        snake_config: SnakeConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_config: RenderConfig | None = None,
    ) -> None:
        super().__init__()
        if isinstance(grid_size, int):
            gw, gh = grid_size, grid_size
        else:
            gw, gh = grid_size

        if snake_config is None:
            self.snake_config = SnakeConfig(grid_width=gw, grid_height=gh)
        else:
            self.snake_config = snake_config

        self.reward_config = reward_config or RewardConfig()
        self.render_config = render_config or RenderConfig()

        if obs_mode not in ("features", "grid"):
            raise ValueError(f"obs_mode 必须为 'features' 或 'grid'，收到: {obs_mode}")
        if action_mode not in ("relative", "absolute"):
            raise ValueError(f"action_mode 必须为 'relative' 或 'absolute'，收到: {action_mode}")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"不支持的 render_mode: {render_mode}")

        self.obs_mode = obs_mode
        self.action_mode = action_mode
        self.render_mode = render_mode

        # 定义动作空间与观测空间
        self.action_space = spaces.Discrete(3 if action_mode == "relative" else 4)
        if obs_mode == "features":
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(FEATURE_DIM,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4, self.snake_config.grid_height, self.snake_config.grid_width),
                dtype=np.float32,
            )

        self.core = SnakeCore(config=self.snake_config)
        self._renderer: Any = None

        # 回合遥测统计状态
        self.episode_count: int = 0
        self.best_score: int = 0
        self.episode_return: float = 0.0
        self.last_reward: float = 0.0
        self.reward_history: deque[float] = deque(maxlen=80)
        self.last_reachable_ratios: np.ndarray = np.ones(3, dtype=np.float32)
        self.last_q_values: np.ndarray | None = None
        self.last_action_probs: np.ndarray | None = None
        self.last_state_value: float | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境至新回合初始状态，返回 (observation, info)。"""
        super().reset(seed=seed)
        self.core.reset(rng=self.np_random)

        self.episode_count += 1
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.reward_history.clear()
        self.last_q_values = None
        self.last_action_probs = None
        self.last_state_value = None

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int | np.integer) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """执行单步动作，返回 (observation, reward, terminated, truncated, info)。"""
        act = int(action)
        step_res = self.core.step(act, action_mode=self.action_mode)
        reward = self._compute_reward(step_res)

        self.last_reward = reward
        self.episode_return += reward
        self.reward_history.append(reward)
        if self.core.score > self.best_score:
            self.best_score = self.core.score

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), step_res.terminated, step_res.truncated, info

    def _compute_reward(self, res: StepResult) -> float:
        """根据物理演进结果与奖励配置计算即时标量奖励。"""
        rc = self.reward_config
        if res.death_reason == "win":
            return rc.food_base + rc.win_bonus
        if res.death_reason in ("wall", "self"):
            return rc.death_penalty
        if res.death_reason == "starvation":
            return rc.starvation_penalty
        if res.ate_food:
            length_ratio = float(self.core.length) / float(self.snake_config.total_cells)
            return rc.food_base * (1.0 + rc.food_length_scale * length_ratio * 10.0)

        # 普通移动步：基于曼哈顿距离减少量的潜势差奖励 + 微弱步数惩罚
        dist_delta = float(res.prev_food_dist - res.curr_food_dist)
        return rc.distance_shaping_coef * dist_delta + rc.step_cost

    def _get_obs(self) -> np.ndarray:
        """提取当前观测张量。"""
        if self.obs_mode == "features":
            return self._build_feature_obs()
        return self._build_grid_obs()

    def _build_feature_obs(self) -> np.ndarray:
        """构建 24 维全归一化特征向量（值域严格位于 [-1.0, 1.0]）。

        维度说明:
          [0..2]   : 正前、左转、右转 1 格即时碰撞危险标志 (0.0 或 1.0)
          [3..10]  : 蛇头相对坐标系 8 向障碍物逆距离 1/dist (0.0 ~ 1.0)
          [11..13] : 正前、左转、右转连通域可达空间比例 (0.0 ~ 1.0)
          [14..17] : 当前朝向 One-Hot (UP, RIGHT, DOWN, LEFT)
          [18..19] : 食物相对蛇头的归一化偏移 (dx / (W-1), dy / (H-1))，[-1.0, 1.0]
          [20..21] : 蛇尾相对蛇头的归一化偏移 (tx / (W-1), ty / (H-1))，[-1.0, 1.0]
          [22]     : 蛇当前归一化长度 length / total_cells，(0.0, 1.0]
          [23]     : 饥饿剩余耐心比例 1.0 - steps_since_food / max_starvation，[0.0, 1.0]
        """
        obs = np.zeros(FEATURE_DIM, dtype=np.float32)
        candidates = self.core.get_relative_candidates()
        for i, pos in enumerate(candidates):
            obs[i] = 1.0 if self.core.is_collision(pos, ignore_tail=True) else 0.0

        inv_rays, _ = self.core.compute_8way_rays()
        obs[3:11] = inv_rays

        reachable = self.core.compute_reachable_ratios()
        self.last_reachable_ratios = reachable
        obs[11:14] = reachable

        obs[14 + int(self.core.direction)] = 1.0

        hx, hy = self.core.head
        denom_x = float(max(1, self.core.width - 1))
        denom_y = float(max(1, self.core.height - 1))

        if self.core.food is not None:
            fx, fy = self.core.food
            obs[18] = np.clip((fx - hx) / denom_x, -1.0, 1.0)
            obs[19] = np.clip((fy - hy) / denom_y, -1.0, 1.0)

        tx, ty = self.core.tail
        obs[20] = np.clip((tx - hx) / denom_x, -1.0, 1.0)
        obs[21] = np.clip((ty - hy) / denom_y, -1.0, 1.0)

        obs[22] = float(self.core.length) / float(self.snake_config.total_cells)
        obs[23] = float(
            np.clip(
                1.0 - float(self.core.steps_since_food) / float(self.snake_config.max_starvation_steps),
                0.0,
                1.0,
            )
        )
        return obs

    def _build_grid_obs(self) -> np.ndarray:
        """构建 (4, H, W) 多通道空间特征张量，适用于卷积神经网络 (CNN)。

        Channel 0: 蛇头掩码 (1.0)
        Channel 1: 蛇身时序梯度（蛇尾为 1/L，越靠近蛇头越接近 1.0，帮助 CNN 感知蛇身运动流向）
        Channel 2: 食物掩码 (1.0)
        Channel 3: 当前朝向编码层（全图填充 (direction + 1) / 4.0）
        """
        h, w = self.core.height, self.core.width
        grid = np.zeros((4, h, w), dtype=np.float32)

        hx, hy = self.core.head
        grid[0, hy, hx] = 1.0

        snake_len = len(self.core.snake)
        if snake_len > 1:
            # 索引 0 为蛇头，1..snake_len-1 为从颈部到蛇尾
            for idx in range(1, snake_len):
                bx, by = self.core.snake[idx]
                grid[1, by, bx] = float(snake_len - idx) / float(snake_len)

        if self.core.food is not None:
            fx, fy = self.core.food
            grid[2, fy, fx] = 1.0

        grid[3, :, :] = float(int(self.core.direction) + 1) / 4.0
        self.last_reachable_ratios = self.core.compute_reachable_ratios()
        return grid

    def _get_info(self) -> dict[str, Any]:
        """构造供日志记录与可视化面板消费的诊断信息字典。"""
        return {
            "score": self.core.score,
            "best_score": self.best_score,
            "length": self.core.length,
            "steps": self.core.steps,
            "steps_since_food": self.core.steps_since_food,
            "max_starvation_steps": self.snake_config.max_starvation_steps,
            "death_reason": self.core.death_reason,
            "episode_return": round(self.episode_return, 4),
            "last_reward": round(self.last_reward, 4),
            "fill_ratio": round(self.core.length / self.snake_config.total_cells, 4),
            "reachable_ratios": self.last_reachable_ratios.copy(),
        }

    def render(self, mode_label: str = "RL AGENT", fps_override: int | None = None) -> np.ndarray | None:
        """渲染当前游戏画面。

        - render_mode == "human": 打开并刷新 Pygame 桌面窗口
        - render_mode == "rgb_array": 离屏渲染并返回 (H, W, 3) uint8 图像数组（可用于录制训练视频）
        """
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from snake_rl.renderer import SnakeRenderer

            self._renderer = SnakeRenderer(
                snake_config=self.snake_config,
                render_config=self.render_config,
                window_enabled=(self.render_mode == "human"),
            )

        return self._renderer.render(
            env=self,
            mode_label=mode_label,
            return_rgb=(self.render_mode == "rgb_array"),
            fps_target=fps_override,
        )

    def close(self) -> None:
        """释放 Pygame 窗口与渲染资源。"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
