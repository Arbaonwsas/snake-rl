#!/usr/bin/env python3
"""强化学习环境合规性校验、无头采样吞吐量测试与启发式基线评估脚本。"""

from __future__ import annotations

import time
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

import snake_rl  # noqa: F401 触发 SnakeRL-v0 注册
from snake_rl import SnakeConfig, SnakeEnv


def run_compliance_checks() -> None:
    """使用 Gymnasium 官方 env_checker 检验双观测模式与双动作模式的接口规范。"""
    print("=== 1. Gymnasium API 标准合规性校验 ===")
    for obs_mode in ("features", "grid"):
        for act_mode in ("relative", "absolute"):
            env = gym.make("SnakeRL-v0", grid_size=10, obs_mode=obs_mode, action_mode=act_mode)
            check_env(env.unwrapped, skip_render_check=True)
            env.close()
            print(f"  [PASS] obs_mode={obs_mode:<8} | action_mode={act_mode:<8}")


def benchmark_throughput(num_steps: int = 20_000) -> None:
    """测量不同配置下无头强化学习环境的每秒交互步数 (Steps Per Second, SPS)。"""
    print(f"\n=== 2. 无头环境采样吞吐量基准 ({num_steps:,} 步) ===")
    configs = [
        ("features (含 Flood-Fill 连通域)", SnakeEnv(grid_size=12, obs_mode="features")),
        (
            "features (极速模式, 关闭 Flood-Fill)",
            SnakeEnv(grid_size=12, obs_mode="features", snake_config=SnakeConfig(enable_flood_fill=False)),
        ),
        ("grid (4x12x12 CNN 张量, 关闭 Flood-Fill)", SnakeEnv(grid_size=12, obs_mode="grid", snake_config=SnakeConfig(enable_flood_fill=False))),
    ]

    for label, env in configs:
        env.reset(seed=0)
        t0 = time.perf_counter()
        for i in range(num_steps):
            action = i % 3
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        elapsed = time.perf_counter() - t0
        sps = num_steps / elapsed
        print(f"  - {label:<42}: {sps:>10,.0f} steps/sec ({elapsed:.3f}s)")
        env.close()


def evaluate_heuristic_baseline(episodes: int = 20, grid_size: int = 12) -> None:
    """运行内置启发式智能体多局游戏，作为后续强化学习训练的对照基线。"""
    print(f"\n=== 3. 内置启发式基线智能体评估 ({episodes} 局, {grid_size}x{grid_size}) ===")
    env = SnakeEnv(grid_size=grid_size, obs_mode="features", action_mode="relative")
    scores: list[int] = []
    steps_list: list[int] = []

    for ep in range(episodes):
        env.reset(seed=1000 + ep)
        while True:
            act = env.core.heuristic_action()
            _, _, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                scores.append(info["score"])
                steps_list.append(info["steps"])
                break
    env.close()

    arr_s = np.array(scores)
    print(
        f"  平均得分: {arr_s.mean():.2f} ± {arr_s.std():.2f} | "
        f"最高得分: {arr_s.max()} / {grid_size * grid_size - 3} | "
        f"平均步数: {np.mean(steps_list):.1f}"
    )


if __name__ == "__main__":
    run_compliance_checks()
    benchmark_throughput()
    evaluate_heuristic_baseline()
