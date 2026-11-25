#!/usr/bin/env python3
"""一键训练 Dueling Double DQN 贪吃蛇智能体并进行三方对比评测。"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from snake_rl.dqn.agent import DQNAgent, DQNConfig
from snake_rl.dqn.trainer import evaluate_agent, train_dqn
from snake_rl.env import FEATURE_DIM, SnakeEnv


def compare_all_policies(best_ckpt: Path, grid_size: int = 12, episodes: int = 30) -> None:
    """在同一组固定测试随机种子下，严格对比随机策略、启发式智能体与训练后 DQN 的表现。"""
    print(f"\n=== 最终盲测对比 ({episodes} 局固定种子, {grid_size}x{grid_size} 满盘上限 {grid_size * grid_size - 3} 分) ===")

    # 1. 随机策略
    env = SnakeEnv(grid_size=grid_size, obs_mode="features", action_mode="relative")
    rand_scores: list[int] = []
    for ep in range(episodes):
        env.reset(seed=50_000 + ep)
        while True:
            _, _, term, trunc, info = env.step(int(env.action_space.sample()))
            if term or trunc:
                rand_scores.append(int(info["score"]))
                break

    # 2. 内置启发式规则智能体
    heur_scores: list[int] = []
    for ep in range(episodes):
        env.reset(seed=50_000 + ep)
        while True:
            _, _, term, trunc, info = env.step(env.core.heuristic_action())
            if term or trunc:
                heur_scores.append(int(info["score"]))
                break
    env.close()

    # 3. 训练完成的 Dueling Double DQN
    best_agent, meta = DQNAgent.load(best_ckpt, device="auto")
    dqn_eval = evaluate_agent(best_agent, grid_size=grid_size, num_episodes=episodes, seed_offset=50_000)

    r_arr = np.array(rand_scores, dtype=np.float32)
    h_arr = np.array(heur_scores, dtype=np.float32)

    print(f"  [1] 随机探索 (Random)            : 均分 {r_arr.mean():5.2f} ± {r_arr.std():4.2f} | 最高分 {int(r_arr.max()):>3}")
    print(f"  [2] 启发式基线 (Heuristic Rule)  : 均分 {h_arr.mean():5.2f} ± {h_arr.std():4.2f} | 最高分 {int(h_arr.max()):>3}")
    print(
        f"  [3] Dueling Double DQN (Ours)    : 均分 {dqn_eval['mean_score']:5.2f} ± {dqn_eval['std_score']:4.2f} | "
        f"中位数 {dqn_eval['median_score']:5.1f} | 最高分 {dqn_eval['max_score']:>3} | "
        f"平均盘面占满率 {dqn_eval['mean_fill_pct']:4.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 Dueling Double DQN 贪吃蛇模型")
    parser.add_argument("--total-steps", type=int, default=400_000, help="总交互环境步数（默认: 400,000）")
    parser.add_argument("--num-envs", type=int, default=16, help="并行采样环境数量（默认: 16）")
    parser.add_argument("--grid-size", type=int, default=12, help="棋盘边长（默认: 12）")
    parser.add_argument("--eval-interval", type=int, default=25_000, help="确定性评估间隔步数（默认: 25,000）")
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW 学习率（默认: 3e-4）")
    parser.add_argument("--eps-decay", type=int, default=120_000, help="Epsilon 探索退火步数（默认: 120,000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    dqn_cfg = DQNConfig(
        obs_dim=FEATURE_DIM,
        action_dim=3,
        learning_rate=args.lr,
        eps_decay_steps=args.eps_decay,
    )
    summary = train_dqn(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        grid_size=args.grid_size,
        eval_interval=args.eval_interval,
        seed=args.seed,
        dqn_config=dqn_cfg,
    )
    print(
        f"\n[训练完成] 耗时: {summary['elapsed_sec']}s | 最佳评估均分: {summary['best_eval_score']:.2f} | "
        f"最优权重: {summary['best_checkpoint']} | 曲线图: {summary['plot_path']}"
    )
    compare_all_policies(Path(summary["best_checkpoint"]), grid_size=args.grid_size, episodes=30)


if __name__ == "__main__":
    main()
