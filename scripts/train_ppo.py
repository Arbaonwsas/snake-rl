#!/usr/bin/env python3
"""启动 RTX 4060 大规模并行 Maskable PPO + SE-ResNet 训练，冲击 100% 满盘彻底通关。"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from snake_rl.config import SnakeConfig
from snake_rl.dqn.agent import DQNAgent
from snake_rl.env import SnakeEnv
from snake_rl.ppo.agent import MaskablePPOAgent, PPOConfig
from snake_rl.ppo.trainer import evaluate_ppo_agent, train_ppo


def run_grand_benchmark(ppo_ckpt: Path, grid_size: int = 10, episodes: int = 100) -> None:
    """在 100 局全新随机种子下，全面对比启发式规则、DQN 与 Maskable PPO 的满分通关率与极限得分。"""
    max_pts = grid_size * grid_size - 3
    print(f"\n{'=' * 78}")
    print(f" 终极百局盲测 ({episodes} 局独立测试种子, {grid_size}x{grid_size} 棋盘, 100% 霸屏满分 = {max_pts} 分)")
    print(f"{'=' * 78}")

    scfg = SnakeConfig(grid_width=grid_size, grid_height=grid_size, starvation_factor=7, min_starvation_steps=200)
    env = SnakeEnv(grid_size=grid_size, obs_mode="features", action_mode="relative", snake_config=scfg)

    # 1. 启发式 + Flood-Fill 基线
    h_scores, h_wins = [], 0
    for ep in range(episodes):
        env.reset(seed=200_000 + ep)
        while True:
            _, _, term, trunc, info = env.step(env.core.heuristic_action())
            if term or trunc:
                h_scores.append(int(info["score"]))
                if info["death_reason"] == "win":
                    h_wins += 1
                break

    # 2. 此前训练的 24 维特征 Dueling Double DQN（跨盘面泛化对比）
    dqn_ckpt = Path("checkpoints/best_dqn.pt")
    dqn_line = ""
    if dqn_ckpt.is_file():
        dqn_agent, _ = DQNAgent.load(dqn_ckpt, device="auto")
        d_scores, d_wins = [], 0
        for ep in range(episodes):
            obs, _ = env.reset(seed=200_000 + ep)
            while True:
                act, _ = dqn_agent.predict(obs)
                obs, _, term, trunc, info = env.step(act)
                if term or trunc:
                    d_scores.append(int(info["score"]))
                    if info["death_reason"] == "win":
                        d_wins += 1
                    break
        d_arr = np.array(d_scores, dtype=np.float32)
        dqn_line = (
            f"  [2] Dueling Double DQN (24D MLP)   : 均分 {d_arr.mean():5.2f} ± {d_arr.std():4.2f} | "
            f"最高 {int(d_arr.max()):>2}/{max_pts} | 满盘通关率 {100.0 * d_wins / episodes:5.1f}%"
        )
    env.close()

    # 3. Maskable PPO + SE-ResNet
    ppo_agent, _ = MaskablePPOAgent.load(ppo_ckpt, device="auto")
    ppo_res = evaluate_ppo_agent(ppo_agent, num_episodes=episodes, seed_offset=200_000)

    h_arr = np.array(h_scores, dtype=np.float32)
    p_scores = np.array(ppo_res["scores"], dtype=np.float32)
    near_full_rate = float(np.mean(p_scores >= max_pts * 0.90) * 100.0)

    print(
        f"  [1] 启发式 + Flood-Fill 基线       : 均分 {h_arr.mean():5.2f} ± {h_arr.std():4.2f} | "
        f"最高 {int(h_arr.max()):>2}/{max_pts} | 满盘通关率 {100.0 * h_wins / episodes:5.1f}%"
    )
    if dqn_line:
        print(dqn_line)
    print(
        f"  [3] Maskable PPO + SE-ResNet (Ours): 均分 {ppo_res['mean_score']:5.2f} ± {ppo_res['std_score']:4.2f} | "
        f"中位数 {ppo_res['median_score']:5.1f} | 最高 {ppo_res['max_score']:>2}/{max_pts} | "
        f"平均占满 {ppo_res['mean_fill_pct']:5.1f}% | ≥90%霸屏率 {near_full_rate:5.1f}% | "
        f"★ 100% 满分通关率: {ppo_res['win_rate'] * 100:5.1f}% ({ppo_res['win_count']}/{episodes} 局)"
    )
    print(f"{'=' * 78}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 4060 大规模并行训练 Maskable PPO 贪吃蛇满盘模型")
    parser.add_argument("--total-steps", type=int, default=16_000_000, help="总环境步数（默认: 1,600 万步，等价 6,400 万步非对称样本）")
    parser.add_argument("--grid-size", type=int, default=10, help="偶数棋盘边长（默认黄金标准: 10，满分 97）")
    parser.add_argument("--num-envs", type=int, default=1024, help="GPU 并行环境数（默认: 1024）")
    parser.add_argument("--rollout-steps", type=int, default=128, help="单轮 Rollout 长度（默认: 128）")
    parser.add_argument("--channels", type=int, default=48, help="SE-ResNet 通道宽度（默认: 48）")
    parser.add_argument("--blocks", type=int, default=3, help="SE-ResNet 残差块层数（默认: 3）")
    parser.add_argument("--eval-interval", type=int, default=10, help="每隔多少个迭代评估一次（默认: 10）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if args.grid_size % 2 != 0:
        print(f"[警告] 奇数边长 ({args.grid_size}) 不存在闭合哈密顿回路，建议使用偶数边长（如 8, 10, 12）以实现 100% 通关！")

    cfg = PPOConfig(
        grid_size=args.grid_size,
        channels=args.channels,
        num_blocks=args.blocks,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
    )
    summary = train_ppo(
        total_steps=args.total_steps,
        eval_interval_iters=args.eval_interval,
        eval_episodes=30,
        seed=args.seed,
        ppo_config=cfg,
    )
    print(
        f"\n[PPO 训练完成] 耗时: {summary['elapsed_sec']}s | "
        f"训练期累计达成 100% 满分通关: {summary['total_wins_in_training']:,} 局 | "
        f"评估最佳均分: {summary['best_mean_score']:.2f}/{args.grid_size ** 2 - 3} "
        f"(通关率 {summary['best_win_rate'] * 100:.1f}%)"
    )
    run_grand_benchmark(Path(summary["best_checkpoint"]), grid_size=args.grid_size, episodes=100)


if __name__ == "__main__":
    main()
