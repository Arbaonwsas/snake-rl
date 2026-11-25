"""多环境并行 Dueling Double DQN 训练器、确定性评估器与收敛曲线绘制模块。"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from snake_rl.config import RewardConfig, SnakeConfig
from snake_rl.dqn.agent import DQNAgent, DQNConfig
from snake_rl.dqn.replay_buffer import NStepAccumulator, ReplayBuffer
from snake_rl.env import FEATURE_DIM, SnakeEnv


def evaluate_agent(
    agent: DQNAgent,
    grid_size: int = 12,
    num_episodes: int = 20,
    seed_offset: int = 10_000,
) -> dict[str, Any]:
    """在零随机探索 (epsilon=0) 下对智能体执行多局确定性评估。"""
    env = SnakeEnv(grid_size=grid_size, obs_mode="features", action_mode="relative")
    scores: list[int] = []
    returns: list[float] = []
    lengths: list[int] = []
    steps_list: list[int] = []
    reasons: Counter[str] = Counter()

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        while True:
            act, _ = agent.predict(obs)
            obs, _, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                scores.append(int(info["score"]))
                returns.append(float(info["episode_return"]))
                lengths.append(int(info["length"]))
                steps_list.append(int(info["steps"]))
                reasons[str(info["death_reason"])] += 1
                break
    env.close()

    arr_scores = np.array(scores, dtype=np.float32)
    total_ep = max(1, num_episodes)
    return {
        "mean_score": float(arr_scores.mean()),
        "std_score": float(arr_scores.std()),
        "max_score": int(arr_scores.max()),
        "min_score": int(arr_scores.min()),
        "median_score": float(np.median(arr_scores)),
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(steps_list)),
        "mean_fill_pct": float(100.0 * np.mean(lengths) / (grid_size * grid_size)),
        "wall_rate": reasons["wall"] / total_ep,
        "self_rate": reasons["self"] / total_ep,
        "starvation_rate": reasons["starvation"] / total_ep,
        "win_rate": reasons["win"] / total_ep,
        "scores": scores,
    }


def plot_training_Report(
    history: dict[str, list[float]],
    output_path: Path,
    heuristic_baseline: float | None = None,
) -> None:
    """绘制四宫格训练收敛曲线报告图（PNG）。"""
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
    fig.patch.set_facecolor("#0f131c")
    for ax in axes.ravel():
        ax.set_facecolor("#161c28")
        ax.tick_params(colors="#92a2bc")
        for spine in ax.spines.values():
            spine.set_color("#2e3a52")
        ax.grid(True, linestyle="--", alpha=0.22, color="#4c6288")

    # 1. 得分收敛曲线 (Eval Mean / Max & Train Rolling Score)
    ax0 = axes[0, 0]
    if history["train_steps"]:
        ax0.plot(
            history["train_steps"],
            history["train_rolling_score"],
            color="#4ca8ff",
            alpha=0.55,
            linewidth=1.5,
            label="Train Rolling Score (100 ep)",
        )
    if history["eval_steps"]:
        ax0.plot(
            history["eval_steps"],
            history["eval_mean_score"],
            color="#38e08c",
            linewidth=2.4,
            marker="o",
            markersize=4,
            label="Eval Mean Score (greedy)",
        )
        ax0.plot(
            history["eval_steps"],
            history["eval_max_score"],
            color="#fac444",
            linewidth=1.5,
            linestyle="--",
            label="Eval Max Score",
        )
    if heuristic_baseline is not None:
        ax0.axhline(
            heuristic_baseline,
            color="#ff5c6c",
            linestyle=":",
            linewidth=1.8,
            label=f"Heuristic Baseline ({heuristic_baseline:.1f})",
        )
    ax0.set_title("Episode Score Convergence (12x12 Board)", color="#ecf2fa", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Environment Steps", color="#92a2bc")
    ax0.set_ylabel("Apples Eaten (Score)", color="#92a2bc")
    ax0.legend(facecolor="#1c2332", edgecolor="#2e3a52", labelcolor="#ecf2fa", fontsize=9)

    # 2. TD Huber Loss 演化
    ax1 = axes[0, 1]
    if history["update_steps"]:
        ax1.plot(history["update_steps"], history["loss"], color="#ff7a88", linewidth=1.6, label="Smooth L1 (Huber) Loss")
    ax1.set_title("TD Optimization Loss", color="#ecf2fa", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Environment Steps", color="#92a2bc")
    ax1.set_ylabel("Loss", color="#92a2bc")
    ax1.legend(facecolor="#1c2332", edgecolor="#2e3a52", labelcolor="#ecf2fa", fontsize=9)

    # 3. 预测 Q 值与探索率 Epsilon 演化
    ax2 = axes[1, 0]
    if history["update_steps"]:
        ax2.plot(history["update_steps"], history["mean_q"], color="#4ca8ff", linewidth=2.0, label="Mean Predicted Q(s, a)")
    ax2.set_title("Predicted Action-Value Q(s, a)", color="#ecf2fa", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Environment Steps", color="#92a2bc")
    ax2.set_ylabel("Mean Q Value", color="#92a2bc")
    ax2.legend(facecolor="#1c2332", edgecolor="#2e3a52", labelcolor="#ecf2fa", fontsize=9, loc="upper left")

    # 4. 评估回合终止原因构成 (Wall / Self / Starvation / Win)
    ax3 = axes[1, 1]
    if history["eval_steps"]:
        st = history["eval_steps"]
        ax3.plot(st, [v * 100 for v in history["eval_wall_rate"]], color="#ff5c6c", linewidth=1.8, label="Wall Collision %")
        ax3.plot(st, [v * 100 for v in history["eval_self_rate"]], color="#fac444", linewidth=1.8, label="Self Collision %")
        ax3.plot(st, [v * 100 for v in history["eval_starv_rate"]], color="#92a2bc", linewidth=1.6, linestyle="--", label="Starvation %")
        if any(v > 0 for v in history["eval_win_rate"]):
            ax3.plot(st, [v * 100 for v in history["eval_win_rate"]], color="#38e08c", linewidth=2.0, label="Perfect Win %")
    ax3.set_title("Evaluation Termination Breakdown (%)", color="#ecf2fa", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Environment Steps", color="#92a2bc")
    ax3.set_ylabel("Percentage (%)", color="#92a2bc")
    ax3.set_ylim(-3, 103)
    ax3.legend(facecolor="#1c2332", edgecolor="#2e3a52", labelcolor="#ecf2fa", fontsize=9)

    fig.tight_layout(pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def train_dqn(
    total_steps: int = 360_000,
    num_envs: int = 16,
    grid_size: int = 12,
    updates_per_round: int = 4,
    eval_interval: int = 20_000,
    eval_episodes: int = 20,
    seed: int = 42,
    dqn_config: DQNConfig | None = None,
    reward_config: RewardConfig | None = None,
    checkpoint_dir: Path = Path("checkpoints"),
    log_dir: Path = Path("runs"),
    output_dir: Path = Path("outputs"),
) -> dict[str, Any]:
    """执行多环境并行 N-step Dueling Double DQN 完整训练流程。"""
    cfg = dqn_config or DQNConfig(obs_dim=FEATURE_DIM, action_dim=3)
    rcfg = reward_config or RewardConfig()
    scfg = SnakeConfig(grid_width=grid_size, grid_height=grid_size, enable_flood_fill=True)

    agent = DQNAgent(config=cfg, device="auto", seed=seed)
    replay_buf = ReplayBuffer(capacity=cfg.buffer_capacity, obs_dim=cfg.obs_dim, seed=seed)
    nstep_accums = [NStepAccumulator(n_step=cfg.n_step, gamma=cfg.gamma) for _ in range(num_envs)]

    # 创建并行无头训练环境组
    envs = [
        SnakeEnv(
            grid_size=grid_size,
            obs_mode="features",
            action_mode="relative",
            snake_config=scfg,
            reward_config=rcfg,
        )
        for _ in range(num_envs)
    ]
    obs_batch = np.zeros((num_envs, cfg.obs_dim), dtype=np.float32)
    for i, env in enumerate(envs):
        obs_batch[i], _ = env.reset(seed=seed + i * 101)

    run_name = f"dueling_ddqn_{grid_size}x{grid_size}_{int(time.time())}"
    writer = SummaryWriter(log_dir=str(log_dir / run_name))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    recent_scores: deque[int] = deque(maxlen=100)
    recent_losses: deque[float] = deque(maxlen=200)
    recent_qs: deque[float] = deque(maxlen=200)

    history: dict[str, list[float]] = {
        "train_steps": [],
        "train_rolling_score": [],
        "update_steps": [],
        "loss": [],
        "mean_q": [],
        "eval_steps": [],
        "eval_mean_score": [],
        "eval_max_score": [],
        "eval_wall_rate": [],
        "eval_self_rate": [],
        "eval_starv_rate": [],
        "eval_win_rate": [],
    }

    best_eval_score = -1.0
    env_steps = 0
    grad_updates = 0
    next_eval_step = eval_interval
    next_log_step = 5_000
    t_start = time.perf_counter()

    print(
        f"[DQN Trainer] 设备: {agent.device} | 并行环境: {num_envs} | "
        f"棋盘: {grid_size}x{grid_size} | 总步数: {total_steps:,} | N-Step: {cfg.n_step}"
    )

    while env_steps < total_steps:
        eps = agent.epsilon_by_step(env_steps)
        actions = agent.select_actions_batch(obs_batch, epsilon=eps)

        for i, env in enumerate(envs):
            act_i = int(actions[i])
            next_obs, reward, terminated, truncated, info = env.step(act_i)

            # 推入 N 步累加器并将就绪样本写入 ReplayBuffer
            ready_transitions = nstep_accums[i].append(
                obs=obs_batch[i].copy(),
                action=act_i,
                reward=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
            )
            for s0, a0, r_n, s_n, term_n, gamma_n in ready_transitions:
                replay_buf.add(s0, a0, r_n, s_n, term_n, gamma_n)

            if terminated or truncated:
                recent_scores.append(int(info["score"]))
                obs_batch[i], _ = env.reset()
            else:
                obs_batch[i] = next_obs

        env_steps += num_envs

        # 经验池预热完成后执行小批量 GPU 更新
        if len(replay_buf) >= cfg.warmup_steps:
            for _ in range(updates_per_round):
                batch = replay_buf.sample(cfg.batch_size, agent.device)
                metrics = agent.update(batch)
                grad_updates += 1
                recent_losses.append(metrics["loss"])
                recent_qs.append(metrics["mean_q"])

        # 周期性记录训练滑动指标
        if env_steps >= next_log_step:
            next_log_step += 5_000
            if recent_scores:
                rolling_sc = float(np.mean(recent_scores))
                history["train_steps"].append(float(env_steps))
                history["train_rolling_score"].append(rolling_sc)
                writer.add_scalar("train/rolling_score_100", rolling_sc, env_steps)
            writer.add_scalar("train/epsilon", eps, env_steps)
            if recent_losses:
                mean_l = float(np.mean(recent_losses))
                mean_q = float(np.mean(recent_qs))
                history["update_steps"].append(float(env_steps))
                history["loss"].append(mean_l)
                history["mean_q"].append(mean_q)
                writer.add_scalar("train/loss", mean_l, env_steps)
                writer.add_scalar("train/mean_q", mean_q, env_steps)

        # 周期性执行无探索确定性评估与最优模型保存
        if env_steps >= next_eval_step or env_steps >= total_steps:
            next_eval_step += eval_interval
            eval_res = evaluate_agent(agent, grid_size=grid_size, num_episodes=eval_episodes)
            elapsed = time.perf_counter() - t_start
            sps = env_steps / max(1e-6, elapsed)

            mean_sc = eval_res["mean_score"]
            max_sc = eval_res["max_score"]
            history["eval_steps"].append(float(env_steps))
            history["eval_mean_score"].append(mean_sc)
            history["eval_max_score"].append(float(max_sc))
            history["eval_wall_rate"].append(eval_res["wall_rate"])
            history["eval_self_rate"].append(eval_res["self_rate"])
            history["eval_starv_rate"].append(eval_res["starvation_rate"])
            history["eval_win_rate"].append(eval_res["win_rate"])

            writer.add_scalar("eval/mean_score", mean_sc, env_steps)
            writer.add_scalar("eval/max_score", max_sc, env_steps)
            writer.add_scalar("eval/mean_fill_pct", eval_res["mean_fill_pct"], env_steps)

            meta = {
                "env_steps": env_steps,
                "grad_updates": grad_updates,
                "grid_size": grid_size,
                "eval_mean_score": mean_sc,
                "eval_max_score": max_sc,
                "reward_config": asdict(rcfg),
            }
            agent.save(checkpoint_dir / "latest_dqn.pt", metadata=meta)
            is_best = mean_sc > best_eval_score
            if is_best:
                best_eval_score = mean_sc
                agent.save(checkpoint_dir / "best_dqn.pt", metadata=meta)

            tag = " ★ NEW BEST" if is_best else ""
            loss_disp = float(np.mean(recent_losses)) if recent_losses else 0.0
            q_disp = float(np.mean(recent_qs)) if recent_qs else 0.0
            print(
                f"Step {env_steps:>7,}/{total_steps:,} | eps {eps:.2f} | "
                f"Eval Score: {mean_sc:5.2f} ± {eval_res['std_score']:4.1f} (Max {max_sc:>3}) | "
                f"Fill: {eval_res['mean_fill_pct']:4.1f}% | Loss: {loss_disp:.4f} | Q: {q_disp:5.1f} | "
                f"{sps:,.0f} sps{tag}"
            )

    for env in envs:
        env.close()
    writer.close()

    # 保存历史日志与收敛曲线报告图
    metrics_path = output_dir / "dqn_training_history.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = output_dir / "dqn_training_curves.png"
    plot_training_Report(history, plot_path, heuristic_baseline=40.1)

    total_time = time.perf_counter() - t_start
    return {
        "best_eval_score": best_eval_score,
        "total_steps": env_steps,
        "grad_updates": grad_updates,
        "elapsed_sec": round(total_time, 2),
        "best_checkpoint": str(checkpoint_dir / "best_dqn.pt"),
        "plot_path": str(plot_path),
    }
