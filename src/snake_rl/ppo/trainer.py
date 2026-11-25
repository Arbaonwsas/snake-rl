"""GPU 大规模并行 Maskable PPO 训练循环、满盘通关率评估器与报告绘图工具。"""

from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from snake_rl.config import SnakeConfig
from snake_rl.env import SnakeEnv
from snake_rl.ppo.agent import MaskablePPOAgent, PPOConfig
from snake_rl.ppo.vec_env_gpu import TorchSnakeVecEnv


@torch.no_grad()
def evaluate_ppo_gpu_fast(
    agent: MaskablePPOAgent,
    num_episodes: int = 128,
    seed: int = 88_888,
) -> dict[str, Any]:
    """在 GPU 上并行运行 128 局零探索确定性评估，耗时仅 ~0.1 秒且方差极低。"""
    g = agent.config.grid_size
    max_possible_score = g * g - 3
    eval_env = TorchSnakeVecEnv(num_envs=num_episodes, grid_size=g, device=agent.device, seed=seed)
    obs, masks = eval_env.reset_all()

    finished = torch.zeros(num_episodes, device=agent.device, dtype=torch.bool)
    ep_scores = torch.zeros(num_episodes, device=agent.device, dtype=torch.int32)
    ep_wins = torch.zeros(num_episodes, device=agent.device, dtype=torch.bool)

    agent.net.eval()
    # 10x10 满盘最长一局通常不超过 2,500 步，设安全步数上限 4,000 步
    max_eval_steps = g * g * 40
    for _ in range(max_eval_steps):
        with torch.autocast(device_type=agent.device.type, dtype=agent.amp_dtype, enabled=agent.amp_enabled):
            logits, _ = agent.net.forward_logits_and_value(obs, masks)
        actions = logits.argmax(dim=-1)
        obs, _, _, masks, info = eval_env.step(actions)

        newly_done = info["dones"] & (~finished)
        ep_scores = torch.where(newly_done, info["final_scores"], ep_scores)
        ep_wins = torch.where(newly_done, info["won"], ep_wins)
        finished = finished | info["dones"]
        if bool(finished.all().item()):
            break

    # 若极少数环境在 4000 步内仍未结束，记录其当前已得分数
    still_running = ~finished
    if bool(still_running.any().item()):
        cur_sc = (eval_env.length - 3).to(torch.int32)
        ep_scores = torch.where(still_running, cur_sc, ep_scores)

    scores_np = ep_scores.cpu().numpy().astype(np.float32)
    wins_cnt = int(ep_wins.sum().item())
    return {
        "mean_score": float(scores_np.mean()),
        "std_score": float(scores_np.std()),
        "max_score": int(scores_np.max()),
        "max_possible_score": max_possible_score,
        "mean_fill_pct": float(100.0 * (scores_np.mean() + 3.0) / (g * g)),
        "win_rate": float(wins_cnt / num_episodes),
        "win_count": wins_cnt,
    }


def evaluate_ppo_agent(
    agent: MaskablePPOAgent,
    num_episodes: int = 40,
    seed_offset: int = 80_000,
    use_lookahead: bool = True,
) -> dict[str, Any]:
    """在标准独立 CPU `SnakeEnv` 上执行确定性评估，统计真实得分与 100% 满盘完美通关率。"""
    g = agent.config.grid_size
    max_possible_score = g * g - 3
    scfg = SnakeConfig(
        grid_width=g,
        grid_height=g,
        starvation_factor=8,
        min_starvation_steps=300,
        enable_flood_fill=False,
    )
    env = SnakeEnv(grid_size=g, obs_mode="grid", action_mode="relative", snake_config=scfg)

    scores: list[int] = []
    lengths: list[int] = []
    steps_list: list[int] = []
    reasons: Counter[str] = Counter()

    for ep in range(num_episodes):
        env.reset(seed=seed_offset + ep)
        while True:
            act, _, _, _ = agent.predict_from_core(env.core, deterministic=True, use_lookahead=use_lookahead)
            _, _, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                scores.append(int(info["score"]))
                lengths.append(int(info["length"]))
                steps_list.append(int(info["steps"]))
                reasons[str(info["death_reason"])] += 1
                break
    env.close()

    arr_s = np.array(scores, dtype=np.float32)
    total = max(1, num_episodes)
    return {
        "mean_score": float(arr_s.mean()),
        "std_score": float(arr_s.std()),
        "median_score": float(np.median(arr_s)),
        "max_score": int(arr_s.max()),
        "min_score": int(arr_s.min()),
        "max_possible_score": max_possible_score,
        "mean_fill_pct": float(100.0 * np.mean(lengths) / (g * g)),
        "win_rate": float(reasons["win"] / total),
        "win_count": int(reasons["win"]),
        "wall_rate": float(reasons["wall"] / total),
        "self_rate": float(reasons["self"] / total),
        "starvation_rate": float(reasons["starvation"] / total),
        "mean_steps": float(np.mean(steps_list)),
        "scores": scores,
    }


def plot_ppo_report(history: dict[str, list[float]], grid_size: int, output_path: Path) -> None:
    """生成面向满盘通关分析的四宫格可视化图表（PNG）。"""
    max_sc = grid_size * grid_size - 3
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=140)
    fig.patch.set_facecolor("#0f131c")
    for ax in axes.ravel():
        ax.set_facecolor("#161c28")
        ax.tick_params(colors="#92a2bc")
        for spine in ax.spines.values():
            spine.set_color("#2e3a52")
        ax.grid(True, linestyle="--", alpha=0.22, color="#4c6288")

    steps_m = [s / 1e6 for s in history["eval_steps"]]
    tr_steps_m = [s / 1e6 for s in history["iter_steps"]]

    # 1. 得分攀升与满分天花板线
    ax0 = axes[0, 0]
    ax0.axhline(max_sc, color="#38e08c", linestyle=":", linewidth=1.8, label=f"100% Perfect Clear ({max_sc} pts)")
    if tr_steps_m:
        ax0.plot(tr_steps_m, history["train_mean_score"], color="#4ca8ff", alpha=0.5, linewidth=1.5, label="GPU Train Mean Score")
    if steps_m:
        ax0.plot(steps_m, history["eval_mean_score"], color="#38e08c", linewidth=2.5, marker="o", markersize=4, label="Eval Mean Score")
        ax0.plot(steps_m, history["eval_max_score"], color="#fac444", linewidth=1.6, linestyle="--", label="Eval Max Score")
    ax0.set_title(f"Maskable PPO + SE-ResNet Score ({grid_size}x{grid_size} Board)", color="#ecf2fa", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Environment Steps (Millions)", color="#92a2bc")
    ax0.set_ylabel("Score (Apples Eaten)", color="#92a2bc")
    ax0.set_ylim(-2, max_sc + 6)
    ax0.legend(facecolor="#1c2332", edgecolor="#2e3a52", labelcolor="#ecf2fa", fontsize=9)

    # 2. 100% 完美通关率 (Perfect Win Rate %) 与盘面平均占满率 (Fill %)
    ax1 = axes[0, 1]
    if steps_m:
        ax1.plot(steps_m, [w * 100 for w in history["eval_win_rate"]], color="#38e08c", linewidth=2.6, marker="s", markersize=4, label="Perfect Clear Win Rate (%)")
        ax1.plot(steps_m, history["eval_fill_pct"], color="#4ca8ff", linewidth=1.9, linestyle="--", label="Mean Board Fill (%)")
    ax1.set_title("100% Full-Screen Clear Rate & Board Fill %", color="#ecf2fa", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Environment Steps (Millions)", color="#92a2bc")
    ax1.set_ylabel("Percentage (%)", color="#92a2bc")
    ax1.set_ylim(-3, 105)
    ax1.legend(facecolor="#1c2332", edgecolor="#2e3a52", labelcolor="#ecf2fa", fontsize=9)

    # 3. 策略熵 (Policy Entropy) 与 KL 散度
    ax2 = axes[1, 0]
    if tr_steps_m:
        ax2.plot(tr_steps_m, history["entropy"], color="#fac444", linewidth=1.8, label="Policy Entropy")
        ax2.plot(tr_steps_m, [k * 10 for k in history["approx_kl"]], color="#ff7a88", alpha=0.75, linewidth=1.4, label="Approx KL (x10)")
    ax2.set_title("Exploration Entropy & Policy Stability", color="#ecf2fa", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Environment Steps (Millions)", color="#92a2bc")
    ax2.set_ylabel("Metric Value", color="#92a2bc")
    ax2.legend(facecolor="#1c2332", edgecolor="#2e3a52", labelcolor="#ecf2fa", fontsize=9)

    # 4. Critic 价值函数损失 (Value Loss)
    ax3 = axes[1, 1]
    if tr_steps_m:
        ax3.plot(tr_steps_m, history["v_loss"], color="#4ca8ff", linewidth=1.8, label="Critic Value Loss")
    ax3.set_title("Critic Value Estimation Loss", color="#ecf2fa", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Environment Steps (Millions)", color="#92a2bc")
    ax3.set_ylabel("Loss", color="#92a2bc")
    ax3.legend(facecolor="#1c2332", edgecolor="#2e3a52", labelcolor="#ecf2fa", fontsize=9)

    fig.tight_layout(pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def train_ppo(
    total_steps: int = 25_000_000,
    eval_interval_iters: int = 20,
    eval_episodes: int = 30,
    seed: int = 42,
    ppo_config: PPOConfig | None = None,
    checkpoint_dir: Path = Path("checkpoints"),
    log_dir: Path = Path("runs"),
    output_dir: Path = Path("outputs"),
) -> dict[str, Any]:
    """在 GPU 上运行大规模向量化 Maskable PPO 训练，冲击 100% 满盘完美通关。"""
    cfg = ppo_config or PPOConfig()
    agent = MaskablePPOAgent(config=cfg, device="auto", seed=seed)
    device = agent.device

    vec_env = TorchSnakeVecEnv(
        num_envs=cfg.num_envs,
        grid_size=cfg.grid_size,
        device=device,
        seed=seed,
    )

    steps_per_iter = cfg.num_envs * cfg.rollout_steps
    num_iters = max(1, total_steps // steps_per_iter)
    actual_total_steps = num_iters * steps_per_iter

    # 预分配显存常驻 Rollout 缓冲区，消除每轮张量重新分配开销
    T, N = cfg.rollout_steps, cfg.num_envs
    H = W = cfg.grid_size
    obs_buf = torch.zeros((T, N, cfg.in_channels, H, W), device=device, dtype=torch.float32)
    masks_buf = torch.zeros((T, N, cfg.action_dim), device=device, dtype=torch.bool)
    actions_buf = torch.zeros((T, N), device=device, dtype=torch.long)
    logprobs_buf = torch.zeros((T, N), device=device, dtype=torch.float32)
    rewards_buf = torch.zeros((T, N), device=device, dtype=torch.float32)
    dones_buf = torch.zeros((T, N), device=device, dtype=torch.float32)
    values_buf = torch.zeros((T, N), device=device, dtype=torch.float32)

    ep_dones_buf = torch.zeros((T, N), device=device, dtype=torch.bool)
    ep_scores_buf = torch.zeros((T, N), device=device, dtype=torch.int32)
    ep_wins_buf = torch.zeros((T, N), device=device, dtype=torch.bool)

    next_obs, next_masks = vec_env.reset_all()

    run_name = f"maskable_ppo_{H}x{W}_{int(time.time())}"
    writer = SummaryWriter(log_dir=str(log_dir / run_name))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    recent_train_scores: deque[float] = deque(maxlen=2000)
    recent_train_wins: deque[float] = deque(maxlen=2000)

    history: dict[str, list[float]] = {
        "iter_steps": [],
        "train_mean_score": [],
        "train_win_rate": [],
        "pg_loss": [],
        "v_loss": [],
        "entropy": [],
        "approx_kl": [],
        "eval_steps": [],
        "eval_mean_score": [],
        "eval_max_score": [],
        "eval_fill_pct": [],
        "eval_win_rate": [],
    }

    best_composite_metric = -1.0  # 综合 (win_rate * 100 + mean_score) 选最优权重
    best_win_rate = 0.0
    best_mean_score = 0.0
    total_wins_in_training = 0
    t_start = time.perf_counter()

    print(
        f"[Maskable PPO] 设备: {device} (AMP={agent.amp_dtype if agent.amp_enabled else 'FP32'}) | "
        f"并行棋盘: {N} 个 ({H}x{W}, 满分 {H * W - 3}) | "
        f"总迭代: {num_iters} ({actual_total_steps:,} 步)"
    )

    for it in range(1, num_iters + 1):
        frac = (it - 1.0) / float(max(1, num_iters - 1))
        cur_lr = cfg.lr_start + (cfg.lr_end - cfg.lr_start) * frac
        cur_ent = cfg.ent_coef_start + (cfg.ent_coef_end - cfg.ent_coef_start) * frac
        agent.set_learning_rate(cur_lr)

        # 1. GPU 原生高速 Rollout 收集
        agent.net.eval()
        with torch.no_grad():
            for step in range(T):
                obs_buf[step] = next_obs
                masks_buf[step] = next_masks

                with torch.autocast(device_type=device.type, dtype=agent.amp_dtype, enabled=agent.amp_enabled):
                    action, logprob, _, value = agent.net.get_action_and_value(next_obs, next_masks)

                actions_buf[step] = action
                logprobs_buf[step] = logprob
                values_buf[step] = value

                next_obs, rewards, _, next_masks, info = vec_env.step(action)
                rewards_buf[step] = rewards
                dones_buf[step] = info["dones"].to(torch.float32)
                ep_dones_buf[step] = info["dones"]
                ep_scores_buf[step] = info["final_scores"]
                ep_wins_buf[step] = info["won"]

            # 整轮 64 步跑完后仅执行 1 次主机同步汇总统计
            if bool(ep_dones_buf.any().item()):
                done_scores = ep_scores_buf[ep_dones_buf].cpu().tolist()
                done_wins = ep_wins_buf[ep_dones_buf].float().cpu().tolist()
                recent_train_scores.extend(done_scores)
                recent_train_wins.extend(done_wins)
                total_wins_in_training += int(ep_wins_buf.sum().item())

            # 计算末步 Bootstrap 价值 V(s_{T})
            with torch.autocast(device_type=device.type, dtype=agent.amp_dtype, enabled=agent.amp_enabled):
                _, next_value = agent.net.forward_logits_and_value(next_obs, next_masks)

            # 2. 并行计算 GAE (Generalized Advantage Estimation)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = torch.zeros(N, device=device, dtype=torch.float32)
            for t in reversed(range(T)):
                nextnonterminal = 1.0 - dones_buf[t]
                nextvalues = next_value if t == T - 1 else values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

        # 3. 展平 Batch 并执行多轮 Minibatch Clipped PPO 更新
        update_stats = agent.update_rollout(
            b_obs=obs_buf.flatten(0, 1),
            b_masks=masks_buf.flatten(0, 1),
            b_actions=actions_buf.flatten(0, 1),
            b_logprobs=logprobs_buf.flatten(0, 1),
            b_advantages=advantages.flatten(0, 1),
            b_returns=returns.flatten(0, 1),
            b_values=values_buf.flatten(0, 1),
            ent_coef=cur_ent,
        )

        env_steps = it * steps_per_iter
        tr_sc = float(np.mean(recent_train_scores)) if recent_train_scores else 0.0
        tr_wr = float(np.mean(recent_train_wins)) if recent_train_wins else 0.0

        history["iter_steps"].append(float(env_steps))
        history["train_mean_score"].append(tr_sc)
        history["train_win_rate"].append(tr_wr)
        history["pg_loss"].append(update_stats["pg_loss"])
        history["v_loss"].append(update_stats["v_loss"])
        history["entropy"].append(update_stats["entropy"])
        history["approx_kl"].append(update_stats["approx_kl"])

        writer.add_scalar("ppo_train/mean_score", tr_sc, env_steps)
        writer.add_scalar("ppo_train/win_rate", tr_wr, env_steps)
        writer.add_scalar("ppo_train/entropy", update_stats["entropy"], env_steps)
        writer.add_scalar("ppo_train/v_loss", update_stats["v_loss"], env_steps)

        # 4. 定期执行 128 局零探索 GPU 并行确定性评估
        if it % eval_interval_iters == 0 or it == num_iters or it == 1:
            eval_res = evaluate_ppo_gpu_fast(agent, num_episodes=128)
            elapsed = time.perf_counter() - t_start
            sps = env_steps / max(1e-6, elapsed)

            m_sc = eval_res["mean_score"]
            w_rt = eval_res["win_rate"]
            max_sc = eval_res["max_score"]
            fill_pct = eval_res["mean_fill_pct"]

            history["eval_steps"].append(float(env_steps))
            history["eval_mean_score"].append(m_sc)
            history["eval_max_score"].append(float(max_sc))
            history["eval_fill_pct"].append(fill_pct)
            history["eval_win_rate"].append(w_rt)

            writer.add_scalar("ppo_eval/mean_score", m_sc, env_steps)
            writer.add_scalar("ppo_eval/win_rate", w_rt, env_steps)
            writer.add_scalar("ppo_eval/fill_pct", fill_pct, env_steps)

            meta = {
                "iter": it,
                "env_steps": env_steps,
                "grid_size": cfg.grid_size,
                "eval_mean_score": m_sc,
                "eval_max_score": max_sc,
                "eval_win_rate": w_rt,
                "eval_fill_pct": fill_pct,
                "total_wins_in_training": total_wins_in_training,
            }
            agent.save(checkpoint_dir / "latest_ppo.pt", metadata=meta)

            composite = w_rt * 50.0 + m_sc
            is_best = composite > best_composite_metric
            if is_best:
                best_composite_metric = composite
                best_win_rate = w_rt
                best_mean_score = m_sc
                agent.save(checkpoint_dir / "best_ppo.pt", metadata=meta)

            tag = " ★ NEW BEST" if is_best else ""
            print(
                f"Iter {it:>3}/{num_iters} ({env_steps / 1e6:5.2f}M) | "
                f"Train: {tr_sc:5.1f} (Win {tr_wr * 100:4.1f}%, 累计通关 {total_wins_in_training:,} 局) | "
                f"Eval: {m_sc:5.2f}/{H * W - 3} (Max {max_sc:>2}, Fill {fill_pct:5.1f}%, 通关率 {w_rt * 100:5.1f}%) | "
                f"Ent {update_stats['entropy']:.3f} | {sps:,.0f} sps{tag}"
            )

    writer.close()

    hist_path = output_dir / "ppo_training_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = output_dir / "ppo_training_curves.png"
    plot_ppo_report(history, grid_size=cfg.grid_size, output_path=plot_path)

    return {
        "best_mean_score": best_mean_score,
        "best_win_rate": best_win_rate,
        "total_wins_in_training": total_wins_in_training,
        "total_steps": actual_total_steps,
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "best_checkpoint": str(checkpoint_dir / "best_ppo.pt"),
        "plot_path": str(plot_path),
    }
