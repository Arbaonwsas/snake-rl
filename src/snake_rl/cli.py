"""贪吃蛇交互式可视化启动器（支持人工试玩、Maskable PPO 满盘宗师模型、DQN 模型与启发式 AI 实时切换）。"""

from __future__ import annotations

import argparse
from pathlib import Path
import pygame

from snake_rl.config import RenderConfig, RewardConfig, SnakeConfig
from snake_rl.core import Direction
from snake_rl.dqn.agent import DQNAgent
from snake_rl.env import SnakeEnv
from snake_rl.ppo.agent import MaskablePPOAgent


FPS_LEVELS = [5, 10, 15, 25, 45, 90, 240]
DEFAULT_DQN_CKPT = Path("checkpoints/best_dqn.pt")
PRETRAINED_DQN_CKPT = Path("pretrained/best_dqn.pt")
DEFAULT_PPO_CKPT = Path("checkpoints/best_ppo.pt")
PRETRAINED_PPO_CKPT = Path("pretrained/best_ppo.pt")

KEY_TO_DIRECTION: dict[int, Direction] = {
    pygame.K_UP: Direction.UP,
    pygame.K_w: Direction.UP,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_d: Direction.RIGHT,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_a: Direction.LEFT,
}


def _direction_to_relative_action(curr_dir: Direction, target_dir: Direction) -> int:
    """将人类输入的绝对目标方向转换为环境的相对动作 (0=直行, 1=左转, 2=右转)。"""
    diff = (int(target_dir) - int(curr_dir)) % 4
    if diff == 0 or diff == 2:
        return 0
    if diff == 3:
        return 1
    return 2


def _try_load_dqn(checkpoint: Path) -> DQNAgent | None:
    target = checkpoint if checkpoint.is_file() else (PRETRAINED_DQN_CKPT if checkpoint == DEFAULT_DQN_CKPT and PRETRAINED_DQN_CKPT.is_file() else checkpoint)
    if target.is_file():
        agent, meta = DQNAgent.load(target, device="auto")
        sc = meta.get("eval_mean_score")
        sc_str = f" (均分: {sc:.1f})" if isinstance(sc, (int, float)) else ""
        print(f"[Snake RL] 已加载 DQN 权重: {target}{sc_str}")
        return agent
    return None


def _try_load_ppo(checkpoint: Path) -> MaskablePPOAgent | None:
    target = checkpoint if checkpoint.is_file() else (PRETRAINED_PPO_CKPT if checkpoint == DEFAULT_PPO_CKPT and PRETRAINED_PPO_CKPT.is_file() else checkpoint)
    if target.is_file():
        agent, meta = MaskablePPOAgent.load(target, device="auto")
        wr = meta.get("eval_win_rate")
        sc = meta.get("eval_mean_score")
        extra = ""
        if isinstance(sc, (int, float)) and isinstance(wr, (int, float)):
            extra = f" (棋盘 {agent.config.grid_size}x{agent.config.grid_size}, 均分: {sc:.1f}, 满分通关率: {wr * 100:.1f}%)"
        print(f"[Snake RL] 已加载 Maskable PPO 权重: {target}{extra}")
        return agent
    return None


def save_ppo_win_showcase(
    ppo_checkpoint: Path,
    almost_full_path: Path,
    perfect_win_path: Path,
    max_search_episodes: int = 30,
) -> dict[str, int | float | bool]:
    """运行训练好的 PPO 智能体，自动截取【临门一脚满屏盘龙瞬间】与【100% 满分通关结算画面】双图。"""
    agent = _try_load_ppo(ppo_checkpoint)
    if agent is None:
        raise FileNotFoundError(f"未找到 PPO 检查点: {ppo_checkpoint}")

    g = agent.config.grid_size
    max_score = g * g - 3
    cell_px = 52 if g <= 10 else 44
    scfg = SnakeConfig(grid_width=g, grid_height=g, starvation_factor=15, min_starvation_steps=500)
    rcfg = RenderConfig(cell_size=cell_px)

    env = SnakeEnv(
        grid_size=g,
        obs_mode="features",
        action_mode="relative",
        render_mode="rgb_array",
        snake_config=scfg,
        render_config=rcfg,
    )

    best_almost_surf: pygame.Surface | None = None
    best_end_surf: pygame.Surface | None = None
    best_ep_score = -1
    best_ep_steps = 0
    achieved_win = False

    for ep in range(max_search_episodes):
        env.reset(seed=90_000 + ep)
        ep_pre_win_surf: pygame.Surface | None = None
        while True:
            act, probs, val, _ = agent.predict_from_core(env.core, deterministic=True)
            env.last_action_probs = probs
            env.last_state_value = val

            # 当距离满分仅差 1~2 颗苹果时，截取无遮罩的满屏巨蟒震撼走位图！
            if env.core.score >= max(best_ep_score, max_score - 2):
                env.render(mode_label="PPO MASTER", fps_override=60)
                ep_pre_win_surf = env._renderer.surface.copy()

            _, _, terminated, truncated, info = env.step(act)
            if terminated or truncated:
                sc = int(info["score"])
                if sc > best_ep_score:
                    best_ep_score = sc
                    best_ep_steps = int(info["steps"])
                    env.render(mode_label="PPO MASTER", fps_override=60)
                    best_end_surf = env._renderer.surface.copy()
                    if ep_pre_win_surf is not None:
                        best_almost_surf = ep_pre_win_surf
                if info["death_reason"] == "win":
                    achieved_win = True
                break

        if achieved_win:
            break

    almost_full_path.parent.mkdir(parents=True, exist_ok=True)
    perfect_win_path.parent.mkdir(parents=True, exist_ok=True)
    if best_almost_surf is not None:
        pygame.image.save(best_almost_surf, str(almost_full_path))
    elif best_end_surf is not None:
        pygame.image.save(best_end_surf, str(almost_full_path))

    if best_end_surf is not None:
        pygame.image.save(best_end_surf, str(perfect_win_path))

    env.close()
    return {
        "achieved_win": achieved_win,
        "score": best_ep_score,
        "max_score": max_score,
        "steps": best_ep_steps,
    }


def save_preview_image(
    output_path: Path,
    grid_size: int = 12,
    steps: int = 45,
    checkpoint: Path | None = None,
) -> None:
    """离屏运行智能体并导出单张预览截图。"""
    env = SnakeEnv(grid_size=grid_size, obs_mode="features", action_mode="relative", render_mode="rgb_array")
    dqn_agent = _try_load_dqn(checkpoint) if checkpoint is not None else None
    obs, _ = env.reset(seed=42)

    for _ in range(steps):
        if dqn_agent is not None:
            act, q_vals = dqn_agent.predict(obs)
            env.last_q_values = q_vals
        else:
            act = env.core.heuristic_action()
        obs, _, terminated, truncated, _ = env.step(act)
        if terminated or truncated:
            obs, _ = env.reset()

    env.render(mode_label="DQN AGENT" if dqn_agent else "HEURISTIC AI", fps_override=30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(env._renderer.surface, str(output_path))
    env.close()


def run_interactive(
    initial_mode: str = "human",
    grid_size: int | None = None,
    cell_size: int | None = None,
    fps: int | None = None,
    dqn_checkpoint: Path = DEFAULT_DQN_CKPT,
    ppo_checkpoint: Path = DEFAULT_PPO_CKPT,
) -> None:
    """启动桌面交互窗口，支持在人类手玩、PPO 满盘模型、DQN 模型与启发式 AI 之间实时切换。"""
    ppo_agent = _try_load_ppo(ppo_checkpoint)
    dqn_agent = _try_load_dqn(dqn_checkpoint)

    if initial_mode == "ppo" and ppo_agent is None:
        print(f"[提示] 未找到 PPO 权重 {ppo_checkpoint}，自动切换至可用智能体。")
        initial_mode = "dqn" if dqn_agent is not None else "heuristic"
    elif initial_mode == "dqn" and dqn_agent is None:
        print(f"[提示] 未找到 DQN 权重 {dqn_checkpoint}，自动切换至可用智能体。")
        initial_mode = "ppo" if ppo_agent is not None else "heuristic"

    # 若未显式指定 grid_size：优先匹配已选模式模型的训练棋盘尺寸
    if grid_size is None:
        if initial_mode == "ppo" and ppo_agent is not None:
            grid_size = ppo_agent.config.grid_size
        elif ppo_agent is not None and initial_mode == "human":
            grid_size = ppo_agent.config.grid_size
        else:
            grid_size = 12

    if cell_size is None:
        cell_size = 52 if grid_size <= 10 else 44

    snake_cfg = SnakeConfig(
        grid_width=grid_size,
        grid_height=grid_size,
        starvation_factor=8,
        min_starvation_steps=300,
    )
    render_cfg = RenderConfig(cell_size=cell_size)
    env = SnakeEnv(
        grid_size=grid_size,
        obs_mode="features",
        action_mode="relative",
        render_mode=None,
        snake_config=snake_cfg,
        reward_config=RewardConfig(),
        render_config=render_cfg,
    )
    from snake_rl.renderer import SnakeRenderer

    renderer = SnakeRenderer(snake_config=snake_cfg, render_config=render_cfg, window_enabled=True)
    env._renderer = renderer

    # 仅将棋盘尺寸匹配的神经网络加入 Tab 键切换列表（DQN 为特征输入任意尺寸皆可兼容）
    available_modes = ["human"]
    if ppo_agent is not None and ppo_agent.config.grid_size == grid_size:
        available_modes.append("ppo")
    if dqn_agent is not None:
        available_modes.append("dqn")
    available_modes.append("heuristic")

    mode = initial_mode
    default_fps = fps or (render_cfg.human_fps if mode == "human" else (45 if mode == "ppo" else render_cfg.ai_fps))
    fps_idx = min(range(len(FPS_LEVELS)), key=lambda i: abs(FPS_LEVELS[i] - default_fps))
    current_fps = FPS_LEVELS[fps_idx]

    turn_queue: list[Direction] = []
    obs, _ = env.reset()

    running = True
    step_once = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    turn_queue.clear()
                    obs, _ = env.reset()
                elif event.key in (pygame.K_TAB, pygame.K_m):
                    curr_i = available_modes.index(mode) if mode in available_modes else 0
                    mode = available_modes[(curr_i + 1) % len(available_modes)]
                    turn_queue.clear()
                    env.last_q_values = None
                    env.last_action_probs = None
                    env.last_state_value = None
                    target = render_cfg.human_fps if mode == "human" else (45 if mode == "ppo" else render_cfg.ai_fps)
                    fps_idx = min(range(len(FPS_LEVELS)), key=lambda i: abs(FPS_LEVELS[i] - target))
                    current_fps = FPS_LEVELS[fps_idx]
                elif event.key == pygame.K_SPACE:
                    renderer.paused = not renderer.paused
                elif event.key == pygame.K_n:
                    renderer.paused = True
                    step_once = True
                elif event.key == pygame.K_v:
                    renderer.show_sensors = not renderer.show_sensors
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    fps_idx = max(0, fps_idx - 1)
                    current_fps = FPS_LEVELS[fps_idx]
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS, pygame.K_PLUS):
                    fps_idx = min(len(FPS_LEVELS) - 1, fps_idx + 1)
                    current_fps = FPS_LEVELS[fps_idx]
                elif mode == "human" and event.key in KEY_TO_DIRECTION:
                    desired = KEY_TO_DIRECTION[event.key]
                    ref_dir = turn_queue[-1] if turn_queue else env.core.direction
                    if (int(desired) - int(ref_dir)) % 4 != 2 and desired != ref_dir:
                        if len(turn_queue) < 2:
                            turn_queue.append(desired)

        should_advance = (not renderer.paused) or step_once
        step_once = False

        if should_advance:
            if env.core.done:
                if mode != "human":
                    pygame.time.delay(500)
                    obs, _ = env.reset()
            else:
                if mode == "human":
                    env.last_q_values = None
                    env.last_action_probs = None
                    action = _direction_to_relative_action(env.core.direction, turn_queue.pop(0)) if turn_queue else 0
                elif mode == "ppo" and ppo_agent is not None:
                    env.last_q_values = None
                    action, probs, val, _ = ppo_agent.predict_from_core(env.core, deterministic=True)
                    env.last_action_probs = probs
                    env.last_state_value = val
                elif mode == "dqn" and dqn_agent is not None:
                    env.last_action_probs = None
                    action, q_vals = dqn_agent.predict(obs)
                    env.last_q_values = q_vals
                elif mode == "heuristic":
                    env.last_q_values = None
                    env.last_action_probs = None
                    action = env.core.heuristic_action()
                else:
                    action = int(env.action_space.sample())

                obs, _, _, _, _ = env.step(action)

        label_map = {
            "human": "HUMAN PLAY",
            "ppo": "PPO MASTER",
            "dqn": "DQN AGENT",
            "heuristic": "HEURISTIC AI",
            "random": "RANDOM AGENT",
        }
        renderer.render(env, mode_label=label_map.get(mode, mode.upper()), fps_target=current_fps)

    env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Snake RL Studio — 贪吃蛇可视化试玩与满盘智能体观战台")
    parser.add_argument(
        "--mode",
        choices=["human", "ppo", "dqn", "heuristic", "random"],
        default="human",
        help="启动控制模式：human / ppo（满盘通关模型）/ dqn / heuristic / random",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_DQN_CKPT, help="DQN 模型权重路径")
    parser.add_argument("--ppo-checkpoint", type=Path, default=DEFAULT_PPO_CKPT, help="PPO 模型权重路径")
    parser.add_argument("--grid-size", type=int, default=None, help="正方形网格边长（默认自动匹配模型配置）")
    parser.add_argument("--cell-size", type=int, default=None, help="单格像素尺寸")
    parser.add_argument("--fps", type=int, default=None, help="指定初始目标帧率")
    parser.add_argument("--preview-steps", type=int, default=45, help="导出预览图步数")
    parser.add_argument("--save-preview", type=Path, default=None, help="离屏导出单张预览图后退出")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.save_preview is not None:
        g = args.grid_size or 12
        ckpt = args.checkpoint if args.mode == "dqn" else None
        save_preview_image(args.save_preview, grid_size=g, steps=args.preview_steps, checkpoint=ckpt)
        print(f"已导出预览截图至: {args.save_preview}")
        return

    run_interactive(
        initial_mode=args.mode,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        fps=args.fps,
        dqn_checkpoint=args.checkpoint,
        ppo_checkpoint=args.ppo_checkpoint,
    )


if __name__ == "__main__":
    main()
