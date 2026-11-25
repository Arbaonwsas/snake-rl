"""贪吃蛇交互式可视化启动器（支持人工试玩、训练好的 DQN 智能体观战、启发式基线切换与截图导出）。"""

from __future__ import annotations

import argparse
from pathlib import Path
import pygame

from snake_rl.config import RenderConfig, RewardConfig, SnakeConfig
from snake_rl.core import Direction
from snake_rl.dqn.agent import DQNAgent
from snake_rl.env import SnakeEnv


FPS_LEVELS = [5, 10, 15, 25, 45, 90, 240]
DEFAULT_CHECKPOINT = Path("checkpoints/best_dqn.pt")

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
    """若给定检查点文件存在，则自动加载 DQNAgent 供可视化实时推理。"""
    if checkpoint.is_file():
        agent, meta = DQNAgent.load(checkpoint, device="auto")
        sc = meta.get("eval_mean_score", None)
        sc_str = f" (评估均分: {sc:.1f})" if isinstance(sc, (int, float)) else ""
        print(f"[Snake RL] 已加载 DQN 权重: {checkpoint}{sc_str}")
        return agent
    return None


def save_preview_image(
    output_path: Path,
    grid_size: int = 12,
    steps: int = 45,
    checkpoint: Path | None = None,
) -> None:
    """离屏运行智能体（优先使用已提供的 DQN 检查点，否则使用启发式智能体）并导出高清界面截图。"""
    env = SnakeEnv(
        grid_size=grid_size,
        obs_mode="features",
        action_mode="relative",
        render_mode="rgb_array",
    )
    dqn_agent = _try_load_dqn(checkpoint) if checkpoint is not None else None
    obs, _ = env.reset(seed=42)

    if dqn_agent is not None:
        # 挑选一局发育良好、蛇身修长且正在追逐苹果的高光时刻截图
        best_surface: pygame.Surface | None = None
        best_score_seen = -1
        for ep_seed in range(3):
            obs, _ = env.reset(seed=2026 + ep_seed)
            for _ in range(steps):
                act, q_vals = dqn_agent.predict(obs)
                env.last_q_values = q_vals
                obs, _, terminated, truncated, info = env.step(act)
                if not (terminated or truncated) and info["score"] > best_score_seen:
                    best_score_seen = info["score"]
                    env.render(mode_label="DQN AGENT", fps_override=30)
                    best_surface = env._renderer.surface.copy()
                if terminated or truncated:
                    break
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if best_surface is not None:
            pygame.image.save(best_surface, str(output_path))
        else:
            env.render(mode_label="DQN AGENT", fps_override=30)
            pygame.image.save(env._renderer.surface, str(output_path))
    else:
        for _ in range(steps):
            act = env.core.heuristic_action()
            obs, _, terminated, truncated, _ = env.step(act)
            if terminated or truncated:
                obs, _ = env.reset()
        env.render(mode_label="HEURISTIC AI", fps_override=25)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(env._renderer.surface, str(output_path))

    env.close()


def run_interactive(
    initial_mode: str = "human",
    grid_size: int = 12,
    cell_size: int = 44,
    fps: int | None = None,
    checkpoint: Path = DEFAULT_CHECKPOINT,
) -> None:
    """启动桌面交互窗口，支持在人类手玩、训练好的 DQN 智能体与启发式智能体之间无缝切换。"""
    snake_cfg = SnakeConfig(grid_width=grid_size, grid_height=grid_size)
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

    dqn_agent = _try_load_dqn(checkpoint)
    if initial_mode == "dqn" and dqn_agent is None:
        print(f"[提示] 未找到 DQN 模型权重 {checkpoint}，已回退至 heuristic 演示模式。")
        initial_mode = "heuristic"

    # 构建 Tab 键循环切换列表
    available_modes = ["human", "dqn", "heuristic"] if dqn_agent is not None else ["human", "heuristic"]
    mode = initial_mode

    default_fps = fps or (render_cfg.human_fps if mode == "human" else render_cfg.ai_fps)
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
                    if mode != "dqn":
                        env.last_q_values = None
                    target = render_cfg.human_fps if mode == "human" else render_cfg.ai_fps
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
                    pygame.time.delay(350)
                    obs, _ = env.reset()
            else:
                if mode == "human":
                    env.last_q_values = None
                    if turn_queue:
                        target_dir = turn_queue.pop(0)
                        action = _direction_to_relative_action(env.core.direction, target_dir)
                    else:
                        action = 0
                elif mode == "dqn" and dqn_agent is not None:
                    action, q_vals = dqn_agent.predict(obs)
                    env.last_q_values = q_vals
                elif mode == "heuristic":
                    env.last_q_values = None
                    action = env.core.heuristic_action()
                else:
                    env.last_q_values = None
                    action = int(env.action_space.sample())

                obs, _, _, _, _ = env.step(action)

        label_map = {
            "human": "HUMAN PLAY",
            "dqn": "DQN AGENT",
            "heuristic": "HEURISTIC AI",
            "random": "RANDOM AGENT",
        }
        renderer.render(env, mode_label=label_map.get(mode, mode.upper()), fps_target=current_fps)

    env.close()


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Snake RL Studio — 贪吃蛇可视化试玩与 RL 智能体观战台")
    parser.add_argument(
        "--mode",
        choices=["human", "dqn", "heuristic", "random"],
        default="human",
        help="启动模式：human（人类试玩）、dqn（加载训练好的 DQN 智能体）、heuristic（启发式 AI）、random（随机）",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="DQN 模型权重路径（默认: checkpoints/best_dqn.pt）",
    )
    parser.add_argument("--grid-size", type=int, default=12, help="正方形网格边长（默认: 12）")
    parser.add_argument("--cell-size", type=int, default=44, help="单格像素尺寸（默认: 44）")
    parser.add_argument("--fps", type=int, default=None, help="指定初始目标帧率")
    parser.add_argument("--preview-steps", type=int, default=45, help="导出预览图时的运行步数上限")
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="离屏生成一张可视化预览图（PNG）后直接退出",
    )
    return parser


def main() -> None:
    """CLI 主入口。"""
    args = build_parser().parse_args()
    if args.save_preview is not None:
        ckpt = args.checkpoint if args.mode == "dqn" else None
        save_preview_image(
            args.save_preview,
            grid_size=args.grid_size,
            steps=args.preview_steps,
            checkpoint=ckpt,
        )
        print(f"已导出预览截图至: {args.save_preview}")
        return

    run_interactive(
        initial_mode=args.mode,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        fps=args.fps,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
