"""贪吃蛇交互式可视化启动器（支持人类试玩、启发式智能体观战、单步调试与截图导出）。"""

from __future__ import annotations

import argparse
from pathlib import Path
import pygame

from snake_rl.config import RenderConfig, RewardConfig, SnakeConfig
from snake_rl.core import Direction
from snake_rl.env import SnakeEnv


FPS_LEVELS = [5, 10, 15, 25, 45, 90, 240]

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
        # 同向或 180 度反向均保持直行
        return 0
    if diff == 3:
        return 1  # 左转
    return 2      # 右转


def save_preview_image(output_path: Path, grid_size: int = 12, steps: int = 45) -> None:
    """在离屏模式下运行启发式智能体若干步并保存高清可视化界面截图（PNG）。"""
    env = SnakeEnv(
        grid_size=grid_size,
        obs_mode="features",
        action_mode="relative",
        render_mode="rgb_array",
    )
    env.reset(seed=42)
    for _ in range(steps):
        act = env.core.heuristic_action()
        _, _, terminated, truncated, _ = env.step(act)
        if terminated or truncated:
            env.reset()

    env.render(mode_label="HEURISTIC AI", fps_override=25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(env._renderer.surface, str(output_path))
    env.close()


def run_interactive(
    initial_mode: str = "human",
    grid_size: int = 12,
    cell_size: int = 44,
    fps: int | None = None,
) -> None:
    """启动桌面交互窗口，支持人类试玩与 AI 实时切换调试。"""
    snake_cfg = SnakeConfig(grid_width=grid_size, grid_height=grid_size)
    render_cfg = RenderConfig(cell_size=cell_size)
    env = SnakeEnv(
        grid_size=grid_size,
        obs_mode="features",
        action_mode="relative",
        render_mode=None,  # 由交互主循环手动驱动高精度渲染与按键队列响应
        snake_config=snake_cfg,
        reward_config=RewardConfig(),
        render_config=render_cfg,
    )
    from snake_rl.renderer import SnakeRenderer

    renderer = SnakeRenderer(snake_config=snake_cfg, render_config=render_cfg, window_enabled=True)
    env._renderer = renderer

    mode = initial_mode  # "human" | "heuristic" | "random"
    default_fps = fps or (render_cfg.human_fps if mode == "human" else render_cfg.ai_fps)
    fps_idx = min(range(len(FPS_LEVELS)), key=lambda i: abs(FPS_LEVELS[i] - default_fps))
    current_fps = FPS_LEVELS[fps_idx]

    # 按键输入缓冲队列（最多缓存 2 个连续转向指令，使人类快速连按拐弯手感极其跟手）
    turn_queue: list[Direction] = []
    env.reset()

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
                    env.reset()
                elif event.key in (pygame.K_TAB, pygame.K_m):
                    mode = "heuristic" if mode == "human" else "human"
                    turn_queue.clear()
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
                # AI 演示模式下结束后自动重开下一局，人类模式下等待玩家按 R 查看结算信息
                if mode != "human":
                    pygame.time.delay(350)
                    env.reset()
            else:
                if mode == "human":
                    if turn_queue:
                        target_dir = turn_queue.pop(0)
                        action = _direction_to_relative_action(env.core.direction, target_dir)
                    else:
                        action = 0
                elif mode == "heuristic":
                    action = env.core.heuristic_action()
                else:
                    action = int(env.action_space.sample())

                env.step(action)

        label_map = {
            "human": "HUMAN PLAY",
            "heuristic": "HEURISTIC AI",
            "random": "RANDOM AGENT",
        }
        renderer.render(env, mode_label=label_map.get(mode, mode.upper()), fps_target=current_fps)

    env.close()


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Snake RL Studio — 贪吃蛇可视化试玩与智能体观战工具")
    parser.add_argument(
        "--mode",
        choices=["human", "heuristic", "random"],
        default="human",
        help="启动控制模式：human（人类键盘试玩，默认）、heuristic（启发式智能体演示）、random（随机探索）",
    )
    parser.add_argument("--grid-size", type=int, default=12, help="正方形网格边长（默认: 12）")
    parser.add_argument("--cell-size", type=int, default=44, help="单格像素尺寸（默认: 44）")
    parser.add_argument("--fps", type=int, default=None, help="指定初始目标帧率")
    parser.add_argument(
        "--save-preview",
        type=Path,
        default=None,
        help="离屏生成一张可视化预览图（PNG）后直接退出，适合无桌面验证",
    )
    return parser


def main() -> None:
    """CLI 主入口。"""
    args = build_parser().parse_args()
    if args.save_preview is not None:
        save_preview_image(args.save_preview, grid_size=args.grid_size)
        print(f"已导出预览截图至: {args.save_preview}")
        return

    run_interactive(
        initial_mode=args.mode,
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
