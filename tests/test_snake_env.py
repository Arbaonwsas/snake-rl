"""贪吃蛇核心引擎、Gymnasium 环境与离屏渲染全量单元测试。"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import snake_rl  # noqa: F401
from snake_rl.config import RewardConfig, SnakeConfig
from snake_rl.core import CELL_FOOD, Direction, SnakeCore
from snake_rl.env import FEATURE_DIM, SnakeEnv


def test_gym_registration_and_compliance() -> None:
    """验证环境可通过 gym.make 创建并通过官方 API 规范检查。"""
    for obs_mode in ("features", "grid"):
        env = gym.make("SnakeRL-v0", grid_size=8, obs_mode=obs_mode)
        check_env(env.unwrapped, skip_render_check=True)
        env.close()


def test_deterministic_seeding() -> None:
    """验证相同随机种子下初始状态与多步演化结果完全一致。"""
    env1 = SnakeEnv(grid_size=10, obs_mode="features")
    env2 = SnakeEnv(grid_size=10, obs_mode="features")

    obs1, _ = env1.reset(seed=12345)
    obs2, _ = env2.reset(seed=12345)
    np.testing.assert_allclose(obs1, obs2)

    actions = [0, 1, 0, 2, 0, 0, 1]
    for act in actions:
        o1, r1, term1, trunc1, _ = env1.step(act)
        o2, r2, term2, trunc2, _ = env2.step(act)
        np.testing.assert_allclose(o1, o2)
        assert r1 == pytest.approx(r2)
        assert term1 == term2
        assert trunc1 == trunc2


def test_reverse_direction_protection_in_absolute_mode() -> None:
    """验证绝对动作模式下发出 180° 反向指令不会导致原地咬脖子自杀。"""
    core = SnakeCore(SnakeConfig(grid_width=10, grid_height=10, initial_length=3))
    assert core.direction == Direction.RIGHT
    head_before = core.head

    # 当前向右移动，输入向左 (Direction.LEFT == 3) 应被自动忽略并继续向右直行
    res = core.step(int(Direction.LEFT), action_mode="absolute")
    assert not res.terminated
    assert core.direction == Direction.RIGHT
    assert core.head == (head_before[0] + 1, head_before[1])


def test_stepping_into_moving_tail_is_legal() -> None:
    """验证长度为 4 的环形蛇下一步踏入即将同步移开的旧蛇尾时不会误判自撞。"""
    core = SnakeCore(SnakeConfig(grid_width=8, grid_height=8, initial_length=4))
    core.board.fill(0)
    # 构造 2x2 闭环蛇身：头在 (2,2)，颈在 (3,2)，腹在 (3,3)，尾在 (2,3)，当前朝下 (DOWN)
    # 下一步向下恰好踏入旧蛇尾 (2,3)
    from collections import deque

    core.snake = deque([(2, 2), (3, 2), (3, 3), (2, 3)])
    for idx, (x, y) in enumerate(core.snake):
        core.board[y, x] = 2 if idx == 0 else 1
    core.direction = Direction.DOWN
    core.food = (0, 0)
    core.board[0, 0] = CELL_FOOD

    assert not core.is_collision((2, 3), ignore_tail=True)
    res = core.step(0, action_mode="relative")  # 直行向下进入 (2, 3)
    assert not res.terminated
    assert core.head == (2, 3)
    assert core.length == 4


def test_eating_food_grows_snake_and_grants_positive_reward() -> None:
    """验证吃到食物后蛇身增长、分数递增且返回正奖励。"""
    env = SnakeEnv(
        grid_size=8,
        obs_mode="features",
        reward_config=RewardConfig(food_base=10.0, food_length_scale=0.0),
    )
    env.reset(seed=7)
    hx, hy = env.core.head
    # 手动把食物放在蛇头正前方 (右侧 1 格)
    if env.core.food is not None:
        ox, oy = env.core.food
        env.core.board[oy, ox] = 0
    target_food = (hx + 1, hy)
    env.core.food = target_food
    env.core.board[target_food[1], target_food[0]] = CELL_FOOD

    init_len = env.core.length
    obs, reward, terminated, truncated, info = env.step(0)  # 直行吃苹果
    assert not terminated and not truncated
    assert info["score"] == 1
    assert env.core.length == init_len + 1
    assert reward == pytest.approx(10.0)
    assert obs.shape == (FEATURE_DIM,)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)


def test_wall_collision_and_starvation_truncation() -> None:
    """验证撞墙触发 terminated=True，而步数超限未进食触发 truncated=True。"""
    # 1. 撞墙测试
    env = SnakeEnv(grid_size=6)
    env.reset(seed=1)
    terminated = False
    for _ in range(10):
        _, reward, terminated, truncated, info = env.step(0)  # 一直向右直行必撞右墙
        if terminated:
            assert not truncated
            assert info["death_reason"] == "wall"
            assert reward < 0
            break
    assert terminated

    # 2. 饥饿截断测试
    cfg = SnakeConfig(grid_width=8, grid_height=8, starvation_factor=0, min_starvation_steps=8)
    env_starv = SnakeEnv(grid_size=8, snake_config=cfg)
    env_starv.reset(seed=3)
    # 移除食物干扰，放置在角落 (0, 0)，让蛇在中部绕 2x2 小圈触发饥饿
    if env_starv.core.food is not None:
        fx, fy = env_starv.core.food
        env_starv.core.board[fy, fx] = 0
    env_starv.core.food = (0, 0)
    env_starv.core.board[0, 0] = CELL_FOOD

    truncated = False
    for _ in range(8):
        _, _, terminated, truncated, info = env_starv.step(2)  # 连续右转绕圈
        if truncated:
            assert not terminated
            assert info["death_reason"] == "starvation"
            break
    assert truncated


def test_grid_observation_and_rgb_array_rendering() -> None:
    """验证 grid 模式张量形状与离屏 rgb_array 渲染输出格式。"""
    env = SnakeEnv(grid_size=(10, 8), obs_mode="grid", render_mode="rgb_array")
    obs, _ = env.reset(seed=99)
    assert obs.shape == (4, 8, 10)
    assert obs.dtype == np.float32
    assert np.sum(obs[0]) == pytest.approx(1.0)  # 仅有一个蛇头

    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    env.close()
