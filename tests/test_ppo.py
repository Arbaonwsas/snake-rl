"""GPU 向量化环境物理一致性、动作硬掩码与 SE-ResNet Maskable PPO 单元测试。"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from snake_rl.config import SnakeConfig
from snake_rl.core import SnakeCore
from snake_rl.ppo.agent import MaskablePPOAgent, PPOConfig
from snake_rl.ppo.vec_env_gpu import TorchSnakeVecEnv, extract_obs_and_mask_from_core


def test_gpu_vec_env_matches_cpu_core_obs_and_mask() -> None:
    """严格校验 GPU 向量化环境的初始盘面、多步无进食移动、TTL 梯度与合法动作掩码与 CPU SnakeCore 100% 一致。"""
    g = 8
    gpu_env = TorchSnakeVecEnv(num_envs=1, grid_size=g, device="cpu", seed=1)
    cpu_core = SnakeCore(SnakeConfig(grid_width=g, grid_height=g, initial_length=3))

    # 强制对齐食物坐标至角落 (0, 0) 以测试纯移动与转向序列
    gpu_env.food_x[0] = 0
    gpu_env.food_y[0] = 0
    if cpu_core.food is not None:
        ox, oy = cpu_core.food
        cpu_core.board[oy, ox] = 0
    cpu_core.food = (0, 0)
    cpu_core.board[0, 0] = 3

    obs_gpu = gpu_env.get_obs()[0].numpy()
    mask_gpu = gpu_env.get_action_masks()[0].numpy()
    obs_cpu, mask_cpu = extract_obs_and_mask_from_core(cpu_core)

    np.testing.assert_allclose(obs_gpu, obs_cpu, atol=1e-6)
    np.testing.assert_array_equal(mask_gpu, mask_cpu)

    # 连续执行一串转向序列：[直行, 左转, 直行, 右转, 右转]
    for act in [0, 1, 0, 2, 2]:
        obs_g_next, _, term_g, mask_g_next, _ = gpu_env.step(torch.tensor([act], dtype=torch.long))
        res_c = cpu_core.step(act, action_mode="relative")
        assert bool(term_g[0].item()) == res_c.terminated

        obs_c_next, mask_c_next = extract_obs_and_mask_from_core(cpu_core)
        np.testing.assert_allclose(obs_g_next[0].numpy(), obs_c_next, atol=1e-6)
        np.testing.assert_array_equal(mask_g_next[0].numpy(), mask_c_next)


def test_invalid_action_masking_zeros_out_illegal_probabilities(tmp_path: Path) -> None:
    """验证 MaskablePPOAgent 将被屏蔽的危险动作概率严格压制为 0.0，且支持检查点往返加载。"""
    cfg = PPOConfig(grid_size=8, channels=32, num_blocks=2, num_envs=4, rollout_steps=8, minibatch_size=16, use_amp=False)
    agent = MaskablePPOAgent(config=cfg, device="cpu", seed=7)

    core = SnakeCore(SnakeConfig(grid_width=8, grid_height=8))
    # 将蛇头移至最上排 (4, 0) 且朝右 (RIGHT)：此时左转 (UP) 必撞上墙，mask 应为 [True, False, True]
    from collections import deque
    from snake_rl.core import Direction

    core.board.fill(0)
    core.snake = deque([(4, 0), (3, 0), (2, 0)])
    for i, (x, y) in enumerate(core.snake):
        core.board[y, x] = 2 if i == 0 else 1
    core.direction = Direction.RIGHT
    core.food = (4, 4)
    core.board[4, 4] = 3

    act, probs, val, mask = agent.predict_from_core(core, deterministic=True)
    assert list(mask) == [True, False, True]
    assert probs[1] == pytest.approx(0.0, abs=1e-6)
    assert act in (0, 2)
    assert np.isfinite(val)

    ckpt = tmp_path / "ppo_test.pt"
    agent.save(ckpt, metadata={"eval_win_rate": 0.85})
    loaded, meta = MaskablePPOAgent.load(ckpt, device="cpu")
    assert meta["eval_win_rate"] == pytest.approx(0.85)
    act2, probs2, _, _ = loaded.predict_from_core(core, deterministic=True)
    assert act == act2
    np.testing.assert_allclose(probs, probs2, atol=1e-6)
