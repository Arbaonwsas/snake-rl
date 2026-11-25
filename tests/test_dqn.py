"""Dueling Double DQN 网络结构、多步经验累加器与模型持久化单元测试。"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from snake_rl.dqn import DQNAgent, DQNConfig, DuelingDQNNet, NStepAccumulator, ReplayBuffer
from snake_rl.env import FEATURE_DIM


def test_dueling_network_shape_and_advantage_centering() -> None:
    """验证 Dueling 网络前向传播维度正确，且 Q 值均值精确等于状态价值流输出 V(s)。"""
    net = DuelingDQNNet(obs_dim=FEATURE_DIM, action_dim=3, hidden_dim=64, stream_dim=32)
    x = torch.randn(5, FEATURE_DIM)
    q = net(x)
    assert q.shape == (5, 3)

    feat = net.feature_trunk(x)
    v = net.value_stream(feat).squeeze(-1)
    np.testing.assert_allclose(q.mean(dim=-1).detach().numpy(), v.detach().numpy(), atol=1e-5)


def test_nstep_accumulator_discount_and_episode_flush() -> None:
    """验证 3-Step 累加器的折扣回报公式与回合结束时的无泄漏残留冲刷。"""
    acc = NStepAccumulator(n_step=3, gamma=0.9)
    s = [np.full(FEATURE_DIM, float(i), dtype=np.float32) for i in range(5)]

    # 第 1、2 步未满 3 步且未结束，不输出
    assert acc.append(s[0], 0, 1.0, s[1], False, False) == []
    assert acc.append(s[1], 1, 2.0, s[2], False, False) == []

    # 第 3 步满窗口，应输出从 s[0] 出发的 3 步回报: 1.0 + 0.9*2.0 + 0.81*3.0 = 5.23, gamma^3 = 0.729
    out = acc.append(s[2], 2, 3.0, s[3], False, False)
    assert len(out) == 1
    s0, a0, r_n, next_s, term, gamma_n = out[0]
    assert a0 == 0
    assert r_n == pytest.approx(1.0 + 0.9 * 2.0 + 0.81 * 3.0)
    assert gamma_n == pytest.approx(0.9**3)
    assert not term
    np.testing.assert_allclose(next_s, s[3])

    # 第 4 步撞墙终止 (terminated=True)，应一次性冲刷剩余的 s[1], s[2], s[3] 共 3 条样本
    flushed = acc.append(s[3], 0, -10.0, s[4], True, False)
    assert len(flushed) == 3
    assert all(item[4] is True for item in flushed)
    assert len(acc.buffer) == 0


def test_dqn_agent_update_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    """验证 DQNAgent 批量训练更新及检查点保存/加载前后预测完全一致。"""
    cfg = DQNConfig(obs_dim=FEATURE_DIM, action_dim=3, hidden_dim=64, stream_dim=32, batch_size=16)
    agent = DQNAgent(config=cfg, device="cpu", seed=123)
    buf = ReplayBuffer(capacity=100, obs_dim=FEATURE_DIM, seed=123)

    rng = np.random.default_rng(0)
    for _ in range(32):
        s = rng.uniform(-1.0, 1.0, size=FEATURE_DIM).astype(np.float32)
        ns = rng.uniform(-1.0, 1.0, size=FEATURE_DIM).astype(np.float32)
        buf.add(s, int(rng.integers(0, 3)), 1.5, ns, False, 0.99**3)

    metrics = agent.update(buf.sample(16, agent.device))
    assert metrics["loss"] >= 0.0

    ckpt_file = tmp_path / "test_dqn.pt"
    agent.save(ckpt_file, metadata={"eval_mean_score": 42.5})

    loaded_agent, meta = DQNAgent.load(ckpt_file, device="cpu")
    assert meta["eval_mean_score"] == pytest.approx(42.5)

    test_obs = rng.uniform(-1.0, 1.0, size=FEATURE_DIM).astype(np.float32)
    act1, q1 = agent.predict(test_obs)
    act2, q2 = loaded_agent.predict(test_obs)
    assert act1 == act2
    np.testing.assert_allclose(q1, q2, atol=1e-6)
