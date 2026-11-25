# Snake RL Studio (`snake-rl`)

面向强化学习（Reinforcement Learning）训练深度优化的贪吃蛇实验工作区，集成**无头高速核心物理引擎**、**标准 Gymnasium 环境 (`SnakeRL-v0`)** 与 **Pygame 实时遥测可视化台**。

## 1. 环境与目录结构

本项目使用 `uv` 管理独立虚拟环境（`.venv`，已开启 PyTorch + CUDA 硬件加速）：

```text
/home/qzy/snake-rl/
├── src/snake_rl/
│   ├── config.py          # 游戏规则、潜势差奖励塑造 (Reward Shaping) 与主题配置
│   ├── core.py            # 纯逻辑无头贪吃蛇引擎 (SnakeCore)、8 向激光测距与 Flood-Fill 连通域算法
│   ├── env.py             # 标准 Gymnasium 环境 (SnakeEnv)，支持 features (24维向量) 与 grid (4通道张量)
│   ├── renderer.py        # Pygame 可视化界面与实时 RL 遥测侧边栏 (HUD + 奖励折线图)
│   └── cli.py             # 命令行交互与截图导出工具
├── scripts/
│   ├── play.py            # 启动可视化游戏窗口（人类试玩 / AI 实时切换演示）
│   └── benchmark_env.py   # Gymnasium API 合规检验、采样吞吐量与启发式基线测试
└── tests/
    └── test_snake_env.py  # Pytest 自动化单元测试套件
```

## 2. 快速运行

### 启动可视化窗口（观看 100% 满屏通关 AI / DQN / 人工试玩）
```bash
cd /home/qzy/snake-rl

# 1. 【强烈推荐】观看 Maskable PPO + 神经拓扑前瞻宗师模型冲击 100% 满屏彻底通关！
#    （10x10 棋盘满分 97 分，平均占满率 98.9%，中位数 97.0 满分通关，按 ] 键可加速至 90/240 FPS）
.venv/bin/python scripts/play.py --mode ppo

# 2. 观看 24 维特征 Dueling Double DQN 智能体实战（实时展示三向 Q 值预测）
.venv/bin/python scripts/play.py --mode dqn --grid-size 12

# 3. 启动人工键盘试玩模式（随时按 Tab 键在 [人工 -> PPO -> DQN -> 启发式 AI] 间无缝切换！）
.venv/bin/python scripts/play.py
```

### 重新运行 GPU 训练或启动 TensorBoard
```bash
# RTX 4060 纯 GPU 1,024 环境并行训练 Maskable PPO + Ego-SE-ResNet（1,600 万步）
.venv/bin/python scripts/train_ppo.py --total-steps 16000000 --grid-size 10 --num-envs 1024

# 16 环境并行训练 3-Step Dueling Double DQN（40 万步）
.venv/bin/python scripts/train_dqn.py --total-steps 400000 --num-envs 16

# 启动 TensorBoard 查看训练日志
.venv/bin/tensorboard --logdir runs
```

### 窗口内交互快捷键
| 按键 | 功能说明 |
| :--- | :--- |
| `方向键` / `WASD` | 人工模式下控制贪吃蛇转向（内置双帧转向缓冲与防 180° 自杀保护） |
| `Tab` / `M` | 一键无缝切换 **[人工试玩 HUMAN]** ↔ **[启发式智能体 HEURISTIC AI]** |
| `Space` / `N` | 暂停或继续 / 暂停状态下按 `N` **单步推进**（逐帧观察观测特征与即时 Reward） |
| `V` | 开启 / 关闭棋盘上的 **8 向障碍物测距射线与三向安全度透视层** |
| `[` / `]`（或 `-` / `+`） | 实时调节运行速度（5 / 10 / 15 / 25 / 45 / 90 / 240 FPS） |
| `R` / `Esc` | 立即重开当前回合 / 退出窗口 |

### 运行单元测试与基准评测
```bash
# 运行全量单元测试
.venv/bin/pytest -v

# 运行 Gymnasium 合规性检查、采样吞吐量与基线评估
.venv/bin/python scripts/benchmark_env.py
```

## 3. 强化学习训练接口速览

```python
import gymnasium as gym
import snake_rl  # 自动注册 SnakeRL-v0

# 1. MLP 训练推荐：24 维全归一化特征向量 [-1, 1] + 3 维相对动作 (0直行, 1左转, 2右转)
env = gym.make("SnakeRL-v0", grid_size=12, obs_mode="features", action_mode="relative")

# 2. CNN 训练推荐：(4, 12, 12) 多通道空间张量（蛇头、蛇身梯度、食物、朝向层）
cnn_env = gym.make("SnakeRL-v0", grid_size=12, obs_mode="grid", action_mode="relative")

obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```
