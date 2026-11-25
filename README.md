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

### 启动可视化窗口（支持 DQN 模型 / 启发式 AI / 人工试玩无缝切换）
```bash
cd /home/qzy/snake-rl

# 1. 直接观看已训练好的 Dueling Double DQN 智能体实战（实时展示三向 Q 值预测）
.venv/bin/python scripts/play.py --mode dqn

# 2. 启动人工键盘试玩模式（随时按 Tab 键在 [人工 -> DQN AI -> 启发式 AI] 之间循环切换！）
.venv/bin/python scripts/play.py
```

### 重新训练 Dueling Double DQN 或启动 TensorBoard
```bash
# 16 环境并行 + RTX 4060 GPU 训练 400,000 步（约 4.5 分钟）
.venv/bin/python scripts/train_dqn.py --total-steps 400000 --num-envs 16

# 启动 TensorBoard 查看训练实时日志
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
