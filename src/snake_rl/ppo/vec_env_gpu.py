"""基于纯 PyTorch 张量的 GPU 原生大规模并行贪吃蛇环境 (TorchSnakeVecEnv)。

核心设计亮点：
1. 采用 TTL (Time-To-Live) 整数矩阵编码蛇身：空地为 0，蛇尾为 1，向蛇头递增至 length。
   未吃到食物时，仅需一行矩阵减法 `clamp(board - 1, min=0)` 即可同时在数千个棋盘上完成
   蛇尾腾空（完美兼容“下一步踏入旧蛇尾合法”物理规则）、身体前移与 O(1) 碰撞检测。
2. 零 CPU-GPU 拷贝：状态转移、奖励计算、非法动作硬掩码 (Action Mask) 与 6 通道全局观测张量
   全部常驻显存，配合 RTX 4060 可达数十万步/秒采样吞吐量。
3. 提供 CPU 单环境 `SnakeCore` 到相同 6 通道观测与动作掩码的精确转换函数，确保 GPU 训练与
   Pygame 桌面可视化推理 100% 逐像素对齐。
"""

from __future__ import annotations

import numpy as np
import torch

from snake_rl.core import Direction, SnakeCore


OBS_CHANNELS = 8


def extract_obs_and_mask_from_core(core: SnakeCore) -> tuple[np.ndarray, np.ndarray]:
    """从单局 CPU `SnakeCore` 提取第一人称朝向对齐（Ego-centric）的 (8, H, W) 观测与 (3,) 动作掩码。

    通过 `rot90(k=direction)` 将整个棋盘旋转至【蛇头永远朝上 (UP)】的标准第一人称坐标系：
    - 动作 0 (直行) 永远对应图像正上方 (row - 1)
    - 动作 1 (左转) 永远对应图像正左方 (col - 1)
    - 动作 2 (右转) 永远对应图像正右方 (col + 1)
    直接赋予卷积网络 4 倍旋转对称不变性，大幅加速全局盘蛇拓扑学习。

    8 通道定义（旋转对齐后）：
      Ch 0: 蛇头掩码 (1.0)
      Ch 1: 蛇尾掩码 (1.0) —— 追尾绕圈灯塔
      Ch 2: 蛇身 TTL 寿命梯度 (从蛇尾 1/L 线性递增至蛇头 1.0)
      Ch 3: 食物掩码 (1.0)
      Ch 4: 棋盘黑白格奇偶性编码 ((x + y) % 2) —— 辅助识别哈密顿二分图奇偶陷阱
      Ch 5: 全 1.0 棋盘区域掩码 —— 配合卷积零填充精确感知四面墙壁边界距离
      Ch 6: 当前盘面填充进度 (length / total_cells)
      Ch 7: 当前饥饿步数比例 (steps_since_food / (200 + 5 * length)) —— 打破确定性评估环路锁死
    """
    h, w = core.height, core.width
    total_cells = float(h * w)
    snake_len = len(core.snake)

    obs = np.zeros((OBS_CHANNELS, h, w), dtype=np.float32)

    if snake_len > 0:
        hx, hy = core.head
        tx, ty = core.tail
        obs[0, hy, hx] = 1.0
        obs[1, ty, tx] = 1.0

        inv_len = 1.0 / float(snake_len)
        for idx, (bx, by) in enumerate(core.snake):
            ttl = snake_len - idx
            obs[2, by, bx] = float(ttl) * inv_len

    if core.food is not None:
        fx, fy = core.food
        obs[3, fy, fx] = 1.0

    ys, xs = np.indices((h, w))
    obs[4] = ((xs + ys) % 2).astype(np.float32)
    obs[5].fill(1.0)
    obs[6].fill(float(snake_len) / total_cells)

    starv_denom = 200.0 + 5.0 * float(snake_len)
    obs[7].fill(min(1.0, float(core.steps_since_food) / starv_denom))

    # 旋转至蛇头朝上 (UP) 的第一人称标准视角
    k_rot = int(core.direction)
    if k_rot != 0:
        obs = np.rot90(obs, k=k_rot, axes=(1, 2)).copy()

    # 计算相对动作合法掩码 (True=合法安全，False=下一步立即撞墙或撞自身)
    candidates = core.get_relative_candidates()
    mask = np.array(
        [not core.is_collision(pos, ignore_tail=True) for pos in candidates],
        dtype=np.bool_,
    )
    if not np.any(mask):
        mask[:] = True

    return obs, mask


class TorchSnakeVecEnv:
    """运行在单个 PyTorch 设备 (CUDA/CPU) 上的大规模向量化贪吃蛇环境。"""

    def __init__(
        self,
        num_envs: int = 1024,
        grid_size: int = 10,
        device: str | torch.device = "cuda",
        seed: int = 42,
    ) -> None:
        self.num_envs = num_envs
        self.H = grid_size
        self.W = grid_size
        self.total_cells = grid_size * grid_size
        self.max_score = self.total_cells - 3
        self.device = torch.device(device)

        torch.manual_seed(seed)
        self.env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)

        # 顺时针方向增量表: 0=UP(0,-1), 1=RIGHT(1,0), 2=DOWN(0,1), 3=LEFT(-1,0)
        self.dir_dx = torch.tensor([0, 1, 0, -1], device=self.device, dtype=torch.long)
        self.dir_dy = torch.tensor([-1, 0, 1, 0], device=self.device, dtype=torch.long)
        # 相对动作到方向偏移量: 0->+0(直行), 1->-1(左转), 2->+1(右转)
        self.rel_turn = torch.tensor([0, -1, 1], device=self.device, dtype=torch.long)

        # 预构建静态奇偶棋盘格模板 (1, H, W)
        gy, gx = torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing="ij",
        )
        self.checkerboard = ((gx + gy) % 2).to(torch.float32).unsqueeze(0)

        # 核心环境状态张量
        self.board = torch.zeros((num_envs, self.H, self.W), device=self.device, dtype=torch.int16)
        self.head_x = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.head_y = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.direction = torch.ones(num_envs, device=self.device, dtype=torch.long)  # 初始朝右 RIGHT=1
        self.length = torch.full((num_envs,), 3, device=self.device, dtype=torch.int16)
        self.food_x = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.food_y = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.has_food = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        self.steps = torch.zeros(num_envs, device=self.device, dtype=torch.int32)
        self.steps_since_food = torch.zeros(num_envs, device=self.device, dtype=torch.int32)

        self.reset_all()

    def reset_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """重置全部并行环境并返回初始 (obs, action_masks)。"""
        all_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._reset_by_bool_mask(all_mask)
        return self.get_obs(), self.get_action_masks()

    def _reset_by_bool_mask(self, mask: torch.Tensor) -> None:
        """零主机同步原地重置 mask 为 True 的子环境。"""
        cx = self.W // 2
        cy = self.H // 2
        m_board = mask.view(-1, 1, 1)

        init_single = torch.zeros((1, self.H, self.W), device=self.device, dtype=torch.int16)
        init_single[0, cy, cx - 2] = 1
        init_single[0, cy, cx - 1] = 2
        init_single[0, cy, cx] = 3

        self.board = torch.where(m_board, init_single, self.board)
        self.head_x = torch.where(mask, torch.full_like(self.head_x, cx), self.head_x)
        self.head_y = torch.where(mask, torch.full_like(self.head_y, cy), self.head_y)
        self.direction = torch.where(mask, torch.full_like(self.direction, int(Direction.RIGHT)), self.direction)
        self.length = torch.where(mask, torch.full_like(self.length, 3), self.length)
        self.steps = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        self.steps_since_food = torch.where(mask, torch.zeros_like(self.steps_since_food), self.steps_since_food)
        self._spawn_food_by_bool_mask(mask)

    def _spawn_food_masked(self, env_indices: torch.Tensor) -> None:
        """对指定索引列表的子环境生成食物。"""
        if env_indices.numel() == 0:
            return
        sub_board = self.board[env_indices].view(env_indices.numel(), -1)
        empty_mask = sub_board == 0
        rand_scores = torch.rand(sub_board.shape, device=self.device)
        rand_scores = torch.where(empty_mask, rand_scores, torch.full_like(rand_scores, -1.0))
        flat_idx = rand_scores.argmax(dim=1)

        has_any_empty = empty_mask.any(dim=1)
        self.food_x[env_indices] = flat_idx % self.W
        self.food_y[env_indices] = flat_idx // self.W
        self.has_food[env_indices] = has_any_empty

    def _spawn_food_by_bool_mask(self, mask: torch.Tensor) -> None:
        """零主机同步（无 .any()/.nonzero()）为布尔掩码为 True 的环境并行刷新随机食物。"""
        flat_board = self.board.view(self.num_envs, -1)
        empty_mask = flat_board == 0
        rand_scores = torch.rand(flat_board.shape, device=self.device)
        rand_scores = torch.where(empty_mask, rand_scores, torch.full_like(rand_scores, -1.0))
        flat_idx = rand_scores.argmax(dim=1)

        new_fx = flat_idx % self.W
        new_fy = flat_idx // self.W
        new_has = empty_mask.any(dim=1)

        self.food_x = torch.where(mask, new_fx, self.food_x)
        self.food_y = torch.where(mask, new_fy, self.food_y)
        self.has_food = torch.where(mask, new_has, self.has_food)

    def get_action_masks(self) -> torch.Tensor:
        """并行计算全部环境的 (num_envs, 3) 合法动作布尔掩码。"""
        # 计算 [直行, 左转, 右转] 三个候选方向 (num_envs, 3)
        cand_dirs = (self.direction.unsqueeze(1) + self.rel_turn.unsqueeze(0)) % 4
        nx = self.head_x.unsqueeze(1) + self.dir_dx[cand_dirs]
        ny = self.head_y.unsqueeze(1) + self.dir_dy[cand_dirs]

        in_bounds = (nx >= 0) & (nx < self.W) & (ny >= 0) & (ny < self.H)
        safe_x = nx.clamp(0, self.W - 1)
        safe_y = ny.clamp(0, self.H - 1)

        # 读取候选格的 TTL 值：若下一步不吃苹果，TTL == 1（当前蛇尾）在下一步会同步清零，因此仅 TTL > 1 会发生碰撞！
        # 若候选格恰好是食物，则蛇尾不会缩回，但食物所在格的 TTL 必定为 0，所以判断 `cell_ttl <= 1` 对两种情况均 100% 精确成立！
        cand_ttl = self.board[self.env_ids.unsqueeze(1), safe_y, safe_x]
        legal = in_bounds & (cand_ttl <= 1)

        # 兜底：若三面皆死路，放开全部 3 个动作避免分布归一化出 NaN
        any_legal = legal.any(dim=1, keepdim=True)
        return torch.where(any_legal, legal, torch.ones_like(legal))

    def get_obs(self) -> torch.Tensor:
        """并行生成第一人称朝向旋转对齐的 (num_envs, 8, H, W) float32 空间观测张量。"""
        obs = torch.zeros((self.num_envs, OBS_CHANNELS, self.H, self.W), device=self.device, dtype=torch.float32)

        len_f = self.length.to(torch.float32)
        # Ch 0: 蛇头
        obs[self.env_ids, 0, self.head_y, self.head_x] = 1.0
        # Ch 1: 蛇尾 (TTL == 1)
        obs[:, 1] = (self.board == 1).to(torch.float32)
        # Ch 2: 蛇身寿命梯度 (board / length)
        obs[:, 2] = self.board.to(torch.float32) * (1.0 / len_f).view(-1, 1, 1)
        # Ch 3: 食物掩码
        obs[self.env_ids, 3, self.food_y, self.food_x] = self.has_food.to(torch.float32)
        # Ch 4: 奇偶棋盘格
        obs[:, 4] = self.checkerboard
        # Ch 5: 全 1 棋盘有效域掩码
        obs[:, 5] = 1.0
        # Ch 6: 棋盘占满进度
        obs[:, 6] = (len_f / float(self.total_cells)).view(-1, 1, 1)
        # Ch 7: 饥饿计时进度
        starv_denom = 200.0 + 5.0 * len_f
        starv_prog = (self.steps_since_food.to(torch.float32) / starv_denom).clamp(max=1.0)
        obs[:, 7] = starv_prog.view(-1, 1, 1)

        # 并行按各自蛇头朝向逆时针旋转至标准第一人称视角 (UP=0)
        d_view = self.direction.view(-1, 1, 1, 1)
        obs_k1 = torch.rot90(obs, k=1, dims=(-2, -1))
        obs_k2 = torch.rot90(obs, k=2, dims=(-2, -1))
        obs_k3 = torch.rot90(obs, k=3, dims=(-2, -1))
        obs = torch.where(d_view == 1, obs_k1, obs)
        obs = torch.where(d_view == 2, obs_k2, obs)
        obs = torch.where(d_view == 3, obs_k3, obs)

        return obs

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """并行推进所有环境一步，对终止/截断环境自动原地重置，返回新 (obs, rewards, dones, action_masks, info)。

        专为满盘通关设计的非线性奖励函数：
        - 吃到食物：+2.0 + 6.0 * (length / total_cells)^2（越接近满盘单颗苹果奖励越高，最高 +8.0）
        - 完美通关（length == total_cells）：额外 +35.0 巨额通关大奖！
        - 撞墙或自撞死亡：-10.0 - 5.0 * (length / total_cells)（长蛇后期暴毙惩罚更重，促使稳健盘蛇）
        - 饥饿超时（动态步数上限 150 + 5 * length）：-6.0
        - 普通移动潜势差（仅在前期 length < 0.55 * total_cells 时开启微弱引导 0.03 * delta_d，后期关闭距离贪心让位给哈密顿全局绕行！）
        """
        prev_dist = (self.head_x - self.food_x).abs() + (self.head_y - self.food_y).abs()

        # 1. 更新朝向与新蛇头坐标
        new_dir = (self.direction + self.rel_turn[actions]) % 4
        self.direction = new_dir
        nx = self.head_x + self.dir_dx[new_dir]
        ny = self.head_y + self.dir_dy[new_dir]

        self.steps += 1
        self.steps_since_food += 1

        # 2. 检查撞墙
        hit_wall = (nx < 0) | (nx >= self.W) | (ny < 0) | (ny >= self.H)
        safe_nx = nx.clamp(0, self.W - 1)
        safe_ny = ny.clamp(0, self.H - 1)

        # 3. 检查是否吃到食物
        ate_food = (~hit_wall) & self.has_food & (safe_nx == self.food_x) & (safe_ny == self.food_y)

        # 4. 对未死且未吃到食物的环境，全盘正数 TTL 减 1（旧蛇尾自动清零腾空！）
        move_only_mask = (~hit_wall) & (~ate_food)
        self.board = torch.where(
            move_only_mask.view(-1, 1, 1),
            (self.board - 1).clamp(min=0),
            self.board,
        )

        # 5. 检查撞自身（此时旧蛇尾若未吃食物已变成 0，故 board > 0 即真实碰撞！）
        target_cell = self.board[self.env_ids, safe_ny, safe_nx]
        hit_self = (~hit_wall) & (target_cell > 0)
        died = hit_wall | hit_self

        # 6. 更新存活环境的蛇头与长度（零 `.any()` 分支同步）
        alive = ~died
        self.head_x = torch.where(alive, safe_nx, self.head_x)
        self.head_y = torch.where(alive, safe_ny, self.head_y)
        self.length = torch.where(ate_food, self.length + 1, self.length)
        self.steps_since_food = torch.where(ate_food, torch.zeros_like(self.steps_since_food), self.steps_since_food)

        cur_cell = self.board[self.env_ids, self.head_y, self.head_x]
        self.board[self.env_ids, self.head_y, self.head_x] = torch.where(alive, self.length, cur_cell)

        # 7. 判定是否达成 100% 满盘完美通关！
        won = ate_food & (self.length >= self.total_cells)

        # 为吃到食物且尚未通关的环境刷新下一颗食物
        need_spawn = ate_food & (~won)
        self._spawn_food_by_bool_mask(need_spawn)

        # 8. 动态饥饿上限截断（10x10 满盘绕行一圈仅需 100 步，给予 200 + 5*length 充裕空间支持后期全局追尾）
        starv_limit = 200 + 5 * self.length.to(torch.int32)
        starved = alive & (~ate_food) & (self.steps_since_food >= starv_limit)

        # 9. 计算奖励标量
        fill_prog = self.length.to(torch.float32) / float(self.total_cells)
        curr_dist = (self.head_x - self.food_x).abs() + (self.head_y - self.food_y).abs()
        dist_delta = (prev_dist - curr_dist).to(torch.float32)

        # 平滑线性衰减潜势引导：前期 (fill < 65%) 快速引导捕食，后期完全归零以解锁全局追尾绕圈
        shaping_weight = 0.05 * (1.0 - fill_prog / 0.65).clamp(min=0.0)
        rewards = shaping_weight * dist_delta - 0.001

        # 覆盖特殊事件奖励
        eat_reward = 2.0 + 8.0 * (fill_prog * fill_prog)
        rewards = torch.where(ate_food, eat_reward, rewards)
        rewards = torch.where(won, eat_reward + 50.0, rewards)
        rewards = torch.where(starved, torch.full_like(rewards, -6.0), rewards)
        death_pen = -10.0 - 5.0 * fill_prog
        rewards = torch.where(died, death_pen, rewards)

        dones = died | won | starved
        terminated = died | won  # 区分真实马尔可夫终止与超时截断

        # 记录刚刚结束回合的真实得分与通关标记，供训练器统计
        final_scores = (self.length - 3).to(torch.int32)
        info = {
            "dones": dones,
            "terminated": terminated,
            "won": won,
            "died": died,
            "starved": starved,
            "final_scores": final_scores,
        }

        # 对已结束回合的子环境立即执行零同步原地重置
        self._reset_by_bool_mask(dones)

        return self.get_obs(), rewards, terminated, self.get_action_masks(), info
