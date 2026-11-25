"""高性能无头贪吃蛇核心物理引擎与空间感知算法。

与图形渲染完全解耦，支持高速强化学习采样，并提供：
- 相对动作（直行/左转/右转）与绝对动作（上/右/下/左，带防 180° 反向自杀保护）
- 精准的蛇尾同步移动碰撞判定（走向即将离开的旧蛇尾合法）
- 8 向障碍物射线测距与 3 向连通域可达空间计算（Flood-Fill）
- 内置启发式基线智能体（Heuristic Agent），用于可视化演示与训练基线对比
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

from snake_rl.config import SnakeConfig


class Direction(IntEnum):
    """顺时针排列的四个移动方向，便于通过模运算直接左转/右转。"""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


# 各方向对应的 (dx, dy) 坐标增量（左上角为原点，x 向右递增，y 向下递增）
DIRECTION_DELTAS: tuple[tuple[int, int], ...] = (
    (0, -1),  # UP
    (1, 0),   # RIGHT
    (0, 1),   # DOWN
    (-1, 0),  # LEFT
)

# 相对于正前方的 8 个方位角（每步旋转 45°，以基准方向为 UP=(0,-1) 定义，运行时按当前朝向旋转）
_BASE_8_RAYS_UP: tuple[tuple[int, int], ...] = (
    (0, -1),   # 正前 (F)
    (1, -1),   # 右前 (FR)
    (1, 0),    # 正右 (R)
    (1, 1),    # 右后 (BR)
    (0, 1),    # 正后 (B)
    (-1, 1),   # 左后 (BL)
    (-1, 0),   # 正左 (L)
    (-1, -1),  # 左前 (FL)
)


def _rotate_vec(vec: tuple[int, int], direction: Direction) -> tuple[int, int]:
    """将基准朝上（UP）的二维方向向量顺时针旋转至指定朝向。"""
    vx, vy = vec
    if direction == Direction.UP:
        return vx, vy
    if direction == Direction.RIGHT:
        return -vy, vx
    if direction == Direction.DOWN:
        return -vx, -vy
    return vy, -vx  # Direction.LEFT


# 预计算所有朝向下的 8 向射线增量表，消除运行期分支计算开销
ROTATED_8_RAYS: dict[Direction, tuple[tuple[int, int], ...]] = {
    d: tuple(_rotate_vec(v, d) for v in _BASE_8_RAYS_UP) for d in Direction
}

# 网格状态编码常量
CELL_EMPTY = 0
CELL_BODY = 1
CELL_HEAD = 2
CELL_FOOD = 3


@dataclass(slots=True)
class StepResult:
    """单步物理演进结果数据载体。"""

    ate_food: bool
    terminated: bool
    truncated: bool
    death_reason: str  # "none" | "wall" | "self" | "starvation" | "win"
    prev_food_dist: int
    curr_food_dist: int


class SnakeCore:
    """无头贪吃蛇核心状态机。"""

    def __init__(
        self,
        config: SnakeConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or SnakeConfig()
        if self.config.grid_width < 5 or self.config.grid_height < 5:
            raise ValueError("棋盘宽高至少需要为 5x5 以保证初始蛇身与食物正常生成。")
        self.rng = rng or np.random.default_rng()

        self.width = self.config.grid_width
        self.height = self.config.grid_height
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.snake: deque[tuple[int, int]] = deque()
        self.direction: Direction = Direction.RIGHT
        self.food: tuple[int, int] | None = None

        self.score: int = 0
        self.steps: int = 0
        self.steps_since_food: int = 0
        self.done: bool = False
        self.death_reason: str = "none"

        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """重置棋盘、蛇身初始位置与随机食物。"""
        if rng is not None:
            self.rng = rng

        self.board.fill(CELL_EMPTY)
        self.snake.clear()

        cx = self.width // 2
        cy = self.height // 2
        self.direction = Direction.RIGHT

        # 蛇头在 (cx, cy)，身体向左延伸
        init_len = min(self.config.initial_length, cx + 1)
        for i in range(init_len):
            pos = (cx - i, cy)
            self.snake.append(pos)
            self.board[pos[1], pos[0]] = CELL_HEAD if i == 0 else CELL_BODY

        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.death_reason = "none"
        self._spawn_food()

    @property
    def head(self) -> tuple[int, int]:
        """返回当前蛇头坐标 (x, y)。"""
        return self.snake[0]

    @property
    def tail(self) -> tuple[int, int]:
        """返回当前蛇尾坐标 (x, y)。"""
        return self.snake[-1]

    @property
    def length(self) -> int:
        """返回当前蛇身总长度。"""
        return len(self.snake)

    def resolve_direction(self, action: int, action_mode: str = "relative") -> Direction:
        """将离散动作映射为下一步实际移动方向。

        - relative 模式: 0=直行, 1=左转, 2=右转
        - absolute 模式: 0=上, 1=右, 2=下, 3=左（若与当前方向相反则忽略反向，保持直行）
        """
        if action_mode == "relative":
            if action == 0:
                return self.direction
            if action == 1:
                return Direction((int(self.direction) - 1) % 4)
            if action == 2:
                return Direction((int(self.direction) + 1) % 4)
            raise ValueError(f"非法的相对动作索引: {action}（应为 0, 1, 2）")

        if action_mode == "absolute":
            if action not in (0, 1, 2, 3):
                raise ValueError(f"非法的绝对动作索引: {action}（应为 0, 1, 2, 3）")
            candidate = Direction(action)
            # 禁止 180 度原地掉头自杀（蛇长 > 1 时）
            if len(self.snake) > 1 and (int(candidate) - int(self.direction)) % 4 == 2:
                return self.direction
            return candidate

        raise ValueError(f"未知的 action_mode: {action_mode}")

    def manhattan_to_food(self, pos: tuple[int, int] | None = None) -> int:
        """计算指定坐标（默认蛇头）到当前食物的曼哈顿距离。"""
        if self.food is None:
            return 0
        px, py = pos if pos is not None else self.head
        return abs(px - self.food[0]) + abs(py - self.food[1])

    def is_collision(self, pos: tuple[int, int], ignore_tail: bool = True) -> bool:
        """判断给定坐标 (x, y) 是否发生越界或身体碰撞。

        当 ignore_tail=True 且下一步并非吃到食物时，旧蛇尾会同步腾出空间，不视为碰撞。
        """
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        cell = self.board[y, x]
        if cell in (CELL_EMPTY, CELL_FOOD):
            return False
        if ignore_tail and pos == self.tail:
            return False
        return True

    def step(self, action: int, action_mode: str = "relative") -> StepResult:
        """执行一步环境交互并返回物理演进结果。"""
        if self.done:
            raise RuntimeError("当前回合已结束，请先调用 reset()。")

        prev_dist = self.manhattan_to_food()
        self.direction = self.resolve_direction(action, action_mode)
        dx, dy = DIRECTION_DELTAS[int(self.direction)]
        hx, hy = self.head
        nx, ny = hx + dx, hy + dy
        next_pos = (nx, ny)

        self.steps += 1
        self.steps_since_food += 1

        # 1. 墙壁碰撞检查
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            self.done = True
            self.death_reason = "wall"
            return StepResult(
                ate_food=False,
                terminated=True,
                truncated=False,
                death_reason="wall",
                prev_food_dist=prev_dist,
                curr_food_dist=prev_dist,
            )

        will_eat = self.food is not None and next_pos == self.food

        # 2. 若未吃到食物，先移出蛇尾（从而允许蛇头合法踏入旧蛇尾位置）
        if not will_eat:
            tx, ty = self.snake.pop()
            self.board[ty, tx] = CELL_EMPTY

        # 3. 自身碰撞检查
        if self.board[ny, nx] in (CELL_BODY, CELL_HEAD):
            self.done = True
            self.death_reason = "self"
            return StepResult(
                ate_food=False,
                terminated=True,
                truncated=False,
                death_reason="self",
                prev_food_dist=prev_dist,
                curr_food_dist=prev_dist,
            )

        # 4. 推进蛇头
        self.board[hy, hx] = CELL_BODY
        self.snake.appendleft(next_pos)
        self.board[ny, nx] = CELL_HEAD

        if will_eat:
            self.score += 1
            self.steps_since_food = 0
            # 检查是否填满全图通关
            if len(self.snake) >= self.config.total_cells:
                self.food = None
                self.done = True
                self.death_reason = "win"
                return StepResult(
                    ate_food=True,
                    terminated=True,
                    truncated=False,
                    death_reason="win",
                    prev_food_dist=prev_dist,
                    curr_food_dist=0,
                )
            self._spawn_food()
            curr_dist = self.manhattan_to_food()
            return StepResult(
                ate_food=True,
                terminated=False,
                truncated=False,
                death_reason="none",
                prev_food_dist=prev_dist,
                curr_food_dist=curr_dist,
            )

        curr_dist = self.manhattan_to_food()

        # 5. 饥饿步数截断检查（属于时间/步数上限截断 truncated，而非自然死亡 terminated）
        if self.steps_since_food >= self.config.max_starvation_steps:
            self.done = True
            self.death_reason = "starvation"
            return StepResult(
                ate_food=False,
                terminated=False,
                truncated=True,
                death_reason="starvation",
                prev_food_dist=prev_dist,
                curr_food_dist=curr_dist,
            )

        return StepResult(
            ate_food=False,
            terminated=False,
            truncated=False,
            death_reason="none",
            prev_food_dist=prev_dist,
            curr_food_dist=curr_dist,
        )

    def _spawn_food(self) -> None:
        """在棋盘所有空闲格子上等概率随机生成食物。"""
        empty_ys, empty_xs = np.where(self.board == CELL_EMPTY)
        if len(empty_xs) == 0:
            self.food = None
            return
        idx = int(self.rng.integers(0, len(empty_xs)))
        fx, fy = int(empty_xs[idx]), int(empty_ys[idx])
        self.food = (fx, fy)
        self.board[fy, fx] = CELL_FOOD

    def get_relative_candidates(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """返回正前方、左转、右转三个相对动作对应的下一格坐标。"""
        hx, hy = self.head
        d_straight = self.direction
        d_left = Direction((int(self.direction) - 1) % 4)
        d_right = Direction((int(self.direction) + 1) % 4)

        sx, sy = DIRECTION_DELTAS[int(d_straight)]
        lx, ly = DIRECTION_DELTAS[int(d_left)]
        rx, ry = DIRECTION_DELTAS[int(d_right)]
        return (hx + sx, hy + sy), (hx + lx, hy + ly), (hx + rx, hy + ry)

    def compute_8way_rays(self) -> tuple[np.ndarray, list[tuple[tuple[int, int], bool]]]:
        """计算蛇头相对坐标系 8 个方向的障碍物逆距离 (1 / steps) 与射线终点列表。

        返回:
            inv_distances: shape=(8,) float32 数组，值域 (0, 1]，1.0 表示紧邻 1 格就是障碍物。
            endpoints: 长度为 8 的列表，每项为 (终点网格坐标, 是否紧邻危险)，供可视化渲染激光测距线。
        """
        hx, hy = self.head
        tail_pos = self.tail
        rays = ROTATED_8_RAYS[self.direction]
        inv_dists = np.zeros(8, dtype=np.float32)
        endpoints: list[tuple[tuple[int, int], bool]] = []

        for i, (dx, dy) in enumerate(rays):
            steps = 0
            cx, cy = hx, hy
            while True:
                steps += 1
                cx += dx
                cy += dy
                if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                    # 到达边界墙外半格位置（裁剪回边缘用于绘制）
                    end_x = min(max(cx, 0), self.width - 1)
                    end_y = min(max(cy, 0), self.height - 1)
                    endpoints.append(((end_x, end_y), steps == 1))
                    break
                cell = self.board[cy, cx]
                if cell in (CELL_BODY, CELL_HEAD) and not (steps == 1 and (cx, cy) == tail_pos):
                    endpoints.append(((cx, cy), steps == 1))
                    break
            inv_dists[i] = 1.0 / float(steps)

        return inv_dists, endpoints

    def compute_reachable_ratios(self) -> np.ndarray:
        """计算直行、左转、右转三个方向的连通可达空间比例，值域 [0.0, 1.0]。

        利用无向图连通分量等价性进行加速：若多个候选格处于同一连通空地分量内，
        一次一维 BFS 遍历即可同时得出所有同分量候选格的精确可达面积。
        """
        candidates = self.get_relative_candidates()
        if not self.config.enable_flood_fill:
            return np.array(
                [0.0 if self.is_collision(pos, ignore_tail=True) else 1.0 for pos in candidates],
                dtype=np.float32,
            )

        w, h = self.width, self.height
        total = w * h
        free_total = max(1, total - len(self.snake) + 1)
        tail_x, tail_y = self.tail
        tail_idx = tail_y * w + tail_x

        # 构建一维可通行布尔表（空地、食物、即将移开的旧蛇尾均可通行）
        flat_board = self.board.ravel()
        passable = (flat_board == CELL_EMPTY) | (flat_board == CELL_FOOD)
        passable[tail_idx] = True
        pass_list = passable.tolist()

        comp_id = [0] * total
        comp_sizes: dict[int, int] = {}
        next_cid = 0
        ratios = np.zeros(3, dtype=np.float32)

        for act_idx, (sx, sy) in enumerate(candidates):
            if sx < 0 or sx >= w or sy < 0 or sy >= h:
                continue
            s_flat = sy * w + sx
            if not pass_list[s_flat]:
                continue

            existing_cid = comp_id[s_flat]
            if existing_cid != 0:
                cnt = comp_sizes[existing_cid]
                ratios[act_idx] = min(1.0, cnt / free_total)
                continue

            next_cid += 1
            cid = next_cid
            comp_id[s_flat] = cid
            queue = [s_flat]
            q_head = 0
            while q_head < len(queue):
                curr = queue[q_head]
                q_head += 1
                cx = curr % w
                # 上
                if curr >= w:
                    nb = curr - w
                    if pass_list[nb] and comp_id[nb] == 0:
                        comp_id[nb] = cid
                        queue.append(nb)
                # 下
                if curr + w < total:
                    nb = curr + w
                    if pass_list[nb] and comp_id[nb] == 0:
                        comp_id[nb] = cid
                        queue.append(nb)
                # 左
                if cx > 0:
                    nb = curr - 1
                    if pass_list[nb] and comp_id[nb] == 0:
                        comp_id[nb] = cid
                        queue.append(nb)
                # 右
                if cx + 1 < w:
                    nb = curr + 1
                    if pass_list[nb] and comp_id[nb] == 0:
                        comp_id[nb] = cid
                        queue.append(nb)

            cnt = len(queue)
            comp_sizes[cid] = cnt
            ratios[act_idx] = min(1.0, cnt / free_total)

        return ratios

    def heuristic_action(self) -> int:
        """内置启发式智能体（相对动作 0=直行, 1=左转, 2=右转）。

        策略：排除碰撞方向 -> 优先选择连通空间充足的方向并最小化到食物的曼哈顿距离；
        若所有安全方向的连通空间均不足以容纳蛇身，则选择连通空间最大的方向（保命绕圈）。
        """
        candidates = self.get_relative_candidates()
        ratios = self.compute_reachable_ratios()
        free_total = max(1, self.config.total_cells - len(self.snake) + 1)
        min_safe_ratio = min(0.85, float(len(self.snake)) / float(free_total))

        safe_actions: list[int] = []
        spacious_actions: list[int] = []
        for act_idx, pos in enumerate(candidates):
            if not self.is_collision(pos, ignore_tail=True):
                safe_actions.append(act_idx)
                if ratios[act_idx] >= min_safe_ratio:
                    spacious_actions.append(act_idx)

        if not safe_actions:
            return 0

        # 若有空间充足的安全动作，在其中挑离食物最近的；相等时偏爱连通度更高者
        if spacious_actions:
            return min(
                spacious_actions,
                key=lambda a: (self.manhattan_to_food(candidates[a]), -float(ratios[a])),
            )

        # 若陷入局部包围，选择连通空间最大的动作争取等到蛇尾移开
        return max(safe_actions, key=lambda a: (float(ratios[a]), -self.manhattan_to_food(candidates[a])))
