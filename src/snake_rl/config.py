"""贪吃蛇强化学习环境与可视化全局配置定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SnakeConfig:
    """贪吃蛇核心物理规则与观测参数配置。"""

    grid_width: int = 12
    grid_height: int = 12
    initial_length: int = 3
    # 最大未进食步数 = max(min_starvation_steps, grid_width * grid_height * starvation_factor)
    starvation_factor: int = 2
    min_starvation_steps: int = 100
    # 是否计算前/左/右三向连通域可达空间比例（帮助智能体识别死胡同）
    enable_flood_fill: bool = True

    @property
    def total_cells(self) -> int:
        """返回棋盘总格数。"""
        return self.grid_width * self.grid_height

    @property
    def max_starvation_steps(self) -> int:
        """返回触发饥饿截断的最大步数上限。"""
        return max(self.min_starvation_steps, self.total_cells * self.starvation_factor)


@dataclass(slots=True)
class RewardConfig:
    """强化学习奖励塑造（Reward Shaping）参数配置。

    采用基于潜势差（Potential-based）的距离引导设计，既能加速早期冷启动探索，
    又避免智能体通过来回移动刷分改变最优策略。
    """

    food_base: float = 10.0
    food_length_scale: float = 0.15
    win_bonus: float = 50.0
    death_penalty: float = -10.0
    starvation_penalty: float = -5.0
    # 距离潜势系数：每靠近食物 1 格曼哈顿距离获得 +coef，远离 1 格获得 -coef（设为 0.0 即纯稀疏奖励）
    distance_shaping_coef: float = 0.1
    step_cost: float = -0.01


@dataclass(slots=True)
class RenderConfig:
    """Pygame 可视化渲染与实时遥测面板样式配置。"""

    cell_size: int = 44
    board_padding: int = 24
    panel_width: int = 350
    human_fps: int = 10
    ai_fps: int = 24
    window_title: str = "Snake RL Studio — 贪吃蛇强化学习实验台"
    show_sensors_default: bool = True

    # 深色科技主题配色 (R, G, B)
    bg_color: tuple[int, int, int] = (15, 19, 28)
    board_border_color: tuple[int, int, int] = (45, 56, 78)
    cell_dark: tuple[int, int, int] = (22, 28, 40)
    cell_light: tuple[int, int, int] = (27, 34, 48)
    panel_bg: tuple[int, int, int] = (20, 25, 36)
    card_bg: tuple[int, int, int] = (28, 35, 50)
    card_border: tuple[int, int, int] = (46, 58, 82)

    snake_head_color: tuple[int, int, int] = (74, 242, 161)
    snake_body_start: tuple[int, int, int] = (46, 213, 137)
    snake_body_end: tuple[int, int, int] = (32, 132, 214)
    food_color: tuple[int, int, int] = (255, 92, 108)
    food_glow_color: tuple[int, int, int] = (255, 138, 148)

    text_primary: tuple[int, int, int] = (236, 242, 250)
    text_secondary: tuple[int, int, int] = (146, 162, 188)
    accent_green: tuple[int, int, int] = (56, 224, 140)
    accent_yellow: tuple[int, int, int] = (250, 196, 68)
    accent_red: tuple[int, int, int] = (255, 92, 108)
    accent_blue: tuple[int, int, int] = (76, 168, 255)
