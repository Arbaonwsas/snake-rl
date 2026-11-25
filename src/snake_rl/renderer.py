"""Pygame 现代化游戏画面与强化学习实时遥测可视化渲染器。

特性：
- 同时支持桌面窗口交互模式（human）与纯内存离屏帧渲染模式（rgb_array，无需 X11 弹窗即可截图/录视频）
- 无缝连体渐变蛇身、随朝向转动的蛇头双眼、脉动光晕食物
- 可选叠加 RL 8 向激光测距射线与前/左/右安全指示点
- 右侧集成实时遥测仪表盘：分数、填充率、饥饿进度条、三向连通空间条形图与最近 80 步即时奖励折线图
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import pygame

from snake_rl.config import RenderConfig, SnakeConfig
from snake_rl.core import Direction

if TYPE_CHECKING:
    from snake_rl.env import SnakeEnv


def _lerp_color(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """在两个 RGB 颜色之间按比例 t in [0, 1] 线性插值。"""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _load_cjk_or_fallback_font(size: int, bold: bool = False) -> pygame.font.Font:
    """自动匹配 Linux 常见中文字体，若不存在则回退到 Pygame 默认字体。"""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "Ubuntu",
        "DejaVu Sans",
    ]
    matched = pygame.font.match_font(candidates, bold=bold)
    if matched:
        return pygame.font.Font(matched, size)
    return pygame.font.SysFont(None, size, bold=bold)


class SnakeRenderer:
    """贪吃蛇可视化与强化学习传感器遥测渲染器。"""

    def __init__(
        self,
        snake_config: SnakeConfig,
        render_config: RenderConfig | None = None,
        window_enabled: bool = True,
    ) -> None:
        self.sc = snake_config
        self.rc = render_config or RenderConfig()
        self.window_enabled = window_enabled
        self.show_sensors: bool = self.rc.show_sensors_default
        self.paused: bool = False

        if not window_enabled and "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.init()
        pygame.font.init()

        self.board_px_w = self.sc.grid_width * self.rc.cell_size
        self.board_px_h = self.sc.grid_height * self.rc.cell_size
        self.win_w = self.board_px_w + self.rc.board_padding * 2 + self.rc.panel_width
        self.win_h = max(620, self.board_px_h + self.rc.board_padding * 2)

        # 棋盘在画布上的垂直居中偏移
        self.board_x = self.rc.board_padding
        self.board_y = (self.win_h - self.board_px_h) // 2

        if self.window_enabled:
            pygame.display.set_caption(self.rc.window_title)
            self.surface = pygame.display.set_mode((self.win_w, self.win_h))
        else:
            self.surface = pygame.Surface((self.win_w, self.win_h))

        self.clock = pygame.time.Clock()
        self._frame_tick: int = 0

        # 字体初始化
        self.font_title = _load_cjk_or_fallback_font(20, bold=True)
        self.font_value = _load_cjk_or_fallback_font(24, bold=True)
        self.font_label = _load_cjk_or_fallback_font(14, bold=False)
        self.font_small = _load_cjk_or_fallback_font(12, bold=False)
        self.font_banner = _load_cjk_or_fallback_font(28, bold=True)

    def _cell_rect(self, gx: int, gy: int, inset: int = 0) -> pygame.Rect:
        """获取网格坐标 (gx, gy) 对应的像素矩形区域。"""
        cs = self.rc.cell_size
        return pygame.Rect(
            self.board_x + gx * cs + inset,
            self.board_y + gy * cs + inset,
            cs - inset * 2,
            cs - inset * 2,
        )

    def _cell_center(self, gx: int, gy: int) -> tuple[int, int]:
        """获取网格坐标 (gx, gy) 的中心像素坐标。"""
        cs = self.rc.cell_size
        return (
            self.board_x + gx * cs + cs // 2,
            self.board_y + gy * cs + cs // 2,
        )

    def render(
        self,
        env: SnakeEnv,
        mode_label: str = "HUMAN PLAY",
        return_rgb: bool = False,
        fps_target: int | None = None,
    ) -> np.ndarray | None:
        """执行一整帧绘制（棋盘 + 蛇 + 食物 + 传感器叠加层 + 遥测侧边栏）。"""
        self._frame_tick += 1
        self.surface.fill(self.rc.bg_color)

        self._draw_board()
        if self.show_sensors and not env.core.done:
            self._draw_sensor_overlay(env)
        self._draw_food(env)
        self._draw_snake(env)
        if env.core.done:
            self._draw_gameover_overlay(env)

        self._draw_telemetry_panel(env, mode_label=mode_label, fps_target=fps_target)

        if self.window_enabled:
            pygame.display.flip()
            if fps_target is not None and fps_target > 0:
                self.clock.tick(fps_target)

        if return_rgb:
            # pygame.surfarray.array3d 返回 (W, H, 3)，转置为标准图像格式 (H, W, 3)
            arr = pygame.surfarray.array3d(self.surface)
            return np.transpose(arr, (1, 0, 2)).copy()
        return None

    def _draw_board(self) -> None:
        """绘制外边框与交替暗色网格棋盘。"""
        outer = pygame.Rect(
            self.board_x - 3,
            self.board_y - 3,
            self.board_px_w + 6,
            self.board_px_h + 6,
        )
        pygame.draw.rect(self.surface, self.rc.board_border_color, outer, border_radius=10)

        for gy in range(self.sc.grid_height):
            for gx in range(self.sc.grid_width):
                color = self.rc.cell_dark if (gx + gy) % 2 == 0 else self.rc.cell_light
                pygame.draw.rect(self.surface, color, self._cell_rect(gx, gy))

    def _draw_food(self, env: SnakeEnv) -> None:
        """绘制带呼吸脉冲光晕的圆形食物。"""
        if env.core.food is None:
            return
        fx, fy = env.core.food
        cx, cy = self._cell_center(fx, fy)
        cs = self.rc.cell_size

        pulse = 0.5 + 0.5 * math.sin(self._frame_tick * 0.2)
        glow_radius = int(cs * (0.34 + 0.08 * pulse))
        glow_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surf,
            (*self.rc.food_glow_color, int(55 + 45 * pulse)),
            (cs // 2, cs // 2),
            glow_radius,
        )
        self.surface.blit(glow_surf, (cx - cs // 2, cy - cs // 2))

        core_radius = max(6, int(cs * 0.28))
        pygame.draw.circle(self.surface, self.rc.food_color, (cx, cy), core_radius)
        # 顶部高光小点
        pygame.draw.circle(
            self.surface,
            (255, 210, 215),
            (cx - core_radius // 3, cy - core_radius // 3),
            max(2, core_radius // 4),
        )

    def _draw_snake(self, env: SnakeEnv) -> None:
        """绘制连体渐变圆角蛇身与带朝向双眼的蛇头。"""
        segments = list(env.core.snake)
        total = max(1, len(segments) - 1)
        inset = max(3, self.rc.cell_size // 10)
        cs = self.rc.cell_size

        # 先画相邻节段之间的连接桥，消除圆角缝隙
        for idx in range(len(segments) - 1):
            x1, y1 = segments[idx]
            x2, y2 = segments[idx + 1]
            t = idx / total
            color = _lerp_color(self.rc.snake_body_start, self.rc.snake_body_end, t)
            r1 = self._cell_rect(x1, y1, inset=inset)
            r2 = self._cell_rect(x2, y2, inset=inset)
            bridge = r1.union(r2)
            pygame.draw.rect(self.surface, color, bridge, border_radius=max(6, cs // 4))

        # 从蛇尾向蛇头依次绘制节段主体
        for idx in range(len(segments) - 1, 0, -1):
            bx, by = segments[idx]
            t = idx / total
            color = _lerp_color(self.rc.snake_body_start, self.rc.snake_body_end, t)
            pygame.draw.rect(
                self.surface,
                color,
                self._cell_rect(bx, by, inset=inset),
                border_radius=max(6, cs // 4),
            )

        # 绘制蛇头与眼睛
        hx, hy = segments[0]
        head_rect = self._cell_rect(hx, hy, inset=max(2, inset - 1))
        pygame.draw.rect(
            self.surface,
            self.rc.snake_head_color,
            head_rect,
            border_radius=max(8, cs // 3),
        )

        cx, cy = head_rect.center
        eye_offset_fwd = cs // 6
        eye_offset_side = cs // 5
        d = env.core.direction
        if d == Direction.UP:
            eyes = [(cx - eye_offset_side, cy - eye_offset_fwd), (cx + eye_offset_side, cy - eye_offset_fwd)]
        elif d == Direction.DOWN:
            eyes = [(cx - eye_offset_side, cy + eye_offset_fwd), (cx + eye_offset_side, cy + eye_offset_fwd)]
        elif d == Direction.LEFT:
            eyes = [(cx - eye_offset_fwd, cy - eye_offset_side), (cx - eye_offset_fwd, cy + eye_offset_side)]
        else:
            eyes = [(cx + eye_offset_fwd, cy - eye_offset_side), (cx + eye_offset_fwd, cy + eye_offset_side)]

        for ex, ey in eyes:
            pygame.draw.circle(self.surface, (245, 250, 255), (ex, ey), max(3, cs // 9))
            pygame.draw.circle(self.surface, (18, 24, 34), (ex, ey), max(2, cs // 16))

    def _draw_sensor_overlay(self, env: SnakeEnv) -> None:
        """在棋盘上绘制 RL 智能体的 8 向激光测距线与前/左/右安全指示点。"""
        _, endpoints = env.core.compute_8way_rays()
        hx, hy = env.core.head
        start_px = self._cell_center(hx, hy)

        overlay = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        for (ex, ey), immediate_danger in endpoints:
            end_px = self._cell_center(ex, ey)
            line_color = (*self.rc.accent_red, 110) if immediate_danger else (*self.rc.accent_blue, 55)
            pygame.draw.line(overlay, line_color, start_px, end_px, 2)

        # 在直行、左转、右转三个候选落点画状态小圆点
        candidates = env.core.get_relative_candidates()
        ratios = env.last_reachable_ratios
        for idx, (cx_g, cy_g) in enumerate(candidates):
            if 0 <= cx_g < self.sc.grid_width and 0 <= cy_g < self.sc.grid_height:
                px, py = self._cell_center(cx_g, cy_g)
                if env.core.is_collision((cx_g, cy_g), ignore_tail=True):
                    dot_col = (*self.rc.accent_red, 190)
                elif ratios[idx] < 0.35:
                    dot_col = (*self.rc.accent_yellow, 180)
                else:
                    dot_col = (*self.rc.accent_green, 170)
                pygame.draw.circle(overlay, dot_col, (px, py), 5)

        self.surface.blit(overlay, (0, 0))

    def _draw_gameover_overlay(self, env: SnakeEnv) -> None:
        """回合结束时在棋盘中央绘制半透明横幅提示。"""
        overlay = pygame.Surface((self.board_px_w, self.board_px_h), pygame.SRCALPHA)
        overlay.fill((12, 16, 24, 175))
        self.surface.blit(overlay, (self.board_x, self.board_y))

        reason_map = {
            "wall": ("撞墙出局 (WALL)", self.rc.accent_red),
            "self": ("咬到自身 (SELF)", self.rc.accent_red),
            "starvation": ("饥饿截断 (STARVED)", self.rc.accent_yellow),
            "win": ("完美通关！(PERFECT WIN)", self.rc.accent_green),
        }
        title_str, color = reason_map.get(env.core.death_reason, ("回合结束", self.rc.text_primary))
        title_surf = self.font_banner.render(title_str, True, color)
        sub_surf = self.font_label.render(
            f"得分: {env.core.score}   |   按 R 重开 / Tab 切换 AI",
            True,
            self.rc.text_primary,
        )

        cx = self.board_x + self.board_px_w // 2
        cy = self.board_y + self.board_px_h // 2
        self.surface.blit(title_surf, title_surf.get_rect(center=(cx, cy - 16)))
        self.surface.blit(sub_surf, sub_surf.get_rect(center=(cx, cy + 20)))

    def _draw_telemetry_panel(
        self,
        env: SnakeEnv,
        mode_label: str,
        fps_target: int | None,
    ) -> None:
        """在窗口右侧绘制强化学习实时遥测仪表盘。"""
        px = self.board_x + self.board_px_w + self.rc.board_padding
        pw = self.rc.panel_width - self.rc.board_padding
        panel_rect = pygame.Rect(px - 10, 12, pw + 10, self.win_h - 24)
        pygame.draw.rect(self.surface, self.rc.panel_bg, panel_rect, border_radius=12)
        pygame.draw.rect(self.surface, self.rc.card_border, panel_rect, width=1, border_radius=12)

        x0 = px + 6
        cw = pw - 22
        y = 26

        # 1. 标题与模式徽章
        title_surf = self.font_title.render("Snake RL 遥测台", True, self.rc.text_primary)
        self.surface.blit(title_surf, (x0, y))

        badge_text = "PAUSED" if self.paused else mode_label
        badge_col = self.rc.accent_yellow if self.paused else self.rc.accent_blue
        badge_surf = self.font_small.render(f" {badge_text} ", True, (12, 16, 24))
        badge_rect = badge_surf.get_rect(topright=(x0 + cw, y + 3))
        pygame.draw.rect(self.surface, badge_col, badge_rect.inflate(10, 6), border_radius=6)
        self.surface.blit(badge_surf, badge_rect)
        y += 36

        # 2. 四宫格核心统计卡片
        fill_pct = 100.0 * env.core.length / self.sc.total_cells
        stats = [
            ("当前得分 SCORE", f"{env.core.score}", self.rc.accent_green),
            ("历史最佳 BEST", f"{env.best_score}", self.rc.accent_yellow),
            ("存活步数 STEPS", f"{env.core.steps}", self.rc.text_primary),
            ("空间占满 FILL", f"{fill_pct:.1f}%", self.rc.accent_blue),
        ]
        card_w = (cw - 10) // 2
        card_h = 58
        for i, (lbl, val, col) in enumerate(stats):
            row, col_idx = divmod(i, 2)
            rx = x0 + col_idx * (card_w + 10)
            ry = y + row * (card_h + 8)
            crect = pygame.Rect(rx, ry, card_w, card_h)
            pygame.draw.rect(self.surface, self.rc.card_bg, crect, border_radius=8)
            pygame.draw.rect(self.surface, self.rc.card_border, crect, width=1, border_radius=8)
            self.surface.blit(self.font_small.render(lbl, True, self.rc.text_secondary), (rx + 10, ry + 7))
            self.surface.blit(self.font_value.render(val, True, col), (rx + 10, ry + 25))

        y += (card_h + 8) * 2 + 6

        # 3. 饥饿步数耐心进度条
        max_starv = self.sc.max_starvation_steps
        rem_ratio = max(0.0, 1.0 - env.core.steps_since_food / max_starv)
        bar_col = (
            self.rc.accent_green
            if rem_ratio > 0.5
            else (self.rc.accent_yellow if rem_ratio > 0.2 else self.rc.accent_red)
        )
        starv_lbl = f"饥饿剩余步数: {max_starv - env.core.steps_since_food} / {max_starv}"
        self.surface.blit(self.font_small.render(starv_lbl, True, self.rc.text_secondary), (x0, y))
        y += 18
        bg_bar = pygame.Rect(x0, y, cw, 10)
        pygame.draw.rect(self.surface, self.rc.card_bg, bg_bar, border_radius=5)
        if rem_ratio > 0:
            fg_bar = pygame.Rect(x0, y, max(6, int(cw * rem_ratio)), 10)
            pygame.draw.rect(self.surface, bar_col, fg_bar, border_radius=5)
        y += 22

        # 4. 三向动作连通空间安全性透视条（若激活 DQN 模式则同步展示预测 Q 值）
        sec_hdr = (
            "动作空间透视 (Flood-Fill & Q值)"
            if env.last_q_values is not None
            else "相对动作可达空间 (Flood-Fill)"
        )
        self.surface.blit(
            self.font_label.render(sec_hdr, True, self.rc.text_primary),
            (x0, y),
        )
        y += 22
        candidates = env.core.get_relative_candidates()
        ratios = env.last_reachable_ratios
        q_vals = env.last_q_values
        best_q_idx = int(np.argmax(q_vals)) if q_vals is not None else -1
        act_names = ["[0] 直行", "[1] 左转", "[2] 右转"]
        for idx, short_name in enumerate(act_names):
            danger = env.core.is_collision(candidates[idx], ignore_tail=True)
            r_val = 0.0 if danger else float(ratios[idx])
            c = self.rc.accent_red if danger else (self.rc.accent_yellow if r_val < 0.35 else self.rc.accent_green)

            if q_vals is not None:
                prefix = "★" if idx == best_q_idx else " "
                lbl_str = f"{prefix}{short_name} Q:{q_vals[idx]:+5.1f}"
                lbl_col = self.rc.accent_blue if idx == best_q_idx else self.rc.text_secondary
            else:
                lbl_str = f"{short_name} Straight" if idx == 0 else (f"{short_name} Turn L  " if idx == 1 else f"{short_name} Turn R  ")
                lbl_col = self.rc.text_secondary

            self.surface.blit(self.font_small.render(lbl_str, True, lbl_col), (x0, y))
            bar_x = x0 + 126
            bar_w = cw - 174
            pygame.draw.rect(self.surface, self.rc.card_bg, pygame.Rect(bar_x, y + 3, bar_w, 10), border_radius=4)
            if r_val > 0:
                pygame.draw.rect(
                    self.surface,
                    c,
                    pygame.Rect(bar_x, y + 3, max(4, int(bar_w * r_val)), 10),
                    border_radius=4,
                )
            pct_str = "COLLIDE" if danger else f"{r_val * 100:4.0f}%"
            self.surface.blit(self.font_small.render(pct_str, True, c), (bar_x + bar_w + 6, y))
            y += 20

        y += 6

        # 5. 即时奖励曲线图 (Reward Sparkline) 与累计回报
        ret_col = self.rc.accent_green if env.episode_return >= 0 else self.rc.accent_red
        ret_hdr = f"累计回报 Return: {env.episode_return:+.2f}  (步奖: {env.last_reward:+.2f})"
        self.surface.blit(self.font_label.render(ret_hdr, True, ret_col), (x0, y))
        y += 22

        chart_h = 74
        chart_rect = pygame.Rect(x0, y, cw, chart_h)
        pygame.draw.rect(self.surface, self.rc.card_bg, chart_rect, border_radius=8)
        pygame.draw.rect(self.surface, self.rc.card_border, chart_rect, width=1, border_radius=8)

        # 绘制 0 基准虚线
        zero_y = y + chart_h // 2
        pygame.draw.line(self.surface, self.rc.board_border_color, (x0 + 6, zero_y), (x0 + cw - 6, zero_y), 1)

        hist = list(env.reward_history)
        if len(hist) >= 2:
            max_abs = max(1.0, max(abs(v) for v in hist))
            pts: list[tuple[int, int]] = []
            for i, val in enumerate(hist):
                px_i = x0 + 8 + int((cw - 16) * (i / max(1, len(hist) - 1)))
                py_i = int(zero_y - (val / max_abs) * (chart_h * 0.40))
                pts.append((px_i, py_i))
            pygame.draw.lines(self.surface, self.rc.accent_blue, False, pts, 2)

        y += chart_h + 14

        # 6. 底部快捷键说明
        actual_fps = self.clock.get_fps()
        fps_info = f"回合 #{env.episode_count}  |  目标帧率: {fps_target or '-'} FPS (实测 {actual_fps:4.1f})"
        self.surface.blit(self.font_small.render(fps_info, True, self.rc.text_secondary), (x0, y))
        y += 20

        help_lines = [
            "方向键 / WASD : 人工控制移动",
            "Tab / M       : 切换 [人工 / DQN / 启发式AI]",
            "Space / N     : 暂停继续 / 暂停时单步调试",
            "V             : 开关 RL 激光雷达与连通度透视",
            "[ / ] (或 -/+) : 实时调节运行速度 FPS",
            "R / Esc       : 重开当前局 / 退出程序",
        ]
        for line in help_lines:
            if y + 15 <= self.win_h - 14:
                self.surface.blit(self.font_small.render(line, True, self.rc.text_secondary), (x0, y))
                y += 16

    def close(self) -> None:
        """安全关闭 Pygame 显示子系统。"""
        pygame.quit()
