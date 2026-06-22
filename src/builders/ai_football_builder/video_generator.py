from __future__ import annotations

import math
import random
from typing import Any

import pygame


CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
FPS = 60
PREVIEW_SCALE = 0.35
WINDOW_TITLE = "AI Football Builder"

GRASS_DARK = (38, 132, 58)
GRASS_LIGHT = (48, 154, 68)
LINE_COLOR = (235, 245, 235)
GOAL_COLOR = (245, 245, 245)

FIELD_MARGIN_X = 34
FIELD_MARGIN_Y = 58
STRIPE_COUNT = 10
LINE_WIDTH = 8
CENTER_CIRCLE_RADIUS = 150
PENALTY_BOX_WIDTH = 620
PENALTY_BOX_HEIGHT = 260
GOAL_BOX_WIDTH = 330
GOAL_BOX_HEIGHT = 110
GOAL_WIDTH = 300
GOAL_DEPTH = 42
GOAL_ANNOUNCEMENT_SECONDS = 2.0

TEAM_RED = "red"
TEAM_BLUE = "blue"
TEAM_ORDER = (TEAM_RED, TEAM_RED, TEAM_BLUE, TEAM_BLUE)
ROLE_ATTACKER = 0
ROLE_SUPPORT = 1

PLAYER_RADIUS = 28
PLAYER_SPEED = 430.0
PLAYER_KICK_COOLDOWN = 0.35
TEAMMATE_TOO_CLOSE_DISTANCE = 300.0
TEAMMATE_SUPPORT_MIN_DISTANCE = 330.0
TEAMMATE_SUPPORT_MAX_DISTANCE = 700.0
BALL_SUPPORT_MIN_DISTANCE = 260.0
BALL_SUPPORT_MAX_DISTANCE = 760.0
BALL_NO_TOUCH_AFTER_KICK_SECONDS = 0.10
BALL_STUCK_SPEED_THRESHOLD = 45.0
BALL_STUCK_EDGE_DISTANCE = 90.0
BALL_STUCK_RESET_SECONDS = 1.8
PASS_RECEPTION_SECONDS = 3.0
ASSIST_WINDOW_SECONDS = 6.0
TACTICAL_TARGET_DISTANCE = 420.0
PLAYER_BLUE = (50, 125, 245)
PLAYER_RED = (230, 58, 58)
PLAYER_OUTLINE = (18, 30, 38)

SCORE_PANEL_COLOR = (8, 16, 14, 170)
SCORE_TEXT_COLOR = (245, 250, 245)
ANNOUNCEMENT_TEXT_COLOR = (255, 255, 255)

BALL_RADIUS = 20
BALL_MIN_SHOT_SPEED = 520.0
BALL_MAX_SHOT_SPEED = 1080.0
BALL_PASS_MIN_SPEED = 330.0
BALL_PASS_MAX_SPEED = 650.0
BALL_ROLL_FRICTION = 300.0
BALL_BOUNCE_DAMPING = 0.72
BALL_WHITE = (245, 245, 238)
BALL_BLACK = (22, 24, 26)
MAX_BALL_SPEED = BALL_MAX_SHOT_SPEED

SHOT_POWER_FLOOR = 0.78
CLEAR_POWER_FLOOR = 0.74
PASS_POWER_FLOOR = 0.46
TEAM_NO_TOUCH_GRACE_SECONDS = 2.0
TEAM_NO_TOUCH_MAX_PENALTY = 0.075

COLLISION_PADDING = 2.0
COLLISION_SOLVER_PASSES = 10
OBSERVATION_SIZE = 35
ACTION_SIZE = 6
DEFAULT_EPISODE_SECONDS = 18.0

KICK_NONE = 0
KICK_SHOOT = 1
KICK_PASS = 2
KICK_CLEAR = 3


class FootballEnvironment:
    def __init__(self, seed: int | None = None) -> None:
        self.screen: pygame.Surface | None = None
        self.canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
        self.clock = pygame.time.Clock()
        self.is_running = False
        self.rng = random.Random(seed)
        self.score = {TEAM_RED: 0, TEAM_BLUE: 0}
        self.announcement_timer = 0.0
        self.announcement_team: str | None = None
        self.flash_time = 0.0
        self.score_font: pygame.font.Font | None = None
        self.announcement_font: pygame.font.Font | None = None
        self.small_announcement_font: pygame.font.Font | None = None
        self.elapsed_time = 0.0
        self.episode_seconds = DEFAULT_EPISODE_SECONDS
        self.last_goal_team: str | None = None
        self.last_touch_player_index: int | None = None
        self.last_kick_player_index: int | None = None
        self.last_kick_type: int = KICK_NONE
        self.last_kick_power = 0.0
        self.pending_pass_from_index: int | None = None
        self.pending_pass_team: str | None = None
        self.pending_pass_timer = 0.0
        self.pending_pass_start_x = 0.0
        self.pending_pass_start_y = 0.0
        self.last_successful_pass: tuple[int, int] | None = None
        self.recent_pass_team: str | None = None
        self.recent_pass_pair: tuple[int, int] | None = None
        self.recent_pass_timer = 0.0
        self.last_assisted_goal_team: str | None = None
        self.last_assisted_goal_pair: tuple[int, int] | None = None
        self.last_pass_reward: dict[int, float] = {}
        self.last_clear_reward: dict[int, float] = {}
        self.ball_touch_cooldown = 0.0
        self.ball_stuck_timer = 0.0
        self.team_no_touch_timers = {TEAM_RED: 0.0, TEAM_BLUE: 0.0}
        self.players: list[dict[str, float | str | tuple[int, int, int]]] = []
        self.ball: dict[str, float] = {}
        self.reset(seed=seed, keep_score=True)

    @property
    def field_rect(self) -> pygame.Rect:
        return pygame.Rect(
            FIELD_MARGIN_X,
            FIELD_MARGIN_Y,
            CANVAS_WIDTH - FIELD_MARGIN_X * 2,
            CANVAS_HEIGHT - FIELD_MARGIN_Y * 2,
        )

    def reset(self, seed: int | None = None, keep_score: bool = False) -> list[list[float]]:
        if seed is not None:
            self.rng.seed(seed)
        if not keep_score:
            self.score = {TEAM_RED: 0, TEAM_BLUE: 0}
        self.announcement_timer = 0.0
        self.announcement_team = None
        self.flash_time = 0.0
        self.elapsed_time = 0.0
        self.last_goal_team = None
        self.last_touch_player_index = None
        self.last_kick_player_index = None
        self.last_kick_type = KICK_NONE
        self.last_kick_power = 0.0
        self.pending_pass_from_index = None
        self.pending_pass_team = None
        self.pending_pass_timer = 0.0
        self.pending_pass_start_x = 0.0
        self.pending_pass_start_y = 0.0
        self.last_successful_pass = None
        self.recent_pass_team = None
        self.recent_pass_pair = None
        self.recent_pass_timer = 0.0
        self.last_assisted_goal_team = None
        self.last_assisted_goal_pair = None
        self.last_pass_reward = {}
        self.last_clear_reward = {}
        self.ball_touch_cooldown = 0.0
        self.ball_stuck_timer = 0.0
        self.team_no_touch_timers = {TEAM_RED: 0.0, TEAM_BLUE: 0.0}
        self.players = self.build_initial_players()
        self.ball = self.build_initial_ball()
        return self.get_observations()

    def build_initial_players(
        self,
    ) -> list[dict[str, float | str | tuple[int, int, int]]]:
        field = self.field_rect
        half_height = field.height / 2
        starting_zone_height = half_height / 3

        red_zone = pygame.Rect(
            field.left + PLAYER_RADIUS,
            field.top + PLAYER_RADIUS,
            field.width - PLAYER_RADIUS * 2,
            int(starting_zone_height) - PLAYER_RADIUS * 2,
        )
        blue_zone = pygame.Rect(
            field.left + PLAYER_RADIUS,
            int(field.bottom - starting_zone_height) + PLAYER_RADIUS,
            field.width - PLAYER_RADIUS * 2,
            int(starting_zone_height) - PLAYER_RADIUS * 2,
        )

        players: list[dict[str, float | str | tuple[int, int, int]]] = []
        for team, color, zone in (
            (TEAM_RED, PLAYER_RED, red_zone),
            (TEAM_BLUE, PLAYER_BLUE, blue_zone),
        ):
            for role in (ROLE_ATTACKER, ROLE_SUPPORT):
                lane_fraction = 0.28 if role == ROLE_ATTACKER else 0.72
                lane_x = zone.left + zone.width * lane_fraction
                jitter_x = self.rng.uniform(-zone.width * 0.045, zone.width * 0.045)
                if team == TEAM_RED:
                    lane_y_fraction = 0.68 if role == ROLE_ATTACKER else 0.26
                else:
                    lane_y_fraction = 0.32 if role == ROLE_ATTACKER else 0.74
                lane_y = zone.top + zone.height * lane_y_fraction
                jitter_y = self.rng.uniform(-zone.height * 0.06, zone.height * 0.06)
                players.append(
                    {
                        "x": self.clamp(
                            lane_x + jitter_x,
                            float(zone.left),
                            float(zone.right),
                        ),
                        "y": self.clamp(
                            lane_y + jitter_y,
                            float(zone.top),
                            float(zone.bottom),
                        ),
                        "team": team,
                        "role": role,
                        "color": color,
                        "kick_cooldown": 0.0,
                    }
                )
        return players

    def build_initial_ball(self) -> dict[str, float]:
        center_x, center_y = self.field_rect.center
        return {
            "x": float(center_x),
            "y": float(center_y),
            "vx": 0.0,
            "vy": 0.0,
        }

    def setup(self) -> None:
        pygame.init()
        self.ensure_fonts()
        window_size = (
            int(CANVAS_WIDTH * PREVIEW_SCALE),
            int(CANVAS_HEIGHT * PREVIEW_SCALE),
        )
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(WINDOW_TITLE)
        self.is_running = True

    def ensure_fonts(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()
        if self.score_font is None:
            self.score_font = pygame.font.SysFont("arial", 48, bold=True)
        if self.announcement_font is None:
            self.announcement_font = pygame.font.SysFont("arial", 76, bold=True)
        if self.small_announcement_font is None:
            self.small_announcement_font = pygame.font.SysFont("arial", 42, bold=True)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.is_running = False

    def step(
        self,
        actions: list[Any] | tuple[Any, ...],
        dt: float,
        training: bool = True,
    ) -> tuple[list[list[float]], list[float], bool, dict[str, Any]]:
        if self.announcement_timer > 0.0:
            self.flash_time += dt
            self.announcement_timer = max(0.0, self.announcement_timer - dt)
            if self.announcement_timer == 0.0:
                self.reset(seed=None, keep_score=True)
            return self.get_observations(), [0.0] * len(self.players), False, {
                "goal_team": None,
                "time_limit": False,
                "score": self.score.copy(),
                "kick_player": None,
                "kick_type": KICK_NONE,
                "kick_power": 0.0,
                "successful_pass": None,
                "assisted_goal_team": None,
                "min_teammate_distance": self.get_min_teammate_distance(),
                "no_touch_red": self.team_no_touch_timers[TEAM_RED],
                "no_touch_blue": self.team_no_touch_timers[TEAM_BLUE],
                "progress_delta": {TEAM_RED: 0.0, TEAM_BLUE: 0.0},
            }

        self.elapsed_time += dt
        for team in (TEAM_RED, TEAM_BLUE):
            self.team_no_touch_timers[team] += dt
        self.ball_touch_cooldown = max(0.0, self.ball_touch_cooldown - dt)
        self.pending_pass_timer = max(0.0, self.pending_pass_timer - dt)
        if self.pending_pass_timer == 0.0:
            self.pending_pass_from_index = None
            self.pending_pass_team = None
        self.recent_pass_timer = max(0.0, self.recent_pass_timer - dt)
        if self.recent_pass_timer == 0.0:
            self.recent_pass_team = None
            self.recent_pass_pair = None
        before_progress = self.get_team_ball_progress()
        before_ball_distances = self.get_player_ball_distances()
        previous_last_touch = self.last_touch_player_index
        self.last_goal_team = None
        self.last_kick_player_index = None
        self.last_kick_type = KICK_NONE
        self.last_kick_power = 0.0
        self.last_successful_pass = None
        self.last_assisted_goal_team = None
        self.last_assisted_goal_pair = None
        self.last_pass_reward = {}
        self.last_clear_reward = {}

        self.apply_player_actions(actions, dt)
        self.resolve_all_collisions()
        self.update_ball(dt, training=training)
        self.resolve_all_collisions()
        self.resolve_pending_pass_reception()

        after_progress = self.get_team_ball_progress()
        after_ball_distances = self.get_player_ball_distances()
        rewards = self.compute_rewards(
            before_progress=before_progress,
            after_progress=after_progress,
            before_ball_distances=before_ball_distances,
            after_ball_distances=after_ball_distances,
            previous_last_touch=previous_last_touch,
        )
        done = self.last_goal_team is not None or self.elapsed_time >= self.episode_seconds
        info = {
            "goal_team": self.last_goal_team,
            "time_limit": self.elapsed_time >= self.episode_seconds,
            "score": self.score.copy(),
            "kick_player": self.last_kick_player_index,
            "kick_type": self.last_kick_type,
            "kick_power": self.last_kick_power,
            "successful_pass": self.last_successful_pass,
            "assisted_goal_team": self.last_assisted_goal_team,
            "min_teammate_distance": self.get_min_teammate_distance(),
            "no_touch_red": self.team_no_touch_timers[TEAM_RED],
            "no_touch_blue": self.team_no_touch_timers[TEAM_BLUE],
            "progress_delta": {
                TEAM_RED: after_progress[TEAM_RED] - before_progress[TEAM_RED],
                TEAM_BLUE: after_progress[TEAM_BLUE] - before_progress[TEAM_BLUE],
            },
        }
        return self.get_observations(), rewards, done, info

    def update(self, dt: float) -> None:
        actions = self.build_scripted_actions()
        self.step(actions, dt, training=False)

    def apply_player_actions(self, actions: list[Any] | tuple[Any, ...], dt: float) -> None:
        if len(actions) != len(self.players):
            actions = [None] * len(self.players)

        for player_index, player in enumerate(self.players):
            action = self.normalize_action(actions[player_index], player_index)
            cooldown = max(0.0, float(player["kick_cooldown"]) - dt)
            player["kick_cooldown"] = cooldown

            move_x = float(action["move_x"])
            move_y = float(action["move_y"]) * self.team_attack_sign(str(player["team"]))
            move_length = math.hypot(move_x, move_y)
            if move_length > 1.0:
                move_x /= move_length
                move_y /= move_length

            player["x"] = float(player["x"]) + move_x * PLAYER_SPEED * dt
            player["y"] = float(player["y"]) + move_y * PLAYER_SPEED * dt
            self.keep_player_in_field(player)

            if self.can_player_kick(player) and cooldown <= 0.0:
                kick_type = int(action["kick_type"])
                if kick_type != KICK_NONE:
                    self.apply_kick(player_index, action)

    def normalize_action(self, action: Any, player_index: int) -> dict[str, float | int]:
        if isinstance(action, dict):
            return {
                "move_x": self.clamp(float(action.get("move_x", 0.0)), -1.0, 1.0),
                "move_y": self.clamp(float(action.get("move_y", 0.0)), -1.0, 1.0),
                "kick_type": int(action.get("kick_type", KICK_NONE)),
                "kick_direction_x": self.clamp(
                    float(action.get("kick_direction_x", 0.0)),
                    -1.0,
                    1.0,
                ),
                "kick_direction_y": self.clamp(
                    float(action.get("kick_direction_y", 1.0)),
                    -1.0,
                    1.0,
                ),
                "kick_power": self.clamp(float(action.get("kick_power", 0.5)), 0.0, 1.0),
            }

        if action is None:
            return {
                "move_x": 0.0,
                "move_y": 0.0,
                "kick_type": KICK_NONE,
                "kick_direction_x": 0.0,
                "kick_direction_y": 1.0,
                "kick_power": 0.5,
            }

        values = list(action)
        values = [0.0] * ACTION_SIZE if len(values) < ACTION_SIZE else values
        return {
            "move_x": self.clamp(float(values[0]), -1.0, 1.0),
            "move_y": self.clamp(float(values[1]), -1.0, 1.0),
            "kick_type": int(round(self.clamp(float(values[2]), 0.0, 3.0))),
            "kick_direction_x": self.clamp(float(values[3]), -1.0, 1.0),
            "kick_direction_y": self.clamp(float(values[4]), -1.0, 1.0),
            "kick_power": self.clamp(float(values[5]), 0.0, 1.0),
        }

    def apply_kick(self, player_index: int, action: dict[str, float | int]) -> None:
        player = self.players[player_index]
        team = str(player["team"])
        kick_type = int(action["kick_type"])
        self.resolve_pending_pass_on_touch(player_index, team)
        direction_x = float(action["kick_direction_x"])
        direction_y = float(action["kick_direction_y"]) * self.team_attack_sign(team)
        direction_length = math.hypot(direction_x, direction_y)
        if direction_length < 0.01:
            direction_x = 0.0
            direction_y = self.team_attack_sign(team)
            direction_length = 1.0
        direction_x /= direction_length
        direction_y /= direction_length

        power = float(action["kick_power"])
        if kick_type == KICK_SHOOT:
            power = max(SHOT_POWER_FLOOR, power)
        elif kick_type == KICK_CLEAR:
            power = max(CLEAR_POWER_FLOOR, power)
        elif kick_type == KICK_PASS:
            power = max(PASS_POWER_FLOOR, power)
        min_speed = BALL_PASS_MIN_SPEED if kick_type == KICK_PASS else BALL_MIN_SHOT_SPEED
        max_speed = BALL_PASS_MAX_SPEED if kick_type == KICK_PASS else BALL_MAX_SHOT_SPEED
        speed = min_speed + power * (max_speed - min_speed)
        shot_spawn_distance = PLAYER_RADIUS + BALL_RADIUS + COLLISION_PADDING

        self.ball["x"] = float(player["x"]) + direction_x * shot_spawn_distance
        self.ball["y"] = float(player["y"]) + direction_y * shot_spawn_distance
        self.ball["vx"] = direction_x * speed
        self.ball["vy"] = direction_y * speed
        self.clamp_ball_position_in_field()
        self.last_touch_player_index = player_index
        self.last_kick_player_index = player_index
        self.last_kick_type = kick_type
        self.last_kick_power = power
        self.team_no_touch_timers[team] = 0.0
        player["kick_cooldown"] = PLAYER_KICK_COOLDOWN
        self.ball_touch_cooldown = BALL_NO_TOUCH_AFTER_KICK_SECONDS
        self.record_kick_shaping(player_index, kick_type, direction_x, direction_y)
        if kick_type == KICK_PASS:
            self.pending_pass_from_index = player_index
            self.pending_pass_team = team
            self.pending_pass_timer = PASS_RECEPTION_SECONDS
            self.pending_pass_start_x = float(self.ball["x"])
            self.pending_pass_start_y = float(self.ball["y"])

    def resolve_pending_pass_on_touch(self, player_index: int, team: str) -> None:
        if self.pending_pass_from_index is None or self.pending_pass_team is None:
            return
        if player_index != self.pending_pass_from_index and team == self.pending_pass_team:
            self.mark_successful_pass(self.pending_pass_from_index, player_index, team)
        self.pending_pass_from_index = None
        self.pending_pass_team = None
        self.pending_pass_timer = 0.0

    def resolve_pending_pass_reception(self) -> None:
        if self.pending_pass_from_index is None or self.pending_pass_team is None:
            return
        reception_distance = PLAYER_RADIUS + BALL_RADIUS + COLLISION_PADDING + 12.0
        for player_index, player in enumerate(self.players):
            if player_index == self.pending_pass_from_index:
                continue
            team = str(player["team"])
            if team != self.pending_pass_team:
                continue
            distance = math.hypot(
                float(self.ball["x"]) - float(player["x"]),
                float(self.ball["y"]) - float(player["y"]),
            )
            if distance <= reception_distance:
                self.mark_successful_pass(self.pending_pass_from_index, player_index, team)
                self.pending_pass_from_index = None
                self.pending_pass_team = None
                self.pending_pass_timer = 0.0
                return

    def mark_successful_pass(self, passer_index: int, receiver_index: int, team: str) -> None:
        self.last_successful_pass = (passer_index, receiver_index)
        self.last_touch_player_index = receiver_index
        self.team_no_touch_timers[team] = 0.0
        self.recent_pass_team = team
        self.recent_pass_pair = self.last_successful_pass
        self.recent_pass_timer = ASSIST_WINDOW_SECONDS

    def record_kick_shaping(
        self,
        player_index: int,
        kick_type: int,
        direction_x: float,
        direction_y: float,
    ) -> None:
        player = self.players[player_index]
        team = str(player["team"])
        if kick_type == KICK_PASS:
            teammate_index = self.get_teammate_index(player_index)
            teammate = self.players[teammate_index]
            dx = float(teammate["x"]) - float(player["x"])
            dy = float(teammate["y"]) - float(player["y"])
            distance = max(1.0, math.hypot(dx, dy))
            alignment = (dx / distance) * direction_x + (dy / distance) * direction_y
            self.last_pass_reward[player_index] = max(0.0, alignment) * 1.15
            if alignment > 0.25:
                self.last_pass_reward[player_index] += 0.35
            if alignment > 0.65:
                self.last_pass_reward[teammate_index] = (
                    self.last_pass_reward.get(teammate_index, 0.0) + 0.45
                )

        if kick_type == KICK_CLEAR:
            own_goal_x, own_goal_y = self.get_own_goal_center(team)
            dx = float(player["x"]) - own_goal_x
            dy = float(player["y"]) - own_goal_y
            distance_from_goal = math.hypot(dx, dy)
            distance_scale = max(0.0, 1.0 - distance_from_goal / (self.field_rect.height * 0.35))
            direction_length = max(1.0, math.hypot(dx, dy))
            alignment = (dx / direction_length) * direction_x + (dy / direction_length) * direction_y
            self.last_clear_reward[player_index] = max(0.0, alignment) * distance_scale * 0.22

    def build_scripted_actions(self) -> list[dict[str, float | int]]:
        actions: list[dict[str, float | int]] = []
        ball_distances = self.get_player_ball_distances()
        team_chasers = self.get_team_chaser_indices(ball_distances)
        for player_index, player in enumerate(self.players):
            team = str(player["team"])
            attack_sign = self.team_attack_sign(team)
            target_x, target_y = self.get_tactical_target(player_index, team_chasers)
            dx = target_x - float(player["x"])
            dy = target_y - float(player["y"])
            distance = max(1.0, math.hypot(dx, dy))
            can_kick = self.can_player_kick(player)
            kick_type = KICK_NONE
            direction_x = 0.0
            direction_y = 1.0
            kick_power = 0.0
            if can_kick and self.is_ball_near_own_goal(team):
                kick_type = KICK_CLEAR
                goal_x, goal_y = self.get_enemy_goal_center(team)
                direction_x, direction_y = self.world_direction_to_action(
                    team,
                    goal_x - float(player["x"]) + self.rng.uniform(-120.0, 120.0),
                    goal_y - float(player["y"]),
                )
                kick_power = self.rng.uniform(0.82, 1.0)
            elif can_kick and self.should_script_pass(player_index):
                teammate = self.players[self.get_teammate_index(player_index)]
                direction_x, direction_y = self.world_direction_to_action(
                    team,
                    float(teammate["x"]) - float(player["x"]),
                    float(teammate["y"]) - float(player["y"]),
                )
                kick_type = KICK_PASS
                kick_power = self.rng.uniform(0.45, 0.80)
            elif can_kick:
                goal_x, goal_y = self.get_enemy_goal_center(team)
                direction_x, direction_y = self.world_direction_to_action(
                    team,
                    goal_x - float(player["x"]) + self.rng.uniform(-90.0, 90.0),
                    goal_y - float(player["y"]),
                )
                kick_type = KICK_SHOOT
                kick_power = self.rng.uniform(0.80, 1.0)
            actions.append(
                {
                    "move_x": dx / distance,
                    "move_y": (dy * attack_sign) / distance,
                    "kick_type": kick_type,
                    "kick_direction_x": direction_x,
                    "kick_direction_y": direction_y,
                    "kick_power": kick_power,
                }
            )
        return actions

    def should_script_pass(self, player_index: int) -> bool:
        player = self.players[player_index]
        teammate = self.players[self.get_teammate_index(player_index)]
        team = str(player["team"])
        attack_sign = self.team_attack_sign(team)
        dx = float(teammate["x"]) - float(player["x"])
        dy = float(teammate["y"]) - float(player["y"])
        distance = math.hypot(dx, dy)
        if not TEAMMATE_SUPPORT_MIN_DISTANCE <= distance <= TEAMMATE_SUPPORT_MAX_DISTANCE:
            return False
        teammate_forward = dy * attack_sign > -80.0
        lateral_space = abs(dx) > 110.0
        progress = self.get_team_ball_progress()[team]
        pass_bias = 0.72 if progress < 0.78 else 0.45
        return teammate_forward and lateral_space and self.rng.random() < pass_bias

    def world_direction_to_action(self, team: str, dx: float, dy: float) -> tuple[float, float]:
        distance = max(1.0, math.hypot(dx, dy))
        return dx / distance, (dy * self.team_attack_sign(team)) / distance

    def get_tactical_target(
        self,
        player_index: int,
        team_chasers: dict[str, int] | None = None,
    ) -> tuple[float, float]:
        player = self.players[player_index]
        team = str(player["team"])
        field = self.field_rect
        attack_sign = self.team_attack_sign(team)
        role = int(player.get("role", ROLE_ATTACKER))
        ball_x = float(self.ball["x"])
        ball_y = float(self.ball["y"])
        if team_chasers is None:
            team_chasers = self.get_team_chaser_indices(self.get_player_ball_distances())
        if team_chasers[team] == player_index:
            return ball_x, ball_y

        teammate = self.players[self.get_teammate_index(player_index)]
        teammate_x = float(teammate["x"])
        own_goal_x, own_goal_y = self.get_own_goal_center(team)
        enemy_goal_x, enemy_goal_y = self.get_enemy_goal_center(team)
        progress = self.get_team_ball_progress()[team]
        lane_side = -1.0 if teammate_x >= field.centerx else 1.0
        if role == ROLE_ATTACKER:
            lane_side *= -1.0

        if progress < 0.38:
            target_x = field.centerx + lane_side * field.width * 0.24
            target_y = ball_y - attack_sign * 260.0
        elif progress > 0.72:
            target_x = enemy_goal_x + lane_side * GOAL_WIDTH * 0.45
            target_y = enemy_goal_y - attack_sign * 170.0
        else:
            target_x = ball_x + lane_side * field.width * 0.24
            target_y = ball_y + attack_sign * 310.0

        if role == ROLE_SUPPORT and progress < 0.55:
            target_x = (target_x + own_goal_x) / 2.0
            target_y = (target_y + own_goal_y) / 2.0

        return self.clamp_point_to_field(target_x, target_y)

    def clamp_point_to_field(self, x: float, y: float) -> tuple[float, float]:
        field = self.field_rect
        return (
            self.clamp(float(x), float(field.left + PLAYER_RADIUS), float(field.right - PLAYER_RADIUS)),
            self.clamp(float(y), float(field.top + PLAYER_RADIUS), float(field.bottom - PLAYER_RADIUS)),
        )

    def update_ball(self, dt: float, training: bool) -> None:
        self.ball["x"] = float(self.ball["x"]) + float(self.ball["vx"]) * dt
        self.ball["y"] = float(self.ball["y"]) + float(self.ball["vy"]) * dt
        self.apply_ball_friction(dt)
        if self.handle_goal_score(training=training):
            return
        self.keep_ball_in_field()
        self.resolve_ball_stall(dt)

    def apply_ball_friction(self, dt: float) -> None:
        vx = float(self.ball["vx"])
        vy = float(self.ball["vy"])
        speed = math.hypot(vx, vy)
        if speed <= 0.0:
            return

        new_speed = max(0.0, speed - BALL_ROLL_FRICTION * dt)
        if new_speed == 0.0:
            self.ball["vx"] = 0.0
            self.ball["vy"] = 0.0
            return

        scale = new_speed / speed
        self.ball["vx"] = vx * scale
        self.ball["vy"] = vy * scale

    def resolve_all_collisions(self) -> None:
        for _ in range(COLLISION_SOLVER_PASSES):
            self.resolve_player_collisions()
            self.resolve_ball_player_collisions()
            for player in self.players:
                self.keep_player_in_field(player)
            self.keep_ball_in_field()

    def keep_player_in_field(
        self,
        player: dict[str, float | str | tuple[int, int, int]],
    ) -> None:
        field = self.field_rect
        player["x"] = max(
            float(field.left + PLAYER_RADIUS),
            min(float(field.right - PLAYER_RADIUS), float(player["x"])),
        )
        player["y"] = max(
            float(field.top + PLAYER_RADIUS),
            min(float(field.bottom - PLAYER_RADIUS), float(player["y"])),
        )

    def resolve_player_collisions(self) -> None:
        minimum_distance = PLAYER_RADIUS * 2 + COLLISION_PADDING
        for first_index, first_player in enumerate(self.players):
            for second_player in self.players[first_index + 1 :]:
                dx = float(second_player["x"]) - float(first_player["x"])
                dy = float(second_player["y"]) - float(first_player["y"])
                distance = math.hypot(dx, dy)
                if distance >= minimum_distance:
                    continue

                if distance == 0.0:
                    angle = self.rng.uniform(0.0, math.tau)
                    dx = math.cos(angle)
                    dy = math.sin(angle)
                    distance = 1.0

                overlap = minimum_distance - distance
                push_x = (dx / distance) * (overlap / 2)
                push_y = (dy / distance) * (overlap / 2)
                first_player["x"] = float(first_player["x"]) - push_x
                first_player["y"] = float(first_player["y"]) - push_y
                second_player["x"] = float(second_player["x"]) + push_x
                second_player["y"] = float(second_player["y"]) + push_y
                self.keep_player_in_field(first_player)
                self.keep_player_in_field(second_player)

    def keep_ball_in_field(self) -> None:
        field = self.field_rect
        goal_left, goal_right = self.get_goal_mouth_bounds()
        ball_x = float(self.ball["x"])
        min_x = field.left + BALL_RADIUS
        max_x = field.right - BALL_RADIUS
        min_y = field.top + BALL_RADIUS
        max_y = field.bottom - BALL_RADIUS
        top_goal_y = field.top - GOAL_DEPTH + BALL_RADIUS
        bottom_goal_y = field.bottom + GOAL_DEPTH - BALL_RADIUS
        ball_is_in_goal_mouth = goal_left <= ball_x <= goal_right

        if ball_x < min_x:
            self.ball["x"] = float(min_x)
            self.ball["vx"] = abs(float(self.ball["vx"])) * BALL_BOUNCE_DAMPING
        elif ball_x > max_x:
            self.ball["x"] = float(max_x)
            self.ball["vx"] = -abs(float(self.ball["vx"])) * BALL_BOUNCE_DAMPING

        if float(self.ball["y"]) < min_y:
            if ball_is_in_goal_mouth:
                self.ball["y"] = max(float(top_goal_y), float(self.ball["y"]))
            else:
                self.ball["y"] = float(min_y)
                self.ball["vy"] = abs(float(self.ball["vy"])) * BALL_BOUNCE_DAMPING
        elif float(self.ball["y"]) > max_y:
            if ball_is_in_goal_mouth:
                self.ball["y"] = min(float(bottom_goal_y), float(self.ball["y"]))
            else:
                self.ball["y"] = float(max_y)
                self.ball["vy"] = -abs(float(self.ball["vy"])) * BALL_BOUNCE_DAMPING

    def clamp_ball_position_in_field(self) -> None:
        field = self.field_rect
        self.ball["x"] = max(
            float(field.left + BALL_RADIUS),
            min(float(field.right - BALL_RADIUS), float(self.ball["x"])),
        )
        self.ball["y"] = max(
            float(field.top + BALL_RADIUS),
            min(float(field.bottom - BALL_RADIUS), float(self.ball["y"])),
        )

    def resolve_ball_player_collisions(self) -> None:
        if self.is_ball_inside_goal_area():
            return

        minimum_distance = PLAYER_RADIUS + BALL_RADIUS + COLLISION_PADDING
        for player in self.players:
            dx = float(self.ball["x"]) - float(player["x"])
            dy = float(self.ball["y"]) - float(player["y"])
            distance = math.hypot(dx, dy)
            if distance >= minimum_distance:
                continue

            if distance == 0.0:
                angle = self.rng.uniform(0.0, math.tau)
                dx = math.cos(angle)
                dy = math.sin(angle)
                distance = 1.0

            overlap = minimum_distance - distance
            normal_x = dx / distance
            normal_y = dy / distance
            ball_push = overlap * 0.65
            player_push = overlap - ball_push
            self.ball["x"] = float(self.ball["x"]) + normal_x * ball_push
            self.ball["y"] = float(self.ball["y"]) + normal_y * ball_push
            player["x"] = float(player["x"]) - normal_x * player_push
            player["y"] = float(player["y"]) - normal_y * player_push

            velocity_into_player = (
                float(self.ball["vx"]) * normal_x + float(self.ball["vy"]) * normal_y
            )
            if velocity_into_player < 0.0:
                self.ball["vx"] = float(self.ball["vx"]) - 2 * velocity_into_player * normal_x
                self.ball["vy"] = float(self.ball["vy"]) - 2 * velocity_into_player * normal_y

            self.keep_ball_in_field()
            self.keep_player_in_field(player)

    def handle_goal_score(self, training: bool) -> bool:
        field = self.field_rect
        goal_left, goal_right = self.get_goal_mouth_bounds()
        ball_x = float(self.ball["x"])
        ball_y = float(self.ball["y"])

        if not goal_left <= ball_x <= goal_right:
            return False

        if ball_y <= field.top - GOAL_DEPTH + BALL_RADIUS:
            self.register_goal(TEAM_BLUE, training=training)
            return True

        if ball_y >= field.bottom + GOAL_DEPTH - BALL_RADIUS:
            self.register_goal(TEAM_RED, training=training)
            return True

        return False

    def register_goal(self, team: str, training: bool = False) -> None:
        self.score[team] += 1
        self.last_goal_team = team
        if self.recent_pass_team == team and self.recent_pass_timer > 0.0:
            self.last_assisted_goal_team = team
            self.last_assisted_goal_pair = self.recent_pass_pair
        self.pending_pass_from_index = None
        self.pending_pass_team = None
        self.pending_pass_timer = 0.0
        self.recent_pass_team = None
        self.recent_pass_pair = None
        self.recent_pass_timer = 0.0
        self.ball["vx"] = 0.0
        self.ball["vy"] = 0.0
        self.ball_touch_cooldown = 0.0
        self.ball_stuck_timer = 0.0
        if not training:
            self.announcement_team = team
            self.announcement_timer = GOAL_ANNOUNCEMENT_SECONDS
            self.flash_time = 0.0

    def resolve_ball_stall(self, dt: float) -> None:
        field = self.field_rect
        ball_speed = math.hypot(float(self.ball["vx"]), float(self.ball["vy"]))
        distance_to_top = abs(float(self.ball["y"]) - float(field.top + BALL_RADIUS))
        distance_to_bottom = abs(float(self.ball["y"]) - float(field.bottom - BALL_RADIUS))
        near_end = min(distance_to_top, distance_to_bottom) <= BALL_STUCK_EDGE_DISTANCE
        if ball_speed <= BALL_STUCK_SPEED_THRESHOLD and near_end:
            self.ball_stuck_timer += dt
        else:
            self.ball_stuck_timer = 0.0

        if self.ball_stuck_timer < BALL_STUCK_RESET_SECONDS:
            return

        self.ball_stuck_timer = 0.0
        self.ball_touch_cooldown = 0.0
        field_center_x, field_center_y = field.center
        self.ball["x"] = float(field_center_x)
        self.ball["y"] = float(field_center_y)
        self.ball["vx"] = 0.0
        self.ball["vy"] = 0.0

    def compute_rewards(
        self,
        before_progress: dict[str, float],
        after_progress: dict[str, float],
        before_ball_distances: list[float],
        after_ball_distances: list[float],
        previous_last_touch: int | None,
    ) -> list[float]:
        rewards = [-0.0005] * len(self.players)
        field_height = max(1.0, float(self.field_rect.height))
        team_chasers = self.get_team_chaser_indices(after_ball_distances)

        for player_index, player in enumerate(self.players):
            team = str(player["team"])
            progress_delta = after_progress[team] - before_progress[team]
            ball_distance_delta = (
                before_ball_distances[player_index] - after_ball_distances[player_index]
            ) / field_height
            is_chaser = team_chasers[team] == player_index
            if is_chaser:
                rewards[player_index] += ball_distance_delta * 1.15
                rewards[player_index] -= (after_ball_distances[player_index] / field_height) * 0.0015
            else:
                rewards[player_index] += self.get_tactical_target_reward(
                    player_index,
                    team_chasers,
                ) * 0.28
                rewards[player_index] += self.get_support_position_reward(player_index) * 0.08
                rewards[player_index] -= self.get_ball_crowding_penalty(after_ball_distances[player_index]) * 0.11
            teammate_crowding_weight = 0.22 if is_chaser else 0.46
            rewards[player_index] -= (
                self.get_teammate_crowding_penalty(player_index) * teammate_crowding_weight
            )
            rewards[player_index] -= self.get_wall_stall_penalty(player_index) * 0.06
            if progress_delta >= 0.0:
                rewards[player_index] += progress_delta * 6.0
            else:
                rewards[player_index] += progress_delta * 1.5
            rewards[player_index] += self.get_forward_ball_velocity_reward(team) * 0.025
            rewards[player_index] += self.get_attacking_lane_reward(team) * 0.010
            rewards[player_index] -= self.get_no_touch_penalty(team)
            rewards[player_index] += self.last_pass_reward.get(player_index, 0.0)
            rewards[player_index] += self.last_clear_reward.get(player_index, 0.0)
            if self.can_player_kick(player):
                rewards[player_index] += 0.04
            if self.last_kick_player_index == player_index:
                rewards[player_index] += 0.45
                if self.last_kick_type == KICK_PASS:
                    rewards[player_index] += 0.85
                elif self.last_kick_type == KICK_CLEAR and self.is_ball_near_own_goal(team):
                    rewards[player_index] += 0.50

        if (
            previous_last_touch is not None
            and self.last_touch_player_index is not None
            and previous_last_touch != self.last_touch_player_index
        ):
            previous_team = str(self.players[previous_last_touch]["team"])
            current_team = str(self.players[self.last_touch_player_index]["team"])
            if previous_team == current_team:
                rewards[previous_last_touch] += 0.22
                rewards[self.last_touch_player_index] += 0.22

        if self.last_successful_pass is not None:
            passer_index, receiver_index = self.last_successful_pass
            rewards[passer_index] += 7.0
            rewards[receiver_index] += 5.5
            passer_team = str(self.players[passer_index]["team"])
            for player_index, player in enumerate(self.players):
                if str(player["team"]) == passer_team:
                    rewards[player_index] += 1.2

        if self.last_assisted_goal_team is not None:
            for player_index, player in enumerate(self.players):
                if str(player["team"]) == self.last_assisted_goal_team:
                    rewards[player_index] += 10.0
            if self.last_assisted_goal_pair is not None:
                passer_index, receiver_index = self.last_assisted_goal_pair
                rewards[passer_index] += 8.0
                rewards[receiver_index] += 8.0

        if self.last_goal_team is not None:
            for player_index, player in enumerate(self.players):
                if str(player["team"]) == self.last_goal_team:
                    rewards[player_index] += 30.0
                else:
                    rewards[player_index] -= 30.0

        return rewards

    def get_observations(self) -> list[list[float]]:
        return [self.get_observation(player_index) for player_index in range(len(self.players))]

    def get_observation(self, player_index: int) -> list[float]:
        player = self.players[player_index]
        team = str(player["team"])
        field = self.field_rect
        field_width = max(1.0, float(field.width))
        field_height = max(1.0, float(field.height))
        attack_sign = self.team_attack_sign(team)
        player_x = float(player["x"])
        player_y = float(player["y"])
        ball_x = float(self.ball["x"])
        ball_y = float(self.ball["y"])
        ball_vx = float(self.ball["vx"])
        ball_vy = float(self.ball["vy"])
        ball_speed_raw = math.hypot(ball_vx, ball_vy)
        ball_speed = self.clamp(ball_speed_raw / MAX_BALL_SPEED, 0.0, 1.5)
        if ball_speed_raw > 0.0:
            ball_dir_x = ball_vx / ball_speed_raw
            ball_dir_y = (ball_vy / ball_speed_raw) * attack_sign
        else:
            ball_dir_x = 0.0
            ball_dir_y = 0.0

        teammate = self.players[self.get_teammate_index(player_index)]
        opponents = self.get_opponents(player_index)
        enemy_goal_x, enemy_goal_y = self.get_enemy_goal_center(team)
        own_goal_x, own_goal_y = self.get_own_goal_center(team)
        score_diff = self.score[team] - self.score[self.other_team(team)]
        teammate_distance = math.hypot(
            float(teammate["x"]) - player_x,
            float(teammate["y"]) - player_y,
        )
        ball_distances = self.get_player_ball_distances()
        team_chasers = self.get_team_chaser_indices(ball_distances)
        tactical_target_x, tactical_target_y = self.get_tactical_target(player_index, team_chasers)
        if self.last_touch_player_index is None:
            possession = 0.0
        else:
            possession_team = str(self.players[self.last_touch_player_index]["team"])
            possession = 1.0 if possession_team == team else -1.0

        observation = [
            self.clamp((player_x - field.centerx) / (field_width / 2), -1.5, 1.5),
            self.clamp(((player_y - field.centery) * attack_sign) / (field_height / 2), -1.5, 1.5),
            1.0 if team == TEAM_RED else -1.0,
            self.rel_x(ball_x, player_x, field_width),
            self.rel_y(ball_y, player_y, field_height, attack_sign),
            self.clamp(ball_vx / MAX_BALL_SPEED, -1.5, 1.5),
            self.clamp((ball_vy * attack_sign) / MAX_BALL_SPEED, -1.5, 1.5),
            ball_speed,
            self.clamp(ball_dir_x, -1.0, 1.0),
            self.clamp(ball_dir_y, -1.0, 1.0),
            self.rel_x(float(teammate["x"]), player_x, field_width),
            self.rel_y(float(teammate["y"]), player_y, field_height, attack_sign),
            self.rel_x(float(opponents[0]["x"]), player_x, field_width),
            self.rel_y(float(opponents[0]["y"]), player_y, field_height, attack_sign),
            self.rel_x(float(opponents[1]["x"]), player_x, field_width),
            self.rel_y(float(opponents[1]["y"]), player_y, field_height, attack_sign),
            self.rel_x(enemy_goal_x, player_x, field_width),
            self.rel_y(enemy_goal_y, player_y, field_height, attack_sign),
            self.rel_x(own_goal_x, player_x, field_width),
            self.rel_y(own_goal_y, player_y, field_height, attack_sign),
            self.clamp(math.hypot(ball_x - player_x, ball_y - player_y) / field_height, 0.0, 1.5),
            1.0 if self.can_player_kick(player) else 0.0,
            self.clamp(float(player["kick_cooldown"]) / PLAYER_KICK_COOLDOWN, 0.0, 1.0),
            self.clamp(score_diff / 5.0, -1.0, 1.0),
            self.clamp(1.0 - self.elapsed_time / self.episode_seconds, 0.0, 1.0),
            -1.0 if int(player.get("role", ROLE_ATTACKER)) == ROLE_ATTACKER else 1.0,
            1.0 if team_chasers[team] == player_index else 0.0,
            self.clamp(teammate_distance / field_height, 0.0, 1.5),
            self.rel_x(tactical_target_x, player_x, field_width),
            self.rel_y(tactical_target_y, player_y, field_height, attack_sign),
            possession,
            self.rel_x(enemy_goal_x, ball_x, field_width),
            self.rel_y(enemy_goal_y, ball_y, field_height, attack_sign),
            self.rel_x(own_goal_x, ball_x, field_width),
            self.rel_y(own_goal_y, ball_y, field_height, attack_sign),
        ]
        return observation

    def get_team_ball_progress(self) -> dict[str, float]:
        field = self.field_rect
        return {
            TEAM_RED: self.clamp((float(self.ball["y"]) - field.top) / field.height, -0.2, 1.2),
            TEAM_BLUE: self.clamp((field.bottom - float(self.ball["y"])) / field.height, -0.2, 1.2),
        }

    def get_player_ball_distances(self) -> list[float]:
        return [
            math.hypot(
                float(self.ball["x"]) - float(player["x"]),
                float(self.ball["y"]) - float(player["y"]),
            )
            for player in self.players
        ]

    def get_team_chaser_indices(self, ball_distances: list[float]) -> dict[str, int]:
        chasers: dict[str, int] = {}
        for team in (TEAM_RED, TEAM_BLUE):
            team_indices = [
                index
                for index, player in enumerate(self.players)
                if str(player["team"]) == team
            ]
            chasers[team] = min(team_indices, key=lambda index: ball_distances[index])
        return chasers

    def get_support_position_reward(self, player_index: int) -> float:
        player = self.players[player_index]
        team = str(player["team"])
        attack_sign = self.team_attack_sign(team)
        teammate = self.players[self.get_teammate_index(player_index)]
        distance_to_ball = math.hypot(
            float(self.ball["x"]) - float(player["x"]),
            float(self.ball["y"]) - float(player["y"]),
        )
        distance_to_teammate = math.hypot(
            float(teammate["x"]) - float(player["x"]),
            float(teammate["y"]) - float(player["y"]),
        )
        forward_of_ball = (
            (float(player["y"]) - float(self.ball["y"])) * attack_sign
        )
        lateral_offset = abs(float(player["x"]) - float(self.ball["x"]))

        ball_spacing = self.distance_window_score(
            distance_to_ball,
            BALL_SUPPORT_MIN_DISTANCE,
            BALL_SUPPORT_MAX_DISTANCE,
        )
        teammate_spacing = self.distance_window_score(
            distance_to_teammate,
            TEAMMATE_SUPPORT_MIN_DISTANCE,
            TEAMMATE_SUPPORT_MAX_DISTANCE,
        )
        forward_support = self.clamp(forward_of_ball / (self.field_rect.height * 0.25), 0.0, 1.0)
        diagonal_lane = self.clamp(lateral_offset / (self.field_rect.width * 0.28), 0.0, 1.0)
        return (
            ball_spacing * 0.35
            + teammate_spacing * 0.30
            + forward_support * 0.20
            + diagonal_lane * 0.15
        )

    def get_tactical_target_reward(
        self,
        player_index: int,
        team_chasers: dict[str, int] | None = None,
    ) -> float:
        player = self.players[player_index]
        target_x, target_y = self.get_tactical_target(player_index, team_chasers)
        distance = math.hypot(
            target_x - float(player["x"]),
            target_y - float(player["y"]),
        )
        return self.clamp(1.0 - distance / TACTICAL_TARGET_DISTANCE, 0.0, 1.0)

    def get_wall_stall_penalty(self, player_index: int) -> float:
        player = self.players[player_index]
        field = self.field_rect
        y = float(player["y"])
        near_top = max(0.0, 1.0 - (y - field.top) / 140.0)
        near_bottom = max(0.0, 1.0 - (field.bottom - y) / 140.0)
        return self.clamp(max(near_top, near_bottom), 0.0, 1.0)

    def get_no_touch_penalty(self, team: str) -> float:
        stale_time = max(0.0, self.team_no_touch_timers[team] - TEAM_NO_TOUCH_GRACE_SECONDS)
        if stale_time <= 0.0:
            return 0.0
        return min(TEAM_NO_TOUCH_MAX_PENALTY, 0.012 + stale_time * 0.018)

    def get_ball_crowding_penalty(self, distance_to_ball: float) -> float:
        if distance_to_ball >= BALL_SUPPORT_MIN_DISTANCE:
            return 0.0
        return 1.0 - distance_to_ball / BALL_SUPPORT_MIN_DISTANCE

    def get_teammate_crowding_penalty(self, player_index: int) -> float:
        player = self.players[player_index]
        teammate = self.players[self.get_teammate_index(player_index)]
        distance = math.hypot(
            float(teammate["x"]) - float(player["x"]),
            float(teammate["y"]) - float(player["y"]),
        )
        if distance >= TEAMMATE_TOO_CLOSE_DISTANCE:
            return 0.0
        return 1.0 - distance / TEAMMATE_TOO_CLOSE_DISTANCE

    def get_min_teammate_distance(self) -> float:
        distances: list[float] = []
        for player_index, player in enumerate(self.players):
            teammate = self.players[self.get_teammate_index(player_index)]
            distances.append(
                math.hypot(
                    float(teammate["x"]) - float(player["x"]),
                    float(teammate["y"]) - float(player["y"]),
                )
            )
        return min(distances) if distances else 0.0

    def distance_window_score(
        self,
        distance: float,
        minimum: float,
        maximum: float,
    ) -> float:
        if distance < minimum:
            return self.clamp(distance / minimum, 0.0, 1.0)
        if distance > maximum:
            falloff = max(maximum, 1.0)
            return self.clamp(1.0 - (distance - maximum) / falloff, 0.0, 1.0)
        return 1.0

    def get_forward_ball_velocity_reward(self, team: str) -> float:
        attack_velocity = float(self.ball["vy"]) * self.team_attack_sign(team)
        return self.clamp(attack_velocity / MAX_BALL_SPEED, -1.0, 1.0)

    def get_attacking_lane_reward(self, team: str) -> float:
        enemy_goal_x, _enemy_goal_y = self.get_enemy_goal_center(team)
        goal_alignment = 1.0 - abs(float(self.ball["x"]) - enemy_goal_x) / (self.field_rect.width / 2)
        progress = self.get_team_ball_progress()[team]
        return self.clamp(goal_alignment, 0.0, 1.0) * self.clamp(progress, 0.0, 1.0)

    def can_player_kick(self, player: dict[str, float | str | tuple[int, int, int]]) -> bool:
        if self.ball_touch_cooldown > 0.0:
            return False
        distance = math.hypot(
            float(self.ball["x"]) - float(player["x"]),
            float(self.ball["y"]) - float(player["y"]),
        )
        return distance <= PLAYER_RADIUS + BALL_RADIUS + COLLISION_PADDING + 3.0

    def get_teammate_index(self, player_index: int) -> int:
        team = str(self.players[player_index]["team"])
        for index, player in enumerate(self.players):
            if index != player_index and str(player["team"]) == team:
                return index
        return player_index

    def get_opponents(self, player_index: int) -> list[dict[str, float | str | tuple[int, int, int]]]:
        team = str(self.players[player_index]["team"])
        return [player for player in self.players if str(player["team"]) != team]

    def get_goal_mouth_bounds(self) -> tuple[float, float]:
        center_x = self.field_rect.centerx
        return (
            float(center_x - GOAL_WIDTH / 2 + BALL_RADIUS),
            float(center_x + GOAL_WIDTH / 2 - BALL_RADIUS),
        )

    def get_enemy_goal_center(self, team: str) -> tuple[float, float]:
        field = self.field_rect
        return (
            float(field.centerx),
            float(field.bottom + GOAL_DEPTH / 2 if team == TEAM_RED else field.top - GOAL_DEPTH / 2),
        )

    def get_own_goal_center(self, team: str) -> tuple[float, float]:
        field = self.field_rect
        return (
            float(field.centerx),
            float(field.top - GOAL_DEPTH / 2 if team == TEAM_RED else field.bottom + GOAL_DEPTH / 2),
        )

    def is_ball_near_own_goal(self, team: str) -> bool:
        own_goal_x, own_goal_y = self.get_own_goal_center(team)
        return math.hypot(float(self.ball["x"]) - own_goal_x, float(self.ball["y"]) - own_goal_y) < self.field_rect.height * 0.28

    def is_ball_inside_goal_area(self) -> bool:
        field = self.field_rect
        return float(self.ball["y"]) < field.top or float(self.ball["y"]) > field.bottom

    def team_attack_sign(self, team: str) -> float:
        return 1.0 if team == TEAM_RED else -1.0

    def other_team(self, team: str) -> str:
        return TEAM_BLUE if team == TEAM_RED else TEAM_RED

    def rel_x(self, target_x: float, source_x: float, field_width: float) -> float:
        return self.clamp((target_x - source_x) / field_width, -1.5, 1.5)

    def rel_y(
        self,
        target_y: float,
        source_y: float,
        field_height: float,
        attack_sign: float,
    ) -> float:
        return self.clamp(((target_y - source_y) * attack_sign) / field_height, -1.5, 1.5)

    @staticmethod
    def clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def draw_grass(self) -> None:
        self.canvas.fill(GRASS_DARK)
        stripe_width = CANVAS_WIDTH / STRIPE_COUNT
        for index in range(STRIPE_COUNT):
            if index % 2 == 0:
                stripe_rect = pygame.Rect(
                    int(index * stripe_width),
                    0,
                    int(stripe_width) + 2,
                    CANVAS_HEIGHT,
                )
                pygame.draw.rect(self.canvas, GRASS_LIGHT, stripe_rect)

    def draw_field_lines(self) -> None:
        field = self.field_rect
        center_x = field.centerx
        center_y = field.centery

        pygame.draw.rect(self.canvas, LINE_COLOR, field, LINE_WIDTH)
        pygame.draw.line(
            self.canvas,
            LINE_COLOR,
            (field.left, center_y),
            (field.right, center_y),
            LINE_WIDTH,
        )
        pygame.draw.circle(
            self.canvas,
            LINE_COLOR,
            (center_x, center_y),
            CENTER_CIRCLE_RADIUS,
            LINE_WIDTH,
        )
        pygame.draw.circle(self.canvas, LINE_COLOR, (center_x, center_y), 12)

        top_penalty = pygame.Rect(
            center_x - PENALTY_BOX_WIDTH // 2,
            field.top,
            PENALTY_BOX_WIDTH,
            PENALTY_BOX_HEIGHT,
        )
        bottom_penalty = pygame.Rect(
            center_x - PENALTY_BOX_WIDTH // 2,
            field.bottom - PENALTY_BOX_HEIGHT,
            PENALTY_BOX_WIDTH,
            PENALTY_BOX_HEIGHT,
        )
        pygame.draw.rect(self.canvas, LINE_COLOR, top_penalty, LINE_WIDTH)
        pygame.draw.rect(self.canvas, LINE_COLOR, bottom_penalty, LINE_WIDTH)

        top_goal_box = pygame.Rect(
            center_x - GOAL_BOX_WIDTH // 2,
            field.top,
            GOAL_BOX_WIDTH,
            GOAL_BOX_HEIGHT,
        )
        bottom_goal_box = pygame.Rect(
            center_x - GOAL_BOX_WIDTH // 2,
            field.bottom - GOAL_BOX_HEIGHT,
            GOAL_BOX_WIDTH,
            GOAL_BOX_HEIGHT,
        )
        pygame.draw.rect(self.canvas, LINE_COLOR, top_goal_box, LINE_WIDTH)
        pygame.draw.rect(self.canvas, LINE_COLOR, bottom_goal_box, LINE_WIDTH)

        top_goal = pygame.Rect(
            center_x - GOAL_WIDTH // 2,
            field.top - GOAL_DEPTH,
            GOAL_WIDTH,
            GOAL_DEPTH,
        )
        bottom_goal = pygame.Rect(
            center_x - GOAL_WIDTH // 2,
            field.bottom,
            GOAL_WIDTH,
            GOAL_DEPTH,
        )
        pygame.draw.rect(self.canvas, GOAL_COLOR, top_goal, LINE_WIDTH)
        pygame.draw.rect(self.canvas, GOAL_COLOR, bottom_goal, LINE_WIDTH)

    def draw_scoreboard(self) -> None:
        self.ensure_fonts()
        if self.score_font is None:
            return

        field = self.field_rect
        score_items = (
            ("RED", self.score[TEAM_RED], PLAYER_RED, field.top + 18),
            ("BLUE", self.score[TEAM_BLUE], PLAYER_BLUE, field.bottom - 82),
        )

        for label, score, color, top in score_items:
            panel = pygame.Surface((300, 64), pygame.SRCALPHA)
            panel.fill(SCORE_PANEL_COLOR)
            panel_rect = panel.get_rect(centerx=field.centerx, top=top)
            self.canvas.blit(panel, panel_rect)
            score_text = self.score_font.render(
                f"{label} {score}",
                True,
                SCORE_TEXT_COLOR,
            )
            text_rect = score_text.get_rect(center=panel_rect.center)
            pygame.draw.circle(
                self.canvas,
                color,
                (panel_rect.left + 34, panel_rect.centery),
                14,
            )
            self.canvas.blit(score_text, text_rect)

    def draw_goal_announcement(self) -> None:
        if self.announcement_timer <= 0.0 or self.announcement_team is None:
            return
        self.ensure_fonts()
        if self.announcement_font is None or self.small_announcement_font is None:
            return

        team_color = PLAYER_RED if self.announcement_team == TEAM_RED else PLAYER_BLUE
        flash = 0.5 + 0.5 * math.sin(self.flash_time * 7.0)
        overlay_alpha = int(24 + flash * 26)
        overlay = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
        overlay.fill((*team_color, overlay_alpha))
        self.canvas.blit(overlay, (0, 0))

        center = (CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2)
        banner = pygame.Surface((650, 170), pygame.SRCALPHA)
        banner.fill((10, 15, 20, 185))
        banner_rect = banner.get_rect(center=center)
        self.canvas.blit(banner, banner_rect)

        team_label = self.announcement_team.upper()
        title_text = self.announcement_font.render(
            f"{team_label} SCORES!",
            True,
            ANNOUNCEMENT_TEXT_COLOR,
        )
        title_rect = title_text.get_rect(center=(center[0], center[1] - 34))
        self.canvas.blit(title_text, title_rect)

        footer_text = self.small_announcement_font.render(
            f"RED {self.score[TEAM_RED]}  -  BLUE {self.score[TEAM_BLUE]}",
            True,
            team_color,
        )
        footer_rect = footer_text.get_rect(center=(center[0], center[1] + 48))
        self.canvas.blit(footer_text, footer_rect)

    def draw_players(self) -> None:
        for player in self.players:
            center = (int(float(player["x"])), int(float(player["y"])))
            pygame.draw.circle(self.canvas, PLAYER_OUTLINE, center, PLAYER_RADIUS + 5)
            pygame.draw.circle(self.canvas, player["color"], center, PLAYER_RADIUS)  # type: ignore[arg-type]

    def draw_ball(self) -> None:
        center = (int(float(self.ball["x"])), int(float(self.ball["y"])))
        pygame.draw.circle(self.canvas, BALL_BLACK, center, BALL_RADIUS + 3)
        pygame.draw.circle(self.canvas, BALL_WHITE, center, BALL_RADIUS)
        pygame.draw.polygon(
            self.canvas,
            BALL_BLACK,
            [
                (center[0], center[1] - 9),
                (center[0] + 9, center[1] - 3),
                (center[0] + 6, center[1] + 9),
                (center[0] - 6, center[1] + 9),
                (center[0] - 9, center[1] - 3),
            ],
        )
        panel_offsets = ((0, -16), (15, -3), (9, 14), (-9, 14), (-15, -3))
        for offset_x, offset_y in panel_offsets:
            pygame.draw.circle(
                self.canvas,
                BALL_BLACK,
                (center[0] + offset_x, center[1] + offset_y),
                5,
            )
        pygame.draw.circle(self.canvas, BALL_WHITE, center, BALL_RADIUS, 2)

    def render_frame(self) -> pygame.Surface:
        self.draw_grass()
        self.draw_field_lines()
        self.draw_ball()
        self.draw_players()
        self.draw_scoreboard()
        self.draw_goal_announcement()
        return self.canvas

    def render(self) -> None:
        if self.screen is None:
            return
        scaled_canvas = pygame.transform.smoothscale(
            self.render_frame(),
            (
                int(CANVAS_WIDTH * PREVIEW_SCALE),
                int(CANVAS_HEIGHT * PREVIEW_SCALE),
            ),
        )
        self.screen.blit(scaled_canvas, (0, 0))
        pygame.display.flip()

    def run(self) -> None:
        self.setup()
        while self.is_running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()
        pygame.quit()


FootballFieldPreview = FootballEnvironment


def run_preview() -> None:
    FootballEnvironment().run()


if __name__ == "__main__":
    run_preview()
