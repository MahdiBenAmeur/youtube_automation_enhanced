from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import warnings

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
import pygame


CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
RENDER_SCALE = 2
RENDER_WIDTH = CANVAS_WIDTH * RENDER_SCALE
RENDER_HEIGHT = CANVAS_HEIGHT * RENDER_SCALE
WINDOW_TITLE = "Quoridor Simulation"
PREVIEW_SCALE = 0.35
FPS = 60
TURN_INTERVAL_FRAMES = 6
ENABLE_WALL_FADE = True
WALL_FADE_FRAMES = 3
WALL_FADE_START_ALPHA = 140
BLUE_PLAYER_MODE = "search"
RED_PLAYER_MODE = "model"
PLAYER_MODE_ORDER = ("random", "search", "model")
MODEL_CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "training_checkpoints" / "quoridor_builder"
MODEL_CHECKPOINT_PATH = MODEL_CHECKPOINT_DIR / "quoridor_best.pt"
MODEL_FALLBACK_CHECKPOINT_PATH = MODEL_CHECKPOINT_DIR / "quoridor_latest.pt"
MODEL_PLAYER_TEMPERATURE = 0.65
VISUAL_SEARCH_DEPTH = 3
VISUAL_SEARCH_MAX_WALL_CANDIDATES = 10
VISUAL_SEARCH_MAX_ROOT_WALL_CANDIDATES = 10
VISUAL_SEARCH_MAX_ACTIONS = 16
VISUAL_SEARCH_MAX_ROOT_ACTIONS = 16
WALL_ACTION_CHANCE = 0.22
FORWARD_MOVE_WEIGHT = 5
SIDEWAYS_MOVE_WEIGHT = 2
BACKWARD_MOVE_WEIGHT = 1

BOARD_SIZE = 9
STARTING_WALLS = 10
MAX_TURNS = 100
WALL_GRID_SIZE = BOARD_SIZE - 1
MOVE_ACTION_COUNT = 8
WALL_PLACEMENT_COUNT = WALL_GRID_SIZE * WALL_GRID_SIZE
ACTION_SIZE = MOVE_ACTION_COUNT + 2 * WALL_PLACEMENT_COUNT
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
BLOCKED_PATH_DISTANCE = CELL_COUNT
WALL_BITS = tuple(1 << index for index in range(WALL_PLACEMENT_COUNT))
WALL_BIT_GRID = tuple(
    tuple(WALL_BITS[row * WALL_GRID_SIZE + col] for col in range(WALL_GRID_SIZE))
    for row in range(WALL_GRID_SIZE)
)
OBSERVATION_BASE_SIZE = 16
ACTION_SIGNAL_PLANES = 4
OBSERVATION_SIZE = (
    OBSERVATION_BASE_SIZE
    + MOVE_ACTION_COUNT
    + 2 * WALL_PLACEMENT_COUNT
    + ACTION_SIGNAL_PLANES * ACTION_SIZE
)
BOARD_PIXEL_SIZE = 1000
BOARD_RENDER_SIZE = BOARD_PIXEL_SIZE * RENDER_SCALE
BOARD_MARGIN_X = (RENDER_WIDTH - BOARD_RENDER_SIZE) // 2
BOARD_MARGIN_Y = (RENDER_HEIGHT - BOARD_RENDER_SIZE) // 2
BOARD_LABEL_GUTTER = 18 * RENDER_SCALE
GRID_GAP = 22 * RENDER_SCALE
CELL_SIZE = (BOARD_RENDER_SIZE - GRID_GAP * (BOARD_SIZE - 1)) / BOARD_SIZE

BACKGROUND_COLOR = (0, 0, 0)
CELL_COLOR = (251, 251, 247)
TOP_ROW_CELL_COLOR = (255, 241, 239)
BOTTOM_ROW_CELL_COLOR = (235, 242, 248)
GRID_COLOR = (219, 226, 235)
BOARD_EDGE_COLOR = (184, 195, 207)
LABEL_COLOR = (139, 154, 170)
PLAYER_BLUE = (53, 124, 244)
PLAYER_RED = (255, 72, 64)
WALL_BLUE = (33, 104, 239)
WALL_RED = (255, 58, 59)
WALL_SHADOW = (65, 83, 115, 105)
COUNTER_BLUE = (53, 124, 244)
COUNTER_RED = (255, 72, 64)
COUNTER_TEXT = (255, 255, 255)
WIN_PANEL_COLOR = (18, 22, 29, 232)
WIN_TEXT_COLOR = (255, 255, 255)
WIN_DIM_COLOR = (0, 0, 0, 155)

PLAYER_RADIUS = 25 * RENDER_SCALE
WALL_THICKNESS = 19 * RENDER_SCALE
WALL_LENGTH = CELL_SIZE * 2 + GRID_GAP
CELL_RADIUS = 0
BOARD_RADIUS = 14 * RENDER_SCALE
WALL_RADIUS = 2 * RENDER_SCALE
COUNTER_RADIUS = 5 * RENDER_SCALE
WIN_PANEL_RADIUS = 24 * RENDER_SCALE

BLUE = "blue"
RED = "red"
MOVE_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CARDINAL_MOVE_ACTIONS = (0, 1, 2, 3)
DIAGONAL_MOVE_ACTIONS = (4, 5, 6, 7)


@dataclass(frozen=True)
class Wall:
    orientation: str
    row: int
    col: int
    owner: str


class QuoridorGame:
    def __init__(self, seed: int | None = None, starting_walls: int = STARTING_WALLS) -> None:
        self.rng = random.Random(seed)
        self.starting_walls = starting_walls
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self.players = {
            BLUE: (0, BOARD_SIZE // 2),
            RED: (BOARD_SIZE - 1, BOARD_SIZE // 2),
        }
        self.wall_counts = {BLUE: self.starting_walls, RED: self.starting_walls}
        self.walls: list[Wall] = []
        self.horizontal_walls: set[tuple[int, int]] = set()
        self.vertical_walls: set[tuple[int, int]] = set()
        self.horizontal_wall_mask = 0
        self.vertical_wall_mask = 0
        self.turn = BLUE
        self.winner: str | None = None
        self.win_reason: str | None = None
        self.turn_number = 0
        self._path_detail_cache: dict[tuple[object, str], dict[int, dict[str, float]]] = {}
        self._shortest_path_cache: dict[tuple[object, ...], int] = {}

    def clone(self) -> QuoridorGame:
        clone = QuoridorGame()
        clone.rng.setstate(self.rng.getstate())
        clone.starting_walls = self.starting_walls
        clone.players = dict(self.players)
        clone.wall_counts = dict(self.wall_counts)
        clone.walls = list(self.walls)
        clone.horizontal_walls = set(self.horizontal_walls)
        clone.vertical_walls = set(self.vertical_walls)
        clone.horizontal_wall_mask = self.horizontal_wall_mask
        clone.vertical_wall_mask = self.vertical_wall_mask
        clone.turn = self.turn
        clone.winner = self.winner
        clone.win_reason = self.win_reason
        clone.turn_number = self.turn_number
        clone._path_detail_cache = self._path_detail_cache
        clone._shortest_path_cache = self._shortest_path_cache
        return clone

    def state_key(self) -> tuple[object, ...]:
        return (
            self.turn,
            self.players[BLUE],
            self.players[RED],
            self.wall_counts[BLUE],
            self.wall_counts[RED],
            self.horizontal_wall_mask,
            self.vertical_wall_mask,
            self.turn_number,
            self.winner,
        )

    def wall_bit(self, row: int, col: int) -> int:
        return WALL_BIT_GRID[row][col]

    def wall_set_mask(self, walls: set[tuple[int, int]]) -> int:
        mask = 0
        for row, col in walls:
            mask |= WALL_BIT_GRID[row][col]
        return mask

    def opponent(self, player: str) -> str:
        return RED if player == BLUE else BLUE

    def goal_row(self, player: str) -> int:
        return BOARD_SIZE - 1 if player == BLUE else 0

    def goal_direction(self, player: str) -> int:
        return 1 if player == BLUE else -1

    def is_inside_board(self, row: int, col: int) -> bool:
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def is_blocked(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        horizontal_walls: set[tuple[int, int]] | None = None,
        vertical_walls: set[tuple[int, int]] | None = None,
    ) -> bool:
        start_row, start_col = start
        end_row, end_col = end
        row_delta = end_row - start_row
        col_delta = end_col - start_col
        if abs(row_delta) + abs(col_delta) != 1:
            return True

        if horizontal_walls is not None or vertical_walls is not None:
            horizontal_walls = self.horizontal_walls if horizontal_walls is None else horizontal_walls
            vertical_walls = self.vertical_walls if vertical_walls is None else vertical_walls
            if row_delta != 0:
                wall_row = min(start_row, end_row)
                for wall_col in (start_col - 1, start_col):
                    if 0 <= wall_col < BOARD_SIZE - 1 and (wall_row, wall_col) in horizontal_walls:
                        return True
                return False

            wall_col = min(start_col, end_col)
            for wall_row in (start_row - 1, start_row):
                if 0 <= wall_row < BOARD_SIZE - 1 and (wall_row, wall_col) in vertical_walls:
                    return True
            return False

        horizontal_mask = self.horizontal_wall_mask
        vertical_mask = self.vertical_wall_mask
        if row_delta != 0:
            wall_row = min(start_row, end_row)
            for wall_col in (start_col - 1, start_col):
                if 0 <= wall_col < WALL_GRID_SIZE and horizontal_mask & WALL_BIT_GRID[wall_row][wall_col]:
                    return True
            return False

        wall_col = min(start_col, end_col)
        for wall_row in (start_row - 1, start_row):
            if 0 <= wall_row < WALL_GRID_SIZE and vertical_mask & WALL_BIT_GRID[wall_row][wall_col]:
                return True
        return False

    def cardinal_action_delta(self, player: str, action: int) -> tuple[int, int]:
        forward = self.goal_direction(player)
        if action == 0:
            return forward, 0
        if action == 1:
            return -forward, 0
        if action == 2:
            return 0, -1
        if action == 3:
            return 0, 1
        raise ValueError(f"Action {action} is not a cardinal move.")

    def diagonal_action_delta(self, player: str, action: int) -> tuple[int, int]:
        forward = self.goal_direction(player)
        if action == 4:
            return forward, -1
        if action == 5:
            return forward, 1
        if action == 6:
            return -forward, -1
        if action == 7:
            return -forward, 1
        raise ValueError(f"Action {action} is not a diagonal move.")

    def nominal_move_action_position(self, player: str, action: int) -> tuple[int, int]:
        row, col = self.players[player]
        if action in CARDINAL_MOVE_ACTIONS:
            row_delta, col_delta = self.cardinal_action_delta(player, action)
        elif action in DIAGONAL_MOVE_ACTIONS:
            row_delta, col_delta = self.diagonal_action_delta(player, action)
        else:
            return row, col
        return row + row_delta, col + col_delta

    def is_straight_jump_available(
        self,
        opponent_position: tuple[int, int],
        row_delta: int,
        col_delta: int,
    ) -> bool:
        jump_position = (
            opponent_position[0] + row_delta,
            opponent_position[1] + col_delta,
        )
        return (
            self.is_inside_board(*jump_position)
            and not self.is_blocked(opponent_position, jump_position)
        )

    def diagonal_jump_position(
        self,
        player: str,
        action: int,
    ) -> tuple[int, int] | None:
        current = self.players[player]
        opponent_position = self.players[self.opponent(player)]
        row_delta, col_delta = self.diagonal_action_delta(player, action)
        candidate = (current[0] + row_delta, current[1] + col_delta)
        if not self.is_inside_board(*candidate):
            return None

        jump_options = (
            ((row_delta, 0), (0, col_delta)),
            ((0, col_delta), (row_delta, 0)),
        )
        for opponent_delta, around_delta in jump_options:
            adjacent = (
                current[0] + opponent_delta[0],
                current[1] + opponent_delta[1],
            )
            if adjacent != opponent_position:
                continue
            if self.is_blocked(current, opponent_position):
                continue
            if self.is_straight_jump_available(opponent_position, *opponent_delta):
                continue
            around_position = (
                opponent_position[0] + around_delta[0],
                opponent_position[1] + around_delta[1],
            )
            if around_position != candidate:
                continue
            if self.is_blocked(opponent_position, candidate):
                continue
            return candidate
        return None

    def move_action_positions(self, player: str) -> dict[int, tuple[int, int]]:
        current = self.players[player]
        opponent_position = self.players[self.opponent(player)]
        positions: dict[int, tuple[int, int]] = {}

        for action in CARDINAL_MOVE_ACTIONS:
            row_delta, col_delta = self.cardinal_action_delta(player, action)
            adjacent = (current[0] + row_delta, current[1] + col_delta)
            if not self.is_inside_board(*adjacent):
                continue
            if self.is_blocked(current, adjacent):
                continue
            if adjacent == opponent_position:
                jump_position = (
                    opponent_position[0] + row_delta,
                    opponent_position[1] + col_delta,
                )
                if (
                    self.is_inside_board(*jump_position)
                    and not self.is_blocked(opponent_position, jump_position)
                ):
                    positions[action] = jump_position
                continue
            positions[action] = adjacent

        for action in DIAGONAL_MOVE_ACTIONS:
            diagonal_position = self.diagonal_jump_position(player, action)
            if diagonal_position is not None:
                positions[action] = diagonal_position
        return positions

    def legal_moves(self, player: str) -> list[tuple[int, int]]:
        return list(dict.fromkeys(self.move_action_positions(player).values()))

    def path_exists(
        self,
        player: str,
        horizontal_walls: set[tuple[int, int]] | None = None,
        vertical_walls: set[tuple[int, int]] | None = None,
        horizontal_wall_mask: int | None = None,
        vertical_wall_mask: int | None = None,
    ) -> bool:
        return (
            self.shortest_path_length(
                player,
                horizontal_walls,
                vertical_walls,
                horizontal_wall_mask,
                vertical_wall_mask,
            )
            < BOARD_SIZE * BOARD_SIZE
        )

    def shortest_path_length(
        self,
        player: str,
        horizontal_walls: set[tuple[int, int]] | None = None,
        vertical_walls: set[tuple[int, int]] | None = None,
        horizontal_wall_mask: int | None = None,
        vertical_wall_mask: int | None = None,
    ) -> int:
        horizontal_walls = self.horizontal_walls if horizontal_walls is None else horizontal_walls
        vertical_walls = self.vertical_walls if vertical_walls is None else vertical_walls
        if horizontal_wall_mask is None:
            horizontal_wall_mask = self.horizontal_wall_mask if horizontal_walls is self.horizontal_walls else self.wall_set_mask(horizontal_walls)
        if vertical_wall_mask is None:
            vertical_wall_mask = self.vertical_wall_mask if vertical_walls is self.vertical_walls else self.wall_set_mask(vertical_walls)
        cache_key = (
            player,
            self.players[player],
            horizontal_wall_mask,
            vertical_wall_mask,
        )
        cached = self._shortest_path_cache.get(cache_key)
        if cached is not None:
            return cached
        goal_row = self.goal_row(player)
        start_row, start_col = self.players[player]
        start_index = start_row * BOARD_SIZE + start_col
        queue = [start_index]
        distances = [0] * CELL_COUNT
        visited = 1 << start_index
        head = 0
        horizontal_mask = horizontal_wall_mask
        vertical_mask = vertical_wall_mask
        board_size = BOARD_SIZE
        wall_grid_size = WALL_GRID_SIZE
        wall_bits = WALL_BIT_GRID

        while head < len(queue):
            cell_index = queue[head]
            head += 1
            row = cell_index // board_size
            col = cell_index - row * board_size
            distance = distances[cell_index]
            if row == goal_row:
                self._shortest_path_cache[cache_key] = distance
                return distance

            next_row = row - 1
            if next_row >= 0:
                next_index = cell_index - board_size
                next_bit = 1 << next_index
                if visited & next_bit:
                    next_row = -1
            if next_row >= 0:
                blocked = False
                wall_row = next_row
                for wall_col in (col - 1, col):
                    if 0 <= wall_col < wall_grid_size and horizontal_mask & wall_bits[wall_row][wall_col]:
                        blocked = True
                        break
                if not blocked:
                    visited |= next_bit
                    distances[next_index] = distance + 1
                    queue.append(next_index)

            next_row = row + 1
            if next_row < board_size:
                next_index = cell_index + board_size
                next_bit = 1 << next_index
                if visited & next_bit:
                    next_row = board_size
            if next_row < board_size:
                blocked = False
                wall_row = row
                for wall_col in (col - 1, col):
                    if 0 <= wall_col < wall_grid_size and horizontal_mask & wall_bits[wall_row][wall_col]:
                        blocked = True
                        break
                if not blocked:
                    visited |= next_bit
                    distances[next_index] = distance + 1
                    queue.append(next_index)

            next_col = col - 1
            if next_col >= 0:
                next_index = cell_index - 1
                next_bit = 1 << next_index
                if visited & next_bit:
                    next_col = -1
            if next_col >= 0:
                blocked = False
                wall_col = next_col
                for wall_row in (row - 1, row):
                    if 0 <= wall_row < wall_grid_size and vertical_mask & wall_bits[wall_row][wall_col]:
                        blocked = True
                        break
                if not blocked:
                    visited |= next_bit
                    distances[next_index] = distance + 1
                    queue.append(next_index)

            next_col = col + 1
            if next_col < board_size:
                next_index = cell_index + 1
                next_bit = 1 << next_index
                if visited & next_bit:
                    next_col = board_size
            if next_col < board_size:
                blocked = False
                wall_col = col
                for wall_row in (row - 1, row):
                    if 0 <= wall_row < wall_grid_size and vertical_mask & wall_bits[wall_row][wall_col]:
                        blocked = True
                        break
                if not blocked:
                    visited |= next_bit
                    distances[next_index] = distance + 1
                    queue.append(next_index)
        blocked_distance = BLOCKED_PATH_DISTANCE
        self._shortest_path_cache[cache_key] = blocked_distance
        return blocked_distance

    def is_valid_wall(self, orientation: str, row: int, col: int) -> bool:
        if not (0 <= row < BOARD_SIZE - 1 and 0 <= col < BOARD_SIZE - 1):
            return False

        wall_bit = WALL_BIT_GRID[row][col]
        if orientation == "h":
            if self.vertical_wall_mask & wall_bit:
                return False
            for nearby_col in (col - 1, col, col + 1):
                if 0 <= nearby_col < WALL_GRID_SIZE and self.horizontal_wall_mask & WALL_BIT_GRID[row][nearby_col]:
                    return False
            next_horizontal_mask = self.horizontal_wall_mask | wall_bit
            next_vertical_mask = self.vertical_wall_mask
        else:
            if self.horizontal_wall_mask & wall_bit:
                return False
            for nearby_row in (row - 1, row, row + 1):
                if 0 <= nearby_row < WALL_GRID_SIZE and self.vertical_wall_mask & WALL_BIT_GRID[nearby_row][col]:
                    return False
            next_horizontal_mask = self.horizontal_wall_mask
            next_vertical_mask = self.vertical_wall_mask | wall_bit

        return (
            self.path_exists(BLUE, horizontal_wall_mask=next_horizontal_mask, vertical_wall_mask=next_vertical_mask)
            and self.path_exists(RED, horizontal_wall_mask=next_horizontal_mask, vertical_wall_mask=next_vertical_mask)
        )

    def valid_walls(self, player: str) -> list[tuple[str, int, int]]:
        if self.wall_counts[player] <= 0:
            return []
        walls: list[tuple[str, int, int]] = []
        for orientation in ("h", "v"):
            for row in range(WALL_GRID_SIZE):
                for col in range(WALL_GRID_SIZE):
                    if self.is_valid_wall(orientation, row, col):
                        walls.append((orientation, row, col))
        return walls

    def place_wall(self, player: str, orientation: str, row: int, col: int) -> None:
        self.walls.append(Wall(orientation, row, col, player))
        self.wall_counts[player] -= 1
        if orientation == "h":
            self.horizontal_walls.add((row, col))
            self.horizontal_wall_mask |= WALL_BIT_GRID[row][col]
        else:
            self.vertical_walls.add((row, col))
            self.vertical_wall_mask |= WALL_BIT_GRID[row][col]

    def move_player(self, player: str, position: tuple[int, int]) -> None:
        self.players[player] = position
        if position[0] == self.goal_row(player):
            self.winner = player
            self.win_reason = "goal"

    def tiebreak_winner(self) -> str:
        blue_score = (
            self.shortest_path_length(BLUE),
            abs(self.goal_row(BLUE) - self.players[BLUE][0]),
            -self.wall_counts[BLUE],
        )
        red_score = (
            self.shortest_path_length(RED),
            abs(self.goal_row(RED) - self.players[RED][0]),
            -self.wall_counts[RED],
        )
        if blue_score == red_score:
            return self.rng.choice([BLUE, RED])
        return BLUE if blue_score < red_score else RED

    def finish_if_turn_limit_reached(self) -> bool:
        if self.winner is not None:
            return True
        if self.turn_number < MAX_TURNS:
            return False
        self.winner = self.tiebreak_winner()
        self.win_reason = "turn_limit"
        return True

    def terminal_value(self, player: str) -> float:
        if self.winner is None:
            return 0.0
        if self.winner == player:
            return 1.0
        return -1.0

    def view_position(self, player: str, position: tuple[int, int]) -> tuple[int, int]:
        row, col = position
        if player == RED:
            row = BOARD_SIZE - 1 - row
        return row, col

    def view_wall(self, player: str, row: int, col: int) -> tuple[int, int]:
        if player == RED:
            row = WALL_GRID_SIZE - 1 - row
        return row, col

    def absolute_wall(self, player: str, row: int, col: int) -> tuple[int, int]:
        if player == RED:
            row = WALL_GRID_SIZE - 1 - row
        return row, col

    def move_action_position(self, player: str, action: int) -> tuple[int, int]:
        return self.move_action_positions(player).get(
            action,
            self.nominal_move_action_position(player, action),
        )

    def decode_wall_action(self, player: str, action: int) -> tuple[str, int, int]:
        wall_index = action - MOVE_ACTION_COUNT
        orientation = "h"
        if wall_index >= WALL_PLACEMENT_COUNT:
            orientation = "v"
            wall_index -= WALL_PLACEMENT_COUNT
        view_row = wall_index // WALL_GRID_SIZE
        view_col = wall_index % WALL_GRID_SIZE
        row, col = self.absolute_wall(player, view_row, view_col)
        return orientation, row, col

    def action_mask(self, player: str | None = None) -> list[bool]:
        player = self.turn if player is None else player
        mask = [False] * ACTION_SIZE
        if self.winner is not None:
            return mask
        for action in self.move_action_positions(player):
            mask[action] = True
        if self.wall_counts[player] <= 0:
            return mask
        for orientation_offset, orientation in ((0, "h"), (WALL_PLACEMENT_COUNT, "v")):
            for view_row in range(WALL_GRID_SIZE):
                for view_col in range(WALL_GRID_SIZE):
                    row, col = self.absolute_wall(player, view_row, view_col)
                    if self.is_valid_wall(orientation, row, col):
                        action = MOVE_ACTION_COUNT + orientation_offset + view_row * WALL_GRID_SIZE + view_col
                        mask[action] = True
        return mask

    def valid_actions(self, player: str | None = None) -> list[int]:
        return [index for index, is_valid in enumerate(self.action_mask(player)) if is_valid]

    def action_path_details(self, player: str | None = None) -> dict[int, dict[str, float]]:
        player = self.turn if player is None else player
        cache_key = (self.state_key(), player)
        cached = self._path_detail_cache.get(cache_key)
        if cached is not None:
            return cached
        opponent = self.opponent(player)
        own_before = self.shortest_path_length(player)
        opp_before = self.shortest_path_length(opponent)
        old_distance = abs(self.goal_row(player) - self.players[player][0])
        details: dict[int, dict[str, float]] = {}

        def add_detail(
            action: int,
            own_after: int,
            opp_after: int,
            new_distance: int,
            terminal_value: float,
            is_wall: float,
        ) -> None:
            own_delta = own_before - own_after
            opp_delta = opp_after - opp_before
            distance_delta = old_distance - new_distance
            score = (
                1.45 * own_delta
                + 1.15 * opp_delta
                + 0.35 * distance_delta
                + 0.08 * (opp_after - own_after)
            )
            if is_wall:
                score -= 0.18
                if opp_delta <= 0:
                    score -= 0.30
                if own_delta < 0:
                    score += 0.70 * own_delta
            else:
                score += 0.08
            if terminal_value > 0:
                score += 8.0
            elif terminal_value < 0:
                score -= 8.0

            details[action] = {
                "own_after": float(own_after),
                "opp_after": float(opp_after),
                "own_delta": float(own_delta),
                "opp_delta": float(opp_delta),
                "distance_delta": float(distance_delta),
                "terminal_value": float(terminal_value),
                "is_wall": is_wall,
                "score": float(score),
            }

        move_positions = self.move_action_positions(player)
        for action in range(MOVE_ACTION_COUNT):
            next_position = move_positions.get(action)
            if next_position is None:
                continue
            next_game = self.clone()
            next_game.turn = player
            next_game.apply_action(action, validated=True)
            own_after = next_game.shortest_path_length(player)
            opp_after = next_game.shortest_path_length(opponent)
            new_distance = abs(next_game.goal_row(player) - next_game.players[player][0])
            terminal_value = next_game.terminal_value(player) if next_game.winner is not None else 0.0
            add_detail(action, own_after, opp_after, new_distance, terminal_value, 0.0)

        if self.wall_counts[player] > 0:
            blocked_path = BOARD_SIZE * BOARD_SIZE
            horizontal_mask = self.horizontal_wall_mask
            vertical_mask = self.vertical_wall_mask
            for orientation_offset, orientation in ((0, "h"), (WALL_PLACEMENT_COUNT, "v")):
                for view_row in range(WALL_GRID_SIZE):
                    for view_col in range(WALL_GRID_SIZE):
                        row, col = self.absolute_wall(player, view_row, view_col)
                        wall_bit = WALL_BIT_GRID[row][col]
                        if orientation == "h":
                            if vertical_mask & wall_bit:
                                continue
                            blocked = False
                            for nearby_col in (col - 1, col, col + 1):
                                if 0 <= nearby_col < WALL_GRID_SIZE and horizontal_mask & WALL_BIT_GRID[row][nearby_col]:
                                    blocked = True
                                    break
                            if blocked:
                                continue
                            next_horizontal_mask = horizontal_mask | wall_bit
                            next_vertical_mask = vertical_mask
                        else:
                            if horizontal_mask & wall_bit:
                                continue
                            blocked = False
                            for nearby_row in (row - 1, row, row + 1):
                                if 0 <= nearby_row < WALL_GRID_SIZE and vertical_mask & WALL_BIT_GRID[nearby_row][col]:
                                    blocked = True
                                    break
                            if blocked:
                                continue
                            next_horizontal_mask = horizontal_mask
                            next_vertical_mask = vertical_mask | wall_bit

                        own_after = self.shortest_path_length(
                            player,
                            horizontal_wall_mask=next_horizontal_mask,
                            vertical_wall_mask=next_vertical_mask,
                        )
                        if own_after >= blocked_path:
                            continue
                        opp_after = self.shortest_path_length(
                            opponent,
                            horizontal_wall_mask=next_horizontal_mask,
                            vertical_wall_mask=next_vertical_mask,
                        )
                        if opp_after >= blocked_path:
                            continue
                        action = MOVE_ACTION_COUNT + orientation_offset + view_row * WALL_GRID_SIZE + view_col
                        add_detail(action, own_after, opp_after, old_distance, 0.0, 1.0)
        self._path_detail_cache[cache_key] = details
        return details

    def action_path_signals(self, player: str | None = None) -> dict[str, list[float]]:
        details = self.action_path_details(player)
        max_path = BOARD_SIZE * 2
        signals = {
            "own_after": [0.0] * ACTION_SIZE,
            "opp_after": [0.0] * ACTION_SIZE,
            "own_delta": [0.0] * ACTION_SIZE,
            "opp_delta": [0.0] * ACTION_SIZE,
        }
        for action, values in details.items():
            signals["own_after"][action] = min(values["own_after"], max_path) / max_path
            signals["opp_after"][action] = min(values["opp_after"], max_path) / max_path
            signals["own_delta"][action] = max(-1.0, min(1.0, values["own_delta"] / BOARD_SIZE))
            signals["opp_delta"][action] = max(-1.0, min(1.0, values["opp_delta"] / BOARD_SIZE))
        return signals

    def action_heuristic_scores(self, player: str | None = None) -> list[float]:
        scores = [-1.0e9] * ACTION_SIZE
        for action, values in self.action_path_details(player).items():
            scores[action] = values["score"]
        return scores

    def action_heuristic_policy(self, player: str | None = None, temperature: float = 0.85) -> list[float]:
        mask = self.action_mask(player)
        scores = self.action_heuristic_scores(player)
        valid_scores = [scores[action] for action, is_valid in enumerate(mask) if is_valid]
        policy = [0.0] * ACTION_SIZE
        if not valid_scores:
            return policy
        temperature = max(0.05, temperature)
        best_score = max(valid_scores)
        weights = []
        actions = []
        for action, is_valid in enumerate(mask):
            if not is_valid:
                continue
            actions.append(action)
            weights.append(math.exp((scores[action] - best_score) / temperature))
        total = sum(weights)
        if total <= 0:
            chance = 1.0 / len(actions)
            for action in actions:
                policy[action] = chance
            return policy
        for action, weight in zip(actions, weights):
            policy[action] = weight / total
        return policy

    def observation(self, player: str | None = None) -> list[float]:
        player = self.turn if player is None else player
        opponent = self.opponent(player)
        self_row, self_col = self.view_position(player, self.players[player])
        opp_row, opp_col = self.view_position(player, self.players[opponent])
        own_path = self.shortest_path_length(player)
        opp_path = self.shortest_path_length(opponent)
        move_positions = self.move_action_positions(player)
        move_mask = [
            action in move_positions
            for action in range(MOVE_ACTION_COUNT)
        ]

        base = [
            self_row / (BOARD_SIZE - 1),
            (self_col - BOARD_SIZE // 2) / (BOARD_SIZE // 2),
            opp_row / (BOARD_SIZE - 1),
            (opp_col - BOARD_SIZE // 2) / (BOARD_SIZE // 2),
            (opp_row - self_row) / (BOARD_SIZE - 1),
            (opp_col - self_col) / (BOARD_SIZE - 1),
            self.wall_counts[player] / STARTING_WALLS,
            self.wall_counts[opponent] / STARTING_WALLS,
            (self.wall_counts[player] - self.wall_counts[opponent]) / STARTING_WALLS,
            min(own_path, BOARD_SIZE * 2) / (BOARD_SIZE * 2),
            min(opp_path, BOARD_SIZE * 2) / (BOARD_SIZE * 2),
            max(-1.0, min(1.0, (opp_path - own_path) / BOARD_SIZE)),
            min(self.turn_number, MAX_TURNS) / MAX_TURNS,
            (BOARD_SIZE - 1 - self_row) / (BOARD_SIZE - 1),
            (BOARD_SIZE - 1 - opp_row) / (BOARD_SIZE - 1),
            1.0,
        ]
        observation = base + [1.0 if flag else 0.0 for flag in move_mask]

        horizontal_map = [0.0] * WALL_PLACEMENT_COUNT
        vertical_map = [0.0] * WALL_PLACEMENT_COUNT
        for row, col in self.horizontal_walls:
            view_row, view_col = self.view_wall(player, row, col)
            horizontal_map[view_row * WALL_GRID_SIZE + view_col] = 1.0
        for row, col in self.vertical_walls:
            view_row, view_col = self.view_wall(player, row, col)
            vertical_map[view_row * WALL_GRID_SIZE + view_col] = 1.0
        observation.extend(horizontal_map)
        observation.extend(vertical_map)
        action_signals = self.action_path_signals(player)
        observation.extend(action_signals["own_after"])
        observation.extend(action_signals["opp_after"])
        observation.extend(action_signals["own_delta"])
        observation.extend(action_signals["opp_delta"])
        return observation

    def choose_random_move(self, player: str, moves: list[tuple[int, int]]) -> tuple[int, int]:
        current_row = self.players[player][0]
        goal_direction = self.goal_direction(player)
        weights = []
        for move in moves:
            row_delta = move[0] - current_row
            if row_delta * goal_direction > 0:
                weights.append(FORWARD_MOVE_WEIGHT)
            elif row_delta == 0:
                weights.append(SIDEWAYS_MOVE_WEIGHT)
            else:
                weights.append(BACKWARD_MOVE_WEIGHT)
        return self.rng.choices(moves, weights=weights, k=1)[0]

    def choose_random_action(self, player: str) -> int | None:
        move_positions = self.move_action_positions(player)
        move_actions = list(move_positions)

        def random_wall_action() -> int | None:
            if self.wall_counts[player] <= 0:
                return None
            wall_actions: list[int] = []
            for orientation_offset, orientation in ((0, "h"), (WALL_PLACEMENT_COUNT, "v")):
                for view_row in range(WALL_GRID_SIZE):
                    for view_col in range(WALL_GRID_SIZE):
                        row, col = self.absolute_wall(player, view_row, view_col)
                        if self.is_valid_wall(orientation, row, col):
                            action = MOVE_ACTION_COUNT + orientation_offset + view_row * WALL_GRID_SIZE + view_col
                            wall_actions.append(action)
            if not wall_actions:
                return None
            return self.rng.choice(wall_actions)

        if self.rng.random() < WALL_ACTION_CHANCE:
            wall_action = random_wall_action()
            if wall_action is not None:
                return wall_action
        if move_actions:
            weights = []
            current_row = self.players[player][0]
            goal_direction = self.goal_direction(player)
            for action in move_actions:
                next_row, _next_col = move_positions[action]
                row_delta = next_row - current_row
                if row_delta * goal_direction > 0:
                    weights.append(FORWARD_MOVE_WEIGHT)
                elif row_delta == 0:
                    weights.append(SIDEWAYS_MOVE_WEIGHT)
                else:
                    weights.append(BACKWARD_MOVE_WEIGHT)
            return self.rng.choices(move_actions, weights=weights, k=1)[0]
        return random_wall_action()

    def apply_action(self, action: int, validated: bool = False) -> tuple[float, bool, dict[str, object]]:
        if self.winner is not None:
            return 0.0, True, {"winner": self.winner, "invalid": False, "player": self.turn}

        player = self.turn
        opponent = self.opponent(player)
        valid = None if validated else self.action_mask(player)
        if action < 0 or action >= ACTION_SIZE or (valid is not None and not valid[action]):
            reward = -2.0
            self.turn_number += 1
            done = self.finish_if_turn_limit_reached()
            if done:
                return reward, True, {"winner": self.winner, "invalid": True, "player": player, "timeout": self.win_reason == "turn_limit"}
            self.turn = opponent
            return reward, False, {"winner": None, "invalid": True, "player": player}

        old_distance = abs(self.goal_row(player) - self.players[player][0])
        old_path = self.shortest_path_length(player)
        old_opponent_path = self.shortest_path_length(opponent)
        info: dict[str, object] = {"winner": None, "invalid": False, "player": player}
        reward = -0.01

        if action < MOVE_ACTION_COUNT:
            next_position = self.move_action_position(player, action)
            self.move_player(player, next_position)
            new_distance = abs(self.goal_row(player) - self.players[player][0])
            new_path = self.shortest_path_length(player)
            reward += 0.18 * (old_distance - new_distance)
            reward += 0.06 * (old_path - new_path)
            info["type"] = "move"
        else:
            orientation, row, col = self.decode_wall_action(player, action)
            self.place_wall(player, orientation, row, col)
            new_path = self.shortest_path_length(player)
            new_opponent_path = self.shortest_path_length(opponent)
            reward -= 0.02
            reward += 0.12 * max(0, new_opponent_path - old_opponent_path)
            reward -= 0.05 * max(0, new_path - old_path)
            info["type"] = "wall"

        if self.winner is not None:
            reward += 12.0
            info["winner"] = self.winner

        self.turn_number += 1
        done = self.finish_if_turn_limit_reached()
        if done and self.win_reason == "turn_limit":
            info["winner"] = self.winner
            info["timeout"] = True
            reward += 6.0 if self.winner == player else -6.0
        if not done:
            self.turn = opponent
        return reward, done, info

    def play_random_turn(self) -> None:
        action = self.choose_random_action(self.turn)
        if action is not None:
            self.apply_action(action, validated=True)


class QuoridorPreview:
    def __init__(self) -> None:
        self.render_surface = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT), pygame.SRCALPHA)
        self.canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
        self.screen: pygame.Surface | None = None
        self.clock = pygame.time.Clock()
        self.is_running = False
        self.frame_count = 0
        self.wall_fade_enabled = ENABLE_WALL_FADE
        self.wall_birth_frames: dict[tuple[str, int, int, str], int] = {}
        self.player_modes = {BLUE: BLUE_PLAYER_MODE, RED: RED_PLAYER_MODE}
        self.search_player = self.create_search_player()
        self.model_player = self.create_model_player()
        self.preview_starting_walls = self.model_starting_walls()
        self.game = QuoridorGame(starting_walls=self.preview_starting_walls)
        self.label_font: pygame.font.Font | None = None
        self.counter_font: pygame.font.Font | None = None
        self.winner_font: pygame.font.Font | None = None
        self.winner_small_font: pygame.font.Font | None = None

    @property
    def board_rect(self) -> pygame.Rect:
        return pygame.Rect(
            BOARD_MARGIN_X,
            BOARD_MARGIN_Y,
            BOARD_RENDER_SIZE,
            BOARD_RENDER_SIZE,
        )

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
        if self.label_font is None:
            self.label_font = pygame.font.SysFont("arial", 7 * RENDER_SCALE)
        if self.counter_font is None:
            self.counter_font = pygame.font.SysFont("arial", 18 * RENDER_SCALE, bold=True)
        if self.winner_font is None:
            self.winner_font = pygame.font.SysFont("arial", 58 * RENDER_SCALE, bold=True)
        if self.winner_small_font is None:
            self.winner_small_font = pygame.font.SysFont("arial", 20 * RENDER_SCALE, bold=True)

    def create_search_player(self) -> object | None:
        try:
            try:
                from .path_search_agent import DepthSearchPlayer, SearchConfig
            except ImportError:
                from path_search_agent import DepthSearchPlayer, SearchConfig
            config = SearchConfig(
                depth=VISUAL_SEARCH_DEPTH,
                max_wall_candidates=VISUAL_SEARCH_MAX_WALL_CANDIDATES,
                max_root_wall_candidates=VISUAL_SEARCH_MAX_ROOT_WALL_CANDIDATES,
                max_actions=VISUAL_SEARCH_MAX_ACTIONS,
                max_root_actions=VISUAL_SEARCH_MAX_ROOT_ACTIONS,
            )
            return DepthSearchPlayer(config)
        except Exception as error:
            print(f"Could not load path search player: {error}")
            return None

    def create_model_player(self) -> object | None:
        checkpoint_path = MODEL_CHECKPOINT_PATH if MODEL_CHECKPOINT_PATH.exists() else MODEL_FALLBACK_CHECKPOINT_PATH
        if not checkpoint_path.exists():
            print(f"No model checkpoint found in {MODEL_CHECKPOINT_DIR}; using random moves for model mode.")
            return None
        try:
            try:
                from .rl_trainer_treestrap import CheckpointModelPlayer
            except ImportError:
                from rl_trainer_treestrap import CheckpointModelPlayer
            print(f"Loaded model player checkpoint: {checkpoint_path} | preview temperature {MODEL_PLAYER_TEMPERATURE}")
            return CheckpointModelPlayer(
                checkpoint_path,
                temperature=MODEL_PLAYER_TEMPERATURE,
            )
        except Exception as error:
            print(f"Could not load checkpoint model player: {error}")
            return None

    def model_starting_walls(self) -> int:
        if self.model_player is None:
            return STARTING_WALLS
        checkpoint = getattr(self.model_player, "checkpoint", {})
        walls = checkpoint.get("stage_walls", STARTING_WALLS) if isinstance(checkpoint, dict) else STARTING_WALLS
        try:
            wall_count = int(walls)
        except (TypeError, ValueError):
            wall_count = STARTING_WALLS
        wall_count = max(0, min(STARTING_WALLS, wall_count))
        print(f"Preview game starting walls per player: {wall_count}")
        return wall_count

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.is_running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.game.reset()
                self.frame_count = 0
                self.wall_birth_frames.clear()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                self.wall_fade_enabled = not self.wall_fade_enabled
            if event.type == pygame.KEYDOWN and event.key == pygame.K_1:
                self.toggle_player_mode(BLUE)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_2:
                self.toggle_player_mode(RED)

    def toggle_player_mode(self, player: str) -> None:
        current = self.player_modes.get(player, "random")
        current_index = PLAYER_MODE_ORDER.index(current) if current in PLAYER_MODE_ORDER else 0
        self.player_modes[player] = PLAYER_MODE_ORDER[(current_index + 1) % len(PLAYER_MODE_ORDER)]

    def choose_turn_action(self) -> int | None:
        mode = self.player_modes.get(self.game.turn, "random")
        if mode == "search" and self.search_player is not None:
            try:
                action = self.search_player.select_action(self.game)
                mask = self.game.action_mask(self.game.turn)
                if action is not None and 0 <= action < len(mask) and mask[action]:
                    return action
            except Exception as error:
                print(f"Search player failed; using random move: {error}")
        if mode == "model" and self.model_player is not None:
            try:
                action = self.model_player.select_action(self.game)
                mask = self.game.action_mask(self.game.turn)
                if action is not None and 0 <= action < len(mask) and mask[action]:
                    return action
            except Exception as error:
                print(f"Model player failed; using random move: {error}")
        return self.game.choose_random_action(self.game.turn)

    def update(self) -> None:
        if self.game.winner is not None:
            return
        self.frame_count += 1
        if self.frame_count % TURN_INTERVAL_FRAMES == 0:
            wall_count = len(self.game.walls)
            action = self.choose_turn_action()
            mask = self.game.action_mask(self.game.turn)
            if action is not None and 0 <= action < len(mask) and mask[action]:
                self.game.apply_action(action, validated=True)
            for wall in self.game.walls[wall_count:]:
                self.wall_birth_frames[self.wall_key(wall)] = self.frame_count

    def wall_key(self, wall: Wall) -> tuple[str, int, int, str]:
        return wall.orientation, wall.row, wall.col, wall.owner

    def wall_alpha(self, wall: Wall) -> int:
        if not self.wall_fade_enabled:
            return 255
        birth_frame = self.wall_birth_frames.get(self.wall_key(wall))
        if birth_frame is None:
            return 255
        age = max(0, self.frame_count - birth_frame)
        if age >= WALL_FADE_FRAMES:
            return 255
        progress = age / WALL_FADE_FRAMES
        smooth = progress * progress * (3 - 2 * progress)
        return max(WALL_FADE_START_ALPHA, min(255, int(WALL_FADE_START_ALPHA + (255 - WALL_FADE_START_ALPHA) * smooth)))

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        x = BOARD_MARGIN_X + col * (CELL_SIZE + GRID_GAP) + CELL_SIZE / 2
        y = BOARD_MARGIN_Y + row * (CELL_SIZE + GRID_GAP) + CELL_SIZE / 2
        return int(round(x)), int(round(y))

    def draw_grid(self) -> None:
        surface = self.render_surface
        surface.fill(BACKGROUND_COLOR)
        board = self.board_rect
        pygame.draw.rect(surface, GRID_COLOR, board, border_radius=BOARD_RADIUS)
        pygame.draw.rect(surface, BOARD_EDGE_COLOR, board, width=2 * RENDER_SCALE, border_radius=BOARD_RADIUS)

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = BOARD_MARGIN_X + col * (CELL_SIZE + GRID_GAP)
                y = BOARD_MARGIN_Y + row * (CELL_SIZE + GRID_GAP)
                cell = pygame.Rect(
                    int(round(x)),
                    int(round(y)),
                    int(round(CELL_SIZE)),
                    int(round(CELL_SIZE)),
                )
                if row == 0:
                    cell_color = TOP_ROW_CELL_COLOR
                elif row == BOARD_SIZE - 1:
                    cell_color = BOTTOM_ROW_CELL_COLOR
                else:
                    cell_color = CELL_COLOR
                pygame.draw.rect(surface, cell_color, cell, border_radius=CELL_RADIUS)

    def draw_labels(self) -> None:
        self.ensure_fonts()
        if self.label_font is None:
            return
        letters = "abcdefghi"
        for col, letter in enumerate(letters):
            center_x, _center_y = self.cell_center(BOARD_SIZE - 1, col)
            label = self.label_font.render(letter, True, LABEL_COLOR)
            label_y = BOARD_MARGIN_Y + BOARD_RENDER_SIZE + BOARD_LABEL_GUTTER
            self.render_surface.blit(label, label.get_rect(center=(center_x, label_y)))

        for row in range(BOARD_SIZE):
            _center_x, center_y = self.cell_center(row, BOARD_SIZE - 1)
            label = self.label_font.render(str(BOARD_SIZE - row), True, LABEL_COLOR)
            label_x = BOARD_MARGIN_X + BOARD_RENDER_SIZE + BOARD_LABEL_GUTTER
            self.render_surface.blit(label, label.get_rect(center=(label_x, center_y)))

    def draw_player(
        self,
        row: int,
        col: int,
        color: tuple[int, int, int],
        is_active: bool = False,
    ) -> None:
        center_x, center_y = self.cell_center(row, col)
        if is_active:
            glow_layer = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT), pygame.SRCALPHA)
            pygame.draw.circle(glow_layer, (*color, 42), (center_x, center_y), PLAYER_RADIUS + 10 * RENDER_SCALE)
            pygame.draw.circle(glow_layer, (*color, 92), (center_x, center_y), PLAYER_RADIUS + 5 * RENDER_SCALE, width=3 * RENDER_SCALE)
            self.render_surface.blit(glow_layer, (0, 0))
        shadow = (center_x + 2 * RENDER_SCALE, center_y + 3 * RENDER_SCALE)
        pygame.draw.circle(self.render_surface, (100, 115, 132, 80), shadow, PLAYER_RADIUS)
        pygame.draw.circle(self.render_surface, color, (center_x, center_y), PLAYER_RADIUS)

    def horizontal_wall_rect(self, row: int, col: int) -> pygame.Rect:
        x = BOARD_MARGIN_X + col * (CELL_SIZE + GRID_GAP)
        y = BOARD_MARGIN_Y + (row + 1) * (CELL_SIZE + GRID_GAP) - GRID_GAP / 2 - WALL_THICKNESS / 2
        return pygame.Rect(
            int(round(x)),
            int(round(y)),
            int(round(WALL_LENGTH)),
            WALL_THICKNESS,
        )

    def draw_horizontal_wall(self, row: int, col: int, color: tuple[int, int, int], alpha: int = 255) -> None:
        wall = self.horizontal_wall_rect(row, col)
        self.draw_wall_rect(wall, color, alpha)

    def vertical_wall_rect(self, row: int, col: int) -> pygame.Rect:
        x = BOARD_MARGIN_X + (col + 1) * (CELL_SIZE + GRID_GAP) - GRID_GAP / 2 - WALL_THICKNESS / 2
        y = BOARD_MARGIN_Y + row * (CELL_SIZE + GRID_GAP)
        return pygame.Rect(
            int(round(x)),
            int(round(y)),
            WALL_THICKNESS,
            int(round(WALL_LENGTH)),
        )

    def draw_vertical_wall(self, row: int, col: int, color: tuple[int, int, int], alpha: int = 255) -> None:
        wall = self.vertical_wall_rect(row, col)
        self.draw_wall_rect(wall, color, alpha)

    def draw_wall_rect(self, wall: pygame.Rect, color: tuple[int, int, int], alpha: int = 255) -> None:
        alpha = max(0, min(255, alpha))
        wall_layer = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT), pygame.SRCALPHA)
        shadow = wall.move(0, 9 * RENDER_SCALE)
        shadow_alpha = int(WALL_SHADOW[3] * (alpha / 255))
        pygame.draw.rect(wall_layer, (*WALL_SHADOW[:3], shadow_alpha), shadow, border_radius=WALL_RADIUS)
        pygame.draw.rect(wall_layer, (*color, alpha), wall, border_radius=WALL_RADIUS)
        self.render_surface.blit(wall_layer, (0, 0))

    def draw_walls(self) -> None:
        for wall in self.game.walls:
            color = WALL_BLUE if wall.owner == BLUE else WALL_RED
            alpha = self.wall_alpha(wall)
            if wall.orientation == "h":
                self.draw_horizontal_wall(wall.row, wall.col, color, alpha)
            else:
                self.draw_vertical_wall(wall.row, wall.col, color, alpha)

    def draw_wall_counter(
        self,
        center: tuple[int, int],
        color: tuple[int, int, int],
        count: int,
    ) -> None:
        self.ensure_fonts()
        if self.counter_font is None:
            return
        counter = pygame.Rect(0, 0, 88 * RENDER_SCALE, 42 * RENDER_SCALE)
        counter.center = center
        pygame.draw.rect(self.render_surface, color, counter, border_radius=COUNTER_RADIUS)

        icon = pygame.Rect(0, 0, 21 * RENDER_SCALE, 6 * RENDER_SCALE)
        icon.centery = counter.centery
        icon.left = counter.left + 16 * RENDER_SCALE
        pygame.draw.rect(self.render_surface, COUNTER_TEXT, icon, border_radius=3 * RENDER_SCALE)

        label = self.counter_font.render(str(count), True, COUNTER_TEXT)
        label_rect = label.get_rect(midleft=(counter.left + 45 * RENDER_SCALE, counter.centery))
        self.render_surface.blit(label, label_rect)

    def draw_win_overlay(self) -> None:
        winner = self.game.winner
        if winner is None:
            return
        self.ensure_fonts()
        if self.winner_font is None or self.winner_small_font is None:
            return

        color = PLAYER_BLUE if winner == BLUE else PLAYER_RED
        overlay = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT), pygame.SRCALPHA)
        overlay.fill(WIN_DIM_COLOR)
        self.render_surface.blit(overlay, (0, 0))

        panel = pygame.Rect(0, 0, 640 * RENDER_SCALE, 250 * RENDER_SCALE)
        panel.center = self.board_rect.center
        pygame.draw.rect(self.render_surface, WIN_PANEL_COLOR, panel, border_radius=WIN_PANEL_RADIUS)
        pygame.draw.rect(self.render_surface, color, panel, width=5 * RENDER_SCALE, border_radius=WIN_PANEL_RADIUS)

        glow_layer = pygame.Surface((RENDER_WIDTH, RENDER_HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(glow_layer, (*color, 44), (panel.centerx, panel.top + 60 * RENDER_SCALE), 72 * RENDER_SCALE)
        self.render_surface.blit(glow_layer, (0, 0))
        pygame.draw.circle(self.render_surface, color, (panel.centerx, panel.top + 60 * RENDER_SCALE), 28 * RENDER_SCALE)

        title = self.winner_font.render(f"{winner.upper()} WINS", True, WIN_TEXT_COLOR)
        self.render_surface.blit(title, title.get_rect(center=(panel.centerx, panel.centery + 18 * RENDER_SCALE)))

        subtitle_text = "closest path after 100 turns" if self.game.win_reason == "turn_limit" else "first to the final row"
        subtitle = self.winner_small_font.render(subtitle_text, True, (205, 215, 228))
        self.render_surface.blit(subtitle, subtitle.get_rect(center=(panel.centerx, panel.bottom - 45 * RENDER_SCALE)))

    def render_frame(self) -> pygame.Surface:
        self.draw_grid()
        self.draw_walls()
        self.draw_player(*self.game.players[BLUE], PLAYER_BLUE, self.game.turn == BLUE and self.game.winner is None)
        self.draw_player(*self.game.players[RED], PLAYER_RED, self.game.turn == RED and self.game.winner is None)
        self.draw_wall_counter((RENDER_WIDTH // 2, BOARD_MARGIN_Y - 42 * RENDER_SCALE), COUNTER_BLUE, self.game.wall_counts[BLUE])
        self.draw_wall_counter((RENDER_WIDTH // 2, BOARD_MARGIN_Y + BOARD_RENDER_SIZE + 58 * RENDER_SCALE), COUNTER_RED, self.game.wall_counts[RED])
        self.draw_labels()
        self.draw_win_overlay()
        self.canvas = pygame.transform.smoothscale(
            self.render_surface,
            (CANVAS_WIDTH, CANVAS_HEIGHT),
        )
        return self.canvas

    def render(self) -> None:
        if self.screen is None:
            return
        self.update()
        self.render_frame()
        scaled = pygame.transform.smoothscale(self.canvas, self.screen.get_size())
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def run(self) -> None:
        self.setup()
        while self.is_running:
            self.clock.tick(FPS)
            self.handle_events()
            self.render()
        pygame.quit()


def main() -> None:
    QuoridorPreview().run()


if __name__ == "__main__":
    main()
