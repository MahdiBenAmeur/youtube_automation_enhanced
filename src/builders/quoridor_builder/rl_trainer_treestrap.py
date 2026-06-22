from __future__ import annotations

import argparse
import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .path_search_agent import SearchConfig, best_action
    from .video_generator import (
        ACTION_SIZE,
        BLUE,
        BOARD_SIZE,
        CELL_COUNT,
        MOVE_ACTION_COUNT,
        RED,
        STARTING_WALLS,
        QuoridorGame,
        WALL_BIT_GRID,
        WALL_GRID_SIZE,
        WALL_PLACEMENT_COUNT,
    )
except ImportError:
    from path_search_agent import SearchConfig, best_action
    from video_generator import (
        ACTION_SIZE,
        BLUE,
        BOARD_SIZE,
        CELL_COUNT,
        MOVE_ACTION_COUNT,
        RED,
        STARTING_WALLS,
        QuoridorGame,
        WALL_BIT_GRID,
        WALL_GRID_SIZE,
        WALL_PLACEMENT_COUNT,
    )


DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[3] / "training_checkpoints" / "quoridor_builder"
PLANE_COUNT = 8
DISTANCE_NORM = float(CELL_COUNT)
DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 6
NEG_INF = -1.0e9
POS_INF = 1.0e9


@dataclass(frozen=True)
class LookaheadConfig:
    root_width: int = 12
    reply_width: int = 6
    second_width: int = 5
    inference_batch_size: int = 4096


# ---------------------------------------------------------------------------
# Board -> plane encoding (unchanged spatial representation)
# ---------------------------------------------------------------------------


def _wall_blocked(row_a: int, col_a: int, row_b: int, col_b: int, horizontal_mask: int, vertical_mask: int) -> bool:
    row_delta = row_b - row_a
    if row_delta != 0:
        wall_row = min(row_a, row_b)
        for wall_col in (col_a - 1, col_a):
            if 0 <= wall_col < WALL_GRID_SIZE and horizontal_mask & WALL_BIT_GRID[wall_row][wall_col]:
                return True
        return False
    wall_col = min(col_a, col_b)
    for wall_row in (row_a - 1, row_a):
        if 0 <= wall_row < WALL_GRID_SIZE and vertical_mask & WALL_BIT_GRID[wall_row][wall_col]:
            return True
    return False


def distance_field(horizontal_mask: int, vertical_mask: int, goal_row: int) -> np.ndarray:
    distances = np.full(CELL_COUNT, -1, dtype=np.int32)
    queue: deque[int] = deque()
    for col in range(BOARD_SIZE):
        index = goal_row * BOARD_SIZE + col
        distances[index] = 0
        queue.append(index)
    while queue:
        index = queue.popleft()
        row, col = divmod(index, BOARD_SIZE)
        dist = distances[index]
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if not (0 <= next_row < BOARD_SIZE and 0 <= next_col < BOARD_SIZE):
                continue
            next_index = next_row * BOARD_SIZE + next_col
            if distances[next_index] != -1:
                continue
            if _wall_blocked(row, col, next_row, next_col, horizontal_mask, vertical_mask):
                continue
            distances[next_index] = dist + 1
            queue.append(next_index)
    distances[distances == -1] = CELL_COUNT
    return distances.reshape(BOARD_SIZE, BOARD_SIZE).astype(np.float32)


def encode_planes(game: QuoridorGame, player: str) -> tuple[np.ndarray, int]:
    opponent = game.opponent(player)
    planes = np.zeros((PLANE_COUNT, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    own_row, own_col = game.view_position(player, game.players[player])
    opp_row, opp_col = game.view_position(player, game.players[opponent])
    planes[0, own_row, own_col] = 1.0
    planes[1, opp_row, opp_col] = 1.0

    for row, col in game.horizontal_walls:
        view_row, view_col = game.view_wall(player, row, col)
        planes[2, view_row, view_col] = 1.0
    for row, col in game.vertical_walls:
        view_row, view_col = game.view_wall(player, row, col)
        planes[3, view_row, view_col] = 1.0

    planes[4, :, :] = game.wall_counts[player] / STARTING_WALLS
    planes[5, :, :] = game.wall_counts[opponent] / STARTING_WALLS

    own_field = distance_field(game.horizontal_wall_mask, game.vertical_wall_mask, game.goal_row(player))
    opp_field = distance_field(game.horizontal_wall_mask, game.vertical_wall_mask, game.goal_row(opponent))
    if player == RED:
        own_field = own_field[::-1, :]
        opp_field = opp_field[::-1, :]
    planes[6] = np.clip(own_field, 0.0, DISTANCE_NORM) / DISTANCE_NORM
    planes[7] = np.clip(opp_field, 0.0, DISTANCE_NORM) / DISTANCE_NORM

    own_idx = own_row * BOARD_SIZE + own_col
    return planes, own_idx


# ---------------------------------------------------------------------------
# Network (same architecture: spatial wall head + pawn-local move head)
# ---------------------------------------------------------------------------


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.silu(x + out)


class QuoridorResNet(nn.Module):
    def __init__(self, in_planes: int = PLANE_COUNT, channels: int = DEFAULT_CHANNELS, blocks: int = DEFAULT_BLOCKS) -> None:
        super().__init__()
        self.channels = channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.trunk = nn.ModuleList([ResidualConvBlock(channels) for _ in range(blocks)])
        self.wall_head = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, 2, kernel_size=1),
        )
        self.move_head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU(),
            nn.Linear(channels, MOVE_ACTION_COUNT),
        )
        self.value_head = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.SiLU(),
            nn.Linear(channels // 2, 1),
            nn.Tanh(),
        )

    def forward(self, planes: torch.Tensor, own_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = planes.shape[0]
        features = self.stem(planes)
        for block in self.trunk:
            features = block(features)

        wall_map = self.wall_head(features)[:, :, :WALL_GRID_SIZE, :WALL_GRID_SIZE]
        wall_logits = wall_map.reshape(batch, 2 * WALL_PLACEMENT_COUNT)

        flat = features.permute(0, 2, 3, 1).reshape(batch, BOARD_SIZE * BOARD_SIZE, self.channels)
        own_feature = flat[torch.arange(batch, device=features.device), own_idx]
        global_feature = features.mean(dim=(2, 3))
        move_logits = self.move_head(torch.cat([own_feature, global_feature], dim=1))

        policy_logits = torch.cat([move_logits, wall_logits], dim=1)
        value = self.value_head(global_feature).squeeze(-1)
        return policy_logits, value


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask.bool(), -1.0e9)
    return F.softmax(masked, dim=-1)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask.bool(), -1.0e9)
    return F.log_softmax(masked, dim=-1)


def forward_batch(model: QuoridorResNet, planes: np.ndarray, own_idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    planes_tensor = torch.as_tensor(planes, dtype=torch.float32, device=device)
    idx_tensor = torch.as_tensor(own_idx, dtype=torch.long, device=device)
    with torch.inference_mode():
        logits, values = model(planes_tensor, idx_tensor)
    return logits, values


# ---------------------------------------------------------------------------
# TreeStrap: fixed-width 3-ply lookahead, batched GPU leaf eval, minimax backup
# ---------------------------------------------------------------------------


def candidate_actions(game: QuoridorGame, player: str, width: int) -> list[int]:
    mask = game.action_mask(player)
    move_actions = [action for action in range(MOVE_ACTION_COUNT) if mask[action]]
    wall_actions = [action for action in range(MOVE_ACTION_COUNT, ACTION_SIZE) if mask[action]]
    budget = width - len(move_actions)
    if budget <= 0:
        return move_actions[:width]
    if len(wall_actions) > budget:
        opponent = game.opponent(player)
        scored = []
        for action in wall_actions:
            orientation, row, col = game.decode_wall_action(player, action)
            wall_bit = game.wall_bit(row, col)
            if orientation == "h":
                next_h = game.horizontal_wall_mask | wall_bit
                next_v = game.vertical_wall_mask
            else:
                next_h = game.horizontal_wall_mask
                next_v = game.vertical_wall_mask | wall_bit
            own_after = game.shortest_path_length(player, horizontal_wall_mask=next_h, vertical_wall_mask=next_v)
            opp_after = game.shortest_path_length(opponent, horizontal_wall_mask=next_h, vertical_wall_mask=next_v)
            scored.append((opp_after - own_after, action))
        scored.sort(key=lambda item: item[0], reverse=True)
        wall_actions = [action for _score, action in scored[:budget]]
    return move_actions + wall_actions


LeafRecord = tuple[int, int, int, int, np.ndarray, int, str, str]


def generate_lookahead_batch(
    games: list[QuoridorGame],
    players: list[str],
    config: LookaheadConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[int | None]], list[LeafRecord]]:
    n_games = len(games)
    shape = (n_games, config.root_width, config.reply_width, config.second_width)
    value_tensor = np.zeros(shape, dtype=np.float32)
    valid2 = np.zeros(shape, dtype=bool)
    valid1 = np.zeros((n_games, config.root_width, config.reply_width), dtype=bool)
    mask0 = np.zeros((n_games, config.root_width), dtype=bool)
    root_actions: list[list[int | None]] = [[None] * config.root_width for _ in range(n_games)]
    leaf_records: list[LeafRecord] = []

    for n, game in enumerate(games):
        player = players[n]
        actions0 = candidate_actions(game, player, config.root_width)
        for k0, action0 in enumerate(actions0):
            mask0[n, k0] = True
            root_actions[n][k0] = action0
            game1 = game.clone()
            game1.apply_action(action0, validated=True)
            if game1.winner is not None:
                value_tensor[n, k0, :, :] = game1.terminal_value(player)
                valid1[n, k0, :] = True
                valid2[n, k0, :, :] = True
                continue

            opponent = game1.turn
            actions1 = candidate_actions(game1, opponent, config.reply_width)
            if not actions1:
                value_tensor[n, k0, :, :] = 1.0
                valid1[n, k0, :] = True
                valid2[n, k0, :, :] = True
                continue

            for k1, action1 in enumerate(actions1):
                valid1[n, k0, k1] = True
                game2 = game1.clone()
                game2.apply_action(action1, validated=True)
                if game2.winner is not None:
                    value_tensor[n, k0, k1, :] = game2.terminal_value(player)
                    valid2[n, k0, k1, :] = True
                    continue

                actions2 = candidate_actions(game2, player, config.second_width)
                if not actions2:
                    value_tensor[n, k0, k1, :] = -1.0
                    valid2[n, k0, k1, :] = True
                    continue

                for k2, action2 in enumerate(actions2):
                    valid2[n, k0, k1, k2] = True
                    game3 = game2.clone()
                    game3.apply_action(action2, validated=True)
                    if game3.winner is not None:
                        value_tensor[n, k0, k1, k2] = game3.terminal_value(player)
                        continue
                    planes, own_idx = encode_planes(game3, game3.turn)
                    leaf_records.append((n, k0, k1, k2, planes, own_idx, player, game3.turn))

    return value_tensor, valid1, valid2, mask0, root_actions, leaf_records


def fill_network_leaves(
    model: QuoridorResNet,
    device: torch.device,
    value_tensor: np.ndarray,
    leaf_records: list[LeafRecord],
    batch_size: int,
) -> None:
    if not leaf_records:
        return
    for start in range(0, len(leaf_records), batch_size):
        chunk = leaf_records[start : start + batch_size]
        planes = np.stack([item[4] for item in chunk])
        own_idx = np.array([item[5] for item in chunk], dtype=np.int64)
        _logits, values = forward_batch(model, planes, own_idx, device)
        values_np = values.detach().cpu().numpy()
        for (n, k0, k1, k2, _planes, _own_idx, player, leaf_turn), v_net in zip(chunk, values_np):
            value_tensor[n, k0, k1, k2] = -float(v_net) if leaf_turn != player else float(v_net)


def backup_minimax(
    value_tensor: np.ndarray,
    valid1: np.ndarray,
    valid2: np.ndarray,
    mask0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masked2 = np.where(valid2, value_tensor, NEG_INF)
    v2 = masked2.max(axis=3)
    masked1 = np.where(valid1, v2, POS_INF)
    v1 = masked1.min(axis=2)
    masked0 = np.where(mask0, v1, NEG_INF)
    v0 = masked0.max(axis=1)
    return v0, v1


def root_policy_from_backup(v1_row: np.ndarray, mask0_row: np.ndarray, root_actions_row: list[int | None]) -> np.ndarray:
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    valid_k0 = np.flatnonzero(mask0_row)
    if len(valid_k0) == 0:
        return policy
    values = v1_row[valid_k0].astype(np.float64)
    shifted = values - values.max()
    weights = np.exp(shifted)
    weights /= weights.sum()
    for k0, weight in zip(valid_k0, weights):
        policy[root_actions_row[k0]] = float(weight)
    return policy


# ---------------------------------------------------------------------------
# Replay buffer and sampling helpers
# ---------------------------------------------------------------------------


@dataclass
class ReplaySample:
    planes: np.ndarray
    own_idx: int
    mask: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int, rng: random.Random) -> None:
        self.capacity = capacity
        self.rng = rng
        self.samples: list[ReplaySample] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.samples)

    def add_many(self, samples: list[ReplaySample]) -> None:
        for sample in samples:
            if len(self.samples) < self.capacity:
                self.samples.append(sample)
            else:
                self.samples[self.position] = sample
                self.position = (self.position + 1) % self.capacity

    def clear(self) -> None:
        self.samples.clear()
        self.position = 0

    def batch(self, batch_size: int) -> list[ReplaySample]:
        size = min(batch_size, len(self.samples))
        return self.rng.sample(self.samples, size)


def sample_action(policy: np.ndarray, rng: random.Random) -> int | None:
    total = float(policy.sum())
    if total <= 0:
        return None
    threshold = rng.random() * total
    running = 0.0
    for action, probability in enumerate(policy):
        running += float(probability)
        if running >= threshold:
            return action
    return int(np.argmax(policy))


def apply_temperature(policy: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(0.05, temperature)
    if abs(temperature - 1.0) < 1.0e-9:
        return policy
    powered = np.power(policy, 1.0 / temperature)
    total = float(powered.sum())
    if total <= 0:
        return policy
    return powered / total


def policy_entropy(policy: np.ndarray) -> float:
    valid = policy[policy > 0]
    if len(valid) == 0:
        return 0.0
    return float(-(valid * np.log(valid + 1.0e-12)).sum())


# ---------------------------------------------------------------------------
# Lockstep self-play: many games advance one ply per macro-step
# ---------------------------------------------------------------------------


def run_lockstep_self_play(
    model: QuoridorResNet,
    device: torch.device,
    config: LookaheadConfig,
    rng: random.Random,
    *,
    parallel_games: int,
    early_temperature: float,
    late_temperature: float,
    temperature_turns: int,
    starting_walls: int,
) -> tuple[list[ReplaySample], list[dict[str, float]]]:
    games = [QuoridorGame(seed=rng.randrange(1_000_000_000), starting_walls=starting_walls) for _ in range(parallel_games)]
    active = list(range(parallel_games))
    move_counts = [0] * parallel_games
    wall_counts = [0] * parallel_games
    samples: list[ReplaySample] = []
    finished_stats: list[dict[str, float]] = []

    while active:
        batch_games = [games[index] for index in active]
        players = [game.turn for game in batch_games]
        value_tensor, valid1, valid2, mask0, root_actions, leaf_records = generate_lookahead_batch(batch_games, players, config)
        fill_network_leaves(model, device, value_tensor, leaf_records, config.inference_batch_size)
        v0, v1 = backup_minimax(value_tensor, valid1, valid2, mask0)

        next_active = []
        for row, game_index in enumerate(active):
            game = games[game_index]
            player = players[row]
            policy = root_policy_from_backup(v1[row], mask0[row], root_actions[row])
            mask = np.asarray(game.action_mask(player), dtype=np.float32)
            planes, own_idx = encode_planes(game, player)
            samples.append(ReplaySample(planes=planes, own_idx=own_idx, mask=mask, policy=policy, value=float(v0[row])))

            temperature = early_temperature if game.turn_number < temperature_turns else late_temperature
            sampling_policy = apply_temperature(policy, temperature)
            action = sample_action(sampling_policy, rng)
            if action is None:
                game.winner = game.opponent(player)
                game.win_reason = "no_actions"
            else:
                if action < MOVE_ACTION_COUNT:
                    move_counts[game_index] += 1
                else:
                    wall_counts[game_index] += 1
                game.apply_action(action, validated=True)

            if game.winner is None:
                next_active.append(game_index)
            else:
                finished_stats.append(
                    {
                        "length": float(game.turn_number),
                        "cap": 1.0 if game.win_reason == "turn_limit" else 0.0,
                        "blue_win": 1.0 if game.winner == BLUE else 0.0,
                        "red_win": 1.0 if game.winner == RED else 0.0,
                        "wall_rate": wall_counts[game_index] / max(1, move_counts[game_index] + wall_counts[game_index]),
                    }
                )
        active = next_active

    return samples, finished_stats


def train_epoch(
    model: QuoridorResNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    batch = replay.batch(batch_size)
    planes = torch.as_tensor(np.stack([sample.planes for sample in batch]), dtype=torch.float32, device=device)
    own_idx = torch.as_tensor([sample.own_idx for sample in batch], dtype=torch.long, device=device)
    masks = torch.as_tensor(np.stack([sample.mask for sample in batch]), dtype=torch.bool, device=device)
    policies = torch.as_tensor(np.stack([sample.policy for sample in batch]), dtype=torch.float32, device=device)
    values = torch.as_tensor([sample.value for sample in batch], dtype=torch.float32, device=device)

    logits, predicted_values = model(planes, own_idx)
    log_probs = masked_log_softmax(logits, masks)
    probs = masked_softmax(logits, masks)
    policy_loss = -(policies * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(predicted_values, values)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    loss = policy_loss + value_loss - 0.01 * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()

    with torch.no_grad():
        target_top = torch.argmax(policies, dim=-1)
        pred_top = torch.argmax(probs, dim=-1)
        top1 = (target_top == pred_top).float().mean()
        value_mae = torch.mean(torch.abs(predicted_values - values))

    return {
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "top1": float(top1.item()),
        "value_mae": float(value_mae.item()),
        "grad_norm": float(grad_norm),
    }


# ---------------------------------------------------------------------------
# Single-game action selection (evaluation matches + live preview)
# ---------------------------------------------------------------------------


def select_model_action(
    model: QuoridorResNet,
    game: QuoridorGame,
    device: torch.device,
    config: LookaheadConfig,
    rng: random.Random,
    *,
    temperature: float = 0.5,
    sample: bool = True,
) -> int | None:
    player = game.turn
    value_tensor, valid1, valid2, mask0, root_actions, leaf_records = generate_lookahead_batch([game], [player], config)
    fill_network_leaves(model, device, value_tensor, leaf_records, config.inference_batch_size)
    _v0, v1 = backup_minimax(value_tensor, valid1, valid2, mask0)

    valid_k0 = np.flatnonzero(mask0[0])
    if len(valid_k0) == 0:
        legal = game.valid_actions(player)
        return rng.choice(legal) if legal else None
    values = v1[0, valid_k0].astype(np.float64)
    if not sample:
        best = valid_k0[int(np.argmax(values))]
        return root_actions[0][best]
    shifted = values - values.max()
    weights = np.exp(shifted / max(0.05, temperature))
    weights /= weights.sum()
    chosen = rng.choices(list(valid_k0), weights=list(weights), k=1)[0]
    return root_actions[0][chosen]


def play_eval_game(
    blue_kind: str,
    red_kind: str,
    current_model: QuoridorResNet,
    best_model: QuoridorResNet | None,
    device: torch.device,
    rng: random.Random,
    lookahead_eval_config: LookaheadConfig,
    search_config: SearchConfig,
    starting_walls: int,
) -> tuple[str | None, str]:
    game = QuoridorGame(seed=rng.randrange(1_000_000_000), starting_walls=starting_walls)
    while game.winner is None:
        kind = blue_kind if game.turn == BLUE else red_kind
        if kind == "current":
            action = select_model_action(current_model, game, device, lookahead_eval_config, rng, temperature=0.4, sample=False)
        elif kind == "best" and best_model is not None:
            action = select_model_action(best_model, game, device, lookahead_eval_config, rng, temperature=0.4, sample=False)
        elif kind == "search":
            action, _score, _stats = best_action(game, search_config)
        else:
            legal = game.valid_actions(game.turn)
            action = rng.choice(legal) if legal else None

        if action is None:
            game.winner = game.opponent(game.turn)
            game.win_reason = "no_actions"
            break
        mask = game.action_mask(game.turn)
        if action < 0 or action >= len(mask) or not mask[action]:
            game.winner = game.opponent(game.turn)
            game.win_reason = "illegal"
            break
        game.apply_action(action, validated=True)
    return game.winner, game.win_reason or ""


def evaluate_matchup(
    opponent_kind: str,
    current_model: QuoridorResNet,
    best_model: QuoridorResNet | None,
    device: torch.device,
    rng: random.Random,
    games: int,
    lookahead_eval_config: LookaheadConfig,
    search_config: SearchConfig,
    starting_walls: int,
) -> tuple[float, float, float, float]:
    wins = 0
    caps = 0
    first_games = 0
    first_wins = 0
    second_games = 0
    second_wins = 0
    for index in range(games):
        current_is_blue = index % 2 == 0
        blue_kind = "current" if current_is_blue else opponent_kind
        red_kind = opponent_kind if current_is_blue else "current"
        winner, reason = play_eval_game(
            blue_kind, red_kind, current_model, best_model, device, rng, lookahead_eval_config, search_config, starting_walls,
        )
        current_won = (current_is_blue and winner == BLUE) or ((not current_is_blue) and winner == RED)
        if current_won:
            wins += 1
        if current_is_blue:
            first_games += 1
            if current_won:
                first_wins += 1
        else:
            second_games += 1
            if current_won:
                second_wins += 1
        if reason == "turn_limit":
            caps += 1
    return (
        100.0 * wins / max(1, games),
        100.0 * caps / max(1, games),
        100.0 * first_wins / max(1, first_games),
        100.0 * second_wins / max(1, second_games),
    )


def evaluate_model_tournament(
    current_model: QuoridorResNet,
    opponent_models: list[QuoridorResNet],
    device: torch.device,
    rng: random.Random,
    games_per_opponent: int,
    lookahead_eval_config: LookaheadConfig,
    search_config: SearchConfig,
    starting_walls: int,
) -> tuple[float | None, float | None]:
    if not opponent_models:
        return None, None
    win_rates = []
    cap_rates = []
    for opponent_model in opponent_models:
        win_rate, cap_rate, _first_wr, _second_wr = evaluate_matchup(
            "best", current_model, opponent_model, device, rng, games_per_opponent, lookahead_eval_config, search_config, starting_walls,
        )
        win_rates.append(win_rate)
        cap_rates.append(cap_rate)
    return float(np.mean(win_rates)), float(np.mean(cap_rates))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: QuoridorResNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    update: int,
    score: float,
    stage_index: int,
    stage_walls: int,
    stage_passes: int,
    league_mode: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "update": update,
            "score": score,
            "stage_index": stage_index,
            "stage_walls": stage_walls,
            "stage_passes": stage_passes,
            "league_mode": league_mode,
            "action_size": ACTION_SIZE,
            "args": vars(args),
        },
        path,
    )


def load_model_from_checkpoint(path: Path, device: torch.device) -> tuple[QuoridorResNet, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {}) or {}
    channels = int(saved_args.get("channels", DEFAULT_CHANNELS))
    blocks = int(saved_args.get("blocks", DEFAULT_BLOCKS))
    model = QuoridorResNet(channels=channels, blocks=blocks).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def league_snapshot_paths(league_dir: Path, limit: int) -> list[Path]:
    if limit <= 0 or not league_dir.exists():
        return []
    paths = sorted(league_dir.glob("league_*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[:limit]


def load_league_models(league_dir: Path, device: torch.device, limit: int) -> list[QuoridorResNet]:
    models = []
    for path in league_snapshot_paths(league_dir, limit):
        try:
            model, _checkpoint = load_model_from_checkpoint(path, device)
            models.append(model)
        except Exception as error:
            print(f"Could not load league snapshot {path}: {error}")
    return models


def save_league_snapshot(
    league_dir: Path,
    model: QuoridorResNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    update: int,
    score: float,
    stage_index: int,
    stage_walls: int,
    stage_passes: int,
    max_snapshots: int,
) -> None:
    if max_snapshots <= 0:
        return
    league_dir.mkdir(parents=True, exist_ok=True)
    path = league_dir / f"league_{update:06d}.pt"
    save_checkpoint(path, model, optimizer, args, update, score, stage_index, stage_walls, stage_passes, league_mode=True)
    snapshots = sorted(league_dir.glob("league_*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    for stale_path in snapshots[max_snapshots:]:
        try:
            stale_path.unlink()
        except OSError:
            pass


class CheckpointModelPlayer:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        temperature: float = 0.5,
        root_width: int = 10,
        reply_width: int = 5,
        second_width: int = 4,
        seed: int = 11,
        device: str = "auto",
    ) -> None:
        self.path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT_DIR / "quoridor_best.pt"
        selected_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected_device)
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.model, self.checkpoint = load_model_from_checkpoint(self.path, self.device)
        self.config = LookaheadConfig(root_width=root_width, reply_width=reply_width, second_width=second_width)

    def select_action(self, game: QuoridorGame) -> int | None:
        return select_model_action(
            self.model, game, self.device, self.config, self.rng, temperature=self.temperature, sample=True,
        )


# ---------------------------------------------------------------------------
# Curriculum, logging, and the training loop
# ---------------------------------------------------------------------------


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = list(row.keys())
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_header = handle.readline().strip().split(",")
        if existing_header != fieldnames:
            rotated = path.with_name(f"{path.stem}_old_schema_{int(time.time())}{path.suffix}")
            path.replace(rotated)
            write_header = True
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def mean_metric(stats: list[dict[str, float]], key: str) -> float:
    if not stats:
        return 0.0
    return float(np.mean([item[key] for item in stats]))


def strength_score(champ_wr: float, search_wr: float, random_wr: float, eval_cap: float) -> float:
    return 0.55 * champ_wr + 0.30 * search_wr + 0.15 * random_wr - 0.35 * eval_cap


def self_play_score(league_wr: float | None, champ_wr: float, league_cap: float | None, champ_cap: float) -> float:
    win_rate = champ_wr if league_wr is None else league_wr
    cap_rate = champ_cap if league_cap is None else league_cap
    return win_rate - 0.35 * cap_rate


def parse_curriculum_stages(raw: str) -> list[int]:
    stages = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError("Curriculum wall counts cannot be negative.")
        stages.append(min(value, STARTING_WALLS))
    if not stages:
        stages = [STARTING_WALLS]
    return list(dict.fromkeys(stages))


def current_stage_walls(args: argparse.Namespace, stages: list[int], stage_index: int) -> int:
    if not args.curriculum:
        return STARTING_WALLS
    return stages[min(stage_index, len(stages) - 1)]


def champion_stage_component(champ_wr: float, threshold: float) -> float:
    return 100.0 if champ_wr >= threshold else 0.0


def format_percent(value: float | None) -> str:
    if value is None:
        return "  -- "
    return f"{value:5.1f}"


def run_training(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    if args.smoke:
        args.updates = 1
        args.parallel_games = min(args.parallel_games, 4)
        args.root_width = min(args.root_width, 6)
        args.reply_width = min(args.reply_width, 4)
        args.second_width = min(args.second_width, 3)
        args.batch_size = min(args.batch_size, 32)
        args.epochs = 1
        args.eval_every = 1
        args.eval_games = 2
        args.min_replay = 1
        args.no_save = True

    curriculum_stages = parse_curriculum_stages(args.curriculum_stages)
    stage_index = 0
    stage_passes = 0
    stage_up_message = ""

    checkpoint_dir = Path(args.checkpoint_dir)
    latest_path = checkpoint_dir / "quoridor_latest.pt"
    best_path = checkpoint_dir / "quoridor_best.pt"
    league_dir = checkpoint_dir / "league"
    csv_path = checkpoint_dir / "training_log.csv" if args.log_csv is None else Path(args.log_csv)

    model = QuoridorResNet(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_update = 1
    best_model: QuoridorResNet | None = None
    best_score = -float("inf")
    league_mode = False

    resume_path = latest_path if latest_path.exists() else best_path
    if args.resume and resume_path.exists():
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_update = int(checkpoint.get("update", 0)) + 1
            best_score = float(checkpoint.get("score", best_score))
            stage_index = int(checkpoint.get("stage_index", stage_index))
            stage_passes = int(checkpoint.get("stage_passes", stage_passes))
            league_mode = bool(checkpoint.get("league_mode", league_mode))
            print(f"Loaded resume checkpoint: {resume_path}")
        except Exception as error:
            print(f"Could not resume from {resume_path} ({error}); starting from a fresh model.")
    if stage_index >= len(curriculum_stages):
        old_stage_index = stage_index
        stage_index = len(curriculum_stages) - 1
        stage_passes = 0
        print(
            f"Resume stage {old_stage_index} exceeds current curriculum; "
            f"clamped to stage {stage_index}:{curriculum_stages[stage_index]}w."
        )

    if best_path.exists():
        try:
            best_model, best_checkpoint = load_model_from_checkpoint(best_path, device)
            best_stage = min(int(best_checkpoint.get("stage_index", stage_index)), len(curriculum_stages) - 1)
            if best_stage == stage_index:
                best_score = float(best_checkpoint.get("score", best_score))
                print(f"Loaded best checkpoint: {best_path} | score {best_score:.2f}")
            else:
                best_model = QuoridorResNet(channels=args.channels, blocks=args.blocks).to(device)
                best_model.load_state_dict(model.state_dict())
                best_model.eval()
                best_score = -float("inf")
                print(f"Best checkpoint is stage {best_stage}, current stage is {stage_index}; resetting champion for this stage.")
        except Exception as error:
            print(f"Could not load best checkpoint {best_path} ({error}); resetting champion.")
            best_model = None
    if best_model is None:
        best_model = QuoridorResNet(channels=args.channels, blocks=args.blocks).to(device)
        best_model.load_state_dict(model.state_dict())
        best_model.eval()
    league_models = load_league_models(league_dir, device, args.league_pool_size)
    if league_models:
        print(f"Loaded league tournament pool: {len(league_models)} snapshots from {league_dir}")

    replay = ReplayBuffer(args.replay_size, rng)
    lookahead_config = LookaheadConfig(
        root_width=args.root_width,
        reply_width=args.reply_width,
        second_width=args.second_width,
        inference_batch_size=args.inference_batch_size,
    )
    lookahead_eval_config = LookaheadConfig(
        root_width=args.eval_root_width,
        reply_width=args.eval_reply_width,
        second_width=args.eval_second_width,
        inference_batch_size=args.inference_batch_size,
    )
    search_eval_config = SearchConfig(depth=args.eval_search_depth)

    last_champ_wr: float | None = None
    last_search_wr: float | None = None
    last_random_wr: float | None = None
    last_eval_cap: float | None = None
    last_champ_cap: float | None = None
    last_search_cap: float | None = None
    last_random_cap: float | None = None
    last_stage_cap: float | None = None
    last_score: float | None = None
    last_stage_score: float | None = None
    last_league_wr: float | None = None
    last_league_cap: float | None = None

    print(
        "logging | "
        f"csv {csv_path} | latest {latest_path} | best {best_path} | "
        f"device {device} | channels {args.channels} | blocks {args.blocks} | actions {ACTION_SIZE} | "
        f"updates {args.updates} | checkpointEvery {args.checkpoint_every} | resume {args.resume} | "
        f"curriculum {args.curriculum} | stages {curriculum_stages} | league {league_mode} | "
        f"lookahead {args.root_width}x{args.reply_width}x{args.second_width} | parallelGames {args.parallel_games}"
    )

    for update in range(start_update, args.updates + 1):
        stage_walls = current_stage_walls(args, curriculum_stages, stage_index)
        eval_stage_index = stage_index
        eval_stage_walls = stage_walls
        started = time.perf_counter()
        model.eval()

        samples, game_stats = run_lockstep_self_play(
            model,
            device,
            lookahead_config,
            rng,
            parallel_games=args.parallel_games,
            early_temperature=args.early_temperature,
            late_temperature=args.late_temperature,
            temperature_turns=args.temperature_turns,
            starting_walls=stage_walls,
        )
        replay.add_many(samples)
        generated = len(samples)

        train_stats = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "top1": 0.0,
            "value_mae": 0.0,
            "grad_norm": 0.0,
        }
        if len(replay) >= args.min_replay:
            model.train()
            epoch_stats = [
                train_epoch(model, optimizer, replay, args.batch_size, device)
                for _ in range(args.epochs)
            ]
            train_stats = {
                key: float(np.mean([item[key] for item in epoch_stats]))
                for key in train_stats
            }

        should_eval = update == 1 or update % args.eval_every == 0 or update == args.updates
        became_best = False
        eval_stage_passes = stage_passes
        if should_eval:
            model.eval()
            champ_wr, champ_cap, _champ_first_wr, _champ_second_wr = evaluate_matchup(
                "best", model, best_model, device, rng, args.eval_games, lookahead_eval_config, search_eval_config, stage_walls,
            )
            search_wr, search_cap, search_first_wr, _search_second_wr = evaluate_matchup(
                "search", model, best_model, device, rng, args.eval_games, lookahead_eval_config, search_eval_config, stage_walls,
            )
            random_wr, random_cap, random_first_wr, _random_second_wr = evaluate_matchup(
                "random", model, best_model, device, rng, args.eval_games, lookahead_eval_config, search_eval_config, stage_walls,
            )
            eval_cap = float(np.mean([champ_cap, search_cap, random_cap]))
            current_score = strength_score(champ_wr, search_wr, random_wr, eval_cap)
            league_wr: float | None = None
            league_cap: float | None = None
            if league_mode:
                if not league_models:
                    league_wr, league_cap = champ_wr, champ_cap
                else:
                    league_opponents = []
                    if best_model is not None:
                        league_opponents.append(best_model)
                    league_opponents.extend(league_models)
                    league_wr, league_cap = evaluate_model_tournament(
                        model, league_opponents, device, rng, max(2, args.league_games_per_opponent),
                        lookahead_eval_config, search_eval_config, stage_walls,
                    )
                current_score = self_play_score(league_wr, champ_wr, league_cap, champ_cap)
            if stage_walls == 0:
                stage_gate_wr = search_first_wr
                stage_random_wr = random_first_wr
                stage_cap = max(search_cap, random_cap)
                stage_score = 0.70 * stage_gate_wr + 0.30 * stage_random_wr - 0.35 * stage_cap
                champ_ready = True
            else:
                stage_gate_wr = search_wr
                stage_random_wr = random_wr
                stage_cap = max(search_cap, random_cap)
                champ_ready = champ_wr >= args.stage_champ_wr
                stage_champ_component = champion_stage_component(champ_wr, args.stage_champ_wr)
                stage_score = strength_score(stage_champ_component, stage_gate_wr, stage_random_wr, stage_cap)
            last_champ_wr = champ_wr
            last_search_wr = search_wr
            last_random_wr = random_wr
            last_eval_cap = eval_cap
            last_champ_cap = champ_cap
            last_search_cap = search_cap
            last_random_cap = random_cap
            last_stage_cap = stage_cap
            last_score = current_score
            last_stage_score = stage_score
            last_league_wr = league_wr
            last_league_cap = league_cap
            final_stage = args.curriculum and stage_index >= len(curriculum_stages) - 1
            gate_ready = champ_ready and stage_score >= args.stage_score and stage_cap <= args.stage_max_cap
            stage_ready = args.curriculum and not league_mode and not final_stage and gate_ready
            final_ready = args.curriculum and not league_mode and final_stage and gate_ready
            if stage_ready or final_ready:
                stage_passes += 1
            elif not league_mode:
                stage_passes = 0
            eval_stage_passes = stage_passes
            best_candidate_score = current_score if league_mode else stage_score if args.curriculum else current_score
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                became_best = True
                best_model = QuoridorResNet(channels=args.channels, blocks=args.blocks).to(device)
                best_model.load_state_dict(model.state_dict())
                best_model.eval()
                if not args.no_save:
                    save_checkpoint(best_path, model, optimizer, args, update, best_score, stage_index, stage_walls, stage_passes, league_mode)
                    if league_mode:
                        save_league_snapshot(
                            league_dir, model, optimizer, args, update, best_score, stage_index, stage_walls, stage_passes, args.league_pool_size,
                        )
                        league_models = load_league_models(league_dir, device, args.league_pool_size)

            if stage_ready and stage_passes >= args.stage_passes:
                old_stage_index = stage_index
                old_walls = stage_walls
                stage_index += 1
                stage_walls = current_stage_walls(args, curriculum_stages, stage_index)
                stage_passes = 0
                if args.clear_replay_on_stage_up:
                    replay.clear()
                best_score = stage_score
                best_model = QuoridorResNet(channels=args.channels, blocks=args.blocks).to(device)
                best_model.load_state_dict(model.state_dict())
                best_model.eval()
                if not args.no_save:
                    save_checkpoint(best_path, model, optimizer, args, update, best_score, stage_index, stage_walls, stage_passes, league_mode)
                stage_up_message = (
                    f"STAGE UP at update {update}: stage {old_stage_index}:{old_walls}w -> {stage_index}:{stage_walls}w | "
                    f"searchWR {search_wr:.1f} | gateWR {stage_gate_wr:.1f} | stageCap {stage_cap:.1f} | "
                    f"stageScore {stage_score:.1f} | replay {'cleared' if args.clear_replay_on_stage_up else 'kept'}"
                )
                print(stage_up_message, flush=True)
            elif final_ready and stage_passes >= args.stage_passes:
                league_mode = True
                stage_passes = args.stage_passes
                if league_wr is None:
                    league_wr, league_cap = champ_wr, champ_cap
                    current_score = self_play_score(league_wr, champ_wr, league_cap, champ_cap)
                    last_league_wr = league_wr
                    last_league_cap = league_cap
                    last_score = current_score
                best_score = current_score
                best_model = QuoridorResNet(channels=args.channels, blocks=args.blocks).to(device)
                best_model.load_state_dict(model.state_dict())
                best_model.eval()
                if not args.no_save:
                    save_checkpoint(best_path, model, optimizer, args, update, best_score, stage_index, stage_walls, stage_passes, league_mode)
                    save_league_snapshot(
                        league_dir, model, optimizer, args, update, best_score, stage_index, stage_walls, stage_passes, args.league_pool_size,
                    )
                    league_models = load_league_models(league_dir, device, args.league_pool_size)
                stage_up_message = (
                    f"LEAGUE MODE at update {update}: final stage {stage_index}:{stage_walls}w mastered | "
                    f"stageScore {stage_score:.1f} | self-play tournaments are now the main score"
                )
                print(stage_up_message, flush=True)

        checkpoint_due = update == args.updates or update % max(1, args.checkpoint_every) == 0
        latest_saved = False
        if not args.no_save and checkpoint_due:
            save_checkpoint(latest_path, model, optimizer, args, update, best_score, stage_index, stage_walls, stage_passes, league_mode)
            latest_saved = True

        elapsed = time.perf_counter() - started
        blue_wins = int(sum(item["blue_win"] for item in game_stats))
        red_wins = int(sum(item["red_win"] for item in game_stats))
        row = {
            "update": update,
            "generated": generated,
            "stage": stage_index,
            "stage_walls": stage_walls,
            "eval_stage": eval_stage_index if should_eval else None,
            "eval_stage_walls": eval_stage_walls if should_eval else None,
            "eval_stage_passes": eval_stage_passes if should_eval else None,
            "stage_passes": stage_passes,
            "stage_up": 1 if stage_up_message else 0,
            "league_mode": int(league_mode),
            "replay": len(replay),
            "games_finished": len(game_stats),
            "self_len": round(mean_metric(game_stats, "length"), 4),
            "cap_pct": round(100.0 * mean_metric(game_stats, "cap"), 4),
            "blue_wins": blue_wins,
            "red_wins": red_wins,
            "wall_rate": round(100.0 * mean_metric(game_stats, "wall_rate"), 4),
            "champ_wr": None if last_champ_wr is None else round(last_champ_wr, 4),
            "search_wr": None if last_search_wr is None else round(last_search_wr, 4),
            "rand_wr": None if last_random_wr is None else round(last_random_wr, 4),
            "eval_cap": None if last_eval_cap is None else round(last_eval_cap, 4),
            "champ_cap": None if last_champ_cap is None else round(last_champ_cap, 4),
            "search_cap": None if last_search_cap is None else round(last_search_cap, 4),
            "rand_cap": None if last_random_cap is None else round(last_random_cap, 4),
            "stage_cap": None if last_stage_cap is None else round(last_stage_cap, 4),
            "league_wr": None if last_league_wr is None else round(last_league_wr, 4),
            "league_cap": None if last_league_cap is None else round(last_league_cap, 4),
            "score": None if last_score is None else round(last_score, 4),
            "stage_score": None if last_stage_score is None else round(last_stage_score, 4),
            "best": round(best_score, 4),
            "loss": round(train_stats["loss"], 6),
            "policy_loss": round(train_stats["policy_loss"], 6),
            "value_loss": round(train_stats["value_loss"], 6),
            "train_entropy": round(train_stats["entropy"], 6),
            "top1": round(100.0 * train_stats["top1"], 4),
            "value_mae": round(train_stats["value_mae"], 6),
            "grad_norm": round(train_stats["grad_norm"], 6),
            "checkpoint_saved": int(latest_saved),
            "seconds": round(elapsed, 3),
        }
        if not args.no_csv:
            append_csv(csv_path, row)

        marker = (" *best" if became_best else "") + (" *save" if latest_saved else "")
        promoted_this_update = bool(stage_up_message)
        if stage_up_message:
            marker += " *league" if stage_up_message.startswith("LEAGUE MODE") else " *stage"
        eval_marker = "eval" if should_eval else "    "
        if should_eval:
            if league_mode:
                stage_label = f"league {eval_stage_walls}w"
            elif args.curriculum and eval_stage_index >= len(curriculum_stages) - 1:
                stage_label = f"finalStage {eval_stage_index}:{eval_stage_walls}w {eval_stage_passes}/{args.stage_passes}"
            else:
                stage_label = f"evalStage {eval_stage_index}:{eval_stage_walls}w {eval_stage_passes}/{args.stage_passes}"
            if stage_up_message and not stage_up_message.startswith("LEAGUE MODE"):
                stage_label += f" -> next {stage_index}:{stage_walls}w"
        else:
            if league_mode:
                stage_label = f"league {stage_walls}w"
            elif args.curriculum and stage_index >= len(curriculum_stages) - 1:
                stage_label = f"final {stage_index}:{stage_walls}w {stage_passes}/{args.stage_passes}"
            else:
                stage_label = f"stage {stage_index}:{stage_walls}w {stage_passes}/{args.stage_passes}"
        print(
            f"Update {update:03d} | {eval_marker} | "
            f"{stage_label} | "
            f"finished {row['games_finished']:3d} | selfLen {row['self_len']:5.1f} | cap% {row['cap_pct']:5.1f} | "
            f"B/R {blue_wins:2d}/{red_wins:2d} | replay {len(replay):6d} | gen {generated:4d} | "
            f"walls {row['wall_rate']:5.1f}% | "
            f"champWR {format_percent(last_champ_wr)} | searchWR {format_percent(last_search_wr)} | "
            f"randWR {format_percent(last_random_wr)} | stageCap {format_percent(last_stage_cap)} | "
            f"evalCap {format_percent(last_eval_cap)} | leagueWR {format_percent(last_league_wr)} | "
            f"score {format_percent(last_score)} | stageScore {format_percent(last_stage_score)} | best {best_score:5.1f} | "
            f"pLoss {train_stats['policy_loss']:6.3f} | vLoss {train_stats['value_loss']:6.3f} | "
            f"ent {train_stats['entropy']:5.2f} | top1 {100.0 * train_stats['top1']:5.1f} | "
            f"vMAE {train_stats['value_mae']:5.2f} | {elapsed:6.2f}s{marker}",
            flush=True,
        )
        if promoted_this_update:
            last_champ_wr = None
            last_search_wr = None
            last_random_wr = None
            last_eval_cap = None
            last_champ_cap = None
            last_search_cap = None
            last_random_cap = None
            last_stage_cap = None
            last_score = None
            last_stage_score = None
            last_league_wr = None
            last_league_cap = None
        stage_up_message = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TreeStrap-style Quoridor self-play trainer: GPU-batched 3-ply lookahead bootstraps the value/policy targets.")
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--parallel-games", type=int, default=16)
    parser.add_argument("--root-width", type=int, default=12)
    parser.add_argument("--reply-width", type=int, default=6)
    parser.add_argument("--second-width", type=int, default=5)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--replay-size", type=int, default=60_000)
    parser.add_argument("--min-replay", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--early-temperature", type=float, default=1.0)
    parser.add_argument("--late-temperature", type=float, default=0.3)
    parser.add_argument("--temperature-turns", type=int, default=18)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=8)
    parser.add_argument("--eval-root-width", type=int, default=14)
    parser.add_argument("--eval-reply-width", type=int, default=7)
    parser.add_argument("--eval-second-width", type=int, default=6)
    parser.add_argument("--eval-search-depth", type=int, default=2)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--curriculum-stages", type=str, default="0,2,4,6,8,10")
    parser.add_argument("--stage-champ-wr", type=float, default=50.0)
    parser.add_argument("--stage-score", type=float, default=80.0)
    parser.add_argument("--stage-max-cap", type=float, default=20.0)
    parser.add_argument("--stage-passes", type=int, default=4)
    parser.add_argument("--league-pool-size", type=int, default=4)
    parser.add_argument("--league-games-per-opponent", type=int, default=4)
    parser.add_argument("--clear-replay-on-stage-up", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-csv", type=str, default=None)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
