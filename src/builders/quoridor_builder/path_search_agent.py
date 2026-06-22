from __future__ import annotations

from dataclasses import dataclass

try:
    from .video_generator import ACTION_SIZE, MOVE_ACTION_COUNT, QuoridorGame
except ImportError:
    from video_generator import ACTION_SIZE, MOVE_ACTION_COUNT, QuoridorGame


@dataclass(frozen=True)
class SearchConfig:
    depth: int = 2
    max_wall_candidates: int = 10
    max_root_wall_candidates: int = 16
    max_actions: int = 16
    max_root_actions: int = 24


def static_score(game: QuoridorGame, player: str) -> float:
    opponent = game.opponent(player)
    if game.winner == player:
        return 1000.0
    if game.winner == opponent:
        return -1000.0
    own_path = game.shortest_path_length(player)
    opp_path = game.shortest_path_length(opponent)
    score = (opp_path - own_path) * 10.0
    score += (game.wall_counts[player] - game.wall_counts[opponent]) * 2.0
    return score


def apply_action_fast(game: QuoridorGame, action: int) -> None:
    game.apply_action(action, validated=True)


def candidate_actions(game: QuoridorGame, player: str, config: SearchConfig, is_root: bool) -> list[int]:
    mask = game.action_mask(player)
    move_actions = [action for action in range(MOVE_ACTION_COUNT) if mask[action]]
    wall_actions = [action for action in range(MOVE_ACTION_COUNT, ACTION_SIZE) if mask[action]]

    if wall_actions:
        opponent = game.opponent(player)
        scored = []
        for action in wall_actions:
            child = game.clone()
            apply_action_fast(child, action)
            opp_after = child.shortest_path_length(opponent)
            own_after = child.shortest_path_length(player)
            scored.append((opp_after - own_after, action))
        scored.sort(key=lambda item: item[0], reverse=True)
        wall_limit = config.max_root_wall_candidates if is_root else config.max_wall_candidates
        wall_actions = [action for _score, action in scored[:wall_limit]]

    actions = move_actions + wall_actions
    total_limit = config.max_root_actions if is_root else config.max_actions
    return actions[:total_limit]


def negamax(game: QuoridorGame, depth: int, config: SearchConfig) -> float:
    if game.winner is not None or depth == 0:
        return static_score(game, game.turn)
    actions = candidate_actions(game, game.turn, config, is_root=False)
    if not actions:
        return static_score(game, game.turn)
    best = -float("inf")
    for action in actions:
        child = game.clone()
        apply_action_fast(child, action)
        value = -negamax(child, depth - 1, config)
        if value > best:
            best = value
    return best


def best_action(game: QuoridorGame, config: SearchConfig) -> tuple[int | None, float, dict[str, int]]:
    player = game.turn
    best_act: int | None = None
    best_val = -float("inf")
    nodes = 0
    for action in candidate_actions(game, player, config, is_root=True):
        child = game.clone()
        apply_action_fast(child, action)
        value = -negamax(child, config.depth - 1, config)
        nodes += 1
        if value > best_val:
            best_val = value
            best_act = action
    if best_act is None:
        legal = game.valid_actions(player)
        best_act = legal[0] if legal else None
    return best_act, best_val, {"nodes": nodes}


class DepthSearchPlayer:
    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    def select_action(self, game: QuoridorGame) -> int | None:
        action, _value, _stats = best_action(game, self.config)
        return action
