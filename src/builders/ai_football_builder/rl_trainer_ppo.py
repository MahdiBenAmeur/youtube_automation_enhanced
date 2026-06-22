from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pygame
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.builders.ai_football_builder.video_generator import (
    ACTION_SIZE,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    FPS,
    KICK_CLEAR,
    KICK_NONE,
    KICK_PASS,
    KICK_SHOOT,
    OBSERVATION_SIZE,
    CLEAR_POWER_FLOOR,
    PASS_POWER_FLOOR,
    PLAYER_RADIUS,
    SHOT_POWER_FLOOR,
    TEAMMATE_TOO_CLOSE_DISTANCE,
    TEAMMATE_SUPPORT_MAX_DISTANCE,
    TEAM_BLUE,
    TEAM_RED,
    FootballEnvironment,
)


TRAINING_STEP_DT = 1.0 / FPS
PPO_TOTAL_UPDATES = 5000
PPO_ROLLOUT_STEPS = 1024
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 256
PPO_GAMMA = 0.985
PPO_GAE_LAMBDA = 0.92
PPO_CLIP_EPSILON = 0.20
PPO_VALUE_COEF = 0.55
PPO_ENTROPY_COEF = 0.008
PPO_LEARNING_RATE = 3e-4
PPO_MAX_GRAD_NORM = 0.5
BEHAVIOR_PRETRAIN_STEPS = 1500
BEHAVIOR_PRETRAIN_BATCH_SIZE = 256
BEHAVIOR_PRETRAIN_SAMPLES = 16000
MODEL_HIDDEN_SIZES = (192, 192)
CHECKPOINT_DIR = Path("src/builders/ai_football_builder/checkpoints")
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
REPLAY_VIDEO_OUTPUT_DIR = Path("generated_videos")
REPLAY_VIDEO_PREFIX = "ai_football_ppo_replay"
REPLAY_VIDEO_CODEC = "mp4v"


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    raw_continuous_actions: torch.Tensor
    kick_types: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class FootballActorCritic(nn.Module):
    def __init__(self, observation_size: int = OBSERVATION_SIZE) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_size = observation_size
        for hidden_size in MODEL_HIDDEN_SIZES:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())
            input_size = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.continuous_mean = nn.Linear(input_size, 5)
        self.continuous_log_std = nn.Parameter(torch.full((5,), -0.75))
        self.kick_logits = nn.Linear(input_size, 4)
        self.value_head = nn.Linear(input_size, 1)

    def forward(
        self,
        observations: torch.Tensor,
    ) -> tuple[Normal, Categorical, torch.Tensor]:
        features = self.backbone(observations)
        mean = self.continuous_mean(features)
        std = torch.exp(self.continuous_log_std).expand_as(mean)
        continuous_distribution = Normal(mean, std)
        kick_distribution = Categorical(logits=self.kick_logits(features))
        values = self.value_head(features).squeeze(-1)
        return continuous_distribution, kick_distribution, values

    def act(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous_distribution, kick_distribution, values = self.forward(observations)
        if deterministic:
            raw_continuous_actions = continuous_distribution.mean
            kick_types = torch.argmax(kick_distribution.logits, dim=-1)
        else:
            raw_continuous_actions = continuous_distribution.rsample()
            kick_types = kick_distribution.sample()

        log_probs = (
            continuous_distribution.log_prob(raw_continuous_actions).sum(dim=-1)
            + kick_distribution.log_prob(kick_types)
        )
        entropy = (
            continuous_distribution.entropy().sum(dim=-1)
            + kick_distribution.entropy()
        )
        return raw_continuous_actions, kick_types, log_probs, entropy, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        raw_continuous_actions: torch.Tensor,
        kick_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous_distribution, kick_distribution, values = self.forward(observations)
        log_probs = (
            continuous_distribution.log_prob(raw_continuous_actions).sum(dim=-1)
            + kick_distribution.log_prob(kick_types)
        )
        entropy = (
            continuous_distribution.entropy().sum(dim=-1)
            + kick_distribution.entropy()
        )
        return log_probs, entropy, values


def raw_actions_to_env_actions(
    raw_continuous_actions: torch.Tensor,
    kick_types: torch.Tensor,
) -> list[list[float]]:
    squashed = torch.tanh(raw_continuous_actions).detach().cpu().numpy()
    kick_values = kick_types.detach().cpu().numpy()
    actions: list[list[float]] = []
    for continuous_action, kick_type in zip(squashed, kick_values):
        move_x = float(continuous_action[0])
        move_y = float(continuous_action[1])
        move_length = float(np.hypot(move_x, move_y))
        if move_length > 0.05:
            target_length = max(0.42, min(1.0, move_length))
            move_x = move_x / move_length * target_length
            move_y = move_y / move_length * target_length
        kick_power = (float(continuous_action[4]) + 1.0) / 2.0
        if int(kick_type) == KICK_SHOOT:
            kick_power = max(SHOT_POWER_FLOOR, kick_power)
        elif int(kick_type) == KICK_CLEAR:
            kick_power = max(CLEAR_POWER_FLOOR, kick_power)
        elif int(kick_type) == KICK_PASS:
            kick_power = max(PASS_POWER_FLOOR, kick_power)
        actions.append(
            [
                move_x,
                move_y,
                float(kick_type),
                float(continuous_action[2]),
                float(continuous_action[3]),
                kick_power,
            ]
        )
    return actions


def env_actions_to_behavior_targets(
    actions: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    continuous_targets: list[list[float]] = []
    kick_targets: list[int] = []
    for action in actions:
        if isinstance(action, dict):
            move_x = float(action["move_x"])
            move_y = float(action["move_y"])
            kick_type = int(action["kick_type"])
            kick_direction_x = float(action["kick_direction_x"])
            kick_direction_y = float(action["kick_direction_y"])
            kick_power = float(action["kick_power"])
        else:
            move_x = float(action[0])
            move_y = float(action[1])
            kick_type = int(action[2])
            kick_direction_x = float(action[3])
            kick_direction_y = float(action[4])
            kick_power = float(action[5])

        kick_power_target = kick_power * 2.0 - 1.0
        continuous_targets.append(
            [
                move_x,
                move_y,
                kick_direction_x,
                kick_direction_y,
                float(np.clip(kick_power_target, -1.0, 1.0)),
            ]
        )
        kick_targets.append(kick_type)
    return (
        np.asarray(continuous_targets, dtype=np.float32),
        np.asarray(kick_targets, dtype=np.int64),
    )


def place_player(env: FootballEnvironment, player_index: int, x: float, y: float) -> None:
    field = env.field_rect
    player = env.players[player_index]
    player["x"] = float(np.clip(x, field.left + PLAYER_RADIUS, field.right - PLAYER_RADIUS))
    player["y"] = float(np.clip(y, field.top + PLAYER_RADIUS, field.bottom - PLAYER_RADIUS))
    player["kick_cooldown"] = 0.0


def seed_behavior_scenario(env: FootballEnvironment, rng: np.random.Generator) -> None:
    env.reset(keep_score=False)
    field = env.field_rect
    team = TEAM_RED if rng.random() < 0.5 else TEAM_BLUE
    attack_sign = env.team_attack_sign(team)
    team_indices = [
        index
        for index, player in enumerate(env.players)
        if str(player["team"]) == team
    ]
    opponent_indices = [
        index
        for index, player in enumerate(env.players)
        if str(player["team"]) != team
    ]
    attacker_index, support_index = team_indices
    scenario = rng.choice(["pass", "shoot", "clear", "defend", "shape"], p=[0.34, 0.22, 0.18, 0.16, 0.10])
    base_x = float(rng.uniform(field.left + 220, field.right - 220))
    base_y = float(rng.uniform(field.top + field.height * 0.32, field.top + field.height * 0.68))

    if scenario == "pass":
        passer_index = attacker_index if rng.random() < 0.65 else support_index
        receiver_index = support_index if passer_index == attacker_index else attacker_index
        place_player(env, passer_index, base_x, base_y)
        receiver_x = base_x + float(rng.choice([-1.0, 1.0])) * float(rng.uniform(260, 390))
        receiver_y = base_y + attack_sign * float(rng.uniform(120, 280))
        place_player(env, receiver_index, receiver_x, receiver_y)
        env.ball["x"] = float(env.players[passer_index]["x"])
        env.ball["y"] = float(env.players[passer_index]["y"])
    elif scenario == "shoot":
        shooter_y = field.top + field.height * (0.70 if team == TEAM_RED else 0.30)
        place_player(env, attacker_index, base_x, shooter_y)
        place_player(env, support_index, base_x + 330.0, shooter_y - attack_sign * 220.0)
        env.ball["x"] = float(env.players[attacker_index]["x"])
        env.ball["y"] = float(env.players[attacker_index]["y"])
    elif scenario == "clear":
        clear_y = field.top + field.height * (0.16 if team == TEAM_RED else 0.84)
        place_player(env, support_index, base_x, clear_y)
        place_player(env, attacker_index, base_x + 360.0, clear_y + attack_sign * 330.0)
        env.ball["x"] = float(env.players[support_index]["x"])
        env.ball["y"] = float(env.players[support_index]["y"])
    elif scenario == "defend":
        ball_y = field.top + field.height * (0.22 if team == TEAM_RED else 0.78)
        env.ball["x"] = base_x
        env.ball["y"] = ball_y
        place_player(env, support_index, base_x, ball_y - attack_sign * 230.0)
        place_player(env, attacker_index, base_x + 360.0, ball_y + attack_sign * 210.0)
    else:
        env.ball["x"] = base_x
        env.ball["y"] = base_y
        place_player(env, attacker_index, base_x - 260.0, base_y - attack_sign * 120.0)
        place_player(env, support_index, base_x + 340.0, base_y + attack_sign * 260.0)

    for offset, opponent_index in enumerate(opponent_indices):
        opponent_x = field.centerx + (-1 if offset == 0 else 1) * float(rng.uniform(210, 360))
        opponent_y = float(env.ball["y"]) + attack_sign * float(rng.uniform(-260, 220))
        place_player(env, opponent_index, opponent_x, opponent_y)
    env.ball["vx"] = 0.0
    env.ball["vy"] = 0.0
    env.ball_touch_cooldown = 0.0


def add_pass_teacher_action(
    env: FootballEnvironment,
    actions: list[dict[str, float | int]],
    rng: np.random.Generator,
) -> None:
    if rng.random() > 0.58:
        return
    candidates: list[int] = [
        index
        for index, player in enumerate(env.players)
        if env.can_player_kick(player) and not env.is_ball_near_own_goal(str(player["team"]))
    ]
    rng.shuffle(candidates)
    for player_index in candidates:
        player = env.players[player_index]
        teammate = env.players[env.get_teammate_index(player_index)]
        dx = float(teammate["x"]) - float(player["x"])
        dy = float(teammate["y"]) - float(player["y"])
        distance = float(np.hypot(dx, dy))
        if distance < TEAMMATE_TOO_CLOSE_DISTANCE or distance > TEAMMATE_SUPPORT_MAX_DISTANCE:
            continue
        direction_x, direction_y = env.world_direction_to_action(str(player["team"]), dx, dy)
        actions[player_index] = {
            "move_x": 0.0,
            "move_y": 0.0,
            "kick_type": KICK_PASS,
            "kick_direction_x": direction_x,
            "kick_direction_y": direction_y,
            "kick_power": float(rng.uniform(0.50, 0.80)),
        }
        return


def collect_behavior_examples(
    sample_count: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = FootballEnvironment(seed=seed)
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    continuous_targets: list[np.ndarray] = []
    kick_targets: list[np.ndarray] = []

    while len(observations) * len(env.players) < sample_count:
        seed_behavior_scenario(env, rng)
        current_observations = np.asarray(env.get_observations(), dtype=np.float32)
        actions = env.build_scripted_actions()
        add_pass_teacher_action(env, actions, rng)
        continuous_target, kick_target = env_actions_to_behavior_targets(actions)
        observations.append(current_observations)
        continuous_targets.append(continuous_target)
        kick_targets.append(kick_target)
        if np.any(kick_target != KICK_NONE):
            extra_copies = 20 if np.any(kick_target == KICK_PASS) else 12
            for _extra_copy in range(extra_copies):
                observations.append(current_observations)
                continuous_targets.append(continuous_target)
                kick_targets.append(kick_target)
        _obs, _rewards, done, _info = env.step(actions, TRAINING_STEP_DT, training=True)
        if done:
            env.reset(keep_score=False)

    return (
        np.concatenate(observations, axis=0)[:sample_count],
        np.concatenate(continuous_targets, axis=0)[:sample_count],
        np.concatenate(kick_targets, axis=0)[:sample_count],
    )


def pretrain_behavior_prior(
    model: FootballActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    seed: int | None,
) -> dict[str, float]:
    if steps <= 0:
        return {"behavior_loss": 0.0, "kick_accuracy": 0.0}

    observations, continuous_targets, kick_targets = collect_behavior_examples(
        sample_count=BEHAVIOR_PRETRAIN_SAMPLES,
        seed=seed,
    )
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    continuous_tensor = torch.as_tensor(continuous_targets, dtype=torch.float32, device=device)
    kick_tensor = torch.as_tensor(kick_targets, dtype=torch.long, device=device)
    sample_count = observation_tensor.shape[0]
    loss_value = 0.0
    accuracy_value = 0.0

    for _step in range(steps):
        indices = torch.randint(0, sample_count, (BEHAVIOR_PRETRAIN_BATCH_SIZE,), device=device)
        batch_observations = observation_tensor[indices]
        batch_continuous = continuous_tensor[indices]
        batch_kicks = kick_tensor[indices]

        features = model.backbone(batch_observations)
        predicted_continuous = torch.tanh(model.continuous_mean(features))
        kick_logits = model.kick_logits(features)
        continuous_weights = torch.ones_like(batch_continuous)
        no_kick_mask = batch_kicks == KICK_NONE
        continuous_weights[no_kick_mask, 2:] = 0.0
        continuous_error = (predicted_continuous - batch_continuous).pow(2) * continuous_weights
        continuous_loss = continuous_error.sum() / continuous_weights.sum().clamp_min(1.0)
        kick_loss = F.cross_entropy(kick_logits, batch_kicks)
        loss = 1.8 * continuous_loss + 0.8 * kick_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), PPO_MAX_GRAD_NORM)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        accuracy_value = float(
            (torch.argmax(kick_logits, dim=-1) == batch_kicks)
            .float()
            .mean()
            .detach()
            .cpu()
            .item()
        )

    return {
        "behavior_loss": loss_value,
        "kick_accuracy": accuracy_value,
    }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(rewards.shape[1], dtype=torch.float32)
    next_value = last_values

    for step_index in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[step_index]
        delta = rewards[step_index] + PPO_GAMMA * next_value * not_done - values[step_index]
        next_advantage = delta + PPO_GAMMA * PPO_GAE_LAMBDA * not_done * next_advantage
        advantages[step_index] = next_advantage
        next_value = values[step_index]

    returns = advantages + values
    return advantages, returns


def collect_rollout(
    env: FootballEnvironment,
    model: FootballActorCritic,
    device: torch.device,
    rollout_steps: int,
) -> tuple[RolloutBatch, dict[str, float]]:
    observation = np.asarray(env.get_observations(), dtype=np.float32)
    observations: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    kick_types: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []
    values: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    episode_returns: list[float] = []
    current_episode_return = 0.0
    goal_count = 0
    kick_count = 0
    shot_count = 0
    shot_power_total = 0.0
    pass_count = 0
    clear_count = 0
    successful_pass_count = 0
    assisted_goal_count = 0
    clump_step_count = 0
    teammate_distance_total = 0.0
    stale_touch_total = 0.0
    forward_progress_total = 0.0
    last_done = False

    for _ in range(rollout_steps):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        with torch.no_grad():
            raw_action_tensor, kick_type_tensor, log_prob_tensor, _entropy, value_tensor = model.act(
                observation_tensor,
                deterministic=False,
            )
        env_actions = raw_actions_to_env_actions(raw_action_tensor, kick_type_tensor)
        next_observation, reward, done, info = env.step(
            env_actions,
            TRAINING_STEP_DT,
            training=True,
        )

        observations.append(observation)
        raw_actions.append(raw_action_tensor.detach().cpu().numpy())
        kick_types.append(kick_type_tensor.detach().cpu().numpy())
        log_probs.append(log_prob_tensor.detach().cpu().numpy())
        values.append(value_tensor.detach().cpu().numpy())
        rewards.append(np.asarray(reward, dtype=np.float32))
        dones.append(np.full(len(reward), float(done), dtype=np.float32))

        current_episode_return += float(np.mean(reward))
        if info["goal_team"] is not None:
            goal_count += 1
        if info["successful_pass"] is not None:
            successful_pass_count += 1
        if info["assisted_goal_team"] is not None:
            assisted_goal_count += 1
        min_teammate_distance = float(info["min_teammate_distance"])
        teammate_distance_total += min_teammate_distance
        if min_teammate_distance < TEAMMATE_TOO_CLOSE_DISTANCE:
            clump_step_count += 1
        stale_touch_total += max(float(info["no_touch_red"]), float(info["no_touch_blue"]))
        forward_progress_total += max(0.0, float(info["progress_delta"]["red"]))
        forward_progress_total += max(0.0, float(info["progress_delta"]["blue"]))
        if info["kick_player"] is not None:
            kick_count += 1
            if info["kick_type"] == KICK_SHOOT:
                shot_count += 1
                shot_power_total += float(info["kick_power"])
            elif info["kick_type"] == KICK_PASS:
                pass_count += 1
            elif info["kick_type"] == KICK_CLEAR:
                clear_count += 1

        if done:
            episode_returns.append(current_episode_return)
            current_episode_return = 0.0
            next_observation = env.reset(keep_score=False)

        observation = np.asarray(next_observation, dtype=np.float32)
        last_done = done

    with torch.no_grad():
        if last_done:
            last_values = torch.zeros(len(env.players), dtype=torch.float32)
        else:
            last_values = model.forward(
                torch.as_tensor(observation, dtype=torch.float32, device=device)
            )[2].detach().cpu()

    rewards_tensor = torch.as_tensor(np.stack(rewards), dtype=torch.float32)
    values_tensor = torch.as_tensor(np.stack(values), dtype=torch.float32)
    dones_tensor = torch.as_tensor(np.stack(dones), dtype=torch.float32)
    advantages, returns = compute_gae(
        rewards_tensor,
        values_tensor,
        dones_tensor,
        last_values,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = RolloutBatch(
        observations=torch.as_tensor(np.stack(observations), dtype=torch.float32).reshape(-1, OBSERVATION_SIZE),
        raw_continuous_actions=torch.as_tensor(np.stack(raw_actions), dtype=torch.float32).reshape(-1, 5),
        kick_types=torch.as_tensor(np.stack(kick_types), dtype=torch.long).reshape(-1),
        old_log_probs=torch.as_tensor(np.stack(log_probs), dtype=torch.float32).reshape(-1),
        returns=returns.reshape(-1),
        advantages=advantages.reshape(-1),
    )
    metrics = {
        "episode_return": float(np.mean(episode_returns)) if episode_returns else current_episode_return,
        "goals": float(goal_count),
        "kicks": float(kick_count),
        "shots": float(shot_count),
        "mean_shot_power": shot_power_total / max(1, shot_count),
        "passes": float(pass_count),
        "clears": float(clear_count),
        "successful_passes": float(successful_pass_count),
        "assisted_goals": float(assisted_goal_count),
        "clump_steps": float(clump_step_count),
        "mean_teammate_distance": teammate_distance_total / max(1, rollout_steps),
        "mean_stale_touch_time": stale_touch_total / max(1, rollout_steps),
        "forward_progress": float(forward_progress_total),
        "mean_reward": float(rewards_tensor.mean().item()),
    }
    return batch, metrics


def update_policy(
    model: FootballActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: RolloutBatch,
    device: torch.device,
) -> dict[str, float]:
    sample_count = batch.observations.shape[0]
    indices = torch.arange(sample_count)
    policy_loss_value = 0.0
    value_loss_value = 0.0
    entropy_value = 0.0
    update_count = 0

    for _epoch in range(PPO_EPOCHS):
        shuffled_indices = indices[torch.randperm(sample_count)]
        for start in range(0, sample_count, PPO_MINIBATCH_SIZE):
            minibatch_indices = shuffled_indices[start : start + PPO_MINIBATCH_SIZE]
            observations = batch.observations[minibatch_indices].to(device)
            raw_actions = batch.raw_continuous_actions[minibatch_indices].to(device)
            kick_types = batch.kick_types[minibatch_indices].to(device)
            old_log_probs = batch.old_log_probs[minibatch_indices].to(device)
            returns = batch.returns[minibatch_indices].to(device)
            advantages = batch.advantages[minibatch_indices].to(device)

            log_probs, entropy, values = model.evaluate_actions(
                observations,
                raw_actions,
                kick_types,
            )
            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(
                ratio,
                1.0 - PPO_CLIP_EPSILON,
                1.0 + PPO_CLIP_EPSILON,
            ) * advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_bonus = entropy.mean()
            loss = (
                policy_loss
                + PPO_VALUE_COEF * value_loss
                - PPO_ENTROPY_COEF * entropy_bonus
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), PPO_MAX_GRAD_NORM)
            optimizer.step()

            policy_loss_value += float(policy_loss.detach().cpu().item())
            value_loss_value += float(value_loss.detach().cpu().item())
            entropy_value += float(entropy_bonus.detach().cpu().item())
            update_count += 1

    divisor = max(1, update_count)
    return {
        "policy_loss": policy_loss_value / divisor,
        "value_loss": value_loss_value / divisor,
        "entropy": entropy_value / divisor,
    }


def save_checkpoint(
    path: Path,
    model: FootballActorCritic,
    optimizer: torch.optim.Optimizer,
    update_index: int,
    metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "update_index": update_index,
            "metric": metric,
            "observation_size": OBSERVATION_SIZE,
            "action_size": ACTION_SIZE,
        },
        path,
    )


def load_checkpoint(path: str | Path | None = None) -> tuple[FootballActorCritic, dict[str, Any]]:
    checkpoint_path = Path(path) if path is not None else BEST_CHECKPOINT
    if not checkpoint_path.exists():
        checkpoint_path = LATEST_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No football checkpoint found at {BEST_CHECKPOINT} or {LATEST_CHECKPOINT}"
        )

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_observation_size = int(checkpoint.get("observation_size", 0))
    if checkpoint_observation_size != OBSERVATION_SIZE:
        raise ValueError(
            f"Checkpoint observation size {checkpoint_observation_size} does not match current size {OBSERVATION_SIZE}. Retrain from scratch."
        )
    model = FootballActorCritic()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def train_self_play_football(
    total_updates: int = PPO_TOTAL_UPDATES,
    rollout_steps: int = PPO_ROLLOUT_STEPS,
    seed: int | None = None,
    checkpoint_dir: str | Path = CHECKPOINT_DIR,
    save_checkpoints: bool = True,
    pretrain_steps: int = BEHAVIOR_PRETRAIN_STEPS,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    env = FootballEnvironment(seed=seed)
    model = FootballActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=PPO_LEARNING_RATE)
    best_metric = float("-inf")
    started_at = perf_counter()
    resolved_checkpoint_dir = Path(checkpoint_dir)
    latest_checkpoint = resolved_checkpoint_dir / "latest.pt"
    best_checkpoint = resolved_checkpoint_dir / "best.pt"

    print(
        f"Football PPO self-play | device {device} | obs {OBSERVATION_SIZE} | "
        f"rollout {rollout_steps} | updates {total_updates}",
        flush=True,
    )
    if pretrain_steps > 0:
        pretrain_started_at = perf_counter()
        pretrain_metrics = pretrain_behavior_prior(
            model=model,
            optimizer=optimizer,
            device=device,
            steps=pretrain_steps,
            seed=seed,
        )
        print(
            f"Behavior warm-start | loss {pretrain_metrics['behavior_loss']:.4f} | "
            f"kick acc {pretrain_metrics['kick_accuracy'] * 100:5.1f}% | "
            f"{perf_counter() - pretrain_started_at:5.2f}s",
            flush=True,
        )

    for update_index in range(1, total_updates + 1):
        rollout_start = perf_counter()
        batch, rollout_metrics = collect_rollout(env, model, device, rollout_steps)
        update_metrics = update_policy(model, optimizer, batch, device)
        kick_rate = rollout_metrics["kicks"] / max(1, rollout_steps)
        pass_attempts = rollout_metrics["passes"]
        pass_rate = (
            rollout_metrics["successful_passes"] / pass_attempts
            if pass_attempts > 0
            else 0.0
        )
        clump_rate = rollout_metrics["clump_steps"] / max(1, rollout_steps)
        model_score = (
            rollout_metrics["episode_return"]
            + rollout_metrics["goals"] * 35.0
            + rollout_metrics["passes"] * 0.6
            + rollout_metrics["successful_passes"] * 6.0
            + rollout_metrics["assisted_goals"] * 25.0
            + rollout_metrics["forward_progress"] * 2.0
            + kick_rate
            - clump_rate * 20.0
        )
        if save_checkpoints:
            save_checkpoint(latest_checkpoint, model, optimizer, update_index, model_score)
        if model_score > best_metric:
            best_metric = model_score
            if save_checkpoints:
                save_checkpoint(best_checkpoint, model, optimizer, update_index, best_metric)

        print(
            f"Update {update_index:03d} | score {model_score:8.3f} | "
            f"return {rollout_metrics['episode_return']:7.3f} | "
            f"goals {rollout_metrics['goals']:3.0f} | "
            f"prog {rollout_metrics['forward_progress']:6.3f} | "
            f"passA {pass_attempts:3.0f} | "
            f"succP {rollout_metrics['successful_passes']:3.0f} | "
            f"assistG {rollout_metrics['assisted_goals']:3.0f} | "
            f"pass% {pass_rate * 100:5.1f} | "
            f"clump% {clump_rate * 100:5.1f} | "
            f"mateD {rollout_metrics['mean_teammate_distance']:5.0f} | "
            f"staleT {rollout_metrics['mean_stale_touch_time']:4.1f} | "
            f"shotP {rollout_metrics['mean_shot_power']:4.2f} | "
            f"kicks {rollout_metrics['kicks']:4.0f} | "
            f"S/P/C {rollout_metrics['shots']:3.0f}/"
            f"{rollout_metrics['passes']:3.0f}/"
            f"{rollout_metrics['clears']:3.0f} | "
            f"r/step {rollout_metrics['mean_reward']:7.4f} | "
            f"pi {update_metrics['policy_loss']:7.3f} | "
            f"v {update_metrics['value_loss']:7.3f} | "
            f"entropy {update_metrics['entropy']:6.3f} | "
            f"{perf_counter() - rollout_start:5.2f}s",
            flush=True,
        )

    return {
        "latest_checkpoint": str(latest_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "best_metric": best_metric,
        "elapsed_seconds": perf_counter() - started_at,
        "pretrain_steps": pretrain_steps,
    }


def policy_actions_from_observations(
    model: FootballActorCritic,
    observations: list[list[float]],
    deterministic: bool = True,
) -> list[list[float]]:
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
    with torch.no_grad():
        raw_actions, kick_types, _log_probs, _entropy, _values = model.act(
            observation_tensor,
            deterministic=deterministic,
        )
    return raw_actions_to_env_actions(raw_actions, kick_types)


def run_trained_football_preview(model_path: str | Path | None = None) -> None:
    try:
        model, checkpoint = load_checkpoint(model_path)
        print(
            f"Loaded football checkpoint from update {checkpoint.get('update_index', '?')}",
            flush=True,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"{error}. Falling back to scripted preview.", flush=True)
        model = None

    env = FootballEnvironment()
    env.setup()
    while env.is_running:
        dt = env.clock.tick(FPS) / 1000.0
        env.handle_events()
        if env.announcement_timer > 0.0:
            actions: list[Any] = [[0.0, 0.0, KICK_NONE, 0.0, 1.0, 0.0] for _ in env.players]
        elif model is None:
            actions = env.build_scripted_actions()
        else:
            actions = policy_actions_from_observations(
                model,
                env.get_observations(),
                deterministic=True,
            )
        _obs, _rewards, done, _info = env.step(actions, dt, training=False)
        if done and env.announcement_timer <= 0.0:
            env.reset(keep_score=True)
        env.render()
    pygame.quit()


def surface_to_video_frame(surface: pygame.Surface) -> np.ndarray:
    rgb_frame = pygame.surfarray.array3d(surface).swapaxes(0, 1)
    return np.ascontiguousarray(rgb_frame[:, :, ::-1])


def build_football_replay_video(
    output_path: str | Path | None = None,
    model_path: str | Path | None = None,
    duration_seconds: float = 30.0,
) -> Path:
    import cv2

    try:
        model, _checkpoint = load_checkpoint(model_path)
    except (FileNotFoundError, ValueError):
        model = None

    if output_path is None:
        REPLAY_VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_output_path = REPLAY_VIDEO_OUTPUT_DIR / f"{REPLAY_VIDEO_PREFIX}_{timestamp}.mp4"
    else:
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    env = FootballEnvironment()
    pygame.font.init()
    fourcc = cv2.VideoWriter_fourcc(*REPLAY_VIDEO_CODEC)
    writer = cv2.VideoWriter(
        str(resolved_output_path),
        fourcc,
        FPS,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {resolved_output_path}")

    try:
        total_frames = int(duration_seconds * FPS)
        for _frame_index in range(total_frames):
            if env.announcement_timer > 0.0:
                actions: list[Any] = [[0.0, 0.0, KICK_NONE, 0.0, 1.0, 0.0] for _ in env.players]
            elif model is None:
                actions = env.build_scripted_actions()
            else:
                actions = policy_actions_from_observations(
                    model,
                    env.get_observations(),
                    deterministic=True,
                )
            _obs, _rewards, done, _info = env.step(
                actions,
                TRAINING_STEP_DT,
                training=False,
            )
            if done and env.announcement_timer <= 0.0:
                env.reset(keep_score=True)
            writer.write(surface_to_video_frame(env.render_frame()))
    finally:
        writer.release()
        pygame.quit()

    print(f"Saved football replay mp4 to {resolved_output_path}", flush=True)
    return resolved_output_path


def run_ppo_smoke_test() -> dict[str, Any]:
    return train_self_play_football(
        total_updates=1,
        rollout_steps=32,
        seed=123,
        checkpoint_dir=CHECKPOINT_DIR,
        save_checkpoints=False,
        pretrain_steps=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or preview AI football PPO.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--updates", type=int, default=PPO_TOTAL_UPDATES)
    train_parser.add_argument("--rollout-steps", type=int, default=PPO_ROLLOUT_STEPS)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument("--pretrain-steps", type=int, default=BEHAVIOR_PRETRAIN_STEPS)
    train_parser.add_argument("--no-pretrain", action="store_true")

    preview_parser = subparsers.add_parser("preview")
    preview_parser.add_argument("--model-path", default=None)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--model-path", default=None)
    export_parser.add_argument("--output", default=None)
    export_parser.add_argument("--duration", type=float, default=30.0)

    subparsers.add_parser("smoke")
    args = parser.parse_args()

    if args.command == "preview":
        run_trained_football_preview(model_path=args.model_path)
    elif args.command == "export":
        build_football_replay_video(
            output_path=args.output,
            model_path=args.model_path,
            duration_seconds=args.duration,
        )
    elif args.command == "smoke":
        print(run_ppo_smoke_test())
    else:
        print(
            train_self_play_football(
                total_updates=getattr(args, "updates", PPO_TOTAL_UPDATES),
                rollout_steps=getattr(args, "rollout_steps", PPO_ROLLOUT_STEPS),
                seed=getattr(args, "seed", None),
                pretrain_steps=0
                if getattr(args, "no_pretrain", False)
                else getattr(args, "pretrain_steps", BEHAVIOR_PRETRAIN_STEPS),
            )
        )


if __name__ == "__main__":
    main()
