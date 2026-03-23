from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pygame

from src.builders.ai_car_driving_builder.video_generator import (
    FPS,
    LASER_ANGLE_OFFSETS,
    ROAD_STRIPE_WIDTH,
    ROAD_WIDTH,
    CarDrivingSimulation,
)


SUCCESS_TIME_SECONDS = 10.0
TRAINING_STEP_DT = 1.0 / FPS
TRAINING_EPISODE_TIME_LIMIT = 16.0
TRAINING_POPULATION_SIZE = 32
TRAINING_ELITE_COUNT = 5
TRAINING_MAX_GENERATIONS = 500
TRAINING_MIN_GENERATIONS = 5
TRAINING_STAGNATION_SECONDS = 2.0
TRAINING_PROGRESS_EPSILON = 12.0
TRAINING_INITIAL_STD = 0.42
TRAINING_MIN_STD = 0.05
TRAINING_MEAN_BLEND = 0.15
TRAINING_STD_BLEND = 0.35
TRAINING_PROGRESS_REWARD = 2.6
TRAINING_SPEED_REWARD = 0.6
TRAINING_HEADING_PENALTY = 0.18
TRAINING_LATERAL_PENALTY = 0.12
TRAINING_STEP_PENALTY = 0.02
TRAINING_COLLISION_PENALTY = 55.0
TRAINING_CRASH_PENALTY = 135.0
TRAINING_STAGNATION_PENALTY = 24.0
TRAINING_TIMEOUT_PENALTY = 18.0
TRAINING_FINISH_BONUS = 320.0
TRAINING_FAST_FINISH_BONUS = 120.0
REPLAY_TIME_LIMIT = 20.0
REPLAY_PAUSE_SECONDS = 0.4
REPLAY_INTRO_SECONDS = 1.4
REPLAY_RUN_COUNT = 5
REPLAY_SHOW_LASERS = False
MODEL_HIDDEN_SIZES: tuple[int, ...] = ()

OBS_SPEED = 0
OBS_HEADING_ERROR = 1
OBS_LATERAL_OFFSET = 2
OBS_PROGRESS = 3
OBS_LASER_START = 4
LEFT_FAR_ANGLES = (-120.0, -95.0)
LEFT_MID_ANGLES = (-75.0, -55.0)
LEFT_NEAR_ANGLES = (-35.0, -15.0)
RIGHT_NEAR_ANGLES = (15.0, 35.0)
RIGHT_MID_ANGLES = (55.0, 75.0)
RIGHT_FAR_ANGLES = (95.0, 120.0)
FRONT_ANGLES = (-15.0, 0.0, 15.0)
BACK_ANGLE = 180.0
ACTION_THROTTLE = 0
ACTION_STEER = 1
ACTION_DRIFT = 2


def laser_obs_index(angle: float) -> int:
    return OBS_LASER_START + LASER_ANGLE_OFFSETS.index(angle)


@dataclass
class PolicyRollout:
    generation: int
    fitness: float
    finished: bool
    elapsed_time: float
    progress_ratio: float
    track_distance: float
    vector: np.ndarray


@dataclass
class CompiledPolicy:
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def act(self, observation: np.ndarray) -> np.ndarray:
        activations = observation
        last_layer_index = len(self.weights) - 1
        for layer_index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            activations = activations @ weights + biases
            activations = np.tanh(activations)
            if layer_index == last_layer_index:
                return activations.astype(np.float32)
        return activations.astype(np.float32)


class PolicyModel:
    def __init__(self, input_size: int) -> None:
        self.layer_sizes = (input_size, *MODEL_HIDDEN_SIZES, 3)
        self.weight_slices: list[slice] = []
        self.bias_slices: list[slice] = []
        self.weight_shapes: list[tuple[int, int]] = []
        self.total_parameter_count = 0

        offset = 0
        for layer_index in range(len(self.layer_sizes) - 1):
            input_width = self.layer_sizes[layer_index]
            output_width = self.layer_sizes[layer_index + 1]
            weight_count = input_width * output_width
            self.weight_slices.append(slice(offset, offset + weight_count))
            self.weight_shapes.append((input_width, output_width))
            offset += weight_count
            self.bias_slices.append(slice(offset, offset + output_width))
            offset += output_width

        self.total_parameter_count = offset

    def compile(self, vector: np.ndarray) -> CompiledPolicy:
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for weight_slice, bias_slice, weight_shape in zip(
            self.weight_slices,
            self.bias_slices,
            self.weight_shapes,
        ):
            weights.append(vector[weight_slice].reshape(weight_shape))
            biases.append(vector[bias_slice])
        return CompiledPolicy(weights=weights, biases=biases)

    def build_initial_mean(self) -> np.ndarray:
        mean = np.zeros(self.total_parameter_count, dtype=np.float32)
        output_biases = mean[self.bias_slices[-1]]
        output_biases[ACTION_THROTTLE] = 0.34
        output_biases[ACTION_STEER] = 0.0
        output_biases[ACTION_DRIFT] = -0.95

        if MODEL_HIDDEN_SIZES:
            return mean

        output_weights = mean[self.weight_slices[0]].reshape(self.weight_shapes[0])
        output_weights[OBS_SPEED, ACTION_THROTTLE] = -0.14
        output_weights[OBS_HEADING_ERROR, ACTION_THROTTLE] = -0.08
        for angle in FRONT_ANGLES:
            output_weights[laser_obs_index(angle), ACTION_THROTTLE] = 0.12
        for angle in LEFT_NEAR_ANGLES + RIGHT_NEAR_ANGLES:
            output_weights[laser_obs_index(angle), ACTION_THROTTLE] = 0.05

        output_weights[OBS_HEADING_ERROR, ACTION_STEER] = -1.50
        output_weights[OBS_LATERAL_OFFSET, ACTION_STEER] = -0.85
        for angle in LEFT_FAR_ANGLES:
            output_weights[laser_obs_index(angle), ACTION_STEER] = -0.24
        for angle in LEFT_MID_ANGLES:
            output_weights[laser_obs_index(angle), ACTION_STEER] = -0.48
        for angle in LEFT_NEAR_ANGLES:
            output_weights[laser_obs_index(angle), ACTION_STEER] = -0.86
        for angle in RIGHT_NEAR_ANGLES:
            output_weights[laser_obs_index(angle), ACTION_STEER] = 0.86
        for angle in RIGHT_MID_ANGLES:
            output_weights[laser_obs_index(angle), ACTION_STEER] = 0.48
        for angle in RIGHT_FAR_ANGLES:
            output_weights[laser_obs_index(angle), ACTION_STEER] = 0.24
        output_weights[laser_obs_index(BACK_ANGLE), ACTION_THROTTLE] = -0.12

        output_weights[OBS_SPEED, ACTION_DRIFT] = 0.18
        output_weights[OBS_HEADING_ERROR, ACTION_DRIFT] = 0.45
        return mean


class ProceduralTrackTrainer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()
        self.simulation = CarDrivingSimulation(enable_obstacles=True)
        initial_observation = np.asarray(self.simulation.reset_episode(), dtype=np.float32)
        self.model = PolicyModel(input_size=int(initial_observation.shape[0]))
        self.mean = self.model.build_initial_mean()
        self.std = np.full(self.model.total_parameter_count, TRAINING_INITIAL_STD, dtype=np.float32)
        self.best_rollout: PolicyRollout | None = None
        self.generation_history: list[PolicyRollout] = []
        self.drivable_half_width = max(1.0, ROAD_WIDTH / 2 - ROAD_STRIPE_WIDTH - 4)

    def evaluate_vector(self, vector: np.ndarray) -> PolicyRollout:
        policy = self.model.compile(vector)
        observation = np.asarray(self.simulation.reset_episode(), dtype=np.float32)
        previous_progress = float(self.simulation.car_state["track_distance"])
        best_progress = previous_progress
        stagnation_time = 0.0
        fitness = 0.0

        while float(self.simulation.car_state["elapsed_time"]) < TRAINING_EPISODE_TIME_LIMIT:
            action = policy.act(observation)
            step_state = self.simulation.apply_control_signals(
                TRAINING_STEP_DT,
                float(action[ACTION_THROTTLE]),
                float(action[ACTION_STEER]),
                float(action[ACTION_DRIFT]),
            )
            observation = np.asarray(self.simulation.get_observation(), dtype=np.float32)

            current_progress = float(step_state["track_distance"])
            progress_delta = current_progress - previous_progress
            previous_progress = current_progress

            heading_error = abs(float(step_state["heading_error"])) / 180.0
            lateral_error = min(1.0, abs(float(step_state["lateral_offset"])) / self.drivable_half_width)
            speed_value = max(0.0, float(step_state["speed"]))

            fitness += progress_delta * TRAINING_PROGRESS_REWARD
            fitness += speed_value * TRAINING_SPEED_REWARD * TRAINING_STEP_DT
            fitness -= heading_error * TRAINING_HEADING_PENALTY
            fitness -= lateral_error * TRAINING_LATERAL_PENALTY
            fitness -= TRAINING_STEP_PENALTY

            if current_progress > best_progress + TRAINING_PROGRESS_EPSILON:
                best_progress = current_progress
                stagnation_time = 0.0
            else:
                stagnation_time += TRAINING_STEP_DT

            if bool(step_state["collided"]):
                fitness -= TRAINING_COLLISION_PENALTY
                break

            if bool(step_state["crashed"]):
                fitness -= TRAINING_CRASH_PENALTY
                break

            if bool(step_state["finished"]):
                elapsed_time = float(step_state["elapsed_time"])
                fitness += TRAINING_FINISH_BONUS
                fitness += max(0.0, SUCCESS_TIME_SECONDS - elapsed_time) * TRAINING_FAST_FINISH_BONUS
                break

            if stagnation_time >= TRAINING_STAGNATION_SECONDS:
                fitness -= TRAINING_STAGNATION_PENALTY
                break

        elapsed_time = float(self.simulation.car_state["elapsed_time"])
        if not bool(self.simulation.car_state["finished"]) and elapsed_time >= TRAINING_EPISODE_TIME_LIMIT:
            fitness -= TRAINING_TIMEOUT_PENALTY

        return PolicyRollout(
            generation=0,
            fitness=fitness,
            finished=bool(self.simulation.car_state["finished"]),
            elapsed_time=elapsed_time,
            progress_ratio=float(self.simulation.car_state["progress_ratio"]),
            track_distance=float(self.simulation.car_state["track_distance"]),
            vector=vector.copy(),
        )

    def sample_population(self) -> list[np.ndarray]:
        population: list[np.ndarray] = [self.mean.copy()]
        if self.best_rollout is not None:
            population.append(self.best_rollout.vector.copy())

        while len(population) < TRAINING_POPULATION_SIZE:
            noise = self.rng.normal(0.0, self.std, size=self.model.total_parameter_count).astype(np.float32)
            population.append((self.mean + noise).astype(np.float32))

        return population[:TRAINING_POPULATION_SIZE]

    def update_distribution(self, elites: list[PolicyRollout]) -> None:
        elite_vectors = np.stack([elite.vector for elite in elites], axis=0)
        elite_mean = elite_vectors.mean(axis=0).astype(np.float32)
        elite_std = elite_vectors.std(axis=0).astype(np.float32)

        self.mean = (
            self.mean * TRAINING_MEAN_BLEND
            + elite_mean * (1.0 - TRAINING_MEAN_BLEND)
        ).astype(np.float32)
        blended_std = self.std * TRAINING_STD_BLEND + elite_std * (1.0 - TRAINING_STD_BLEND)
        self.std = np.maximum(TRAINING_MIN_STD, blended_std).astype(np.float32)

    def train(self) -> tuple[list[PolicyRollout], int]:
        print(
            f"Training on CPU | track length {self.simulation.track_total_length:.0f} px | "
            f"segments {len(self.simulation.road_segments)} | "
            f"obstacles {len(self.simulation.track_obstacles)}",
            flush=True,
        )

        first_success_generation = -1

        for generation in range(1, TRAINING_MAX_GENERATIONS + 1):
            generation_start = perf_counter()
            population = self.sample_population()
            rollouts = [self.evaluate_vector(candidate) for candidate in population]
            rollouts.sort(key=lambda rollout: rollout.fitness, reverse=True)

            generation_best = rollouts[0]
            generation_best.generation = generation
            self.generation_history.append(generation_best)

            if self.best_rollout is None or generation_best.fitness > self.best_rollout.fitness:
                self.best_rollout = generation_best

            elites = rollouts[:TRAINING_ELITE_COUNT]
            self.update_distribution(elites)

            generation_duration = perf_counter() - generation_start
            hit_target = generation_best.finished and generation_best.elapsed_time <= SUCCESS_TIME_SECONDS
            if hit_target and generation < TRAINING_MIN_GENERATIONS:
                status = "target met, collecting milestones"
            elif hit_target:
                status = "success"
            else:
                status = "searching"
            print(
                f"Generation {generation:03d} | best {generation_best.fitness:8.2f} | "
                f"progress {generation_best.progress_ratio * 100:6.2f}% | "
                f"time {generation_best.elapsed_time:5.2f}s | {status} | "
                f"{generation_duration:4.2f}s/gen",
                flush=True,
            )

            if hit_target and generation >= TRAINING_MIN_GENERATIONS:
                first_success_generation = generation
                break

        if first_success_generation == -1:
            raise RuntimeError(
                "Training stopped without reaching the under-10-second finish target. "
                "Increase the training globals in rl_trainer.py and run again."
            )

        return self.generation_history, first_success_generation

    def select_replay_generations(
        self,
        history: list[PolicyRollout],
        success_generation: int,
    ) -> list[int]:
        failed_rollouts = [
            rollout
            for rollout in history[: success_generation - 1]
            if not (rollout.finished and rollout.elapsed_time <= SUCCESS_TIME_SECONDS)
        ]
        target_failed_count = max(0, REPLAY_RUN_COUNT - 1)
        improving_failed_generations = self.select_improving_failed_generations(failed_rollouts)

        if len(improving_failed_generations) >= target_failed_count:
            selected_failed = self.select_spaced_generations(improving_failed_generations, target_failed_count)
        else:
            quality_sorted_failed = [
                rollout.generation
                for rollout in sorted(failed_rollouts, key=self.get_replay_quality)
            ]
            if len(quality_sorted_failed) >= target_failed_count:
                selected_failed = self.select_spaced_generations(quality_sorted_failed, target_failed_count)
            else:
                selected_failed = quality_sorted_failed

        return [*selected_failed, success_generation]

    def select_improving_failed_generations(self, failed_rollouts: list[PolicyRollout]) -> list[int]:
        improving_generations: list[int] = []
        best_quality: tuple[float, float, float, float] | None = None

        for rollout in failed_rollouts:
            quality = self.get_replay_quality(rollout)
            if best_quality is None or quality > best_quality:
                improving_generations.append(rollout.generation)
                best_quality = quality

        return improving_generations

    def get_replay_quality(self, rollout: PolicyRollout) -> tuple[float, float, float, float]:
        return (
            round(rollout.progress_ratio, 6),
            round(rollout.track_distance, 3),
            round(rollout.fitness, 3),
            -round(rollout.elapsed_time, 3),
        )

    def select_spaced_generations(self, generations: list[int], target_count: int) -> list[int]:
        if target_count <= 0 or not generations:
            return []
        if len(generations) <= target_count:
            return generations[:]

        selected: list[int] = []
        for generation_value in np.linspace(0, len(generations) - 1, num=target_count):
            generation = generations[int(round(float(generation_value)))]
            if generation not in selected:
                selected.append(generation)

        candidate_index = 0
        while len(selected) < target_count and candidate_index < len(generations):
            generation = generations[candidate_index]
            if generation not in selected:
                selected.append(generation)
            candidate_index += 1

        return selected[:target_count]

    def replay_single_generation(self, generation_rollout: PolicyRollout, run_number: int, run_total: int) -> bool:
        policy = self.model.compile(generation_rollout.vector)
        self.simulation.reset_episode()
        self.simulation.show_lasers = REPLAY_SHOW_LASERS
        self.simulation.hud_footer_text = (
            f"Replay {run_number}/{run_total} | best {generation_rollout.elapsed_time:.2f}s"
        )
        self.simulation.hud_title_text = (
            f"Generation {generation_rollout.generation} starting"
        )
        if not self.show_replay_card(REPLAY_INTRO_SECONDS):
            return False

        self.simulation.hud_title_text = (
            f"Replay {run_number}/{run_total} | Generation {generation_rollout.generation}"
        )
        self.simulation.hud_footer_text = (
            f"Policy run | gen best {generation_rollout.elapsed_time:.2f}s"
        )
        finished_run = False
        hold_time = 0.0

        while self.simulation.is_running:
            dt = self.simulation.clock.tick(FPS) / 1000.0
            self.simulation.handle_events()
            if not self.simulation.is_running:
                return False

            if not finished_run:
                observation = np.asarray(self.simulation.get_observation(), dtype=np.float32)
                action = policy.act(observation)
                step_state = self.simulation.apply_control_signals(
                    dt,
                    float(action[ACTION_THROTTLE]),
                    float(action[ACTION_STEER]),
                    float(action[ACTION_DRIFT]),
                )
                if (
                    bool(step_state["finished"])
                    or bool(step_state["collided"])
                    or bool(step_state["crashed"])
                    or float(step_state["elapsed_time"]) >= REPLAY_TIME_LIMIT
                ):
                    finished_run = True
                    if bool(step_state["finished"]):
                        result_label = "Success"
                    elif bool(step_state["crashed"]):
                        result_label = "Crashed"
                    else:
                        result_label = "Attempt"
                    self.simulation.hud_title_text = (
                        f"Replay {run_number}/{run_total} | "
                        f"Generation {generation_rollout.generation} | "
                        f"{result_label}"
                    )
                    self.simulation.hud_footer_text = (
                        f"time {float(step_state['elapsed_time']):.2f}s | "
                        f"progress {float(step_state['progress_ratio']) * 100:.1f}%"
                    )
            else:
                hold_time += dt
                if hold_time >= REPLAY_PAUSE_SECONDS:
                    break

            self.simulation.render()

        return self.simulation.is_running

    def show_replay_card(self, duration_seconds: float) -> bool:
        elapsed = 0.0
        while self.simulation.is_running and elapsed < duration_seconds:
            dt = self.simulation.clock.tick(FPS) / 1000.0
            self.simulation.handle_events()
            if not self.simulation.is_running:
                return False
            self.simulation.render()
            elapsed += dt
        return self.simulation.is_running

    def replay_milestones(self, history: list[PolicyRollout], success_generation: int) -> None:
        milestone_generations = self.select_replay_generations(history, success_generation)
        milestones = [history[generation - 1] for generation in milestone_generations]
        print(
            "Replaying generations: "
            + ", ".join(str(generation) for generation in milestone_generations),
            flush=True,
        )

        self.simulation.setup()
        self.simulation.show_lasers = REPLAY_SHOW_LASERS
        for run_number, milestone in enumerate(milestones, start=1):
            if not self.replay_single_generation(milestone, run_number, len(milestones)):
                break
        pygame.quit()


def run_training_and_replay() -> None:
    trainer = ProceduralTrackTrainer()
    history, success_generation = trainer.train()
    print(
        f"Training finished at generation {success_generation} with "
        f"{history[success_generation - 1].elapsed_time:.2f}s run time.",
        flush=True,
    )
    trainer.replay_milestones(history, success_generation)


if __name__ == "__main__":
    run_training_and_replay()
