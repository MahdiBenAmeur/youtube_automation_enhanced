from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import cv2
import numpy as np
import pygame

from src.builders.ai_car_driving_builder.video_generator import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    FPS,
    LASER_ANGLE_OFFSETS,
    ROAD_STRIPE_WIDTH,
    ROAD_WIDTH,
    CarDrivingSimulation,
)


SUCCESS_TIME_SECONDS = 10.0
TRAINING_STEP_DT = 1.0 / FPS
TRAINING_EPISODE_TIME_LIMIT = 16.0
TRAINING_POPULATION_SIZE = 64
TRAINING_ELITE_COUNT = 5
TRAINING_MAX_GENERATIONS = 60
TRAINING_MIN_GENERATIONS = 5
TRAINING_STAGNATION_SECONDS = 2.0
TRAINING_PROGRESS_EPSILON = 12.0
TRAINING_INITIAL_STD = 0.32
TRAINING_MIN_STD = 0.03
TRAINING_MEAN_BLEND = 0.12
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
REPLAY_TIME_LIMIT = 14.0
REPLAY_PAUSE_SECONDS = 0.2
REPLAY_INTRO_SECONDS = 0.5
REPLAY_RUN_COUNT = 5
REPLAY_SHOW_LASERS = False
REPLAY_EXPORT_VIDEO = True
REPLAY_SHOW_WINDOW = False
REPLAY_VIDEO_OUTPUT_DIR = "generated_videos"
REPLAY_VIDEO_PREFIX = "ai_car_driving_neural_replay"
REPLAY_VIDEO_CODEC = "mp4v"
REPLAY_ENGINE_AUDIO_FILENAMES = (
    "caracceleration1.mp3",
    "caracceleration2.mp3",
    "caracceleration3.mp3",
)
REPLAY_TRANSITION_AUDIO_FILENAME = "transition.mp3"
REPLAY_TRANSITION_AUDIO_SAMPLE_RATE = 44100
REPLAY_ENGINE_AUDIO_VOLUME = 0.8
REPLAY_TRANSITION_AUDIO_VOLUME = 1.2
REPLAY_ENGINE_AUDIO_MIN_SPEED = 8.0
REPLAY_AUDIO_FADE_IN_SECONDS = 0.04
REPLAY_AUDIO_FADE_OUT_SECONDS = 0.10
REPLAY_AUDIO_SILENCE_TRIM_SECONDS = 0.02
REPLAY_AUDIO_SILENCE_THRESHOLD_DB = -45
REPLAY_TRANSITION_DARKEN_STRENGTH = 0.28
REPLAY_TRANSITION_MIN_SCALE = 0.08
MODEL_HIDDEN_SIZES: tuple[int, ...] = (48, 24)

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
class AudioOverlay:
    path: Path
    start_frame: int
    duration_frames: int
    volume: float


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
        for weight_slice, weight_shape in zip(self.weight_slices, self.weight_shapes):
            fan_in = max(1, weight_shape[0])
            scale = 1.0 / np.sqrt(fan_in)
            mean[weight_slice] = np.random.normal(0.0, scale, weight_slice.stop - weight_slice.start).astype(np.float32)

        output_biases = mean[self.bias_slices[-1]]
        output_biases[ACTION_THROTTLE] = 0.18
        output_biases[ACTION_STEER] = 0.0
        output_biases[ACTION_DRIFT] = -1.0
        return mean


class NeuralTrackTrainer:
    def __init__(self) -> None:
        self.rng = np.random.default_rng()
        self.simulation = CarDrivingSimulation(enable_obstacles=True)
        initial_observation = np.asarray(self.simulation.reset_episode(), dtype=np.float32)
        self.base_observation_size = int(initial_observation.shape[0])
        self.model = PolicyModel(input_size=self.base_observation_size * 3)
        self.mean = self.model.build_initial_mean()
        self.std = np.full(self.model.total_parameter_count, TRAINING_INITIAL_STD, dtype=np.float32)
        self.best_rollout: PolicyRollout | None = None
        self.generation_history: list[PolicyRollout] = []
        self.drivable_half_width = max(1.0, ROAD_WIDTH / 2 - ROAD_STRIPE_WIDTH - 4)
        self._replay_observation_history: list[np.ndarray] = []

    def build_policy_observation(
        self,
        current_observation: np.ndarray,
        previous_observations: list[np.ndarray] | None,
    ) -> np.ndarray:
        history = [] if previous_observations is None else previous_observations[-2:]
        padded_history = [np.zeros_like(current_observation) for _ in range(2 - len(history))] + history
        return np.concatenate((*padded_history, current_observation)).astype(np.float32)

    def evaluate_vector(self, vector: np.ndarray) -> PolicyRollout:
        policy = self.model.compile(vector)
        current_observation = np.asarray(self.simulation.reset_episode(), dtype=np.float32)
        previous_observations: list[np.ndarray] = []
        previous_progress = float(self.simulation.car_state["track_distance"])
        best_progress = previous_progress
        stagnation_time = 0.0
        fitness = 0.0

        while float(self.simulation.car_state["elapsed_time"]) < TRAINING_EPISODE_TIME_LIMIT:
            action = policy.act(
                self.build_policy_observation(current_observation, previous_observations)
            )
            step_state = self.simulation.apply_control_signals(
                TRAINING_STEP_DT,
                float(action[ACTION_THROTTLE]),
                float(action[ACTION_STEER]),
                float(action[ACTION_DRIFT]),
            )
            previous_observations = [*previous_observations, current_observation][-2:]
            current_observation = np.asarray(self.simulation.get_observation(), dtype=np.float32)

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
            f"Neural-net training on CPU | track length {self.simulation.track_total_length:.0f} px | "
            f"segments {len(self.simulation.road_segments)} | "
            f"obstacles {len(self.simulation.track_obstacles)} | "
            f"layers {self.model.layer_sizes}",
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
                "Neural-net training stopped without reaching the under-10-second finish target. "
                "Increase the neural trainer globals in rl_trainer_neural.py and run again."
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

    def build_output_video_path(self) -> Path:
        output_dir = Path(REPLAY_VIDEO_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return output_dir / f"{REPLAY_VIDEO_PREFIX}_{timestamp}.mp4"

    def get_transition_audio_path(self) -> Path:
        return Path(__file__).resolve().parent / "assets" / REPLAY_TRANSITION_AUDIO_FILENAME

    def get_engine_audio_paths(self) -> list[Path]:
        assets_dir = Path(__file__).resolve().parent / "assets"
        return [
            assets_dir / filename
            for filename in REPLAY_ENGINE_AUDIO_FILENAMES
            if (assets_dir / filename).exists()
        ]

    def get_media_duration_seconds(self, media_path: Path) -> float | None:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(media_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            return None
        try:
            return float(result.stdout.strip())
        except ValueError:
            return None

    def surface_to_video_frame(self, surface: pygame.Surface) -> np.ndarray:
        rgb_frame = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        return np.ascontiguousarray(rgb_frame[:, :, ::-1])

    def write_video_frame(self, writer: cv2.VideoWriter) -> int:
        frame_surface = self.simulation.render_frame()
        writer.write(self.surface_to_video_frame(frame_surface))
        return 1

    def write_video_hold(self, writer: cv2.VideoWriter, duration_seconds: float) -> int:
        frame_count = max(1, int(round(duration_seconds * FPS)))
        for _ in range(frame_count):
            self.write_video_frame(writer)
        return frame_count

    def draw_transition_text(
        self,
        frame: np.ndarray,
        title_text: str,
        footer_text: str,
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        title_font = cv2.FONT_HERSHEY_DUPLEX
        footer_font = cv2.FONT_HERSHEY_SIMPLEX

        title_scale = 1.05
        footer_scale = 0.9
        title_thickness = 2
        footer_thickness = 2
        text_color = (245, 245, 245)
        accent_color = (80, 220, 255)
        shadow_color = (10, 10, 10)

        title_size = cv2.getTextSize(title_text, title_font, title_scale, title_thickness)[0]
        footer_size = cv2.getTextSize(footer_text, footer_font, footer_scale, footer_thickness)[0]
        title_x = (CANVAS_WIDTH - title_size[0]) // 2
        title_y = CANVAS_HEIGHT // 2 - 24
        footer_x = (CANVAS_WIDTH - footer_size[0]) // 2
        footer_y = title_y + 64

        cv2.putText(
            annotated_frame,
            title_text,
            (title_x + 3, title_y + 3),
            title_font,
            title_scale,
            shadow_color,
            title_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            title_text,
            (title_x, title_y),
            title_font,
            title_scale,
            text_color,
            title_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            footer_text,
            (footer_x + 2, footer_y + 2),
            footer_font,
            footer_scale,
            shadow_color,
            footer_thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_frame,
            footer_text,
            (footer_x, footer_y),
            footer_font,
            footer_scale,
            accent_color,
            footer_thickness,
            cv2.LINE_AA,
        )
        return annotated_frame

    def build_transition_frame(
        self,
        base_frame: np.ndarray,
        progress_ratio: float,
        title_text: str,
        footer_text: str,
    ) -> np.ndarray:
        scale = max(
            REPLAY_TRANSITION_MIN_SCALE,
            1.0 - 0.88 * max(0.0, min(1.0, progress_ratio)),
        )
        scaled_width = max(1, int(round(CANVAS_WIDTH * scale)))
        scaled_height = max(1, int(round(CANVAS_HEIGHT * scale)))
        blurred_frame = cv2.resize(
            cv2.resize(base_frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR),
            (CANVAS_WIDTH, CANVAS_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
        darkened_frame = cv2.addWeighted(
            blurred_frame,
            1.0 - REPLAY_TRANSITION_DARKEN_STRENGTH * progress_ratio,
            np.zeros_like(blurred_frame),
            REPLAY_TRANSITION_DARKEN_STRENGTH * progress_ratio,
            0.0,
        )
        return self.draw_transition_text(darkened_frame, title_text, footer_text)

    def write_blurred_transition(
        self,
        writer: cv2.VideoWriter,
        duration_seconds: float,
        title_text: str,
        footer_text: str,
    ) -> int:
        frame_count = max(1, int(round(duration_seconds * FPS)))
        base_frame = self.surface_to_video_frame(self.simulation.render_frame())
        for frame_index in range(frame_count):
            progress_ratio = (frame_index + 1) / frame_count
            writer.write(
                self.build_transition_frame(
                    base_frame,
                    progress_ratio,
                    title_text,
                    footer_text,
                )
            )
        return frame_count

    def mux_replay_audio(
        self,
        raw_video_path: Path,
        output_path: Path,
        audio_overlays: list[AudioOverlay],
        total_frames: int,
    ) -> None:
        total_duration_seconds = max(1.0 / FPS, total_frames / FPS)

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_video_path),
            "-f",
            "lavfi",
            "-t",
            f"{total_duration_seconds:.3f}",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={REPLAY_TRANSITION_AUDIO_SAMPLE_RATE}",
        ]
        for overlay in audio_overlays:
            command.extend(["-i", str(overlay.path)])

        filter_parts = [
            f"[1:a]atrim=0:{total_duration_seconds:.3f},asetpts=N/SR/TB[a0]"
        ]
        mix_inputs = ["[a0]"]
        for input_index, overlay in enumerate(audio_overlays, start=2):
            label = f"[a{input_index - 1}]"
            delay_ms = int(round((overlay.start_frame / FPS) * 1000.0))
            duration_seconds = max(1.0 / FPS, overlay.duration_frames / FPS)
            fade_in_seconds = min(REPLAY_AUDIO_FADE_IN_SECONDS, duration_seconds / 2)
            fade_out_seconds = min(REPLAY_AUDIO_FADE_OUT_SECONDS, duration_seconds / 2)
            fade_out_start = max(0.0, duration_seconds - fade_out_seconds)
            filter_parts.append(
                f"[{input_index}:a]"
                f"silenceremove=start_periods=1:start_silence={REPLAY_AUDIO_SILENCE_TRIM_SECONDS:.3f}:"
                f"start_threshold={REPLAY_AUDIO_SILENCE_THRESHOLD_DB}dB,"
                f"atrim=0:{duration_seconds:.3f},"
                f"volume={overlay.volume:.3f},"
                f"afade=t=in:st=0:d={fade_in_seconds:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_seconds:.3f},"
                f"asetpts=N/SR/TB,adelay={delay_ms}|{delay_ms}{label}"
            )
            mix_inputs.append(label)
        filter_parts.append(
            "".join(mix_inputs)
            + f"amix=inputs={len(mix_inputs)}:duration=longest:dropout_transition=0,"
            + f"atrim=0:{total_duration_seconds:.3f},asetpts=N/SR/TB[aout]"
        )
        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])

    def export_single_generation_video(
        self,
        writer: cv2.VideoWriter,
        generation_rollout: PolicyRollout,
        run_number: int,
        run_total: int,
        engine_audio_path: Path | None,
        include_transition_after: bool,
        next_generation: int | None,
        transition_duration_seconds: float,
        transition_audio_path: Path | None,
    ) -> tuple[int, list[AudioOverlay]]:
        policy = self.model.compile(generation_rollout.vector)
        self.simulation.reset_episode()
        self._replay_observation_history = []
        self.simulation.show_lasers = REPLAY_SHOW_LASERS
        self.simulation.hud_footer_text = (
            f"Replay {run_number}/{run_total} | best {generation_rollout.elapsed_time:.2f}s"
        )
        self.simulation.hud_title_text = f"Generation {generation_rollout.generation} starting"
        frames_written = self.write_video_hold(writer, REPLAY_INTRO_SECONDS)

        self.simulation.hud_title_text = (
            f"Replay {run_number}/{run_total} | Generation {generation_rollout.generation}"
        )
        self.simulation.hud_footer_text = (
            f"Neural policy | gen best {generation_rollout.elapsed_time:.2f}s"
        )
        audio_overlays: list[AudioOverlay] = []
        engine_start_frame: int | None = None
        engine_end_frame: int | None = None

        while True:
            current_observation = np.asarray(self.simulation.get_observation(), dtype=np.float32)
            action = policy.act(
                self.build_policy_observation(
                    current_observation,
                    self._replay_observation_history,
                )
            )
            step_state = self.simulation.apply_control_signals(
                TRAINING_STEP_DT,
                float(action[ACTION_THROTTLE]),
                float(action[ACTION_STEER]),
                float(action[ACTION_DRIFT]),
            )
            self._replay_observation_history = [
                *self._replay_observation_history,
                current_observation,
            ][-2:]
            if abs(float(step_state["speed"])) >= REPLAY_ENGINE_AUDIO_MIN_SPEED:
                if engine_start_frame is None:
                    engine_start_frame = frames_written
                engine_end_frame = frames_written + 1

            if (
                bool(step_state["finished"])
                or bool(step_state["collided"])
                or bool(step_state["crashed"])
                or float(step_state["elapsed_time"]) >= REPLAY_TIME_LIMIT
            ):
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
                frames_written += self.write_video_frame(writer)
                if (
                    engine_audio_path is not None
                    and engine_start_frame is not None
                    and engine_end_frame is not None
                    and engine_end_frame > engine_start_frame
                ):
                    audio_overlays.append(
                        AudioOverlay(
                            path=engine_audio_path,
                            start_frame=engine_start_frame,
                            duration_frames=engine_end_frame - engine_start_frame,
                            volume=REPLAY_ENGINE_AUDIO_VOLUME,
                        )
                    )
                if include_transition_after and next_generation is not None:
                    transition_start_frame = frames_written
                    transition_frame_count = self.write_blurred_transition(
                        writer,
                        transition_duration_seconds,
                        f"Transitioning to Generation {next_generation}",
                        f"Replay {run_number + 1}/{run_total} incoming",
                    )
                    frames_written += transition_frame_count
                    if transition_audio_path is not None:
                        audio_overlays.append(
                            AudioOverlay(
                                path=transition_audio_path,
                                start_frame=transition_start_frame,
                                duration_frames=transition_frame_count,
                                volume=REPLAY_TRANSITION_AUDIO_VOLUME,
                            )
                        )
                else:
                    frames_written += self.write_video_hold(writer, REPLAY_PAUSE_SECONDS)
                return frames_written, audio_overlays

            frames_written += self.write_video_frame(writer)

        return frames_written, audio_overlays

    def export_milestones_video(
        self,
        history: list[PolicyRollout],
        success_generation: int,
        output_path: str | Path | None = None,
    ) -> Path:
        milestone_generations = self.select_replay_generations(history, success_generation)
        milestones = [history[generation - 1] for generation in milestone_generations]
        if output_path is None:
            resolved_output_path = self.build_output_video_path()
        else:
            resolved_output_path = Path(output_path)
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            "Exporting generations to mp4: "
            + ", ".join(str(generation) for generation in milestone_generations),
            flush=True,
        )

        self.simulation.setup_headless()
        self.simulation.show_lasers = REPLAY_SHOW_LASERS
        transition_audio_path = self.get_transition_audio_path()
        if not transition_audio_path.exists():
            transition_audio_path = None
        engine_audio_paths = self.get_engine_audio_paths()
        transition_duration_seconds = REPLAY_PAUSE_SECONDS
        if transition_audio_path is not None:
            transition_duration_seconds = max(
                REPLAY_PAUSE_SECONDS,
                self.get_media_duration_seconds(transition_audio_path) or REPLAY_PAUSE_SECONDS,
            )
        raw_video_path = resolved_output_path.with_name(f"{resolved_output_path.stem}_raw.mp4")
        if raw_video_path.exists():
            raw_video_path.unlink()

        fourcc = cv2.VideoWriter_fourcc(*REPLAY_VIDEO_CODEC)
        writer = cv2.VideoWriter(str(raw_video_path), fourcc, FPS, (CANVAS_WIDTH, CANVAS_HEIGHT))
        if not writer.isOpened():
            pygame.quit()
            raise RuntimeError(f"Failed to open video writer for {raw_video_path}")

        total_frames = 0
        audio_overlays: list[AudioOverlay] = []
        try:
            for run_number, milestone in enumerate(milestones, start=1):
                print(
                    f"Writing replay {run_number}/{len(milestones)} | generation {milestone.generation}",
                    flush=True,
                )
                next_generation = milestones[run_number].generation if run_number < len(milestones) else None
                selected_engine_audio = None
                if engine_audio_paths:
                    selected_engine_audio = self.rng.choice(engine_audio_paths)
                    print(
                        f"  Engine audio: {selected_engine_audio.name}",
                        flush=True,
                    )
                frames_written, run_audio_overlays = self.export_single_generation_video(
                    writer,
                    milestone,
                    run_number,
                    len(milestones),
                    engine_audio_path=selected_engine_audio,
                    include_transition_after=run_number < len(milestones),
                    next_generation=next_generation,
                    transition_duration_seconds=transition_duration_seconds,
                    transition_audio_path=transition_audio_path,
                )
                for overlay in run_audio_overlays:
                    audio_overlays.append(
                        AudioOverlay(
                            path=overlay.path,
                            start_frame=total_frames + overlay.start_frame,
                            duration_frames=overlay.duration_frames,
                            volume=overlay.volume,
                        )
                    )
                total_frames += frames_written
        finally:
            writer.release()
            pygame.quit()

        try:
            if audio_overlays:
                print("Muxing replay audio into mp4...", flush=True)
                try:
                    self.mux_replay_audio(
                        raw_video_path=raw_video_path,
                        output_path=resolved_output_path,
                        audio_overlays=audio_overlays,
                        total_frames=total_frames,
                    )
                except RuntimeError as error:
                    print(
                        "Replay audio mux failed; saving video-only fallback.\n"
                        + str(error),
                        flush=True,
                    )
                    shutil.copy2(raw_video_path, resolved_output_path)
            else:
                print("No replay audio assets were available; saving video-only replay.", flush=True)
                shutil.copy2(raw_video_path, resolved_output_path)
        finally:
            if raw_video_path.exists():
                try:
                    raw_video_path.unlink()
                except OSError:
                    pass

        return resolved_output_path

    def replay_single_generation(self, generation_rollout: PolicyRollout, run_number: int, run_total: int) -> bool:
        policy = self.model.compile(generation_rollout.vector)
        self.simulation.reset_episode()
        self._replay_observation_history = []
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
            f"Neural policy | gen best {generation_rollout.elapsed_time:.2f}s"
        )
        finished_run = False
        hold_time = 0.0

        while self.simulation.is_running:
            dt = self.simulation.clock.tick(FPS) / 1000.0
            self.simulation.handle_events()
            if not self.simulation.is_running:
                return False

            if not finished_run:
                current_observation = np.asarray(self.simulation.get_observation(), dtype=np.float32)
                action = policy.act(
                    self.build_policy_observation(
                        current_observation,
                        self._replay_observation_history,
                    )
                )
                step_state = self.simulation.apply_control_signals(
                    dt,
                    float(action[ACTION_THROTTLE]),
                    float(action[ACTION_STEER]),
                    float(action[ACTION_DRIFT]),
                )
                self._replay_observation_history = [
                    *self._replay_observation_history,
                    current_observation,
                ][-2:]
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


def build_replay_video_neural(output_path: str | Path | None = None) -> Path:
    trainer = NeuralTrackTrainer()
    history, success_generation = trainer.train()
    print(
        f"Neural-net training finished at generation {success_generation} with "
        f"{history[success_generation - 1].elapsed_time:.2f}s run time.",
        flush=True,
    )
    resolved_output_path = trainer.export_milestones_video(
        history,
        success_generation,
        output_path=output_path,
    )
    print(f"Saved replay mp4 to {resolved_output_path}", flush=True)
    return resolved_output_path


def run_training_and_replay_neural() -> None:
    trainer = NeuralTrackTrainer()
    history, success_generation = trainer.train()
    print(
        f"Neural-net training finished at generation {success_generation} with "
        f"{history[success_generation - 1].elapsed_time:.2f}s run time.",
        flush=True,
    )
    if REPLAY_EXPORT_VIDEO:
        output_path = trainer.export_milestones_video(history, success_generation)
        print(f"Saved replay mp4 to {output_path}", flush=True)
    if REPLAY_SHOW_WINDOW:
        trainer.replay_milestones(history, success_generation)


if __name__ == "__main__":
    run_training_and_replay_neural()
