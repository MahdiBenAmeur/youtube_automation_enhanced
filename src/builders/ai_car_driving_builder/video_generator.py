from __future__ import annotations

import math
import random

import pygame


CANVAS_WIDTH = 1080
CANVAS_HEIGHT = 1920
FPS = 60
PREVIEW_SCALE = 0.3
BACKGROUND_COLOR = (46, 154, 46)
WINDOW_TITLE = "AI Car Driving Builder"
ROAD_WIDTH = int(CANVAS_WIDTH * 0.18)
ROAD_LENGTH = int(CANVAS_HEIGHT * 0.15)
ROAD_COLOR = (55, 55, 55)
ROAD_STRIPE_WIDTH = 18
ROAD_STRIPE_SEGMENT_LENGTH = 40
ROAD_STRIPE_COLORS = ((255, 255, 255), (220, 40, 40))
CENTER_LINE_WIDTH = 10
CENTER_LINE_SEGMENT_LENGTH = 40
CENTER_LINE_GAP = 28
CENTER_LINE_COLOR = (255, 255, 255)
FINISH_LINE_LENGTH = 42
FINISH_TILE_SIZE = 18
FINISH_LINE_OFFSET_FROM_END = 28
HUD_PANEL_COLOR = (18, 18, 18, 190)
HUD_TEXT_COLOR = (245, 245, 245)
HUD_ACCENT_COLOR = (255, 220, 80)
HUD_DANGER_COLOR = (255, 90, 90)

CAR_LENGTH = 54
CAR_WIDTH = 30
CAR_SPAWN_OFFSET = 54
CAR_COLOR = (210, 32, 32)
CAR_WINDOW_COLOR = (160, 220, 255)
CAR_TIRE_COLOR = (18, 18, 18)
CAR_MAX_FORWARD_SPEED = 520.0
CAR_MAX_REVERSE_SPEED = 220.0
CAR_ACCELERATION = 460.0
CAR_BRAKE_ACCELERATION = 680.0
CAR_COAST_DRAG = 220.0
CAR_STEER_RATE = 150.0
CAR_DRIFT_STEER_MULTIPLIER = 1.45
CAR_DRIFT_DRAG_MULTIPLIER = 0.45
CAR_FINISH_STOP_DRAG = 1200.0
CAR_SPEED_TO_KMH = 0.32
CAR_COLLISION_STEP = 4.0
LASER_ANGLE_OFFSETS = (-120.0, -95.0, -75.0, -55.0, -35.0, -15.0, 0.0, 15.0, 35.0, 55.0, 75.0, 95.0, 120.0, 180.0)
LASER_MAX_DISTANCE = 260.0
LASER_STEP = 8.0
LASER_COLOR = (255, 245, 120)
LASER_HIT_COLOR = (255, 110, 70)
SIMULATION_ENABLE_OBSTACLES = True
HAY_BALE_COUNT_MIN = 1
HAY_BALE_COUNT_MAX = 3
MOVING_BARRIER_COUNT_MIN = 2
MOVING_BARRIER_COUNT_MAX = 4
OBSTACLE_MIN_TRACK_DISTANCE_FROM_START = 240.0
OBSTACLE_MIN_TRACK_DISTANCE_FROM_FINISH = 180.0
OBSTACLE_TRACK_SPACING = 180.0
OBSTACLE_PLACEMENT_ATTEMPTS = 500
HAY_BALE_HALF_LENGTH = 18.0
HAY_BALE_HALF_WIDTH = 15.0
HAY_BALE_COLOR = (212, 186, 84)
HAY_BALE_SHADOW_COLOR = (156, 126, 44)
MOVING_BARRIER_HALF_LENGTH = 18.0
MOVING_BARRIER_HALF_WIDTH = 42.0
MOVING_BARRIER_MIN_SPEED = 90.0
MOVING_BARRIER_MAX_SPEED = 130.0
MOVING_BARRIER_COLOR = (230, 236, 238)
MOVING_BARRIER_STRIPE_COLOR = (214, 56, 56)
MOVING_BARRIER_POST_COLOR = (40, 40, 40)

TREE_COUNT_MIN = 18
TREE_COUNT_MAX = 32
BUSH_COUNT_MIN = 16
BUSH_COUNT_MAX = 28
SCENERY_PLACEMENT_ATTEMPTS = 800
SCENERY_ROAD_CLEARANCE = 70
TREE_TRUNK_COLOR = (110, 74, 38)
TREE_LEAF_COLORS = ((42, 110, 42), (54, 128, 52), (36, 96, 36))
BUSH_COLORS = ((34, 115, 34), (42, 126, 42), (56, 138, 56))

TRACK_BORDER_PADDING = 6
TRACK_SAMPLE_STEP = 12
TRACK_GUIDE_STEP = 8
TRACK_COLLISION_MARGIN = 12
TRACK_MAX_SEGMENTS = 80
TRACK_MAX_RESTARTS = 250
TRACK_MAX_ATTEMPTS_PER_SEGMENT = 180
RANDOM_TURN_MIN_DEGREES = 20
RANDOM_TURN_MAX_DEGREES = 140
STRAIGHT_WEIGHT_AT_BOTTOM = 0.7
STRAIGHT_WEIGHT_AT_TOP = 2.0
TURN_WEIGHT = 2.0
UPWARD_BIAS_BASE = 0.0
UPWARD_BIAS_STEP = 0.04
UPWARD_BIAS_CAP = 0.82


class CarDrivingSimulation:
    def __init__(self, enable_obstacles: bool = SIMULATION_ENABLE_OBSTACLES) -> None:
        self.screen: pygame.Surface | None = None
        self.canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
        self.clock = pygame.time.Clock()
        self.is_running = False
        self.rng = random.Random()
        self.enable_obstacles = enable_obstacles
        self.track_start_x = CANVAS_WIDTH / 2
        self.track_start_y = float(CANVAS_HEIGHT)
        self.road_segments: list[dict[str, float | list[tuple[float, float]]]] = []
        self.track_samples: list[dict[str, float]] = []
        self.track_total_length = 0.0
        self.drivable_points: list[tuple[float, float]] = []
        self.drivable_surface = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
        self.drivable_mask: pygame.mask.Mask | None = None
        self.build_random_track()
        self.rebuild_track_cache()
        self.finish_state = self.build_finish_state()
        self.scenery_objects = self.generate_scenery()
        self.track_obstacles = self.generate_track_obstacles()
        self.car_state = self.build_car_state()
        self.last_laser_distances = [LASER_MAX_DISTANCE for _ in LASER_ANGLE_OFFSETS]
        self.last_laser_points = [
            (float(self.car_state["x"]), float(self.car_state["y"]))
            for _ in LASER_ANGLE_OFFSETS
        ]
        self.show_lasers = False
        self.hud_title_text = ""
        self.hud_footer_text = "WASD / Arrows, Space = Drift | Avoid obstacles"
        self.refresh_car_metrics()
        self.hud_font: pygame.font.Font | None = None
        self.hud_small_font: pygame.font.Font | None = None

    def ensure_render_resources(self) -> None:
        fonts_were_inactive = not pygame.font.get_init()
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if self.hud_font is None or fonts_were_inactive:
            self.hud_font = pygame.font.SysFont("arial", 44, bold=True)
        if self.hud_small_font is None or fonts_were_inactive:
            self.hud_small_font = pygame.font.SysFont("arial", 28, bold=True)

    def setup(self) -> None:
        self.ensure_render_resources()

        window_size = (
            int(CANVAS_WIDTH * PREVIEW_SCALE),
            int(CANVAS_HEIGHT * PREVIEW_SCALE),
        )
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(WINDOW_TITLE)
        self.is_running = True

    def setup_headless(self) -> None:
        self.ensure_render_resources()
        self.screen = None
        self.is_running = True

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.is_running = False

    def update(self, dt: float) -> None:
        self.update_car(dt)

    def build_random_track(self) -> None:
        start_margin = ROAD_WIDTH / 2 + TRACK_BORDER_PADDING

        for _ in range(TRACK_MAX_RESTARTS):
            self.road_segments = []
            self.track_start_x = self.rng.uniform(start_margin, CANVAS_WIDTH - start_margin)
            self.track_start_y = float(CANVAS_HEIGHT)

            for _segment_index in range(TRACK_MAX_SEGMENTS):
                accepted_segment, reached_top = self.try_add_random_segment()
                if accepted_segment:
                    if reached_top:
                        return
                    continue
                break

        self.build_fallback_track()

    def build_fallback_track(self) -> None:
        self.road_segments = []
        self.track_start_x = CANVAS_WIDTH / 2
        self.track_start_y = float(CANVAS_HEIGHT)

        fallback_segment = self.create_straight_segment(
            start_x=self.track_start_x,
            start_y=self.track_start_y,
            angle_degrees=-90,
            length=CANVAS_HEIGHT,
            width=ROAD_WIDTH,
        )
        clipped_segment = self.trim_segment_to_top(fallback_segment)
        self.append_segment(clipped_segment)

    def try_add_random_segment(self) -> tuple[bool, bool]:
        for _attempt in range(TRACK_MAX_ATTEMPTS_PER_SEGMENT):
            candidate = self.create_random_candidate_segment()

            if not self.bias_allows_heading(candidate["end_angle"]):  # type: ignore[index]
                continue

            reaches_top = self.segment_reaches_top(candidate)
            if reaches_top:
                candidate = self.trim_segment_to_top(candidate)

            candidate["centerline_points"] = self.get_segment_sample_points(candidate)  # type: ignore[index]

            if not self.segment_stays_in_bounds(candidate, allow_top_finish=reaches_top):
                continue
            if self.segment_overlaps_existing(candidate):
                continue

            self.append_segment(candidate)
            return True, reaches_top

        return False, False

    def create_random_candidate_segment(self) -> dict[str, float | list[tuple[float, float]]]:
        start_x, start_y = self.get_last_endpoint()
        start_angle = self.get_last_heading()
        candidate_length = ROAD_LENGTH
        if start_y <= ROAD_LENGTH * 1.6:
            candidate_length = max(ROAD_LENGTH, start_y + ROAD_WIDTH)
        progress_to_top = max(0.0, min(1.0, 1.0 - (start_y / CANVAS_HEIGHT)))
        straight_weight = STRAIGHT_WEIGHT_AT_BOTTOM + (
            (STRAIGHT_WEIGHT_AT_TOP - STRAIGHT_WEIGHT_AT_BOTTOM) * progress_to_top
        )
        segment_kind = self.rng.choices(
            population=("straight", "left", "right", "angle"),
            weights=(straight_weight, TURN_WEIGHT, TURN_WEIGHT, TURN_WEIGHT),
            k=1,
        )[0]

        if segment_kind == "straight":
            return self.create_straight_segment(
                start_x=start_x,
                start_y=start_y,
                angle_degrees=start_angle,
                length=candidate_length,
                width=ROAD_WIDTH,
            )

        if segment_kind == "left":
            return self.create_turn_segment(
                start_x=start_x,
                start_y=start_y,
                start_angle=start_angle,
                turn_degrees=-90,
                length=candidate_length,
                width=ROAD_WIDTH,
            )

        if segment_kind == "right":
            return self.create_turn_segment(
                start_x=start_x,
                start_y=start_y,
                start_angle=start_angle,
                turn_degrees=90,
                length=candidate_length,
                width=ROAD_WIDTH,
            )

        turn_degrees = self.rng.uniform(RANDOM_TURN_MIN_DEGREES, RANDOM_TURN_MAX_DEGREES)
        if self.rng.random() < 0.5:
            turn_degrees *= -1

        return self.create_turn_segment(
            start_x=start_x,
            start_y=start_y,
            start_angle=start_angle,
            turn_degrees=turn_degrees,
            length=candidate_length,
            width=ROAD_WIDTH,
        )

    def bias_allows_heading(self, end_angle: float) -> bool:
        segment_count = len(self.road_segments)
        desired_upness = min(UPWARD_BIAS_BASE + segment_count * UPWARD_BIAS_STEP, UPWARD_BIAS_CAP)
        actual_upness = -math.sin(math.radians(end_angle))
        if actual_upness >= desired_upness:
            return True

        miss = desired_upness - actual_upness
        return self.rng.random() > miss

    def get_last_endpoint(self) -> tuple[float, float]:
        if not self.road_segments:
            return self.track_start_x, self.track_start_y

        last_segment = self.road_segments[-1]
        return last_segment["end_x"], last_segment["end_y"]  # type: ignore[index]

    def get_last_heading(self) -> float:
        if not self.road_segments:
            return -90.0

        return self.road_segments[-1]["end_angle"]  # type: ignore[index]

    def append_segment(self, segment: dict[str, float | list[tuple[float, float]]]) -> None:
        segment["stripe_offset"] = len(self.road_segments) % len(ROAD_STRIPE_COLORS)
        segment["centerline_points"] = self.get_segment_sample_points(segment)
        self.road_segments.append(segment)

    def build_finish_state(self) -> dict[str, float]:
        last_segment = self.road_segments[-1]
        offset = min(FINISH_LINE_OFFSET_FROM_END, max(0.0, float(last_segment["length"]) - 1))
        center_x, center_y, heading = self.get_point_and_heading_from_segment_distance(last_segment, offset)
        heading_radians = math.radians(heading)
        tangent_x = math.cos(heading_radians)
        tangent_y = math.sin(heading_radians)
        normal_x = -tangent_y
        normal_y = tangent_x
        return {
            "center_x": center_x,
            "center_y": center_y,
            "heading": heading,
            "tangent_x": tangent_x,
            "tangent_y": tangent_y,
            "normal_x": normal_x,
            "normal_y": normal_y,
            "half_width": float(last_segment["width"]) / 2,
            "half_length": FINISH_LINE_LENGTH / 2,
        }

    def build_car_state(self) -> dict[str, float | bool]:
        first_segment = self.road_segments[0]
        spawn_x, spawn_y, heading = self.get_point_and_heading_from_segment_start(first_segment, CAR_SPAWN_OFFSET)
        guidance = self.get_track_guidance(spawn_x, spawn_y)
        return {
            "x": spawn_x,
            "y": spawn_y,
            "heading": heading,
            "speed": 0.0,
            "elapsed_time": 0.0,
            "finished": False,
            "collided": False,
            "crashed": False,
            "track_distance": guidance["track_distance"],
            "progress_ratio": guidance["progress_ratio"],
            "lateral_offset": guidance["lateral_offset"],
            "heading_error": self.normalize_angle(heading - guidance["track_heading"]),
        }

    @staticmethod
    def normalize_angle(angle_degrees: float) -> float:
        return ((angle_degrees + 180.0) % 360.0) - 180.0

    def rebuild_track_cache(self) -> None:
        self.track_samples = self.build_track_samples()
        self.track_total_length = float(self.track_samples[-1]["track_distance"]) if self.track_samples else 0.0
        self.drivable_points = [
            (float(sample["x"]), float(sample["y"]))
            for sample in self.track_samples
        ]
        self.build_drivable_mask()

    def build_drivable_mask(self) -> None:
        drivable_width = max(20.0, ROAD_WIDTH - (ROAD_STRIPE_WIDTH + 4) * 2)
        self.drivable_surface.fill((0, 0, 0, 0))
        for segment in self.road_segments:
            self.draw_track_band(self.drivable_surface, segment, drivable_width, (255, 255, 255, 255))
        self.drivable_mask = pygame.mask.from_surface(self.drivable_surface)

    def build_track_samples(self) -> list[dict[str, float]]:
        samples: list[dict[str, float]] = []
        previous_x: float | None = None
        previous_y: float | None = None
        cumulative_distance = 0.0

        for segment in self.road_segments:
            progress_points = self.get_segment_progress_points(segment, step=TRACK_GUIDE_STEP)
            for point_index, (x, y, progress) in enumerate(progress_points):
                if samples and point_index == 0:
                    continue

                _sample_x, _sample_y, heading = self.get_point_and_heading_from_segment_progress(segment, progress)
                if previous_x is not None and previous_y is not None:
                    cumulative_distance += math.hypot(x - previous_x, y - previous_y)

                samples.append(
                    {
                        "x": x,
                        "y": y,
                        "heading": heading,
                        "track_distance": cumulative_distance,
                    }
                )
                previous_x = x
                previous_y = y

        return samples

    def get_track_guidance(self, x: float, y: float) -> dict[str, float]:
        if not self.track_samples:
            return {
                "sample_x": x,
                "sample_y": y,
                "track_heading": -90.0,
                "track_distance": 0.0,
                "progress_ratio": 0.0,
                "distance_to_center": 0.0,
                "lateral_offset": 0.0,
            }

        best_sample = self.track_samples[0]
        best_distance_sq = float("inf")
        delta_x = 0.0
        delta_y = 0.0

        for sample in self.track_samples:
            sample_dx = x - float(sample["x"])
            sample_dy = y - float(sample["y"])
            distance_sq = sample_dx * sample_dx + sample_dy * sample_dy
            if distance_sq < best_distance_sq:
                best_sample = sample
                best_distance_sq = distance_sq
                delta_x = sample_dx
                delta_y = sample_dy

        track_heading = float(best_sample["heading"])
        heading_radians = math.radians(track_heading)
        normal_x = -math.sin(heading_radians)
        normal_y = math.cos(heading_radians)
        track_distance = float(best_sample["track_distance"])
        total_length = max(1.0, self.track_total_length)

        return {
            "sample_x": float(best_sample["x"]),
            "sample_y": float(best_sample["y"]),
            "track_heading": track_heading,
            "track_distance": track_distance,
            "progress_ratio": track_distance / total_length,
            "distance_to_center": math.sqrt(best_distance_sq),
            "lateral_offset": delta_x * normal_x + delta_y * normal_y,
        }

    def cast_laser(self, angle_offset: float) -> tuple[float, tuple[float, float]]:
        origin_x = float(self.car_state["x"])
        origin_y = float(self.car_state["y"])
        absolute_angle = math.radians(float(self.car_state["heading"]) + angle_offset)
        direction_x = math.cos(absolute_angle)
        direction_y = math.sin(absolute_angle)
        last_valid_x = origin_x
        last_valid_y = origin_y

        distance = 0.0
        while distance < LASER_MAX_DISTANCE:
            distance = min(LASER_MAX_DISTANCE, distance + LASER_STEP)
            sample_x = origin_x + direction_x * distance
            sample_y = origin_y + direction_y * distance
            if (
                sample_x < 0
                or sample_x > CANVAS_WIDTH
                or sample_y < 0
                or sample_y > CANVAS_HEIGHT
                or not self.is_point_on_drivable_road(sample_x, sample_y)
                or self.point_hits_any_track_obstacle(sample_x, sample_y)
            ):
                return distance, (last_valid_x, last_valid_y)

            last_valid_x = sample_x
            last_valid_y = sample_y

        return LASER_MAX_DISTANCE, (last_valid_x, last_valid_y)

    def update_laser_cache(self) -> None:
        self.last_laser_distances = []
        self.last_laser_points = []
        for angle_offset in LASER_ANGLE_OFFSETS:
            distance, hit_point = self.cast_laser(angle_offset)
            self.last_laser_distances.append(distance)
            self.last_laser_points.append(hit_point)

    def refresh_car_metrics(self) -> None:
        guidance = self.get_track_guidance(float(self.car_state["x"]), float(self.car_state["y"]))
        heading = float(self.car_state["heading"])
        self.car_state["track_distance"] = guidance["track_distance"]
        self.car_state["progress_ratio"] = guidance["progress_ratio"]
        self.car_state["lateral_offset"] = guidance["lateral_offset"]
        self.car_state["heading_error"] = self.normalize_angle(heading - guidance["track_heading"])
        self.update_laser_cache()

    def reset_episode(self) -> list[float]:
        self.reset_track_obstacles()
        self.car_state = self.build_car_state()
        self.refresh_car_metrics()
        return self.get_observation()

    def get_observation(self) -> list[float]:
        drivable_half_width = max(1.0, ROAD_WIDTH / 2 - ROAD_STRIPE_WIDTH - 4)
        return [
            float(self.car_state["speed"]) / CAR_MAX_FORWARD_SPEED,
            float(self.car_state["heading_error"]) / 180.0,
            max(-1.0, min(1.0, float(self.car_state["lateral_offset"]) / drivable_half_width)),
            max(0.0, min(1.0, float(self.car_state["progress_ratio"]))),
            *[distance / LASER_MAX_DISTANCE for distance in self.last_laser_distances],
        ]

    def get_step_state(self) -> dict[str, float | bool]:
        return {
            "x": float(self.car_state["x"]),
            "y": float(self.car_state["y"]),
            "heading": float(self.car_state["heading"]),
            "speed": float(self.car_state["speed"]),
            "elapsed_time": float(self.car_state["elapsed_time"]),
            "finished": bool(self.car_state["finished"]),
            "collided": bool(self.car_state["collided"]),
            "crashed": bool(self.car_state["crashed"]),
            "track_distance": float(self.car_state["track_distance"]),
            "progress_ratio": float(self.car_state["progress_ratio"]),
            "lateral_offset": float(self.car_state["lateral_offset"]),
            "heading_error": float(self.car_state["heading_error"]),
        }

    def reset_track_obstacles(self) -> None:
        for obstacle in self.track_obstacles:
            if obstacle["kind"] != "moving_barrier":
                continue
            obstacle["lateral_offset"] = float(obstacle["initial_lateral_offset"])
            obstacle["direction"] = float(obstacle["initial_direction"])

    def generate_track_obstacles(self) -> list[dict[str, float | str]]:
        if not self.enable_obstacles:
            return []

        obstacles: list[dict[str, float | str]] = []
        hay_bale_target = self.rng.randint(HAY_BALE_COUNT_MIN, HAY_BALE_COUNT_MAX)
        moving_barrier_target = self.rng.randint(MOVING_BARRIER_COUNT_MIN, MOVING_BARRIER_COUNT_MAX)

        for obstacle_kind, obstacle_target in (
            ("hay_bale", hay_bale_target),
            ("moving_barrier", moving_barrier_target),
        ):
            for _ in range(obstacle_target):
                obstacle = self.create_track_obstacle(obstacle_kind, obstacles)
                if obstacle is not None:
                    obstacles.append(obstacle)

        obstacles.sort(key=lambda obstacle: float(obstacle["track_distance"]))
        return obstacles

    def create_track_obstacle(
        self,
        obstacle_kind: str,
        existing_obstacles: list[dict[str, float | str]],
    ) -> dict[str, float | str] | None:
        if not self.track_samples:
            return None

        minimum_distance = OBSTACLE_MIN_TRACK_DISTANCE_FROM_START
        maximum_distance = self.track_total_length - OBSTACLE_MIN_TRACK_DISTANCE_FROM_FINISH
        if maximum_distance <= minimum_distance:
            return None

        candidate_samples = [
            sample
            for sample in self.track_samples
            if minimum_distance <= float(sample["track_distance"]) <= maximum_distance
        ]
        if not candidate_samples:
            return None

        drivable_half_width = max(24.0, ROAD_WIDTH / 2 - ROAD_STRIPE_WIDTH - 10)
        for _ in range(OBSTACLE_PLACEMENT_ATTEMPTS):
            sample = self.rng.choice(candidate_samples)
            track_distance = float(sample["track_distance"])
            if self.track_obstacle_conflicts(track_distance, existing_obstacles):
                continue

            heading = float(sample["heading"])
            heading_radians = math.radians(heading)
            tangent_x = math.cos(heading_radians)
            tangent_y = math.sin(heading_radians)
            normal_x = -math.sin(heading_radians)
            normal_y = math.cos(heading_radians)

            if obstacle_kind == "hay_bale":
                lateral_limit = max(0.0, drivable_half_width - HAY_BALE_HALF_WIDTH - 8)
                lateral_offset = self.rng.uniform(-lateral_limit, lateral_limit)
                obstacle = {
                    "kind": "hay_bale",
                    "track_distance": track_distance,
                    "anchor_x": float(sample["x"]),
                    "anchor_y": float(sample["y"]),
                    "heading": heading,
                    "tangent_x": tangent_x,
                    "tangent_y": tangent_y,
                    "normal_x": normal_x,
                    "normal_y": normal_y,
                    "half_length": HAY_BALE_HALF_LENGTH,
                    "half_width": HAY_BALE_HALF_WIDTH,
                    "lateral_offset": lateral_offset,
                }
            else:
                road_half_width = float(ROAD_WIDTH) / 2
                travel_half_range = road_half_width + MOVING_BARRIER_HALF_WIDTH + 10.0
                start_side = self.rng.choice((-1.0, 1.0))
                obstacle = {
                    "kind": "moving_barrier",
                    "track_distance": track_distance,
                    "anchor_x": float(sample["x"]),
                    "anchor_y": float(sample["y"]),
                    "heading": heading,
                    "tangent_x": tangent_x,
                    "tangent_y": tangent_y,
                    "normal_x": normal_x,
                    "normal_y": normal_y,
                    "half_length": MOVING_BARRIER_HALF_LENGTH,
                    "half_width": MOVING_BARRIER_HALF_WIDTH,
                    "lateral_offset": start_side * travel_half_range,
                    "initial_lateral_offset": start_side * travel_half_range,
                    "travel_half_range": travel_half_range,
                    "speed": self.rng.uniform(MOVING_BARRIER_MIN_SPEED, MOVING_BARRIER_MAX_SPEED),
                    "direction": -start_side,
                    "initial_direction": -start_side,
                }

            if self.obstacle_overlaps_existing(obstacle, existing_obstacles):
                continue
            return obstacle

        return None

    def track_obstacle_conflicts(
        self,
        track_distance: float,
        existing_obstacles: list[dict[str, float | str]],
    ) -> bool:
        for obstacle in existing_obstacles:
            if abs(track_distance - float(obstacle["track_distance"])) < OBSTACLE_TRACK_SPACING:
                return True
        return False

    def obstacle_overlaps_existing(
        self,
        obstacle: dict[str, float | str],
        existing_obstacles: list[dict[str, float | str]],
    ) -> bool:
        obstacle_x, obstacle_y = self.get_track_obstacle_center(obstacle)
        obstacle_radius = math.hypot(float(obstacle["half_length"]), float(obstacle["half_width"]))
        for existing in existing_obstacles:
            existing_x, existing_y = self.get_track_obstacle_center(existing)
            existing_radius = math.hypot(float(existing["half_length"]), float(existing["half_width"]))
            if math.hypot(obstacle_x - existing_x, obstacle_y - existing_y) < obstacle_radius + existing_radius + 24:
                return True
        return False

    def update_track_obstacles(self, dt: float) -> None:
        if not self.enable_obstacles or dt <= 0:
            return

        for obstacle in self.track_obstacles:
            if obstacle["kind"] != "moving_barrier":
                continue

            travel_half_range = float(obstacle["travel_half_range"])
            next_offset = float(obstacle["lateral_offset"]) + float(obstacle["direction"]) * float(obstacle["speed"]) * dt
            if abs(next_offset) >= travel_half_range:
                next_offset = max(-travel_half_range, min(travel_half_range, next_offset))
                obstacle["direction"] = -float(obstacle["direction"])
            obstacle["lateral_offset"] = next_offset

    def get_track_obstacle_center(self, obstacle: dict[str, float | str]) -> tuple[float, float]:
        lateral_offset = float(obstacle["lateral_offset"])
        return (
            float(obstacle["anchor_x"]) + float(obstacle["normal_x"]) * lateral_offset,
            float(obstacle["anchor_y"]) + float(obstacle["normal_y"]) * lateral_offset,
        )

    def get_track_obstacle_polygon(self, obstacle: dict[str, float | str]) -> list[tuple[float, float]]:
        center_x, center_y = self.get_track_obstacle_center(obstacle)
        tangent_x = float(obstacle["tangent_x"])
        tangent_y = float(obstacle["tangent_y"])
        normal_x = float(obstacle["normal_x"])
        normal_y = float(obstacle["normal_y"])
        half_length = float(obstacle["half_length"])
        half_width = float(obstacle["half_width"])
        return [
            (
                center_x + tangent_x * half_length + normal_x * half_width,
                center_y + tangent_y * half_length + normal_y * half_width,
            ),
            (
                center_x + tangent_x * half_length - normal_x * half_width,
                center_y + tangent_y * half_length - normal_y * half_width,
            ),
            (
                center_x - tangent_x * half_length - normal_x * half_width,
                center_y - tangent_y * half_length - normal_y * half_width,
            ),
            (
                center_x - tangent_x * half_length + normal_x * half_width,
                center_y - tangent_y * half_length + normal_y * half_width,
            ),
        ]

    def point_hits_track_obstacle(
        self,
        point_x: float,
        point_y: float,
        obstacle: dict[str, float | str],
    ) -> bool:
        center_x, center_y = self.get_track_obstacle_center(obstacle)
        tangent_x = float(obstacle["tangent_x"])
        tangent_y = float(obstacle["tangent_y"])
        normal_x = float(obstacle["normal_x"])
        normal_y = float(obstacle["normal_y"])
        rel_x = point_x - center_x
        rel_y = point_y - center_y
        along = rel_x * tangent_x + rel_y * tangent_y
        across = rel_x * normal_x + rel_y * normal_y
        return (
            abs(along) <= float(obstacle["half_length"])
            and abs(across) <= float(obstacle["half_width"])
        )

    def point_hits_any_track_obstacle(self, point_x: float, point_y: float) -> bool:
        if not self.enable_obstacles:
            return False

        for obstacle in self.track_obstacles:
            if self.point_hits_track_obstacle(point_x, point_y, obstacle):
                return True
        return False

    def car_hits_track_obstacle(self, x: float, y: float, heading: float) -> bool:
        if not self.enable_obstacles:
            return False

        for point_x, point_y in self.get_car_collision_points(x, y, heading):
            for obstacle in self.track_obstacles:
                if self.point_hits_track_obstacle(point_x, point_y, obstacle):
                    return True
        return False

    def generate_scenery(self) -> list[dict[str, float | tuple[int, int, int] | str]]:
        scenery: list[dict[str, float | tuple[int, int, int] | str]] = []
        scenery.extend(self.generate_scenery_items("tree", self.rng.randint(TREE_COUNT_MIN, TREE_COUNT_MAX)))
        scenery.extend(self.generate_scenery_items("bush", self.rng.randint(BUSH_COUNT_MIN, BUSH_COUNT_MAX)))
        return scenery

    def generate_scenery_items(
        self,
        item_kind: str,
        target_count: int,
    ) -> list[dict[str, float | tuple[int, int, int] | str]]:
        placed_items: list[dict[str, float | tuple[int, int, int] | str]] = []

        for _ in range(SCENERY_PLACEMENT_ATTEMPTS):
            if len(placed_items) >= target_count:
                break

            if item_kind == "tree":
                radius = self.rng.uniform(18, 30)
                clear_radius = radius + SCENERY_ROAD_CLEARANCE
            else:
                radius = self.rng.uniform(12, 22)
                clear_radius = radius + SCENERY_ROAD_CLEARANCE - 12

            x = self.rng.uniform(radius + TRACK_BORDER_PADDING, CANVAS_WIDTH - radius - TRACK_BORDER_PADDING)
            y = self.rng.uniform(radius + TRACK_BORDER_PADDING, CANVAS_HEIGHT - radius - TRACK_BORDER_PADDING)

            if not self.point_is_in_grass(x, y, clear_radius):
                continue
            if self.point_hits_scenery(x, y, radius, placed_items):
                continue

            color = self.rng.choice(TREE_LEAF_COLORS if item_kind == "tree" else BUSH_COLORS)
            placed_items.append(
                {
                    "kind": item_kind,
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "color": color,
                }
            )

        return placed_items

    def point_hits_scenery(
        self,
        x: float,
        y: float,
        radius: float,
        placed_items: list[dict[str, float | tuple[int, int, int] | str]],
    ) -> bool:
        for item in placed_items:
            dx = x - float(item["x"])
            dy = y - float(item["y"])
            if math.hypot(dx, dy) < radius + float(item["radius"]) + 10:
                return True
        return False

    def point_is_in_grass(self, x: float, y: float, clear_radius: float) -> bool:
        for segment in self.road_segments:
            centerline_points = list(segment["centerline_points"])  # type: ignore[arg-type]
            width_clearance = float(segment["width"]) / 2 + clear_radius
            width_clearance_sq = width_clearance * width_clearance
            for point_x, point_y in centerline_points:
                dx = x - point_x
                dy = y - point_y
                if dx * dx + dy * dy < width_clearance_sq:
                    return False
        return True

    def get_point_and_heading_from_segment_start(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
        distance_from_start: float,
    ) -> tuple[float, float, float]:
        segment_length = max(1.0, float(segment["length"]))
        progress = max(0.0, min(1.0, distance_from_start / segment_length))
        return self.get_point_and_heading_from_segment_progress(segment, progress)

    def add_straight_line(
        self,
        angle_degrees: float,
        length: float = ROAD_LENGTH,
        width: float = ROAD_WIDTH,
    ) -> None:
        start_x, start_y = self.get_last_endpoint()
        segment = self.create_straight_segment(start_x, start_y, angle_degrees, length, width)
        self.append_segment(segment)

    def add_turn_angle(
        self,
        turn_degrees: float,
        length: float = ROAD_LENGTH,
        width: float = ROAD_WIDTH,
    ) -> None:
        start_x, start_y = self.get_last_endpoint()
        start_angle = self.get_last_heading()
        segment = self.create_turn_segment(start_x, start_y, start_angle, turn_degrees, length, width)
        self.append_segment(segment)

    def create_straight_segment(
        self,
        start_x: float,
        start_y: float,
        angle_degrees: float,
        length: float,
        width: float,
    ) -> dict[str, float | list[tuple[float, float]]]:
        angle_radians = math.radians(angle_degrees)
        end_x = start_x + math.cos(angle_radians) * length
        end_y = start_y + math.sin(angle_radians) * length
        return {
            "kind": "straight",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "width": width,
            "length": length,
            "start_angle": angle_degrees,
            "end_angle": angle_degrees,
        }

    def create_turn_segment(
        self,
        start_x: float,
        start_y: float,
        start_angle: float,
        turn_degrees: float,
        length: float,
        width: float,
    ) -> dict[str, float | list[tuple[float, float]]]:
        turn_angle_radians = math.radians(turn_degrees)
        radius = length / abs(turn_angle_radians)
        start_angle_radians = math.radians(start_angle)
        turn_direction = 1 if turn_degrees > 0 else -1
        normal_x = -math.sin(start_angle_radians) * turn_direction
        normal_y = math.cos(start_angle_radians) * turn_direction
        center_x = start_x + normal_x * radius
        center_y = start_y + normal_y * radius

        start_theta = math.atan2(start_y - center_y, start_x - center_x)
        end_theta = start_theta + turn_angle_radians
        end_x = center_x + math.cos(end_theta) * radius
        end_y = center_y + math.sin(end_theta) * radius

        return {
            "kind": "turn",
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "width": width,
            "length": length,
            "radius": radius,
            "center_x": center_x,
            "center_y": center_y,
            "start_theta": start_theta,
            "end_theta": end_theta,
            "turn_angle_degrees": turn_degrees,
            "start_angle": start_angle,
            "end_angle": start_angle + turn_degrees,
        }

    def get_segment_progress_points(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
        step: float = TRACK_SAMPLE_STEP,
    ) -> list[tuple[float, float, float]]:
        length = max(float(segment["length"]), step)
        count = max(2, int(math.ceil(length / step)) + 1)
        points: list[tuple[float, float, float]] = []

        if segment["kind"] == "straight":
            start_x = float(segment["start_x"])
            start_y = float(segment["start_y"])
            end_x = float(segment["end_x"])
            end_y = float(segment["end_y"])
            for index in range(count):
                t = index / (count - 1)
                x = start_x + (end_x - start_x) * t
                y = start_y + (end_y - start_y) * t
                points.append((x, y, t))
            return points

        center_x = float(segment["center_x"])
        center_y = float(segment["center_y"])
        radius = float(segment["radius"])
        start_theta = float(segment["start_theta"])
        end_theta = float(segment["end_theta"])
        for index in range(count):
            t = index / (count - 1)
            theta = start_theta + (end_theta - start_theta) * t
            x = center_x + math.cos(theta) * radius
            y = center_y + math.sin(theta) * radius
            points.append((x, y, t))
        return points

    def get_segment_sample_points(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
        step: float = TRACK_SAMPLE_STEP,
    ) -> list[tuple[float, float]]:
        return [(x, y) for x, y, _ in self.get_segment_progress_points(segment, step)]

    def segment_reaches_top(self, segment: dict[str, float | list[tuple[float, float]]]) -> bool:
        points = self.get_segment_progress_points(segment, step=TRACK_SAMPLE_STEP / 2)
        return any(y <= 0 for _x, y, _t in points[1:])

    def trim_segment_to_top(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
    ) -> dict[str, float | list[tuple[float, float]]]:
        points = self.get_segment_progress_points(segment, step=TRACK_SAMPLE_STEP / 2)

        for index in range(1, len(points)):
            x1, y1, t1 = points[index - 1]
            x2, y2, t2 = points[index]
            if y1 > 0 >= y2:
                if y2 == y1:
                    clip_t = t2
                else:
                    ratio = (0 - y1) / (y2 - y1)
                    clip_t = t1 + (t2 - t1) * ratio
                return self.trim_segment_at_progress(segment, clip_t)

        return segment

    def trim_segment_at_progress(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
        progress: float,
    ) -> dict[str, float | list[tuple[float, float]]]:
        progress = max(0.0, min(1.0, progress))

        if segment["kind"] == "straight":
            start_x = float(segment["start_x"])
            start_y = float(segment["start_y"])
            end_x = start_x + (float(segment["end_x"]) - start_x) * progress
            end_y = start_y + (float(segment["end_y"]) - start_y) * progress
            trimmed = dict(segment)
            trimmed["end_x"] = end_x
            trimmed["end_y"] = end_y
            trimmed["length"] = float(segment["length"]) * progress
            return trimmed

        start_theta = float(segment["start_theta"])
        end_theta = float(segment["end_theta"])
        radius = float(segment["radius"])
        clipped_theta = start_theta + (end_theta - start_theta) * progress
        center_x = float(segment["center_x"])
        center_y = float(segment["center_y"])
        trimmed = dict(segment)
        trimmed["end_theta"] = clipped_theta
        trimmed["end_x"] = center_x + math.cos(clipped_theta) * radius
        trimmed["end_y"] = center_y + math.sin(clipped_theta) * radius
        trimmed["turn_angle_degrees"] = float(segment["turn_angle_degrees"]) * progress
        trimmed["end_angle"] = float(segment["start_angle"]) + float(trimmed["turn_angle_degrees"])
        trimmed["length"] = float(segment["length"]) * progress
        return trimmed

    def get_point_and_heading_from_segment_progress(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
        progress: float,
    ) -> tuple[float, float, float]:
        progress = max(0.0, min(1.0, progress))

        if segment["kind"] == "straight":
            start_x = float(segment["start_x"])
            start_y = float(segment["start_y"])
            end_x = float(segment["end_x"])
            end_y = float(segment["end_y"])
            x = start_x + (end_x - start_x) * progress
            y = start_y + (end_y - start_y) * progress
            heading = float(segment["end_angle"])
            return x, y, heading

        center_x = float(segment["center_x"])
        center_y = float(segment["center_y"])
        radius = float(segment["radius"])
        start_theta = float(segment["start_theta"])
        end_theta = float(segment["end_theta"])
        theta = start_theta + (end_theta - start_theta) * progress
        x = center_x + math.cos(theta) * radius
        y = center_y + math.sin(theta) * radius
        heading = float(segment["start_angle"]) + float(segment["turn_angle_degrees"]) * progress
        return x, y, heading

    def get_point_and_heading_from_segment_distance(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
        back_distance: float,
    ) -> tuple[float, float, float]:
        segment_length = max(1.0, float(segment["length"]))
        progress = max(0.0, min(1.0, 1.0 - (back_distance / segment_length)))
        return self.get_point_and_heading_from_segment_progress(segment, progress)

    def segment_stays_in_bounds(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
        allow_top_finish: bool,
    ) -> bool:
        half_width = float(segment["width"]) / 2
        points = self.get_segment_sample_points(segment, step=TRACK_SAMPLE_STEP / 2)
        segment_length = max(float(segment["length"]), TRACK_SAMPLE_STEP / 2)

        for index, (x, y) in enumerate(points):
            if x - half_width <= TRACK_BORDER_PADDING:
                return False
            if x + half_width >= CANVAS_WIDTH - TRACK_BORDER_PADDING:
                return False
            if index == 0:
                continue
            traveled_distance = segment_length * (index / (len(points) - 1))
            if (
                y + half_width >= CANVAS_HEIGHT - TRACK_BORDER_PADDING
                and traveled_distance > half_width + TRACK_SAMPLE_STEP
            ):
                return False
            if allow_top_finish:
                continue
            if y - half_width <= TRACK_BORDER_PADDING:
                return False

        return True

    def segment_overlaps_existing(self, candidate: dict[str, float | list[tuple[float, float]]]) -> bool:
        candidate_points = list(candidate["centerline_points"])  # type: ignore[arg-type]
        junction_skip_count = max(4, int(math.ceil(float(candidate["width"]) / TRACK_SAMPLE_STEP)))
        if len(candidate_points) <= junction_skip_count:
            return False

        candidate_check_points = candidate_points[junction_skip_count:]
        for index, existing_segment in enumerate(self.road_segments):
            existing_points = list(existing_segment["centerline_points"])  # type: ignore[arg-type]
            if index == len(self.road_segments) - 1 and len(existing_points) > junction_skip_count:
                existing_points = existing_points[:-junction_skip_count]

            if not existing_points:
                continue

            threshold = (float(candidate["width"]) + float(existing_segment["width"])) / 2 - TRACK_COLLISION_MARGIN
            for candidate_x, candidate_y in candidate_check_points:
                for existing_x, existing_y in existing_points:
                    if math.hypot(candidate_x - existing_x, candidate_y - existing_y) < threshold:
                        return True

        return False

    def is_point_on_drivable_road(self, x: float, y: float) -> bool:
        if self.drivable_mask is None:
            return False

        point_x = int(round(x))
        point_y = int(round(y))
        if point_x < 0 or point_x >= CANVAS_WIDTH or point_y < 0 or point_y >= CANVAS_HEIGHT:
            return False

        return bool(self.drivable_mask.get_at((point_x, point_y)))

    def get_car_polygon(
        self,
        x: float | None = None,
        y: float | None = None,
        heading: float | None = None,
    ) -> list[tuple[float, float]]:
        car_x = float(self.car_state["x"] if x is None else x)
        car_y = float(self.car_state["y"] if y is None else y)
        car_heading = float(self.car_state["heading"] if heading is None else heading)
        heading_radians = math.radians(car_heading)
        forward_x = math.cos(heading_radians)
        forward_y = math.sin(heading_radians)
        side_x = -forward_y
        side_y = forward_x
        half_length = CAR_LENGTH / 2
        half_width = CAR_WIDTH / 2
        return [
            (
                car_x + forward_x * half_length + side_x * half_width,
                car_y + forward_y * half_length + side_y * half_width,
            ),
            (
                car_x + forward_x * half_length - side_x * half_width,
                car_y + forward_y * half_length - side_y * half_width,
            ),
            (
                car_x - forward_x * half_length - side_x * half_width,
                car_y - forward_y * half_length - side_y * half_width,
            ),
            (
                car_x - forward_x * half_length + side_x * half_width,
                car_y - forward_y * half_length + side_y * half_width,
            ),
        ]

    def get_car_collision_points(
        self,
        x: float | None = None,
        y: float | None = None,
        heading: float | None = None,
    ) -> list[tuple[float, float]]:
        polygon = self.get_car_polygon(x, y, heading)
        center_x = float(self.car_state["x"] if x is None else x)
        center_y = float(self.car_state["y"] if y is None else y)
        car_heading = float(self.car_state["heading"] if heading is None else heading)
        heading_radians = math.radians(car_heading)
        forward_x = math.cos(heading_radians)
        forward_y = math.sin(heading_radians)
        side_x = -forward_y
        side_y = forward_x

        collision_points = list(polygon)
        for point_index in range(len(polygon)):
            next_index = (point_index + 1) % len(polygon)
            point_a = polygon[point_index]
            point_b = polygon[next_index]
            collision_points.append(
                ((point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2)
            )

        collision_points.extend(
            [
                (center_x, center_y),
                (
                    center_x + forward_x * (CAR_LENGTH * 0.35),
                    center_y + forward_y * (CAR_LENGTH * 0.35),
                ),
                (
                    center_x - forward_x * (CAR_LENGTH * 0.35),
                    center_y - forward_y * (CAR_LENGTH * 0.35),
                ),
                (
                    center_x + side_x * (CAR_WIDTH * 0.35),
                    center_y + side_y * (CAR_WIDTH * 0.35),
                ),
                (
                    center_x - side_x * (CAR_WIDTH * 0.35),
                    center_y - side_y * (CAR_WIDTH * 0.35),
                ),
            ]
        )
        return collision_points

    def car_collides_with_barrier(self, x: float, y: float, heading: float) -> bool:
        for point_x, point_y in self.get_car_collision_points(x, y, heading):
            if not self.is_point_on_drivable_road(point_x, point_y):
                return True
        return False

    def car_reaches_finish(self) -> bool:
        heading_radians = math.radians(float(self.car_state["heading"]))
        front_x = float(self.car_state["x"]) + math.cos(heading_radians) * (CAR_LENGTH / 2)
        front_y = float(self.car_state["y"]) + math.sin(heading_radians) * (CAR_LENGTH / 2)
        rel_x = front_x - float(self.finish_state["center_x"])
        rel_y = front_y - float(self.finish_state["center_y"])
        along = rel_x * float(self.finish_state["tangent_x"]) + rel_y * float(self.finish_state["tangent_y"])
        across = rel_x * float(self.finish_state["normal_x"]) + rel_y * float(self.finish_state["normal_y"])
        return (
            abs(along) <= float(self.finish_state["half_length"]) + 4
            and abs(across) <= float(self.finish_state["half_width"]) - 4
        )

    def move_toward(self, value: float, target: float, delta: float) -> float:
        if value < target:
            return min(target, value + delta)
        return max(target, value - delta)

    def get_manual_control_signals(self) -> tuple[float, float, float]:
        keys = pygame.key.get_pressed()
        throttle_forward = keys[pygame.K_UP] or keys[pygame.K_w]
        throttle_back = keys[pygame.K_DOWN] or keys[pygame.K_s]
        steer_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        steer_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        drift = keys[pygame.K_SPACE]
        throttle_signal = float(bool(throttle_forward)) - float(bool(throttle_back))
        steer_signal = float(bool(steer_right)) - float(bool(steer_left))
        drift_signal = 1.0 if drift else 0.0
        return throttle_signal, steer_signal, drift_signal

    def apply_control_signals(
        self,
        dt: float,
        throttle_signal: float,
        steer_signal: float,
        drift_signal: float,
    ) -> dict[str, float | bool]:
        if dt <= 0:
            return self.get_step_state()

        self.update_track_obstacles(dt)

        if not bool(self.car_state["finished"]) and not bool(self.car_state["crashed"]):
            self.car_state["elapsed_time"] = float(self.car_state["elapsed_time"]) + dt

        if bool(self.car_state["finished"]) or bool(self.car_state["crashed"]):
            speed = float(self.car_state["speed"])
            self.car_state["speed"] = self.move_toward(speed, 0.0, CAR_FINISH_STOP_DRAG * dt)
            self.car_state["collided"] = False
            if bool(self.car_state["crashed"]):
                self.car_state["speed"] = 0.0
            self.refresh_car_metrics()
            return self.get_step_state()

        speed = float(self.car_state["speed"])
        heading = float(self.car_state["heading"])
        x = float(self.car_state["x"])
        y = float(self.car_state["y"])
        throttle_signal = max(-1.0, min(1.0, throttle_signal))
        steer_signal = max(-1.0, min(1.0, steer_signal))
        drift = drift_signal > 0.5

        if throttle_signal > 0.05:
            speed = min(CAR_MAX_FORWARD_SPEED, speed + CAR_ACCELERATION * throttle_signal * dt)
        elif throttle_signal < -0.05:
            brake_strength = abs(throttle_signal)
            if speed > 0:
                speed = max(0.0, speed - CAR_BRAKE_ACCELERATION * brake_strength * dt)
            else:
                speed = max(-CAR_MAX_REVERSE_SPEED, speed - CAR_ACCELERATION * brake_strength * dt)
        else:
            drag = CAR_COAST_DRAG * (CAR_DRIFT_DRAG_MULTIPLIER if drift else 1.0)
            speed = self.move_toward(speed, 0.0, drag * dt)

        speed_factor = min(1.0, abs(speed) / max(1.0, CAR_MAX_FORWARD_SPEED * 0.35))
        steer_rate = CAR_STEER_RATE * (CAR_DRIFT_STEER_MULTIPLIER if drift else 1.0)
        heading += steer_signal * steer_rate * speed_factor * dt

        heading_radians = math.radians(heading)
        move_x = math.cos(heading_radians) * speed * dt
        move_y = math.sin(heading_radians) * speed * dt
        proposed_x = x + move_x
        proposed_y = y + move_y
        collided = False

        move_distance = math.hypot(move_x, move_y)
        move_steps = max(1, int(math.ceil(move_distance / CAR_COLLISION_STEP)))
        safe_x = x
        safe_y = y
        crashed = False
        for step_index in range(1, move_steps + 1):
            step_ratio = step_index / move_steps
            step_x = x + move_x * step_ratio
            step_y = y + move_y * step_ratio
            if self.car_collides_with_barrier(step_x, step_y, heading):
                speed = 0.0
                proposed_x = safe_x
                proposed_y = safe_y
                collided = True
                break
            if self.car_hits_track_obstacle(step_x, step_y, heading):
                speed = 0.0
                proposed_x = safe_x
                proposed_y = safe_y
                crashed = True
                break
            safe_x = step_x
            safe_y = step_y

        self.car_state["x"] = proposed_x
        self.car_state["y"] = proposed_y
        self.car_state["heading"] = heading
        self.car_state["speed"] = speed
        self.car_state["collided"] = collided
        self.car_state["crashed"] = crashed
        self.refresh_car_metrics()

        if self.car_reaches_finish() and not bool(self.car_state["crashed"]):
            self.car_state["finished"] = True
            self.car_state["speed"] = 0.0
            self.car_state["collided"] = False

        return self.get_step_state()

    def update_car(self, dt: float) -> None:
        throttle_signal, steer_signal, drift_signal = self.get_manual_control_signals()
        self.apply_control_signals(dt, throttle_signal, steer_signal, drift_signal)

    def get_segment_polygon(self, segment: dict[str, float | list[tuple[float, float]]]) -> list[tuple[float, float]]:
        return self.get_segment_polygon_with_width(segment, float(segment["width"]))

    def get_segment_polygon_with_width(
        self,
        segment: dict[str, float | list[tuple[float, float]]],
        width: float,
    ) -> list[tuple[float, float]]:
        start_x = float(segment["start_x"])
        start_y = float(segment["start_y"])
        end_x = float(segment["end_x"])
        end_y = float(segment["end_y"])

        delta_x = end_x - start_x
        delta_y = end_y - start_y
        segment_length = math.hypot(delta_x, delta_y)
        if segment_length == 0:
            return []

        unit_x = delta_x / segment_length
        unit_y = delta_y / segment_length
        normal_x = -unit_y
        normal_y = unit_x
        half_width = width / 2

        return [
            (start_x + normal_x * half_width, start_y + normal_y * half_width),
            (end_x + normal_x * half_width, end_y + normal_y * half_width),
            (end_x - normal_x * half_width, end_y - normal_y * half_width),
            (start_x - normal_x * half_width, start_y - normal_y * half_width),
        ]

    def draw_polygon_on_surface(
        self,
        surface: pygame.Surface,
        points: list[tuple[float, float]],
        color: tuple[int, int, int] | tuple[int, int, int, int],
    ) -> None:
        if not points:
            return

        int_points = [(int(round(x)), int(round(y))) for x, y in points]
        pygame.draw.polygon(surface, color, int_points)

    def draw_segment_polygon(self, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
        self.draw_polygon_on_surface(self.canvas, points, color)

    def get_arc_points(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        start_theta: float,
        end_theta: float,
    ) -> list[tuple[float, float]]:
        sweep = end_theta - start_theta
        step_count = max(8, int(abs(sweep) * radius / 12))
        return [
            (
                center_x + math.cos(start_theta + sweep * (index / step_count)) * radius,
                center_y + math.sin(start_theta + sweep * (index / step_count)) * radius,
            )
            for index in range(step_count + 1)
        ]

    def draw_straight_segment(self, segment: dict[str, float | list[tuple[float, float]]]) -> None:
        road_polygon = self.get_segment_polygon(segment)
        self.draw_segment_polygon(road_polygon, ROAD_COLOR)

        delta_x = float(segment["end_x"]) - float(segment["start_x"])
        delta_y = float(segment["end_y"]) - float(segment["start_y"])
        segment_length = math.hypot(delta_x, delta_y)
        if segment_length == 0:
            return

        unit_x = delta_x / segment_length
        unit_y = delta_y / segment_length
        normal_x = -unit_y
        normal_y = unit_x
        half_width = float(segment["width"]) / 2

        stripe_count = max(1, round(segment_length / ROAD_STRIPE_SEGMENT_LENGTH))
        stripe_segment_length = segment_length / stripe_count
        stripe_offset = int(segment["stripe_offset"])
        for index in range(stripe_count):
            color = ROAD_STRIPE_COLORS[(index + stripe_offset) % len(ROAD_STRIPE_COLORS)]
            stripe_start_distance = index * stripe_segment_length
            stripe_end_distance = stripe_start_distance + stripe_segment_length

            start_x = float(segment["start_x"]) + unit_x * stripe_start_distance
            start_y = float(segment["start_y"]) + unit_y * stripe_start_distance
            end_x = float(segment["start_x"]) + unit_x * stripe_end_distance
            end_y = float(segment["start_y"]) + unit_y * stripe_end_distance

            left_points = [
                (start_x + normal_x * half_width, start_y + normal_y * half_width),
                (end_x + normal_x * half_width, end_y + normal_y * half_width),
                (
                    end_x + normal_x * (half_width - ROAD_STRIPE_WIDTH),
                    end_y + normal_y * (half_width - ROAD_STRIPE_WIDTH),
                ),
                (
                    start_x + normal_x * (half_width - ROAD_STRIPE_WIDTH),
                    start_y + normal_y * (half_width - ROAD_STRIPE_WIDTH),
                ),
            ]
            right_points = [
                (start_x - normal_x * half_width, start_y - normal_y * half_width),
                (end_x - normal_x * half_width, end_y - normal_y * half_width),
                (
                    end_x - normal_x * (half_width - ROAD_STRIPE_WIDTH),
                    end_y - normal_y * (half_width - ROAD_STRIPE_WIDTH),
                ),
                (
                    start_x - normal_x * (half_width - ROAD_STRIPE_WIDTH),
                    start_y - normal_y * (half_width - ROAD_STRIPE_WIDTH),
                ),
            ]
            self.draw_segment_polygon(left_points, color)
            self.draw_segment_polygon(right_points, color)

        center_step = CENTER_LINE_SEGMENT_LENGTH + CENTER_LINE_GAP
        center_count = math.ceil(segment_length / center_step)
        for index in range(center_count):
            dash_start_distance = index * center_step
            dash_end_distance = min(dash_start_distance + CENTER_LINE_SEGMENT_LENGTH, segment_length)
            if dash_start_distance >= segment_length:
                break

            start_x = float(segment["start_x"]) + unit_x * dash_start_distance
            start_y = float(segment["start_y"]) + unit_y * dash_start_distance
            end_x = float(segment["start_x"]) + unit_x * dash_end_distance
            end_y = float(segment["start_y"]) + unit_y * dash_end_distance

            center_points = [
                (
                    start_x + normal_x * (CENTER_LINE_WIDTH / 2),
                    start_y + normal_y * (CENTER_LINE_WIDTH / 2),
                ),
                (
                    end_x + normal_x * (CENTER_LINE_WIDTH / 2),
                    end_y + normal_y * (CENTER_LINE_WIDTH / 2),
                ),
                (
                    end_x - normal_x * (CENTER_LINE_WIDTH / 2),
                    end_y - normal_y * (CENTER_LINE_WIDTH / 2),
                ),
                (
                    start_x - normal_x * (CENTER_LINE_WIDTH / 2),
                    start_y - normal_y * (CENTER_LINE_WIDTH / 2),
                ),
            ]
            self.draw_segment_polygon(center_points, CENTER_LINE_COLOR)

    def draw_track_band(
        self,
        surface: pygame.Surface,
        segment: dict[str, float | list[tuple[float, float]]],
        width: float,
        color: tuple[int, int, int] | tuple[int, int, int, int],
    ) -> None:
        if segment["kind"] == "turn":
            center_x = float(segment["center_x"])
            center_y = float(segment["center_y"])
            radius = float(segment["radius"])
            start_theta = float(segment["start_theta"])
            end_theta = float(segment["end_theta"])
            half_width = width / 2
            outer_points = self.get_arc_points(center_x, center_y, radius + half_width, start_theta, end_theta)
            inner_points = self.get_arc_points(center_x, center_y, radius - half_width, start_theta, end_theta)
            self.draw_polygon_on_surface(surface, outer_points + list(reversed(inner_points)), color)
            return

        self.draw_polygon_on_surface(surface, self.get_segment_polygon_with_width(segment, width), color)

    def draw_turn_segment(self, segment: dict[str, float | list[tuple[float, float]]]) -> None:
        center_x = float(segment["center_x"])
        center_y = float(segment["center_y"])
        radius = float(segment["radius"])
        start_theta = float(segment["start_theta"])
        end_theta = float(segment["end_theta"])
        half_width = float(segment["width"]) / 2
        inner_radius = radius - half_width
        outer_radius = radius + half_width

        outer_points = self.get_arc_points(center_x, center_y, outer_radius, start_theta, end_theta)
        inner_points = self.get_arc_points(center_x, center_y, inner_radius, start_theta, end_theta)
        road_polygon = outer_points + list(reversed(inner_points))
        self.draw_segment_polygon(road_polygon, ROAD_COLOR)

        turn_direction = 1 if float(segment["turn_angle_degrees"]) > 0 else -1
        arc_length = radius * abs(end_theta - start_theta)
        stripe_count = max(1, round(arc_length / ROAD_STRIPE_SEGMENT_LENGTH))
        stripe_arc_length = arc_length / stripe_count
        stripe_offset = int(segment["stripe_offset"])

        for index in range(stripe_count):
            color = ROAD_STRIPE_COLORS[(index + stripe_offset) % len(ROAD_STRIPE_COLORS)]
            stripe_start_theta = start_theta + turn_direction * (index * stripe_arc_length) / radius
            stripe_end_theta = start_theta + turn_direction * ((index + 1) * stripe_arc_length) / radius

            outer_edge_points = self.get_arc_points(center_x, center_y, outer_radius, stripe_start_theta, stripe_end_theta)
            outer_inner_points = self.get_arc_points(
                center_x,
                center_y,
                outer_radius - ROAD_STRIPE_WIDTH,
                stripe_start_theta,
                stripe_end_theta,
            )
            outer_stripe_polygon = outer_edge_points + list(reversed(outer_inner_points))
            self.draw_segment_polygon(outer_stripe_polygon, color)

            inner_edge_points = self.get_arc_points(
                center_x,
                center_y,
                inner_radius + ROAD_STRIPE_WIDTH,
                stripe_start_theta,
                stripe_end_theta,
            )
            inner_border_points = self.get_arc_points(center_x, center_y, inner_radius, stripe_start_theta, stripe_end_theta)
            inner_stripe_polygon = inner_edge_points + list(reversed(inner_border_points))
            self.draw_segment_polygon(inner_stripe_polygon, color)

        center_step = CENTER_LINE_SEGMENT_LENGTH + CENTER_LINE_GAP
        center_count = math.ceil(arc_length / center_step)
        center_outer_radius = radius + (CENTER_LINE_WIDTH / 2)
        center_inner_radius = radius - (CENTER_LINE_WIDTH / 2)

        for index in range(center_count):
            dash_start_distance = index * center_step
            dash_end_distance = min(dash_start_distance + CENTER_LINE_SEGMENT_LENGTH, arc_length)
            if dash_start_distance >= arc_length:
                break

            dash_start_theta = start_theta + turn_direction * dash_start_distance / radius
            dash_end_theta = start_theta + turn_direction * dash_end_distance / radius
            center_outer_points = self.get_arc_points(
                center_x,
                center_y,
                center_outer_radius,
                dash_start_theta,
                dash_end_theta,
            )
            center_inner_points = self.get_arc_points(
                center_x,
                center_y,
                center_inner_radius,
                dash_start_theta,
                dash_end_theta,
            )
            center_polygon = center_outer_points + list(reversed(center_inner_points))
            self.draw_segment_polygon(center_polygon, CENTER_LINE_COLOR)

    def draw_road_segment(self, segment: dict[str, float | list[tuple[float, float]]]) -> None:
        if segment["kind"] == "turn":
            self.draw_turn_segment(segment)
            return

        self.draw_straight_segment(segment)

    def draw_scenery(self) -> None:
        for item in self.scenery_objects:
            x = int(round(float(item["x"])))
            y = int(round(float(item["y"])))
            radius = int(round(float(item["radius"])))
            color = item["color"]  # type: ignore[assignment]

            if item["kind"] == "tree":
                trunk_width = max(8, radius // 2)
                trunk_height = max(16, int(radius * 1.35))
                trunk_rect = pygame.Rect(
                    x - trunk_width // 2,
                    y - trunk_height // 4,
                    trunk_width,
                    trunk_height,
                )
                pygame.draw.ellipse(
                    self.canvas,
                    (20, 70, 20),
                    (x - radius, y - radius // 2, radius * 2, radius),
                )
                pygame.draw.rect(self.canvas, TREE_TRUNK_COLOR, trunk_rect)
                pygame.draw.circle(self.canvas, color, (x, y - trunk_height // 2), radius)
                pygame.draw.circle(self.canvas, color, (x - radius // 2, y - trunk_height // 2 + 6), int(radius * 0.7))
                pygame.draw.circle(self.canvas, color, (x + radius // 2, y - trunk_height // 2 + 6), int(radius * 0.68))
            else:
                pygame.draw.ellipse(
                    self.canvas,
                    (28, 92, 28),
                    (x - radius, y - radius // 3, radius * 2, int(radius * 0.9)),
                )
                pygame.draw.circle(self.canvas, color, (x - radius // 2, y), int(radius * 0.7))
                pygame.draw.circle(self.canvas, color, (x + radius // 2, y), int(radius * 0.72))
                pygame.draw.circle(self.canvas, color, (x, y - radius // 3), radius)

    def draw_track_obstacles(self) -> None:
        for obstacle in self.track_obstacles:
            polygon = self.get_track_obstacle_polygon(obstacle)
            if obstacle["kind"] == "hay_bale":
                self.draw_segment_polygon(polygon, HAY_BALE_COLOR)
                center_x, center_y = self.get_track_obstacle_center(obstacle)
                tangent_x = float(obstacle["tangent_x"])
                tangent_y = float(obstacle["tangent_y"])
                normal_x = float(obstacle["normal_x"])
                normal_y = float(obstacle["normal_y"])
                band_half_length = float(obstacle["half_length"]) * 0.42
                band_half_width = float(obstacle["half_width"]) * 0.84
                band_polygon = [
                    (
                        center_x + tangent_x * band_half_length + normal_x * band_half_width,
                        center_y + tangent_y * band_half_length + normal_y * band_half_width,
                    ),
                    (
                        center_x + tangent_x * band_half_length - normal_x * band_half_width,
                        center_y + tangent_y * band_half_length - normal_y * band_half_width,
                    ),
                    (
                        center_x - tangent_x * band_half_length - normal_x * band_half_width,
                        center_y - tangent_y * band_half_length - normal_y * band_half_width,
                    ),
                    (
                        center_x - tangent_x * band_half_length + normal_x * band_half_width,
                        center_y - tangent_y * band_half_length + normal_y * band_half_width,
                    ),
                ]
                self.draw_segment_polygon(band_polygon, HAY_BALE_SHADOW_COLOR)
                continue

            self.draw_segment_polygon(polygon, MOVING_BARRIER_COLOR)
            center_x, center_y = self.get_track_obstacle_center(obstacle)
            tangent_x = float(obstacle["tangent_x"])
            tangent_y = float(obstacle["tangent_y"])
            normal_x = float(obstacle["normal_x"])
            normal_y = float(obstacle["normal_y"])
            half_length = float(obstacle["half_length"])
            half_width = float(obstacle["half_width"])

            stripe_count = 4
            stripe_step = (half_width * 2) / stripe_count
            for stripe_index in range(stripe_count):
                stripe_start = -half_width + stripe_index * stripe_step
                stripe_end = stripe_start + stripe_step * 0.56
                stripe_polygon = [
                    (
                        center_x + tangent_x * half_length + normal_x * stripe_end,
                        center_y + tangent_y * half_length + normal_y * stripe_end,
                    ),
                    (
                        center_x + tangent_x * half_length + normal_x * stripe_start,
                        center_y + tangent_y * half_length + normal_y * stripe_start,
                    ),
                    (
                        center_x - tangent_x * half_length + normal_x * stripe_start,
                        center_y - tangent_y * half_length + normal_y * stripe_start,
                    ),
                    (
                        center_x - tangent_x * half_length + normal_x * stripe_end,
                        center_y - tangent_y * half_length + normal_y * stripe_end,
                    ),
                ]
                self.draw_segment_polygon(stripe_polygon, MOVING_BARRIER_STRIPE_COLOR)

            post_offset = half_width + 8
            for sign in (-1.0, 1.0):
                post_center_x = center_x + normal_x * post_offset * sign
                post_center_y = center_y + normal_y * post_offset * sign
                pygame.draw.circle(
                    self.canvas,
                    MOVING_BARRIER_POST_COLOR,
                    (int(round(post_center_x)), int(round(post_center_y))),
                    6,
                )

    def draw_finish_line(self) -> None:
        center_x = float(self.finish_state["center_x"])
        center_y = float(self.finish_state["center_y"])
        tangent_x = float(self.finish_state["tangent_x"])
        tangent_y = float(self.finish_state["tangent_y"])
        normal_x = float(self.finish_state["normal_x"])
        normal_y = float(self.finish_state["normal_y"])
        half_width = float(self.finish_state["half_width"])
        half_length = float(self.finish_state["half_length"])

        finish_polygon = [
            (
                center_x + normal_x * half_width + tangent_x * half_length,
                center_y + normal_y * half_width + tangent_y * half_length,
            ),
            (
                center_x - normal_x * half_width + tangent_x * half_length,
                center_y - normal_y * half_width + tangent_y * half_length,
            ),
            (
                center_x - normal_x * half_width - tangent_x * half_length,
                center_y - normal_y * half_width - tangent_y * half_length,
            ),
            (
                center_x + normal_x * half_width - tangent_x * half_length,
                center_y + normal_y * half_width - tangent_y * half_length,
            ),
        ]
        self.draw_segment_polygon(finish_polygon, (245, 245, 245))

        tile_rows = max(2, int(math.ceil((half_width * 2) / FINISH_TILE_SIZE)))
        tile_cols = max(2, int(math.ceil(FINISH_LINE_LENGTH / FINISH_TILE_SIZE)))
        row_step = (half_width * 2) / tile_rows
        col_step = FINISH_LINE_LENGTH / tile_cols

        for row in range(tile_rows):
            for col in range(tile_cols):
                color = (20, 20, 20) if (row + col) % 2 == 0 else (245, 245, 245)
                side_a = -half_width + row * row_step
                side_b = side_a + row_step
                along_a = -half_length + col * col_step
                along_b = along_a + col_step
                tile_polygon = [
                    (
                        center_x + normal_x * side_a + tangent_x * along_b,
                        center_y + normal_y * side_a + tangent_y * along_b,
                    ),
                    (
                        center_x + normal_x * side_b + tangent_x * along_b,
                        center_y + normal_y * side_b + tangent_y * along_b,
                    ),
                    (
                        center_x + normal_x * side_b + tangent_x * along_a,
                        center_y + normal_y * side_b + tangent_y * along_a,
                    ),
                    (
                        center_x + normal_x * side_a + tangent_x * along_a,
                        center_y + normal_y * side_a + tangent_y * along_a,
                    ),
                ]
                self.draw_segment_polygon(tile_polygon, color)

    def draw_lasers(self) -> None:
        if not self.show_lasers:
            return

        origin = (
            int(round(float(self.car_state["x"]))),
            int(round(float(self.car_state["y"]))),
        )
        for hit_point in self.last_laser_points:
            hit_x = int(round(hit_point[0]))
            hit_y = int(round(hit_point[1]))
            pygame.draw.line(self.canvas, LASER_COLOR, origin, (hit_x, hit_y), 2)
            pygame.draw.circle(self.canvas, LASER_HIT_COLOR, (hit_x, hit_y), 4)

    def draw_car(self) -> None:
        car_polygon = self.get_car_polygon()
        self.draw_segment_polygon(car_polygon, CAR_COLOR)

        heading_radians = math.radians(float(self.car_state["heading"]))
        forward_x = math.cos(heading_radians)
        forward_y = math.sin(heading_radians)
        side_x = -forward_y
        side_y = forward_x
        center_x = float(self.car_state["x"])
        center_y = float(self.car_state["y"])

        windshield = [
            (
                center_x + forward_x * 10 + side_x * 8,
                center_y + forward_y * 10 + side_y * 8,
            ),
            (
                center_x + forward_x * 10 - side_x * 8,
                center_y + forward_y * 10 - side_y * 8,
            ),
            (
                center_x + forward_x * 2 - side_x * 10,
                center_y + forward_y * 2 - side_y * 10,
            ),
            (
                center_x + forward_x * 2 + side_x * 10,
                center_y + forward_y * 2 + side_y * 10,
            ),
        ]
        self.draw_segment_polygon(windshield, CAR_WINDOW_COLOR)

        tire_centers = [
            (
                center_x + forward_x * 14 + side_x * 13,
                center_y + forward_y * 14 + side_y * 13,
            ),
            (
                center_x + forward_x * 14 - side_x * 13,
                center_y + forward_y * 14 - side_y * 13,
            ),
            (
                center_x - forward_x * 14 + side_x * 13,
                center_y - forward_y * 14 + side_y * 13,
            ),
            (
                center_x - forward_x * 14 - side_x * 13,
                center_y - forward_y * 14 - side_y * 13,
            ),
        ]
        for tire_x, tire_y in tire_centers:
            pygame.draw.circle(self.canvas, CAR_TIRE_COLOR, (int(round(tire_x)), int(round(tire_y))), 5)

    def draw_hud(self) -> None:
        if self.hud_font is None or self.hud_small_font is None:
            return

        overlay_height = 120 if not self.hud_title_text else 168
        overlay = pygame.Surface((400, overlay_height), pygame.SRCALPHA)
        overlay.fill(HUD_PANEL_COLOR)
        self.canvas.blit(overlay, (24, 24))

        speed_value = abs(float(self.car_state["speed"])) * CAR_SPEED_TO_KMH
        time_value = float(self.car_state["elapsed_time"])
        minutes = int(time_value // 60)
        seconds = time_value % 60

        speed_text = self.hud_font.render(f"{int(speed_value):03d} km/h", True, HUD_TEXT_COLOR)
        time_text = self.hud_small_font.render(f"Time {minutes:02d}:{seconds:05.2f}", True, HUD_TEXT_COLOR)
        footer_text = self.hud_small_font.render(self.hud_footer_text, True, HUD_ACCENT_COLOR)

        self.canvas.blit(speed_text, (42, 36))
        self.canvas.blit(time_text, (44, 84))
        self.canvas.blit(footer_text, (24, CANVAS_HEIGHT - 52))

        if self.hud_title_text:
            title_text = self.hud_small_font.render(self.hud_title_text, True, HUD_ACCENT_COLOR)
            self.canvas.blit(title_text, (42, 124))

        if bool(self.car_state["finished"]):
            finish_text = self.hud_font.render("FINISHED", True, HUD_ACCENT_COLOR)
            finish_rect = finish_text.get_rect(center=(CANVAS_WIDTH // 2, 92))
            self.canvas.blit(finish_text, finish_rect)
        elif bool(self.car_state["crashed"]):
            crash_text = self.hud_font.render("CRASHED", True, HUD_DANGER_COLOR)
            crash_rect = crash_text.get_rect(center=(CANVAS_WIDTH // 2, 92))
            self.canvas.blit(crash_text, crash_rect)

    def draw_road(self) -> None:
        for segment in self.road_segments:
            self.draw_road_segment(segment)

        self.draw_finish_line()

    def render_frame(self) -> pygame.Surface:
        self.canvas.fill(BACKGROUND_COLOR)
        self.draw_scenery()
        self.draw_road()
        self.draw_track_obstacles()
        self.draw_lasers()
        self.draw_car()
        self.draw_hud()
        return self.canvas

    def render(self) -> None:
        self.render_frame()
        if self.screen is None:
            return
        scaled_canvas = pygame.transform.smoothscale(
            self.canvas,
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


def run_simulation() -> None:
    CarDrivingSimulation().run()


if __name__ == "__main__":
    run_simulation()
