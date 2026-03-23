from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.builders.ai_car_driving_builder.rl_trainer_neural import build_replay_video_neural
from src.builders.fighting_balls_builder.video_generator_power import run_headless
from src.pipeline.upload_short import upload_short

AI_CAR_DRIVING_TITLE = "ai learns to drive a car"


def build_and_upload_short(
    title: str,
    n_balls: int | None = 6,
) -> dict[str, Any]:
    output_dir = Path("uploaded_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{run_headless.__name__}_{timestamp}.mp4"
    video_path = output_dir / video_filename

    run_headless(n_balls=n_balls, output=str(video_path))
    return upload_short(video_path=video_path, title=title)


def build_and_upload_ai_car_driving_short() -> dict[str, Any]:
    output_dir = Path("uploaded_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"ai_learns_to_drive_a_car_{timestamp}.mp4"
    video_path = output_dir / video_filename

    build_replay_video_neural(output_path=video_path)
    return upload_short(
        video_path=video_path,
        title=AI_CAR_DRIVING_TITLE,
        description=AI_CAR_DRIVING_TITLE,
    )
