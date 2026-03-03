from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.builders.fighting_balls_builder.video_generator import run_headless
from src.pipeline.upload_short import upload_short


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
