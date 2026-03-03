from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
DEFAULT_CLIENT_SECRETS_DIR = Path("src/core")
DEFAULT_TOKEN_FILE = Path("youtube_token.json")
DEFAULT_UPLOAD_COUNT_FILE = Path("uploaded_videos_count.json")


def upload_short(video_path: str | Path, title: str) -> dict[str, Any]:
    source_path = Path(video_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Video file not found: {source_path}")

    upload_number = get_uploaded_video_count() + 1
    description = f"fighting simulation {upload_number}"
    youtube = get_youtube_client()

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": "22",
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=MediaFileUpload(str(source_path), chunksize=-1, resumable=True),
    )

    response = execute_upload(request)
    increment_uploaded_video_count()

    return {
        "video_id": response["id"],
        "title": title,
        "description": description,
        "privacy_status": "public",
        "upload_number": upload_number,
        "video_path": str(source_path),
    }


def run(video_path: str, title: str) -> dict[str, Any]:
    return upload_short(video_path=video_path, title=title)


def get_uploaded_video_count() -> int:
    state = load_upload_state()
    return int(state.get("uploaded_count", 0))


def increment_uploaded_video_count() -> int:
    state = load_upload_state()
    new_count = int(state.get("uploaded_count", 0)) + 1
    state["uploaded_count"] = new_count

    state_file = get_upload_count_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return new_count


def load_upload_state() -> dict[str, Any]:
    state_file = get_upload_count_file()
    if not state_file.exists():
        return {"uploaded_count": 0}
    return json.loads(state_file.read_text(encoding="utf-8"))


def get_upload_count_file() -> Path:
    value = os.getenv("YOUTUBE_UPLOAD_COUNT_FILE")
    return Path(value) if value else DEFAULT_UPLOAD_COUNT_FILE


def get_youtube_client() -> Any:
    token_file = get_token_file()
    creds: Credentials | None = None

    if token_file.exists():
        creds = Credentials.from_authorized_user_file(
            str(token_file),
            scopes=[YOUTUBE_UPLOAD_SCOPE],
        )

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    elif not creds or not creds.valid:
        client_secrets_file = get_client_secrets_file()
        if not client_secrets_file.is_file():
            raise FileNotFoundError(
                f"Client secrets file not found: {client_secrets_file}"
            )

        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_file),
            scopes=[YOUTUBE_UPLOAD_SCOPE],
        )
        creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds)


def get_client_secrets_file() -> Path:
    value = os.getenv("YOUTUBE_CLIENT_SECRETS_FILE")
    if value:
        return Path(value)

    matches = sorted(DEFAULT_CLIENT_SECRETS_DIR.glob("client_secret*.json"))
    if matches:
        return matches[0]

    return DEFAULT_CLIENT_SECRETS_DIR / "client_secrets.json"


def get_token_file() -> Path:
    value = os.getenv("YOUTUBE_TOKEN_FILE")
    return Path(value) if value else DEFAULT_TOKEN_FILE


def execute_upload(request: Any) -> dict[str, Any]:
    response: dict[str, Any] | None = None

    while response is None:
        _, response = request.next_chunk()

    if "id" not in response:
        raise RuntimeError("YouTube upload completed without a video id.")

    return response
