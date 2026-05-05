#!/usr/bin/env python3
"""CLI for remote GPU video processing via backend API."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import mimetypes
import os
import sys
from pathlib import Path

import httpx

CONFIG_DIR = Path.home() / ".config" / "skating-cli"
CREDS_PATH = CONFIG_DIR / "credentials.json"

BACKEND_URL = os.environ.get("SKATING_BACKEND_URL", "http://localhost:8000")
API_BASE = f"{BACKEND_URL}/api/v1"

CHUNK_SIZE = 5 * 1024 * 1024
POLL_INTERVAL = 2.0


class CLIError(Exception):
    """CLI error with exit code."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


def _load_credentials() -> dict | None:
    if not CREDS_PATH.exists():
        return None
    try:
        return json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_credentials(creds: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CREDS_PATH.write_text(json.dumps(creds), encoding="utf-8")
    CREDS_PATH.chmod(0o600)


async def _ensure_access_token(client: httpx.AsyncClient) -> str:
    creds = _load_credentials()
    if creds is None:
        raise CLIError("Not logged in. Run 'skating login' first.", code=2)

    access_token = creds.get("access_token")
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}

    resp = await client.get(f"{API_BASE}/users/me", headers=headers)
    if resp.status_code == 401:
        refresh_token = creds.get("refresh_token")
        if not refresh_token:
            raise CLIError("Session expired. Run 'skating login'.", code=2)
        refresh_resp = await client.post(
            f"{API_BASE}/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        if refresh_resp.status_code != 200:
            raise CLIError("Session expired. Run 'skating login'.", code=2)
        data = refresh_resp.json()
        new_creds = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", refresh_token),
        }
        _save_credentials(new_creds)
        access_token = new_creds["access_token"]
    elif resp.status_code != 200:
        raise CLIError(f"Auth check failed: {resp.status_code}", code=2)

    return access_token


async def login(args: argparse.Namespace) -> None:
    email = input("Email: ")
    password = getpass.getpass("Password: ")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{API_BASE}/auth/login",
            json={"email": email, "password": password},
        )

    if resp.status_code != 200:
        raise CLIError(f"Login failed: {resp.status_code} {resp.text}")

    data = resp.json()
    _save_credentials(
        {
            "access_token": data["access_token"],
            "refresh_token": data["refresh_token"],
        }
    )
    print(json.dumps({"status": "ok"}))


async def upload_video(client: httpx.AsyncClient, access_token: str, video_path: str) -> str:
    path = Path(video_path)
    if not path.exists():
        raise CLIError(f"File not found: {video_path}")

    total_size = path.stat().st_size
    filename = path.name
    content_type, _ = mimetypes.guess_type(str(path))
    content_type = content_type or "application/octet-stream"

    # Init upload
    resp = await client.post(
        f"{API_BASE}/uploads/init",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"file_name": filename, "content_type": content_type, "total_size": total_size},
    )
    if resp.status_code != 200:
        raise CLIError(f"Upload init failed: {resp.status_code} {resp.text}")

    init_data = resp.json()
    upload_id = init_data["upload_id"]
    key = init_data["key"]
    parts = init_data["parts"]

    # Upload chunks
    etags: dict[int, str] = {}
    chunk_size = init_data.get("chunk_size", CHUNK_SIZE)
    with path.open("rb") as f:
        for part in parts:
            chunk = f.read(chunk_size)
            put_resp = await client.put(part["url"], content=chunk)
            if put_resp.status_code not in (200, 204):
                raise CLIError(f"Chunk upload failed: {put_resp.status_code} {put_resp.text}")
            etags[part["part_number"]] = put_resp.headers.get("etag", "")
            print(f"Uploaded part {part['part_number']}/{len(parts)}", file=sys.stderr)

    # Complete upload
    parts_payload = [{"part_number": n, "etag": etag} for n, etag in etags.items()]
    complete_resp = await client.post(
        f"{API_BASE}/uploads/complete",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"upload_id": upload_id, "key": key, "parts": parts_payload},
    )
    if complete_resp.status_code != 200:
        raise CLIError(f"Upload complete failed: {complete_resp.status_code} {complete_resp.text}")

    complete_data = complete_resp.json()
    return complete_data["key"]


async def analyze(args: argparse.Namespace) -> None:
    video_path: str = args.video
    element: str = args.element

    # Parse person_click
    person_click = {"x": 0, "y": 0}
    if args.person_click:
        try:
            x_str, y_str = args.person_click.split(",")
            person_click = {"x": int(x_str), "y": int(y_str)}
        except ValueError as exc:
            raise CLIError("--person-click must be in format 'x,y'") from exc

    async with httpx.AsyncClient() as client:
        access_token = await _ensure_access_token(client)

        print("Uploading video...", file=sys.stderr)
        video_key = await upload_video(client, access_token, video_path)

        payload = {
            "video_key": video_key,
            "person_click": person_click,
            "frame_skip": args.frame_skip,
            "layer": args.layer,
            "tracking": args.tracking,
            "export": args.export,
            "session_id": args.session_id,
            "depth": args.depth,
            "optical_flow": args.optical_flow,
            "segment": args.segment,
            "foot_track": args.foot_track,
            "matting": args.matting,
            "inpainting": args.inpainting,
        }

        print("Enqueuing task...", file=sys.stderr)
        resp = await client.post(
            f"{API_BASE}/process/queue",
            headers={"Authorization": f"Bearer {access_token}"},
            json=payload,
        )
        if resp.status_code != 200:
            raise CLIError(f"Enqueue failed: {resp.status_code} {resp.text}")

        task_data = resp.json()
        task_id = task_data["task_id"]
        print(f"Task ID: {task_id}", file=sys.stderr)

        # Poll status
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            status_resp = await client.get(
                f"{API_BASE}/process/{task_id}/status",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if status_resp.status_code == 401:
                raise CLIError("Auth expired during polling.", code=2)
            if status_resp.status_code != 200:
                raise CLIError(f"Status check failed: {status_resp.status_code} {status_resp.text}")

            status_data = status_resp.json()
            status = status_data.get("status")
            print(f"Status: {status}", file=sys.stderr)

            if status == "completed":
                print(json.dumps(status_data.get("result", {})))
                return
            if status in ("failed", "cancelled"):
                error_msg = status_data.get("error", "Unknown error")
                print(f"Task {status}: {error_msg}", file=sys.stderr)
                sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="skating")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # login
    subparsers.add_parser("login", help="Authenticate with the backend")

    # analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a video")
    analyze_parser.add_argument("video", help="Path to video file")
    analyze_parser.add_argument("--element", default="waltz_jump", help="Element to analyze")
    analyze_parser.add_argument(
        "--person-click", default=None, help="Person click coordinates 'x,y'"
    )
    analyze_parser.add_argument("--frame-skip", type=int, default=1)
    analyze_parser.add_argument("--layer", type=int, default=3)
    analyze_parser.add_argument("--tracking", default="auto")
    analyze_parser.add_argument("--export", action="store_true", default=True)
    analyze_parser.add_argument("--session-id", type=str, default=None)
    analyze_parser.add_argument("--depth", action="store_true", default=False)
    analyze_parser.add_argument("--optical-flow", action="store_true", default=False)
    analyze_parser.add_argument("--segment", action="store_true", default=False)
    analyze_parser.add_argument("--foot-track", action="store_true", default=False)
    analyze_parser.add_argument("--matting", action="store_true", default=False)
    analyze_parser.add_argument("--inpainting", action="store_true", default=False)

    args = parser.parse_args()

    try:
        if args.command == "login":
            asyncio.run(login(args))
        elif args.command == "analyze":
            asyncio.run(analyze(args))
    except CLIError as e:
        print(str(e), file=sys.stderr)
        sys.exit(e.code)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
