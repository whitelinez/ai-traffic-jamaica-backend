"""
services/demo_recorder.py — One-shot stream recorder for demo footage.

Starts an ffmpeg subprocess that captures the live HLS stream for a given
duration, saves a compressed H.264 MP4 to /tmp, then uploads it to the
Supabase Storage bucket 'demo-videos' and returns the public URL.

Usage (from admin router):
    result = await demo_recorder.start_recording(duration_sec=600, cfg=cfg)
    status = demo_recorder.get_status()
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import httpx

from ai.url_refresher import get_current_url, fetch_fresh_stream_url, get_current_alias

logger = logging.getLogger(__name__)

_BUCKET = "demo-videos"

# ── Module-level state ────────────────────────────────────────────────────────
_task: asyncio.Task | None = None
_status: dict = {"state": "idle", "progress": 0, "url": None, "error": None, "started_at": None}


def get_status() -> dict:
    return dict(_status)


async def start_recording(duration_sec: int, cfg) -> dict:
    """Kick off a background recording task. Returns immediately."""
    global _task
    if _status["state"] == "recording":
        return {"ok": False, "error": "Already recording", "status": _status}
    _task = asyncio.create_task(_record(duration_sec, cfg))
    return {"ok": True, "message": f"Recording started for {duration_sec}s"}


# ── Internal ──────────────────────────────────────────────────────────────────

async def _record(duration_sec: int, cfg) -> None:
    global _status
    _status = {"state": "recording", "progress": 0, "url": None, "error": None, "started_at": time.time()}

    out_path = f"/tmp/demo_{int(time.time())}.mp4"
    try:
        # ── 1. Get stream URL ─────────────────────────────────────────────────
        stream_url = get_current_url()
        if not stream_url:
            alias = get_current_alias()
            if alias:
                stream_url = await fetch_fresh_stream_url(alias)
        if not stream_url:
            _status.update({"state": "error", "error": "No stream URL available"})
            return

        logger.info("[demo_recorder] Recording %ds → %s", duration_sec, out_path)

        # ── 2. ffmpeg capture ─────────────────────────────────────────────────
        # -t duration  : stop after N seconds
        # -c:v libx264 : H.264 for broad browser/player compat
        # -preset fast : good quality/speed trade-off on CPU
        # -crf 26      : ~720p quality; raises to 28 on slow CPU to save space
        # -vf scale    : cap at 1280 wide to keep file size sane
        # -an          : no audio (stream is usually silent)
        cmd = [
            "ffmpeg", "-y",
            "-i", stream_url,
            "-t", str(duration_sec),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "26",
            "-vf", "scale='min(1280,iw)':-2",
            "-an",
            "-movflags", "+faststart",
            out_path,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Poll progress via a background watcher while ffmpeg runs
        async def _watch():
            elapsed = 0
            while proc.returncode is None:
                await asyncio.sleep(10)
                elapsed += 10
                _status["progress"] = min(90, int(elapsed / duration_sec * 90))

        watcher = asyncio.create_task(_watch())
        _, stderr_bytes = await proc.communicate()
        watcher.cancel()

        if proc.returncode != 0:
            err = stderr_bytes.decode(errors="replace")[-400:]
            logger.error("[demo_recorder] ffmpeg failed: %s", err)
            _status.update({"state": "error", "error": f"ffmpeg exit {proc.returncode}: {err}"})
            return

        file_size = Path(out_path).stat().st_size
        logger.info("[demo_recorder] Recorded %d bytes. Uploading…", file_size)
        _status["progress"] = 92

        # ── 3. Upload to Supabase storage ─────────────────────────────────────
        ts = int(time.time())
        remote_path = f"demo_{ts}.mp4"
        upload_url = f"{cfg.SUPABASE_URL.rstrip('/')}/storage/v1/object/{_BUCKET}/{remote_path}"

        async with httpx.AsyncClient(timeout=300) as client:
            with open(out_path, "rb") as fh:
                resp = await client.post(
                    upload_url,
                    content=fh.read(),
                    headers={
                        "Authorization": f"Bearer {cfg.SUPABASE_SERVICE_ROLE_KEY}",
                        "Content-Type": "video/mp4",
                        "x-upsert": "true",
                    },
                )

        if resp.status_code not in (200, 201):
            err = resp.text[:300]
            logger.error("[demo_recorder] Upload failed %d: %s", resp.status_code, err)
            _status.update({"state": "error", "error": f"Upload {resp.status_code}: {err}"})
            return

        pub_url = f"{cfg.SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{_BUCKET}/{remote_path}"
        logger.info("[demo_recorder] Done. Public URL: %s", pub_url)
        _status.update({"state": "done", "progress": 100, "url": pub_url, "error": None})

    except Exception as exc:
        logger.exception("[demo_recorder] Unexpected error")
        _status.update({"state": "error", "error": str(exc)})
    finally:
        try:
            os.unlink(out_path)
        except FileNotFoundError:
            pass
