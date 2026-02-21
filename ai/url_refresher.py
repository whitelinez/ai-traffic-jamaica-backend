"""
ai/url_refresher.py — Proactively fetches a fresh HLS stream URL from ipcamlive.

Reverse-engineered flow from ipcamlive player network traffic:
  1. Generate a random 32-byte token (base64) — client-side, like the JS player does
  2. GET registerviewer.php?alias=X  → {"result":"ok","data":{"viewerid":123456}}
  3. GET getcamerastreamstate.php?alias=X&token=...&viewerid=...
       → {"details":{"address":"http://sXX.ipcamlive.com/","streamid":"TOKEN","streamavailable":"1",...}}
  4. Build URL: https://sXX.ipcamlive.com/streams/TOKEN/stream.m3u8

Runs as a background asyncio task. Refreshes every URL_REFRESH_INTERVAL seconds
(default 240s — well before typical session expiry). Updates Supabase cameras table
and exposes get_current_url() so the AI loop can hot-reload without restart.
"""
import asyncio
import base64
import logging
import os
import time

import httpx

logger = logging.getLogger(__name__)

IPCAM_API_BASE = "https://g3.ipcamlive.com/player"
_REQUEST_HEADERS = {
    "Referer": "https://g3.ipcamlive.com/player/player.php",
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 13; SM-G981B) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
}

# Module-level current URL — written by refresh loop, read by AI loop
_current_url: str | None = None


def get_current_url() -> str | None:
    """Return the most recently fetched live stream URL."""
    return _current_url


def _make_token() -> str:
    """Generate a random 32-byte session token (base64), matching what the ipcamlive JS player does."""
    return base64.b64encode(os.urandom(32)).decode()


def _build_stream_url(details: dict) -> str | None:
    """
    Construct the HLS manifest URL from getcamerastreamstate response fields.
    Returns None if the stream is offline or required fields are missing.
    """
    if details.get("streamavailable") != "1":
        logger.warning("Camera reports streamavailable=0 — stream offline")
        return None

    address = details.get("address", "").rstrip("/")
    streamid = details.get("streamid", "")

    if not address or not streamid:
        logger.warning("Missing address or streamid in response: %s", details)
        return None

    # ipcamlive returns http:// — upgrade to https
    address = address.replace("http://", "https://")
    return f"{address}/streams/{streamid}/stream.m3u8"


async def fetch_fresh_stream_url(alias: str) -> str | None:
    """
    Execute the two-step ipcamlive API flow and return a fresh HLS URL.
    Returns None on any failure so the caller can keep the existing URL.
    """
    token = _make_token()

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Step 1: register viewer → get viewerid
        ts = int(time.time() * 1000)
        reg = await client.get(
            f"{IPCAM_API_BASE}/registerviewer.php",
            params={
                "_": ts,
                "alias": alias,
                "type": "HTML5",
                "browser": "Chrome Mobile",
                "browser_ver": "145.0.0.0",
                "os": "Android",
                "os_ver": "13",
                "streaming": "hls",
            },
            headers={**_REQUEST_HEADERS, "Referer": f"https://g3.ipcamlive.com/player/player.php?alias={alias}&autoplay=1"},
        )
        reg.raise_for_status()
        reg_data = reg.json()

        if reg_data.get("result") != "ok":
            logger.warning("registerviewer.php failed: %s", reg_data)
            return None

        viewerid = reg_data.get("data", {}).get("viewerid")

        # Step 2: get stream state → address + streamid
        ts = int(time.time() * 1000)
        state = await client.get(
            f"{IPCAM_API_BASE}/getcamerastreamstate.php",
            params={
                "_": ts,
                "token": token,
                "alias": alias,
                "targetdomain": "g3.ipcamlive.com",
                "viewerid": viewerid,
            },
            headers={**_REQUEST_HEADERS, "Referer": f"https://g3.ipcamlive.com/player/player.php?alias={alias}&autoplay=1"},
        )
        state.raise_for_status()
        state_data = state.json()

        details = state_data.get("details", {})
        url = _build_stream_url(details)

        if url:
            logger.info("Fresh stream URL: %s", url)
        return url


async def _supabase_update_stream_url(alias: str, url: str) -> bool:
    """
    Update cameras.stream_url via direct REST PATCH (bypasses supabase-py client quirks).
    Returns True on success.
    """
    from config import get_config
    cfg = get_config()

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.patch(
            f"{cfg.SUPABASE_URL}/rest/v1/cameras",
            params={"ipcam_alias": f"eq.{alias}"},
            json={"stream_url": url},
            headers={
                "apikey": cfg.SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {cfg.SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
        )
    logger.info("Supabase PATCH status=%d body=%s", resp.status_code, resp.text[:300])
    return resp.status_code in (200, 204)


async def url_refresh_loop(alias: str, interval_seconds: int = 240) -> None:
    """
    Background task: fetch a fresh URL immediately on startup, then every
    `interval_seconds`. Updates _current_url and the Supabase cameras table.

    Must be started before the AI loop so the URL is ready on first use.
    """
    global _current_url

    while True:
        try:
            url = await fetch_fresh_stream_url(alias)
            if url:
                _current_url = url
                await _supabase_update_stream_url(alias, url)
            else:
                logger.warning("Refresh returned no URL — keeping existing (alias=%s)", alias)
        except Exception as exc:
            logger.error("URL refresh error (alias=%s): %s", alias, exc)

        await asyncio.sleep(interval_seconds)
