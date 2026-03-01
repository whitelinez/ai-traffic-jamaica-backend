"""
ai/url_refresher.py - Proactively fetches fresh HLS URLs from ipcamlive.

Flow:
  1. registerviewer.php (get viewerid)
  2. getcamerastreamstate.php (get stream state/details)
  3. Build HLS URL and validate it before use.
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

_current_url: str | None = None
_current_alias: str | None = None


def get_current_url() -> str | None:
    return _current_url


def get_current_alias() -> str | None:
    return _current_alias


def _make_token() -> str:
    return base64.b64encode(os.urandom(32)).decode()


def _build_stream_url(details: dict) -> str | None:
    from urllib.parse import urlparse, urlunparse
    if str(details.get("streamavailable", "")) != "1":
        logger.warning("Camera reports streamavailable=0")
        return None

    address  = str(details.get("address", "")).strip().rstrip("/")
    streamid = str(details.get("streamid", "")).strip()
    if not address or not streamid:
        logger.warning("Missing address/streamid in stream state details")
        return None

    # Force HTTPS using proper URL parsing — avoids partial replace bugs
    try:
        parsed = urlparse(address if "://" in address else f"https://{address}")
        if parsed.scheme not in ("http", "https"):
            logger.warning("Unexpected stream address scheme: %s", parsed.scheme)
            return None
        safe_address = urlunparse(parsed._replace(scheme="https"))
        url = f"{safe_address.rstrip('/')}/streams/{streamid}/stream.m3u8"
        # Final sanity check — must be an absolute https URL
        final = urlparse(url)
        if final.scheme != "https" or not final.netloc:
            logger.warning("Rejecting malformed stream URL: %s", url)
            return None
        return url
    except Exception as exc:
        logger.warning("Failed to build stream URL: %s", exc)
        return None


async def _validate_manifest(client: httpx.AsyncClient, url: str) -> bool:
    try:
        resp = await client.get(url, headers=_REQUEST_HEADERS, follow_redirects=True)
    except Exception as exc:
        logger.warning("Manifest validation network failure: %s", exc)
        return False
    if resp.status_code != 200:
        logger.warning("Manifest validation failed status=%s url=%s", resp.status_code, url)
        return False
    text = (resp.text or "").strip()
    return "#EXTM3U" in text


async def fetch_fresh_stream_url(alias: str) -> str | None:
    token = _make_token()
    async with httpx.AsyncClient(timeout=15.0) as client:
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
            logger.warning("registerviewer failed for alias=%s: %s", alias, reg_data)
            return None
        viewerid = reg_data.get("data", {}).get("viewerid")

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
        details = (state.json() or {}).get("details", {})
        url = _build_stream_url(details)
        if not url:
            return None
        if not await _validate_manifest(client, url):
            logger.warning("Discarded invalid manifest URL for alias=%s", alias)
            return None
        return url


async def _supabase_update_stream_url(alias: str, url: str) -> bool:
    # Use the Supabase async SDK client — service role key stays off the wire
    from supabase_client import get_supabase
    try:
        sb = await get_supabase()
        await (
            sb.table("cameras")
            .update({"stream_url": url})
            .eq("ipcam_alias", alias)
            .execute()
        )
        return True
    except Exception as exc:
        logger.warning("Failed to persist stream URL for alias=%s: %s", alias, exc)
        return False


async def get_candidate_aliases(primary_alias: str | None = None) -> list[str]:
    """
    Build an ordered list of aliases to try for stream URL resolution.

    Priority:
      1. primary_alias (if given) — e.g. the alias explicitly requested
      2. The current is_active camera from Supabase (source of truth)
      3. CAMERA_ALIAS env var (static fallback)
      4. CAMERA_ALIASES env var list
    """
    from config import get_config
    cfg = get_config()
    out: list[str] = []

    def _push(v: str | None) -> None:
        s = str(v or "").strip()
        if s and s not in out:
            out.append(s)

    if primary_alias:
        _push(primary_alias)

    # Supabase is_active cameras are the authoritative source — put them first
    try:
        from supabase_client import get_supabase
        sb = await get_supabase()
        resp = await (
            sb.table("cameras")
            .select("ipcam_alias,created_at")
            .eq("is_active", True)
            .order("created_at", desc=True)
            .execute()
        )
        for row in resp.data or []:
            _push(row.get("ipcam_alias"))
    except Exception as exc:
        logger.debug("Could not load aliases from cameras table: %s", exc)

    # Env var aliases as fallback
    _push(cfg.CAMERA_ALIAS)
    for alias in getattr(cfg, "CAMERA_ALIASES", []) or []:
        _push(alias)

    return out


async def url_refresh_loop(alias: str, interval_seconds: int = 240) -> None:
    global _current_url, _current_alias
    while True:
        try:
            # Always re-query Supabase so switching is_active is reflected
            # without a backend restart.
            aliases = await get_candidate_aliases()
            if not aliases:
                aliases = [alias]

            selected_alias = None
            selected_url = None
            for candidate in aliases:
                url = await fetch_fresh_stream_url(candidate)
                if not url:
                    continue
                selected_alias = candidate
                selected_url = url
                await _supabase_update_stream_url(candidate, url)
                break

            if selected_url and selected_alias:
                _current_url = selected_url
                _current_alias = selected_alias
                logger.info("URL refresh selected alias=%s", selected_alias)
            else:
                logger.warning("No online stream found in aliases=%s", aliases)
        except Exception as exc:
            logger.error("URL refresh error: %s", exc)

        await asyncio.sleep(interval_seconds)
