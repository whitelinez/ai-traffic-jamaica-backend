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
    if str(details.get("streamavailable", "")) != "1":
        logger.warning("Camera reports streamavailable=0")
        return None

    address = str(details.get("address", "")).rstrip("/")
    streamid = str(details.get("streamid", "")).strip()
    if not address or not streamid:
        logger.warning("Missing address/streamid in stream state details")
        return None
    address = address.replace("http://", "https://")
    return f"{address}/streams/{streamid}/stream.m3u8"


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
    return resp.status_code in (200, 204)


async def get_candidate_aliases(primary_alias: str | None = None) -> list[str]:
    from config import get_config
    cfg = get_config()
    out: list[str] = []

    def _push(v: str | None) -> None:
        s = str(v or "").strip()
        if s and s not in out:
            out.append(s)

    _push(primary_alias or cfg.CAMERA_ALIAS)
    for alias in getattr(cfg, "CAMERA_ALIASES", []) or []:
        _push(alias)

    try:
        from supabase_client import get_supabase
        sb = await get_supabase()
        resp = await (
            sb.table("cameras")
            .select("ipcam_alias,created_at")
            .eq("is_active", True)
            .order("created_at", desc=False)
            .execute()
        )
        for row in resp.data or []:
            _push(row.get("ipcam_alias"))
    except Exception as exc:
        logger.debug("Could not load aliases from cameras table: %s", exc)

    return out


async def url_refresh_loop(alias: str, interval_seconds: int = 240) -> None:
    global _current_url, _current_alias
    while True:
        try:
            aliases = await get_candidate_aliases(alias)
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
