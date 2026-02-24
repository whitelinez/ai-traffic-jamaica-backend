"""
routers/stream.py - GET /stream/live.m3u8
Validates HMAC token, reads the current HLS URL from Supabase, proxies the
manifest, and rewrites relative URLs to absolute URLs.
"""
import asyncio
import logging
import time
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ai.url_refresher import (
    _supabase_update_stream_url,
    fetch_fresh_stream_url,
    get_candidate_aliases,
    get_current_url,
)
from config import get_config
from middleware.hmac_auth import validate_ws_token
from supabase_client import get_supabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["stream"])

_PROXY_HEADERS = {
    "Referer": "https://www.ipcamlive.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Short in-process cache to reduce load when many clients request the same
# manifest at the same time.
_MANIFEST_CACHE_TTL_SEC = 1.25
_STREAM_URL_CACHE_TTL_SEC = 8.0
_cache_lock = asyncio.Lock()
_manifest_cache: dict[str, object] = {"body": "", "fetched_at": 0.0}
_stream_url_cache: dict[str, object] = {"url": "", "fetched_at": 0.0}


async def _get_stream_url(cfg, preferred_alias: str | None = None) -> str:
    now = time.monotonic()
    cached_url = str(_stream_url_cache.get("url") or "")
    cached_at = float(_stream_url_cache.get("fetched_at") or 0.0)
    if cached_url and (now - cached_at) < _STREAM_URL_CACHE_TTL_SEC:
        return cached_url

    # Prefer the refresher's live-selected URL first.
    live_url = str(get_current_url() or "").strip()
    if live_url:
        _stream_url_cache["url"] = live_url
        _stream_url_cache["fetched_at"] = now
        return live_url

    supabase = await get_supabase()
    aliases = await get_candidate_aliases(preferred_alias or cfg.CAMERA_ALIAS)
    stream_url = ""
    for alias in aliases:
        cam = await (
            supabase.table("cameras")
            .select("stream_url")
            .eq("ipcam_alias", alias)
            .limit(1)
            .execute()
        )
        candidate = str((cam.data or [{}])[0].get("stream_url") or "").strip()
        if candidate:
            stream_url = candidate
            break
    if not stream_url:
        raise HTTPException(status_code=503, detail="Stream URL not yet available")

    _stream_url_cache["url"] = stream_url
    _stream_url_cache["fetched_at"] = now
    return stream_url


def _rewrite_manifest(manifest_body: str, base_url: str) -> str:
    lines = []
    for line in manifest_body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("http"):
            line = urljoin(base_url, stripped)
        lines.append(line)
    return "\n".join(lines)


@router.get("/live.m3u8")
async def stream_manifest(
    token: str = Query(...),
    alias: str | None = Query(default=None),
):
    cfg = get_config()
    if not validate_ws_token(token, cfg.WS_AUTH_SECRET):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    now = time.monotonic()
    cached_body = str(_manifest_cache.get("body") or "")
    cached_at = float(_manifest_cache.get("fetched_at") or 0.0)
    if cached_body and (now - cached_at) < _MANIFEST_CACHE_TTL_SEC:
        return Response(
            content=cached_body,
            media_type="application/vnd.apple.mpegurl",
            headers={"Cache-Control": "no-cache, no-store"},
        )

    async with _cache_lock:
        now = time.monotonic()
        cached_body = str(_manifest_cache.get("body") or "")
        cached_at = float(_manifest_cache.get("fetched_at") or 0.0)
        if cached_body and (now - cached_at) < _MANIFEST_CACHE_TTL_SEC:
            return Response(
                content=cached_body,
                media_type="application/vnd.apple.mpegurl",
                headers={"Cache-Control": "no-cache, no-store"},
            )

        preferred_alias = str(alias or "").strip() or None
        stream_url = await _get_stream_url(cfg, preferred_alias=preferred_alias)
        base_url = stream_url.rsplit("/", 1)[0] + "/"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    stream_url,
                    headers=_PROXY_HEADERS,
                    follow_redirects=True,
                )
                # If the persisted URL is stale, force one refresh and retry.
                if resp.status_code == 404:
                    logger.warning("Stored HLS URL returned 404, attempting one forced refresh")
                    aliases = await get_candidate_aliases(preferred_alias or cfg.CAMERA_ALIAS)
                    for alias in aliases:
                        fresh = await fetch_fresh_stream_url(alias)
                        if not fresh:
                            continue
                        stream_url = fresh
                        base_url = stream_url.rsplit("/", 1)[0] + "/"
                        _stream_url_cache["url"] = stream_url
                        _stream_url_cache["fetched_at"] = time.monotonic()
                        await _supabase_update_stream_url(alias, fresh)
                        resp = await client.get(
                            stream_url,
                            headers=_PROXY_HEADERS,
                            follow_redirects=True,
                        )
                        if resp.status_code == 200:
                            break
        except Exception as exc:
            logger.error("Failed to fetch HLS manifest: %s", exc)
            raise HTTPException(status_code=502, detail="Stream unavailable")

        if resp.status_code != 200:
            logger.warning("HLS manifest returned %d for URL: %s", resp.status_code, stream_url)
            raise HTTPException(status_code=502, detail="Stream unavailable")

        rewritten = _rewrite_manifest(resp.text, base_url)
        _manifest_cache["body"] = rewritten
        _manifest_cache["fetched_at"] = time.monotonic()

    return Response(
        content=rewritten,
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache, no-store"},
    )
