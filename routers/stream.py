"""
routers/stream.py — GET /stream/live.m3u8
Validates HMAC token, proxies the top-level HLS manifest from HLS_STREAM_URL,
and rewrites any relative URLs to absolute so the browser can load segments
directly from the source. Source URL is never exposed to the client.
"""
import logging
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from config import get_config
from middleware.hmac_auth import validate_ws_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stream", tags=["stream"])


@router.get("/live.m3u8")
async def stream_manifest(token: str = Query(...)):
    cfg = get_config()

    if not validate_ws_token(token, cfg.WS_AUTH_SECRET):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Base URL for resolving relative paths in the manifest
    base_url = cfg.HLS_STREAM_URL.rsplit("/", 1)[0] + "/"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(cfg.HLS_STREAM_URL, follow_redirects=True)
    except Exception as exc:
        logger.error("Failed to fetch HLS manifest: %s", exc)
        raise HTTPException(status_code=502, detail="Stream unavailable")

    if resp.status_code != 200:
        logger.warning("HLS manifest returned %d", resp.status_code)
        raise HTTPException(status_code=502, detail="Stream unavailable")

    # Rewrite relative URLs in the manifest to absolute so the browser can
    # fetch sub-playlists and segments directly from the source.
    lines = []
    for line in resp.text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("http"):
            line = urljoin(base_url, stripped)
        lines.append(line)

    return Response(
        content="\n".join(lines),
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache, no-store"},
    )
