"""
routers/stream.py — GET /stream/live.m3u8
Validates HMAC token, reads the current HLS URL from Supabase (kept fresh by
the url_refresh_loop), proxies the manifest, and rewrites relative URLs to
absolute so the browser can load segments directly from ipcamlive.
"""
import logging
from urllib.parse import urljoin

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

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


@router.get("/live.m3u8")
async def stream_manifest(token: str = Query(...)):
    cfg = get_config()

    if not validate_ws_token(token, cfg.WS_AUTH_SECRET):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Get the current stream URL from Supabase (kept fresh by url_refresh_loop)
    supabase = await get_supabase()
    cam = await supabase.table("cameras").select("stream_url").eq("ipcam_alias", cfg.CAMERA_ALIAS).limit(1).execute()

    if not cam.data or not cam.data[0].get("stream_url"):
        raise HTTPException(status_code=503, detail="Stream URL not yet available — refresher may still be starting")

    stream_url = cam.data[0]["stream_url"]
    base_url = stream_url.rsplit("/", 1)[0] + "/"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(stream_url, headers=_PROXY_HEADERS, follow_redirects=True)
    except Exception as exc:
        logger.error("Failed to fetch HLS manifest: %s", exc)
        raise HTTPException(status_code=502, detail="Stream unavailable")

    if resp.status_code != 200:
        logger.warning("HLS manifest returned %d for URL: %s", resp.status_code, stream_url)
        raise HTTPException(status_code=502, detail="Stream unavailable")

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
