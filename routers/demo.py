"""
routers/demo.py — Public demo manifest endpoint.
GET /demo/manifest → returns the latest demo recording manifest from Supabase Storage.
No auth required — manifest only contains public CDN URLs.
"""
import logging

import httpx
from fastapi import APIRouter

from config import get_config

router = APIRouter(prefix="/demo", tags=["demo"])
logger = logging.getLogger(__name__)


@router.get("/manifest")
async def demo_manifest():
    cfg = get_config()
    url = f"{cfg.SUPABASE_URL.rstrip('/')}/storage/v1/object/public/demo-videos/manifest.json"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception as exc:
        logger.debug("[demo] manifest fetch error: %s", exc)
    return {"available": False}
