"""
ws_public.py — /ws/live endpoint.
HMAC token required. Broadcasts live count + market data to all subscribers.
"""
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status

from config import get_config
from middleware.hmac_auth import validate_ws_token
from supabase_client import get_supabase
from websocket.ws_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


async def _send_bootstrap_count(websocket: WebSocket, camera_alias: str) -> None:
    """
    Send latest persisted count snapshot to a newly connected public client.
    This prevents UI reset after backend restarts/redeploys while first live frame
    is still warming up.
    """
    try:
        sb = await get_supabase()
        cam_resp = await (
            sb.table("cameras")
            .select("id")
            .eq("ipcam_alias", camera_alias)
            .limit(1)
            .execute()
        )
        camera_id = cam_resp.data[0]["id"] if cam_resp.data else None
        if not camera_id:
            return

        snap_resp = await (
            sb.table("count_snapshots")
            .select("camera_id,captured_at,count_in,count_out,total,vehicle_breakdown")
            .eq("camera_id", camera_id)
            .order("captured_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = snap_resp.data or []
        if not rows:
            return
        latest = rows[0] or {}
        payload = {
            "type": "count",
            "camera_id": latest.get("camera_id"),
            "captured_at": latest.get("captured_at"),
            "count_in": int(latest.get("count_in", 0) or 0),
            "count_out": int(latest.get("count_out", 0) or 0),
            "total": int(latest.get("total", 0) or 0),
            "vehicle_breakdown": latest.get("vehicle_breakdown") or {},
            "new_crossings": 0,
            "detections": [],
            "bootstrap": True,
        }
        await websocket.send_text(json.dumps(payload))
    except Exception as exc:
        logger.debug("WS bootstrap snapshot skipped: %s", exc)


@router.websocket("/ws/live")
async def ws_live(
    websocket: WebSocket,
    token: str | None = Query(default=None),
):
    cfg = get_config()

    # Check Origin header
    origin = websocket.headers.get("origin", "")
    if origin and origin != cfg.ALLOWED_ORIGIN:
        logger.warning("WS rejected — bad origin: %s", origin)
        await websocket.close(code=4003)
        return

    # Validate HMAC token
    if not validate_ws_token(token, cfg.WS_AUTH_SECRET):
        logger.warning("WS rejected — invalid HMAC token")
        await websocket.close(code=4001)
        return

    await manager.connect_public(websocket)
    await _send_bootstrap_count(websocket, cfg.CAMERA_ALIAS)
    try:
        while True:
            # Keep alive — client doesn't send data, just listens
            data = await websocket.receive_text()
            # Accept ping frames silently
    except WebSocketDisconnect:
        manager.disconnect_public(websocket)
    except Exception as exc:
        logger.warning("Public WS error: %s", exc)
        manager.disconnect_public(websocket)
