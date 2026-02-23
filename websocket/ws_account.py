"""
ws_account.py — /ws/account endpoint.
Supabase JWT required. Pushes per-user balance updates and resolved bet events.
"""
import logging
from urllib.parse import urlsplit

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from config import get_config
from services.auth_service import validate_supabase_jwt, get_user_id
from services.bet_service import get_user_balance
from websocket.ws_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter()


def _normalize_origin(value: str) -> str:
    raw = (value or "").strip().rstrip("/")
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"
    return raw.lower()


def _origin_allowed(origin: str, allowed: str) -> bool:
    normalized_origin = _normalize_origin(origin)
    if not normalized_origin:
        return True
    candidates = [p.strip() for p in str(allowed or "").split(",") if p.strip()]
    if not candidates:
        return False
    if any(p == "*" for p in candidates):
        return True
    normalized_allowed = {_normalize_origin(p) for p in candidates}
    return normalized_origin in normalized_allowed


@router.websocket("/ws/account")
async def ws_account(
    websocket: WebSocket,
    token: str | None = Query(default=None),
):
    cfg = get_config()

    # Check Origin
    origin = websocket.headers.get("origin", "")
    if not _origin_allowed(origin, cfg.ALLOWED_ORIGIN):
        logger.warning("Account WS rejected due to origin=%s allowed=%s", origin, cfg.ALLOWED_ORIGIN)
        await websocket.close(code=4003)
        return

    # Validate Supabase JWT
    if not token:
        await websocket.close(code=4001)
        return
    try:
        payload = await validate_supabase_jwt(token)
        user_id = get_user_id(payload)
    except Exception:
        await websocket.close(code=4001)
        return

    await manager.connect_user(websocket, user_id)

    # Send initial balance on connect
    try:
        balance = await get_user_balance(user_id)
        await websocket.send_json({"type": "balance", "balance": balance})
    except Exception as exc:
        logger.warning("Failed to send initial balance to user %s: %s", user_id, exc)

    try:
        while True:
            _ = await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        manager.disconnect_user(websocket, user_id)
    except Exception as exc:
        logger.warning("Account WS error for user %s: %s", user_id, exc)
        manager.disconnect_user(websocket, user_id)
