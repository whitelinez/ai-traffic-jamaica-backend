"""
routers/admin.py — Admin-only round management endpoints.
POST /admin/rounds     → create a new round + markets
POST /admin/resolve    → manually resolve a round
POST /admin/set-role   → grant/revoke admin role on a user
"""
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from models.round import CreateRoundRequest, ResolveRoundRequest, RoundOut
from services.auth_service import validate_supabase_jwt, require_admin, get_user_id
from services.round_service import create_round, resolve_round, resolve_round_from_latest_snapshot
from supabase_client import get_supabase
from middleware.rate_limiter import limiter

router = APIRouter(prefix="/admin", tags=["admin"])


async def _require_admin_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    payload = await validate_supabase_jwt(token)
    require_admin(payload)
    return payload


@router.post("/rounds", response_model=RoundOut, status_code=201)
async def admin_create_round(
    body: CreateRoundRequest,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    try:
        return await create_round(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/resolve")
async def admin_resolve_round(
    body: ResolveRoundRequest,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    await resolve_round(str(body.round_id), body.result)
    return {"message": "Round resolved", "round_id": str(body.round_id)}


@router.patch("/rounds")
async def admin_resolve_round_latest(
    body: dict,  # {"round_id": "..."}
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    """
    Manual override resolve:
    - Resolves the given round from latest snapshot result.
    - Works even when auto-resolve is enabled.
    """
    round_id = body.get("round_id")
    if not round_id:
        raise HTTPException(status_code=400, detail="round_id required")

    result = await resolve_round_from_latest_snapshot(str(round_id))
    return {"message": "Round resolved", "round_id": str(round_id), "result": result}


@router.get("/bets")
async def admin_list_bets(
    admin: Annotated[dict, Depends(_require_admin_user)],
    limit: int = Query(default=200, ge=1, le=1000),
):
    """
    Admin read-only feed of recent bets with bettor identity.
    Uses profiles.username when available, otherwise falls back to user_id.
    """
    sb = await get_supabase()

    bets_resp = await (
        sb.table("bets")
        .select(
            "id,user_id,round_id,market_id,amount,potential_payout,status,bet_type,"
            "vehicle_class,exact_count,actual_count,window_duration_sec,placed_at,resolved_at,"
            "markets(label,odds),bet_rounds(market_type,status)"
        )
        .order("placed_at", desc=True)
        .limit(limit)
        .execute()
    )
    bets = bets_resp.data or []

    user_ids = sorted({b.get("user_id") for b in bets if b.get("user_id")})
    profile_map: dict[str, str] = {}
    if user_ids:
        try:
            prof_resp = await (
                sb.table("profiles")
                .select("user_id,username")
                .in_("user_id", user_ids)
                .execute()
            )
            for p in (prof_resp.data or []):
                uid = p.get("user_id")
                uname = p.get("username")
                if uid and uname:
                    profile_map[str(uid)] = str(uname)
        except Exception:
            # profiles table may be absent in older installs
            pass

    enriched = []
    for b in bets:
        uid = str(b.get("user_id") or "")
        b["username"] = profile_map.get(uid)
        enriched.append(b)

    return {"bets": enriched}


@router.post("/set-role")
async def set_user_role(
    body: dict,  # {"user_id": "...", "role": "admin" | "user"}
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    """
    Set app_metadata.role on a user via Supabase Admin API.
    Only a service-role client can do this.
    """
    target_user_id = body.get("user_id")
    role = body.get("role", "user")
    if not target_user_id:
        raise HTTPException(status_code=400, detail="user_id required")
    if role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'user'")

    sb = await get_supabase()
    # Supabase admin SDK: update user metadata
    resp = await sb.auth.admin.update_user_by_id(
        target_user_id,
        {"app_metadata": {"role": role}},
    )
    return {"message": f"User {target_user_id} role set to {role}"}
