"""
routers/admin.py — Admin-only round management endpoints.
POST /admin/rounds     → create a new round + markets
POST /admin/resolve    → manually resolve a round
POST /admin/set-role   → grant/revoke admin role on a user
"""
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status

from models.round import CreateRoundRequest, ResolveRoundRequest, RoundOut
from models.round_session import CreateRoundSessionRequest
from services.auth_service import validate_supabase_jwt, require_admin, get_user_id
from services.round_service import create_round, resolve_round, resolve_round_from_latest_snapshot
from services.round_session_service import create_round_session, list_round_sessions, stop_round_session
from services.ml_pipeline_service import auto_retrain_cycle, list_jobs, list_models
from services.ml_capture_monitor import get_capture_status
from config import get_config
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


@router.post("/round-sessions", status_code=201)
async def admin_create_round_session(
    body: CreateRoundSessionRequest,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    return await create_round_session(body)


@router.get("/round-sessions")
async def admin_list_round_sessions(
    admin: Annotated[dict, Depends(_require_admin_user)],
    limit: int = Query(default=50, ge=1, le=200),
):
    return {"sessions": await list_round_sessions(limit=limit)}


@router.patch("/round-sessions/{session_id}/stop")
async def admin_stop_round_session(
    session_id: str,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    data = await stop_round_session(session_id)
    return {"session": data}


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


@router.get("/users")
async def admin_list_users(
    admin: Annotated[dict, Depends(_require_admin_user)],
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=200, ge=1, le=1000),
):
    """
    List registered auth users (admin-only) with minimal safe fields.
    """
    sb = await get_supabase()

    # supabase-py signatures vary by version; support both call styles.
    try:
        resp = await sb.auth.admin.list_users(page=page, per_page=per_page)
    except TypeError:
        resp = await sb.auth.admin.list_users({"page": page, "per_page": per_page})

    users_raw = getattr(resp, "users", None)
    if users_raw is None and isinstance(resp, dict):
        users_raw = resp.get("users")
    if users_raw is None and hasattr(resp, "model_dump"):
        users_raw = (resp.model_dump() or {}).get("users")
    users_raw = users_raw or []

    users = []
    for u in users_raw:
        if hasattr(u, "model_dump"):
            rec = u.model_dump()
        elif isinstance(u, dict):
            rec = u
        else:
            rec = {}
        app_meta = rec.get("app_metadata") or {}
        users.append(
            {
                "id": rec.get("id"),
                "email": rec.get("email"),
                "created_at": rec.get("created_at"),
                "last_sign_in_at": rec.get("last_sign_in_at"),
                "role": app_meta.get("role", "user"),
                "email_confirmed_at": rec.get("email_confirmed_at"),
            }
        )

    return {"users": users, "page": page, "per_page": per_page}


@router.post("/ml/retrain")
async def admin_ml_retrain(
    body: dict,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    """
    Manual ML retrain trigger.
    Optional body overrides:
    - hours
    - min_rows
    - min_score_gain
    """
    cfg = get_config()
    hours = int(body.get("hours", cfg.ML_AUTO_RETRAIN_HOURS))
    min_rows = int(body.get("min_rows", cfg.ML_AUTO_RETRAIN_MIN_ROWS))
    min_score_gain = float(body.get("min_score_gain", cfg.ML_AUTO_RETRAIN_MIN_SCORE_GAIN))

    dataset_yaml_url = str(body.get("dataset_yaml_url") or cfg.TRAINER_DATASET_YAML_URL).strip()
    if not dataset_yaml_url:
        raise HTTPException(
            status_code=400,
            detail="dataset_yaml_url is required. Set TRAINER_DATASET_YAML_URL or pass it in request body.",
        )
    epochs = int(body.get("epochs", cfg.TRAINER_EPOCHS))
    imgsz = int(body.get("imgsz", cfg.TRAINER_IMGSZ))
    batch = int(body.get("batch", cfg.TRAINER_BATCH))

    result = await auto_retrain_cycle(
        hours=hours,
        min_rows=min_rows,
        min_score_gain=min_score_gain,
        base_model=cfg.YOLO_MODEL,
        provider="webhook",
        params={
            "trainer_webhook_url": cfg.TRAINER_WEBHOOK_URL,
            "trainer_webhook_secret": cfg.TRAINER_WEBHOOK_SECRET,
            "dataset_yaml_url": dataset_yaml_url,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
        },
    )
    return result


@router.get("/ml/jobs")
async def admin_ml_jobs(
    admin: Annotated[dict, Depends(_require_admin_user)],
    limit: int = Query(default=50, ge=1, le=500),
):
    return {"jobs": await list_jobs(limit=limit)}


@router.get("/ml/models")
async def admin_ml_models(
    admin: Annotated[dict, Depends(_require_admin_user)],
    limit: int = Query(default=50, ge=1, le=500),
):
    return {"models": await list_models(limit=limit)}


@router.get("/ml/capture-status")
async def admin_ml_capture_status(
    admin: Annotated[dict, Depends(_require_admin_user)],
    limit: int = Query(default=50, ge=1, le=200),
):
    cfg = get_config()
    status = get_capture_status(limit=limit)
    return {
        "capture_enabled": cfg.AUTO_CAPTURE_ENABLED == 1,
        "upload_enabled": cfg.AUTO_CAPTURE_UPLOAD_ENABLED == 1,
        "capture_classes": [c.strip() for c in cfg.AUTO_CAPTURE_CLASSES.split(",") if c.strip()],
        **status,
    }
