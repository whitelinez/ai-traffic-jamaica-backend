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
from services.ml_pipeline_service import auto_retrain_cycle, list_jobs, list_models, get_ml_diagnostics
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
    List registered users (admin-only) with auth info + betting summary.
    """
    sb = await get_supabase()

    # supabase-py signatures vary by version; support both call styles.
    try:
        resp = await sb.auth.admin.list_users(page=page, per_page=per_page)
    except TypeError:
        resp = await sb.auth.admin.list_users({"page": page, "per_page": per_page})

    # list_users response shape differs across SDK versions:
    # - resp.users
    # - {"users": [...]}
    # - {"data": {"users": [...]}}
    # - model_dump() with one of the above
    payload = None
    if hasattr(resp, "model_dump"):
        payload = resp.model_dump() or {}
    elif isinstance(resp, dict):
        payload = resp
    else:
        payload = {}

    users_raw = (
        getattr(resp, "users", None)
        or payload.get("users")
        or (payload.get("data") or {}).get("users")
        or getattr(resp, "data", {}).get("users") if hasattr(resp, "data") else None
    )
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
        raw_meta = rec.get("user_metadata") or rec.get("raw_user_meta_data") or {}
        identities = rec.get("identities") or []
        identity_email = None
        for ident in identities:
            if not isinstance(ident, dict):
                continue
            id_data = ident.get("identity_data") or {}
            if isinstance(id_data, dict) and id_data.get("email"):
                identity_email = id_data.get("email")
                break
        users.append(
            {
                "id": rec.get("id"),
                "email": rec.get("email") or raw_meta.get("email") or identity_email,
                "created_at": rec.get("created_at"),
                "last_sign_in_at": rec.get("last_sign_in_at") or rec.get("updated_at"),
                "role": app_meta.get("role", "user"),
                "email_confirmed_at": rec.get("email_confirmed_at"),
                "username": raw_meta.get("username"),
            }
        )

    # Fallback: if auth admin list is empty but profiles exist, surface those users.
    # This helps when auth list shape/permissions differ but app users are present.
    if not users:
        try:
            prof_resp = await (
                sb.table("profiles")
                .select("user_id,username,created_at,updated_at")
                .order("created_at", desc=True)
                .limit(per_page)
                .execute()
            )
            for p in (prof_resp.data or []):
                uid = p.get("user_id")
                if not uid:
                    continue
                users.append(
                    {
                        "id": uid,
                        "email": None,
                        "created_at": p.get("created_at"),
                        "last_sign_in_at": p.get("updated_at") or p.get("created_at"),
                        "role": "user",
                        "email_confirmed_at": None,
                        "username": p.get("username"),
                    }
                )
        except Exception:
            pass

    # Enrich from profiles + bets so admin can see user activity at a glance.
    user_ids = [str(u.get("id")) for u in users if u.get("id")]
    profile_map: dict[str, dict] = {}
    bet_summary_map: dict[str, dict] = {}

    if user_ids:
        try:
            prof_resp = await (
                sb.table("profiles")
                .select("user_id,username,avatar_url,created_at,updated_at")
                .in_("user_id", user_ids)
                .execute()
            )
            for p in (prof_resp.data or []):
                uid = p.get("user_id")
                if uid:
                    profile_map[str(uid)] = p
        except Exception:
            pass

        try:
            bets_resp = await (
                sb.table("bets")
                .select(
                    "user_id,amount,status,placed_at,bet_type,vehicle_class,exact_count,"
                    "markets(label)"
                )
                .in_("user_id", user_ids)
                .order("placed_at", desc=True)
                .limit(5000)
                .execute()
            )
            for b in (bets_resp.data or []):
                uid = b.get("user_id")
                if not uid:
                    continue
                key = str(uid)
                if key not in bet_summary_map:
                    bet_summary_map[key] = {
                        "bet_count": 0,
                        "total_staked": 0,
                        "pending_count": 0,
                        "won_count": 0,
                        "lost_count": 0,
                        "last_bet_at": None,
                        "last_bet_status": None,
                        "last_bet_amount": 0,
                        "last_bet_label": None,
                    }
                s = bet_summary_map[key]
                amount = int(b.get("amount") or 0)
                status = str(b.get("status") or "pending")
                s["bet_count"] += 1
                s["total_staked"] += amount
                if status == "won":
                    s["won_count"] += 1
                elif status == "lost":
                    s["lost_count"] += 1
                else:
                    s["pending_count"] += 1

                # First item is most recent due to desc ordering.
                if not s["last_bet_at"]:
                    bet_type = str(b.get("bet_type") or "market")
                    market = b.get("markets") or {}
                    market_label = market.get("label") if isinstance(market, dict) else None
                    if bet_type == "exact_count":
                        cls = b.get("vehicle_class") or "vehicles"
                        label = f"Exact {b.get('exact_count') or 0} {cls}"
                    else:
                        label = market_label or "Market bet"
                    s["last_bet_at"] = b.get("placed_at")
                    s["last_bet_status"] = status
                    s["last_bet_amount"] = amount
                    s["last_bet_label"] = label
        except Exception:
            pass

    for u in users:
        uid = str(u.get("id") or "")
        p = profile_map.get(uid) or {}
        # Prefer auth username, then profile username.
        u["username"] = u.get("username") or p.get("username")
        u["avatar_url"] = p.get("avatar_url")
        if not u.get("created_at"):
            u["created_at"] = p.get("created_at")
        if not u.get("last_sign_in_at"):
            u["last_sign_in_at"] = p.get("updated_at") or p.get("created_at")
        u["bet_summary"] = bet_summary_map.get(uid) or {
            "bet_count": 0,
            "total_staked": 0,
            "pending_count": 0,
            "won_count": 0,
            "lost_count": 0,
            "last_bet_at": None,
            "last_bet_status": None,
            "last_bet_amount": 0,
            "last_bet_label": None,
        }

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


@router.get("/ml/diagnostics")
async def admin_ml_diagnostics(
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    cfg = get_config()
    return await get_ml_diagnostics(cfg=cfg)


@router.post("/ml/one-click")
async def admin_ml_one_click(
    body: dict,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    cfg = get_config()
    diagnostics = await get_ml_diagnostics(cfg=cfg)
    if not diagnostics.get("ready_for_one_click"):
        raise HTTPException(
            status_code=400,
            detail={
                "message": "One-click pipeline blocked by missing required setup",
                "diagnostics": diagnostics,
            },
        )

    hours = int(body.get("hours", cfg.ML_AUTO_RETRAIN_HOURS))
    min_rows = int(body.get("min_rows", cfg.ML_AUTO_RETRAIN_MIN_ROWS))
    min_score_gain = float(body.get("min_score_gain", cfg.ML_AUTO_RETRAIN_MIN_SCORE_GAIN))
    epochs = int(body.get("epochs", cfg.TRAINER_EPOCHS))
    imgsz = int(body.get("imgsz", cfg.TRAINER_IMGSZ))
    batch = int(body.get("batch", cfg.TRAINER_BATCH))
    dataset_yaml_url = str(body.get("dataset_yaml_url") or cfg.TRAINER_DATASET_YAML_URL).strip()

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
    return {
        "message": "One-click model pipeline completed",
        "result": result,
        "diagnostics": await get_ml_diagnostics(cfg=cfg),
    }


@router.get("/ml/night-profile")
async def admin_ml_night_profile_get(
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    cfg = get_config()
    return {
        "enabled": int(getattr(cfg, "NIGHT_PROFILE_ENABLED", 0) or 0) == 1,
        "start_hour": int(getattr(cfg, "NIGHT_PROFILE_START_HOUR", 18) or 18),
        "end_hour": int(getattr(cfg, "NIGHT_PROFILE_END_HOUR", 6) or 6),
        "yolo_conf": float(getattr(cfg, "NIGHT_YOLO_CONF", cfg.YOLO_CONF)),
        "infer_size": int(getattr(cfg, "NIGHT_DETECT_INFER_SIZE", cfg.DETECT_INFER_SIZE)),
        "iou": float(getattr(cfg, "NIGHT_DETECT_IOU", cfg.DETECT_IOU)),
        "max_det": int(getattr(cfg, "NIGHT_DETECT_MAX_DET", cfg.DETECT_MAX_DET)),
        "note": "Runtime settings only. Persist via environment variables for restart durability.",
    }


@router.patch("/ml/night-profile")
async def admin_ml_night_profile_patch(
    body: dict,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    cfg = get_config()

    if "enabled" in body:
        cfg.NIGHT_PROFILE_ENABLED = 1 if bool(body.get("enabled")) else 0
    if "start_hour" in body:
        cfg.NIGHT_PROFILE_START_HOUR = int(body.get("start_hour")) % 24
    if "end_hour" in body:
        cfg.NIGHT_PROFILE_END_HOUR = int(body.get("end_hour")) % 24
    if "yolo_conf" in body:
        cfg.NIGHT_YOLO_CONF = max(0.01, min(0.99, float(body.get("yolo_conf"))))
    if "infer_size" in body:
        cfg.NIGHT_DETECT_INFER_SIZE = max(320, min(1280, int(body.get("infer_size"))))
    if "iou" in body:
        cfg.NIGHT_DETECT_IOU = max(0.05, min(0.95, float(body.get("iou"))))
    if "max_det" in body:
        cfg.NIGHT_DETECT_MAX_DET = max(10, min(500, int(body.get("max_det"))))

    return {
        "ok": True,
        "settings": {
            "enabled": int(getattr(cfg, "NIGHT_PROFILE_ENABLED", 0) or 0) == 1,
            "start_hour": int(getattr(cfg, "NIGHT_PROFILE_START_HOUR", 18) or 18),
            "end_hour": int(getattr(cfg, "NIGHT_PROFILE_END_HOUR", 6) or 6),
            "yolo_conf": float(getattr(cfg, "NIGHT_YOLO_CONF", cfg.YOLO_CONF)),
            "infer_size": int(getattr(cfg, "NIGHT_DETECT_INFER_SIZE", cfg.DETECT_INFER_SIZE)),
            "iou": float(getattr(cfg, "NIGHT_DETECT_IOU", cfg.DETECT_IOU)),
            "max_det": int(getattr(cfg, "NIGHT_DETECT_MAX_DET", cfg.DETECT_MAX_DET)),
        },
        "note": "Applied in-memory immediately. Set env vars to persist across restart/redeploy.",
    }
