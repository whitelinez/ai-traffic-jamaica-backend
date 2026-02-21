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
from services.round_service import create_round, resolve_round
from services.analytics_service import get_analytics_overview
from services.ml_pipeline_service import (
    export_dataset_job,
    start_training_job,
    promote_model,
    list_jobs,
    list_models,
)
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
    return await create_round(body)


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
    Resolve a round using the latest count snapshot for its camera.
    This matches the admin dashboard "Resolve" button payload.
    """
    round_id = body.get("round_id")
    if not round_id:
        raise HTTPException(status_code=400, detail="round_id required")

    sb = await get_supabase()
    rnd_resp = await (
        sb.table("bet_rounds")
        .select("id, camera_id")
        .eq("id", round_id)
        .single()
        .execute()
    )
    if not rnd_resp.data:
        raise HTTPException(status_code=404, detail="Round not found")

    camera_id = rnd_resp.data.get("camera_id")
    result = {"total": 0, "vehicle_breakdown": {}}

    if camera_id:
        snap_resp = await (
            sb.table("count_snapshots")
            .select("total, vehicle_breakdown")
            .eq("camera_id", camera_id)
            .order("captured_at", desc=True)
            .limit(1)
            .execute()
        )
        if snap_resp.data:
            snap = snap_resp.data[0]
            result = {
                "total": snap.get("total", 0) or 0,
                "vehicle_breakdown": snap.get("vehicle_breakdown") or {},
            }

    await resolve_round(round_id, result)
    return {"message": "Round resolved", "round_id": str(round_id), "result": result}


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


@router.get("/analytics/overview")
async def admin_analytics_overview(
    admin: Annotated[dict, Depends(_require_admin_user)],
    hours: int = Query(default=24, ge=1, le=24 * 14),
):
    return await get_analytics_overview(hours=hours)


@router.get("/ml/jobs")
async def admin_ml_jobs(
    admin: Annotated[dict, Depends(_require_admin_user)],
    limit: int = Query(default=100, ge=1, le=500),
):
    return await list_jobs(limit=limit)


@router.post("/ml/jobs/export")
async def admin_ml_export(
    body: dict,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    hours = int(body.get("hours", 24))
    return await export_dataset_job(hours=hours)


@router.post("/ml/jobs/train")
async def admin_ml_train(
    body: dict,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    base_model = body.get("base_model", "yolov8n.pt")
    dataset_job_id = body.get("dataset_job_id")
    provider = body.get("provider", "internal_stub")
    params = body.get("params") or {}
    return await start_training_job(
        base_model=base_model,
        dataset_job_id=dataset_job_id,
        provider=provider,
        params=params,
    )


@router.get("/ml/models")
async def admin_ml_models(
    admin: Annotated[dict, Depends(_require_admin_user)],
    limit: int = Query(default=100, ge=1, le=500),
):
    return await list_models(limit=limit)


@router.post("/ml/models/promote")
async def admin_ml_promote(
    body: dict,
    admin: Annotated[dict, Depends(_require_admin_user)],
):
    model_id = body.get("model_id")
    if model_id is None:
        raise HTTPException(status_code=400, detail="model_id required")
    return await promote_model(int(model_id))
