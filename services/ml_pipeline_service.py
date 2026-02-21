"""
services/ml_pipeline_service.py - Modular ML pipeline orchestration.

This module keeps the workflow provider-agnostic:
1) export_dataset_job -> builds training manifest from telemetry
2) start_training_job -> schedules train run (provider-specific later)
3) promote_model -> marks an approved model active
4) list_jobs / list_models -> admin observability
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException

from supabase_client import get_supabase


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def export_dataset_job(hours: int = 24) -> dict[str, Any]:
    sb = await get_supabase()
    since = datetime.now(timezone.utc) - timedelta(hours=max(1, min(hours, 24 * 30)))
    since_iso = since.isoformat()

    telemetry = await (
        sb.table("ml_detection_events")
        .select("camera_id, captured_at, class_counts, avg_confidence, detections_count")
        .gte("captured_at", since_iso)
        .order("captured_at", desc=False)
        .limit(10000)
        .execute()
    )
    rows = telemetry.data or []
    if not rows:
        raise HTTPException(status_code=400, detail="No telemetry rows available for dataset export")

    # Lightweight manifest (you can swap for S3/GCS artifact manifests later)
    manifest = {
        "window_hours": hours,
        "rows": len(rows),
        "exported_at": _utc_now_iso(),
        "features": ["class_counts", "avg_confidence", "detections_count"],
    }
    job = {
        "job_type": "export",
        "status": "completed",
        "provider": "internal",
        "started_at": _utc_now_iso(),
        "completed_at": _utc_now_iso(),
        "params": {"hours": hours},
        "metrics": {"telemetry_rows": len(rows)},
        "artifact_manifest": manifest,
        "notes": "Telemetry export manifest generated",
    }
    resp = await sb.table("ml_training_jobs").insert(job).execute()
    return resp.data[0] if resp.data else job


async def start_training_job(
    *,
    base_model: str,
    dataset_job_id: int | None,
    provider: str = "internal_stub",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sb = await get_supabase()

    artifact_manifest = None
    if dataset_job_id is not None:
        ds = await (
            sb.table("ml_training_jobs")
            .select("id, artifact_manifest, status, job_type")
            .eq("id", dataset_job_id)
            .single()
            .execute()
        )
        if not ds.data or ds.data.get("job_type") != "export":
            raise HTTPException(status_code=404, detail="Dataset export job not found")
        if ds.data.get("status") != "completed":
            raise HTTPException(status_code=400, detail="Dataset export job is not completed")
        artifact_manifest = ds.data.get("artifact_manifest")

    started = _utc_now_iso()

    # Stub provider: records a train job and an immutable model candidate row.
    # Can be replaced by true train orchestration later (SageMaker/Vertex/Replicate/etc.).
    model_tag = f"whitelinez-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    model_uri = f"models/{model_tag}.pt"
    metrics = {
        "mAP50": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "note": "stub metrics; replace with provider callback metrics",
    }

    train_job = {
        "job_type": "train",
        "status": "completed",
        "provider": provider,
        "started_at": started,
        "completed_at": _utc_now_iso(),
        "params": {
            "base_model": base_model,
            "dataset_job_id": dataset_job_id,
            **(params or {}),
        },
        "metrics": metrics,
        "artifact_manifest": artifact_manifest,
        "notes": "Stub training pipeline completed",
    }
    j = await sb.table("ml_training_jobs").insert(train_job).execute()
    job_row = j.data[0] if j.data else train_job

    candidate = {
        "model_name": model_tag,
        "model_uri": model_uri,
        "base_model": base_model,
        "training_job_id": job_row.get("id"),
        "status": "candidate",
        "metrics": metrics,
        "created_at": _utc_now_iso(),
        "promoted_at": None,
    }
    m = await sb.table("ml_model_registry").insert(candidate).execute()
    model_row = m.data[0] if m.data else candidate

    return {
        "job": job_row,
        "model": model_row,
    }


async def promote_model(model_id: int) -> dict[str, Any]:
    sb = await get_supabase()
    model_resp = await (
        sb.table("ml_model_registry")
        .select("*")
        .eq("id", model_id)
        .single()
        .execute()
    )
    model = model_resp.data
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # demote current active
    await sb.table("ml_model_registry").update({"status": "archived"}).eq("status", "active").execute()
    # promote selected
    now_iso = _utc_now_iso()
    upd = await (
        sb.table("ml_model_registry")
        .update({"status": "active", "promoted_at": now_iso})
        .eq("id", model_id)
        .execute()
    )
    return (upd.data or [model])[0]


async def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
    sb = await get_supabase()
    resp = await (
        sb.table("ml_training_jobs")
        .select("*")
        .order("id", desc=True)
        .limit(max(1, min(limit, 500)))
        .execute()
    )
    return resp.data or []


async def list_models(limit: int = 100) -> list[dict[str, Any]]:
    sb = await get_supabase()
    resp = await (
        sb.table("ml_model_registry")
        .select("*")
        .order("id", desc=True)
        .limit(max(1, min(limit, 500)))
        .execute()
    )
    return resp.data or []


async def get_active_model_uri() -> str | None:
    sb = await get_supabase()
    resp = await (
        sb.table("ml_model_registry")
        .select("model_uri, status")
        .eq("status", "active")
        .order("promoted_at", desc=True)
        .limit(1)
        .execute()
    )
    if resp.data:
        return resp.data[0].get("model_uri")
    return None
