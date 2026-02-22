"""
bet_service.py — Atomic bet placement (market bets + exact-count live bets).
"""
import logging
from uuid import UUID
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException
from supabase_client import get_supabase
from models.bet import PlaceBetRequest, PlaceBetResponse, PlaceLiveBetRequest, PlaceLiveBetResponse

logger = logging.getLogger(__name__)

INITIAL_BALANCE = 1000
LIVE_BET_ODDS = 8.0  # fixed 8x for exact-count micro-bets
MAX_PENDING_BETS_PER_ROUND = 2


def _as_actionable_db_error(exc: Exception) -> HTTPException:
    msg = str(exc)
    low = msg.lower()
    if ("column" in low and "does not exist" in low) or ("null value in column \"market_id\"" in low):
        return HTTPException(
            status_code=500,
            detail=(
                "Database schema is out of date for betting. "
                "Run latest supabase/schema.sql migration (bets columns + nullable market_id)."
            ),
        )
    return HTTPException(status_code=500, detail="Bet placement failed due to database error")


def _parse_round_closes_at(rnd: dict) -> datetime:
    raw = rnd.get("closes_at")
    if not raw:
        raise HTTPException(status_code=400, detail="Round timing is misconfigured (missing closes_at)")
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="Round timing is invalid (bad closes_at format)")


def _extract_bet_id_from_rpc_data(data) -> str:
    """
    Supabase RPC payloads can vary by client version.
    Accept common shapes and fail with a clear error if unknown.
    """
    # {"bet_id": "..."}
    if isinstance(data, dict):
        if data.get("bet_id"):
            return str(data["bet_id"])
        # {"place_bet_atomic": {"bet_id": "..."}}
        inner = data.get("place_bet_atomic")
        if isinstance(inner, dict) and inner.get("bet_id"):
            return str(inner["bet_id"])

    # [{"bet_id": "..."}] or [{"place_bet_atomic": {"bet_id":"..."}}]
    if isinstance(data, list) and data:
        row = data[0]
        if isinstance(row, dict):
            if row.get("bet_id"):
                return str(row["bet_id"])
            inner = row.get("place_bet_atomic")
            if isinstance(inner, dict) and inner.get("bet_id"):
                return str(inner["bet_id"])

    raise HTTPException(
        status_code=500,
        detail=f"Unexpected RPC response format from place_bet_atomic: {type(data).__name__}",
    )


async def _pending_bets_for_round(sb, user_id: str, round_id: str) -> int:
    resp = await (
        sb.table("bets")
        .select("id", count="exact", head=True)
        .eq("user_id", user_id)
        .eq("round_id", round_id)
        .eq("status", "pending")
        .execute()
    )
    return int(resp.count or 0)


async def _get_baseline_count(sb, camera_id: str | None, market_type: str, params: dict | None) -> int:
    if not camera_id:
        return 0
    snap_resp = await (
        sb.table("count_snapshots")
        .select("total, vehicle_breakdown")
        .eq("camera_id", camera_id)
        .order("captured_at", desc=True)
        .limit(1)
        .execute()
    )
    if not snap_resp.data:
        return 0
    snap = snap_resp.data[0]
    if market_type == "vehicle_count":
        cls = (params or {}).get("vehicle_class")
        if not cls:
            return 0
        return int((snap.get("vehicle_breakdown") or {}).get(cls, 0) or 0)
    return int(snap.get("total", 0) or 0)


async def place_bet(user_id: str, req: PlaceBetRequest) -> PlaceBetResponse:
    """
    Atomically place a market bet:
    1. Verify the round is still open
    2. Verify the market belongs to the round
    3. Fetch current user balance
    4. Check sufficient funds
    5. Deduct balance + insert bet in one RPC call
    """
    sb = await get_supabase()
    try:
        try:
            round_resp = await sb.table("bet_rounds").select("*").eq("id", str(req.round_id)).single().execute()
            if not round_resp.data:
                raise HTTPException(status_code=404, detail="Round not found")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=404, detail="Round not found")

        rnd = round_resp.data
        now = datetime.now(timezone.utc)

        if rnd["status"] != "open":
            raise HTTPException(status_code=400, detail=f"Round is {rnd['status']}, bets not accepted")

        closes_at = _parse_round_closes_at(rnd)
        if now >= closes_at:
            raise HTTPException(status_code=403, detail="Betting window has closed")

        pending = await _pending_bets_for_round(sb, user_id, str(req.round_id))
        if pending >= MAX_PENDING_BETS_PER_ROUND:
            raise HTTPException(
                status_code=429,
                detail=f"Maximum {MAX_PENDING_BETS_PER_ROUND} active bets per round reached",
            )

        try:
            mkt_resp = await (
                sb.table("markets")
                .select("*")
                .eq("id", str(req.market_id))
                .eq("round_id", str(req.round_id))
                .single()
                .execute()
            )
            if not mkt_resp.data:
                raise HTTPException(status_code=404, detail="Market not found in this round")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=404, detail="Market not found in this round")

        market = mkt_resp.data
        odds = float(market["odds"])
        potential_payout = int(req.amount * odds)

        try:
            rpc_resp = await sb.rpc("place_bet_atomic", {
                "p_user_id": user_id,
                "p_round_id": str(req.round_id),
                "p_market_id": str(req.market_id),
                "p_amount": req.amount,
                "p_potential_payout": potential_payout,
            }).execute()
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            if "insufficient balance" in low:
                raise HTTPException(status_code=400, detail="Insufficient balance")
            if "duplicate" in low or "unique" in low:
                raise HTTPException(status_code=400, detail="You already placed a bet on this market")
            raise _as_actionable_db_error(exc)

        if rpc_resp.data and isinstance(rpc_resp.data, dict) and rpc_resp.data.get("error"):
            raise HTTPException(status_code=400, detail=rpc_resp.data["error"])

        bet_id = _extract_bet_id_from_rpc_data(rpc_resp.data)
        baseline_count = await _get_baseline_count(
            sb,
            rnd.get("camera_id"),
            str(rnd.get("market_type") or ""),
            rnd.get("params") or {},
        )

        # Optional enrichment fields for newer schemas; ignore if columns do not exist.
        try:
            await (
                sb.table("bets")
                .update({"bet_type": "market", "baseline_count": baseline_count})
                .eq("id", str(bet_id))
                .execute()
            )
        except Exception as exc:
            logger.warning("Skipping optional bet enrichment columns for bet %s: %s", bet_id, exc)

        return PlaceBetResponse(
            bet_id=bet_id,
            status="pending",
            amount=req.amount,
            potential_payout=potential_payout,
            placed_at=datetime.now(timezone.utc),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled place_bet crash user_id=%s round_id=%s market_id=%s", user_id, req.round_id, req.market_id)
        raise HTTPException(status_code=500, detail=f"Bet placement crashed: {exc}")


async def place_live_bet(user_id: str, req: PlaceLiveBetRequest) -> PlaceLiveBetResponse:
    """
    Place an exact-count micro-bet:
    1. Verify round is open and closes_at not passed
    2. Fetch baseline count from latest count_snapshot
    3. Deduct balance atomically
    4. Insert bet with bet_type='exact_count'
    Returns bet details including window_end time.
    """
    sb = await get_supabase()

    # 1. Validate round
    try:
        round_resp = await sb.table("bet_rounds").select("*").eq("id", str(req.round_id)).single().execute()
        if not round_resp.data:
            raise HTTPException(status_code=404, detail="Round not found")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Round not found")

    rnd = round_resp.data
    now = datetime.now(timezone.utc)

    if rnd["status"] != "open":
        raise HTTPException(status_code=400, detail=f"Round is {rnd['status']}, bets not accepted")

    closes_at = _parse_round_closes_at(rnd)
    if now >= closes_at:
        raise HTTPException(status_code=403, detail="Betting window has closed")

    pending = await _pending_bets_for_round(sb, user_id, str(req.round_id))
    if pending >= MAX_PENDING_BETS_PER_ROUND:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum {MAX_PENDING_BETS_PER_ROUND} active bets per round reached",
        )

    # 2. Validate vehicle_class
    valid_classes = {"car", "truck", "bus", "motorcycle"}
    if req.vehicle_class is not None and req.vehicle_class not in valid_classes:
        raise HTTPException(status_code=400, detail=f"Invalid vehicle_class: {req.vehicle_class}")

    # 3. Fetch baseline from latest count_snapshot for this camera
    cam_resp = await sb.table("bet_rounds").select("camera_id").eq("id", str(req.round_id)).single().execute()
    camera_id = cam_resp.data.get("camera_id") if cam_resp.data else None

    baseline_count = 0
    if camera_id:
        snap_resp = await sb.table("count_snapshots") \
            .select("total, vehicle_breakdown") \
            .eq("camera_id", camera_id) \
            .order("captured_at", desc=True) \
            .limit(1) \
            .execute()
        if snap_resp.data:
            snap = snap_resp.data[0]
            if req.vehicle_class is None:
                baseline_count = snap.get("total", 0) or 0
            else:
                bd = snap.get("vehicle_breakdown") or {}
                baseline_count = bd.get(req.vehicle_class, 0)

    # 4. Check balance and deduct atomically
    potential_payout = int(req.amount * LIVE_BET_ODDS)

    # Route through existing DB atomic function so balance + insert stay in one transaction.
    try:
        rpc_resp = await sb.rpc("place_bet_atomic", {
            "p_user_id": user_id,
            "p_round_id": str(req.round_id),
            "p_market_id": None,
            "p_amount": req.amount,
            "p_potential_payout": potential_payout,
        }).execute()
    except Exception as exc:
        msg = str(exc).lower()
        if "insufficient balance" in msg:
            raise HTTPException(status_code=400, detail="Insufficient balance")
        raise _as_actionable_db_error(exc)

    if rpc_resp.data and isinstance(rpc_resp.data, dict) and rpc_resp.data.get("error"):
        raise HTTPException(status_code=400, detail=rpc_resp.data["error"])

    bet_id = _extract_bet_id_from_rpc_data(rpc_resp.data)

    # 5. Update inserted bet with live-bet specific metadata
    window_start = now
    window_end = now + timedelta(seconds=req.window_duration_sec)

    live_update = {
        "bet_type": "exact_count",
        "window_start": window_start.isoformat(),
        "window_duration_sec": req.window_duration_sec,
        "vehicle_class": req.vehicle_class,
        "exact_count": req.exact_count,
        "baseline_count": baseline_count,
    }

    try:
        await sb.table("bets").update(live_update).eq("id", str(bet_id)).execute()
    except Exception as exc:
        raise _as_actionable_db_error(exc)

    return PlaceLiveBetResponse(
        bet_id=bet_id,
        status="pending",
        amount=req.amount,
        potential_payout=potential_payout,
        window_end=window_end,
        exact_count=req.exact_count,
        vehicle_class=req.vehicle_class,
        placed_at=window_start,
    )


async def get_user_balance(user_id: str) -> int:
    """Read user balance. Falls back to INITIAL_BALANCE if no record found."""
    sb = await get_supabase()
    resp = await sb.rpc("get_user_balance", {"p_user_id": user_id}).execute()
    if resp.data is not None:
        return int(resp.data)
    return INITIAL_BALANCE
