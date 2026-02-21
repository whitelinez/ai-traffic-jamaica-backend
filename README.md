# WHITELINEZ Backend (Railway)

## Railway Deploy Checklist
1. Create a Railway service from `whitelinez-backend/`.
2. Ensure build uses `requirements.txt` and starts FastAPI app (Railway/Nixpacks handles this from project defaults).
3. Add all required environment variables listed below.
4. Deploy and check `GET /health`.

## Required Environment Variables
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `CAMERA_ALIAS`
- `WS_AUTH_SECRET`
- `ALLOWED_ORIGIN`

## Recommended Starter Variables
- `URL_REFRESH_INTERVAL=240`
- `YOLO_MODEL=yolov8n.pt`
- `YOLO_CONF=0.45`
- `COUNT_LINE_RATIO=0.55`
- `DB_SNAPSHOT_INTERVAL_SEC=0.75`
- `BET_LOCK_SECONDS=10`
- `WS_PORT=8000`

## Quick Tuning (Safe)
- More responsive detection:
  - Lower `YOLO_CONF` slightly (example `0.40` to `0.45`).
- Lower backend DB pressure:
  - Increase `DB_SNAPSHOT_INTERVAL_SEC` (example `1.0`).
- Better freshness:
  - Keep `DB_SNAPSHOT_INTERVAL_SEC` between `0.5` and `1.0`.
- Stream lag feels behind real-time:
  - Confirm camera/source settings and keep low buffering at source.

## Health Check
- Endpoint: `/health`
- Expect:
  - `"status": "ok"`
  - AI/refresh/round/resolver tasks running
  - active WS connection counters when clients are connected

## Notes
- All timestamps in backend are stored in UTC.
- Frontend display is configured to Jamaica time (`America/Jamaica`).
