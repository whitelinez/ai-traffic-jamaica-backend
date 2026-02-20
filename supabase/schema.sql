-- ─────────────────────────────────────────────────────────────────────────────
-- WHITELINEZ — Supabase Schema
-- Run this in the Supabase SQL editor (Project → SQL Editor → New query)
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Tables ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cameras (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL,
  stream_url  TEXT NOT NULL DEFAULT '',
  count_line  JSONB,                        -- {x1,y1,x2,y2} as 0–1 relative coords
  is_active   BOOLEAN DEFAULT TRUE,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bet_rounds (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  camera_id    UUID REFERENCES cameras ON DELETE SET NULL,
  market_type  TEXT NOT NULL,              -- 'over_under' | 'vehicle_type'
  params       JSONB NOT NULL DEFAULT '{}',
  status       TEXT DEFAULT 'upcoming',   -- upcoming|open|locked|resolved|cancelled
  opens_at     TIMESTAMPTZ,
  closes_at    TIMESTAMPTZ,
  ends_at      TIMESTAMPTZ,
  result       JSONB,
  created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS markets (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  round_id     UUID REFERENCES bet_rounds ON DELETE CASCADE,
  label        TEXT NOT NULL,
  outcome_key  TEXT NOT NULL,
  odds         NUMERIC(5,2) NOT NULL,
  total_staked INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bets (
  id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          UUID REFERENCES auth.users NOT NULL,
  round_id         UUID REFERENCES bet_rounds NOT NULL,
  market_id        UUID REFERENCES markets NOT NULL,
  amount           INT NOT NULL CHECK (amount > 0),
  potential_payout INT NOT NULL,
  status           TEXT DEFAULT 'pending',  -- pending|won|lost|cancelled
  placed_at        TIMESTAMPTZ DEFAULT now(),
  resolved_at      TIMESTAMPTZ,
  UNIQUE (user_id, market_id)
);

CREATE TABLE IF NOT EXISTS count_snapshots (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  camera_id         UUID REFERENCES cameras ON DELETE SET NULL,
  captured_at       TIMESTAMPTZ DEFAULT now(),
  count_in          INT DEFAULT 0,
  count_out         INT DEFAULT 0,
  total             INT DEFAULT 0,
  vehicle_breakdown JSONB
);

-- ── Seed: default camera (required for AI loop to start) ─────────────────────

INSERT INTO cameras (name, stream_url, is_active)
VALUES ('Kingston Feed', '', TRUE)
ON CONFLICT DO NOTHING;

-- ── Row Level Security ────────────────────────────────────────────────────────

ALTER TABLE cameras          ENABLE ROW LEVEL SECURITY;
ALTER TABLE bet_rounds       ENABLE ROW LEVEL SECURITY;
ALTER TABLE markets          ENABLE ROW LEVEL SECURITY;
ALTER TABLE bets             ENABLE ROW LEVEL SECURITY;
ALTER TABLE count_snapshots  ENABLE ROW LEVEL SECURITY;

-- Cameras: public read, admin write
CREATE POLICY "public_read_cameras" ON cameras
  FOR SELECT USING (true);

CREATE POLICY "admin_write_cameras" ON cameras
  FOR ALL USING (
    (auth.jwt() -> 'app_metadata' ->> 'role') = 'admin'
  );

-- Bet rounds: public read, service role write (backend only)
CREATE POLICY "public_read_rounds" ON bet_rounds
  FOR SELECT USING (true);

-- Markets: public read, service role write
CREATE POLICY "public_read_markets" ON markets
  FOR SELECT USING (true);

-- Bets: users see only their own
CREATE POLICY "own_bets_select" ON bets
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "own_bets_insert" ON bets
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Count snapshots: public read, service role write
CREATE POLICY "public_read_snapshots" ON count_snapshots
  FOR SELECT USING (true);
