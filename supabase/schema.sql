-- ─────────────────────────────────────────────────────────────────────────────
-- WHITELINEZ — Supabase Schema
-- Run this in the Supabase SQL editor (Project → SQL Editor → New query)
-- ─────────────────────────────────────────────────────────────────────────────

-- ── Tables ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cameras (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name         TEXT NOT NULL,
  ipcam_alias  TEXT UNIQUE,                  -- ipcamlive camera alias (e.g. "abc123")
  stream_url   TEXT NOT NULL DEFAULT '',     -- refreshed every ~4min by url_refresh_loop
  count_line   JSONB,                        -- {x1,y1,x2,y2} as 0–1 relative coords
  is_active    BOOLEAN DEFAULT TRUE,
  created_at   TIMESTAMPTZ DEFAULT now()
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

-- ── User balances ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS user_balances (
  user_id   UUID REFERENCES auth.users PRIMARY KEY,
  balance   INT  NOT NULL DEFAULT 1000 CHECK (balance >= 0)
);

ALTER TABLE user_balances ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "own_balance_select" ON user_balances;
CREATE POLICY "own_balance_select" ON user_balances
  FOR SELECT USING (auth.uid() = user_id);

-- ── Seed: default camera (required for AI loop to start) ─────────────────────
-- Set ipcam_alias to your ipcamlive camera alias (e.g. "5e8a2f9c1b3d4").

INSERT INTO cameras (name, ipcam_alias, stream_url, is_active)
VALUES ('Kingston Feed', NULL, '', TRUE)
ON CONFLICT DO NOTHING;

-- ── Row Level Security ────────────────────────────────────────────────────────

ALTER TABLE cameras          ENABLE ROW LEVEL SECURITY;
ALTER TABLE bet_rounds       ENABLE ROW LEVEL SECURITY;
ALTER TABLE markets          ENABLE ROW LEVEL SECURITY;
ALTER TABLE bets             ENABLE ROW LEVEL SECURITY;
ALTER TABLE count_snapshots  ENABLE ROW LEVEL SECURITY;

-- Cameras: public read, admin write
DROP POLICY IF EXISTS "public_read_cameras"  ON cameras;
DROP POLICY IF EXISTS "admin_write_cameras"  ON cameras;
CREATE POLICY "public_read_cameras" ON cameras
  FOR SELECT USING (true);
CREATE POLICY "admin_write_cameras" ON cameras
  FOR ALL USING (
    (auth.jwt() -> 'app_metadata' ->> 'role') = 'admin'
  );

-- Bet rounds: public read, service role write (backend only)
DROP POLICY IF EXISTS "public_read_rounds" ON bet_rounds;
CREATE POLICY "public_read_rounds" ON bet_rounds
  FOR SELECT USING (true);

-- Markets: public read, service role write
DROP POLICY IF EXISTS "public_read_markets" ON markets;
CREATE POLICY "public_read_markets" ON markets
  FOR SELECT USING (true);

-- Bets: users see only their own
DROP POLICY IF EXISTS "own_bets_select" ON bets;
DROP POLICY IF EXISTS "own_bets_insert" ON bets;
CREATE POLICY "own_bets_select" ON bets
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "own_bets_insert" ON bets
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Count snapshots: public read, service role write
DROP POLICY IF EXISTS "public_read_snapshots" ON count_snapshots;
CREATE POLICY "public_read_snapshots" ON count_snapshots
  FOR SELECT USING (true);

-- ── Database functions ────────────────────────────────────────────────────────

-- Atomically place a bet: check balance → deduct → insert bet → update staked.
-- Called by bet_service.py via sb.rpc("place_bet_atomic", {...}).
-- SECURITY DEFINER runs as table owner, bypassing RLS for internal writes.
CREATE OR REPLACE FUNCTION place_bet_atomic(
  p_user_id        UUID,
  p_round_id       UUID,
  p_market_id      UUID,
  p_amount         INT,
  p_potential_payout INT
) RETURNS JSON
LANGUAGE plpgsql SECURITY DEFINER AS $$
DECLARE
  v_balance INT;
  v_bet_id  UUID;
BEGIN
  -- Auto-init balance for new users
  INSERT INTO user_balances (user_id, balance)
  VALUES (p_user_id, 1000)
  ON CONFLICT (user_id) DO NOTHING;

  -- Lock the row to prevent race conditions
  SELECT balance INTO v_balance
  FROM user_balances
  WHERE user_id = p_user_id
  FOR UPDATE;

  IF v_balance < p_amount THEN
    RETURN json_build_object('error', 'Insufficient balance');
  END IF;

  -- Deduct stake
  UPDATE user_balances
  SET balance = balance - p_amount
  WHERE user_id = p_user_id;

  -- Insert bet
  INSERT INTO bets (user_id, round_id, market_id, amount, potential_payout, status)
  VALUES (p_user_id, p_round_id, p_market_id, p_amount, p_potential_payout, 'pending')
  RETURNING id INTO v_bet_id;

  -- Update market total_staked
  UPDATE markets
  SET total_staked = total_staked + p_amount
  WHERE id = p_market_id;

  RETURN json_build_object('bet_id', v_bet_id);
END;
$$;

-- Return a user's current balance. Auto-initialises to 1000 for new users.
CREATE OR REPLACE FUNCTION get_user_balance(p_user_id UUID)
RETURNS INT
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO user_balances (user_id, balance)
  VALUES (p_user_id, 1000)
  ON CONFLICT (user_id) DO NOTHING;

  RETURN (SELECT balance FROM user_balances WHERE user_id = p_user_id);
END;
$$;

-- Add credits to a user's balance (called on bet resolution/payout).
CREATE OR REPLACE FUNCTION credit_user_balance(p_user_id UUID, p_amount INT)
RETURNS VOID
LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
  INSERT INTO user_balances (user_id, balance)
  VALUES (p_user_id, p_amount)
  ON CONFLICT (user_id) DO UPDATE
    SET balance = user_balances.balance + EXCLUDED.balance;
END;
$$;

-- Profile table for avatars + display names (public read for chat, user-owned writes)
CREATE TABLE IF NOT EXISTS profiles (
  user_id UUID PRIMARY KEY REFERENCES auth.users ON DELETE CASCADE,
  username TEXT NOT NULL CHECK (length(username) BETWEEN 1 AND 32),
  avatar_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS public_read_profiles ON profiles;
CREATE POLICY public_read_profiles ON profiles
  FOR SELECT USING (true);

DROP POLICY IF EXISTS own_write_profiles ON profiles;
CREATE POLICY own_write_profiles ON profiles
  FOR INSERT WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS own_update_profiles ON profiles;
CREATE POLICY own_update_profiles ON profiles
  FOR UPDATE USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE INDEX IF NOT EXISTS idx_profiles_username ON profiles(username);

-- Optional storage bucket + policies for avatar uploads (run in Supabase SQL Editor)
INSERT INTO storage.buckets (id, name, public)
SELECT 'avatars', 'avatars', true
WHERE NOT EXISTS (SELECT 1 FROM storage.buckets WHERE id = 'avatars');

DROP POLICY IF EXISTS public_read_avatars ON storage.objects;
CREATE POLICY public_read_avatars ON storage.objects
  FOR SELECT
  USING (bucket_id = 'avatars');

DROP POLICY IF EXISTS own_insert_avatars ON storage.objects;
CREATE POLICY own_insert_avatars ON storage.objects
  FOR INSERT
  WITH CHECK (
    bucket_id = 'avatars'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

DROP POLICY IF EXISTS own_update_avatars ON storage.objects;
CREATE POLICY own_update_avatars ON storage.objects
  FOR UPDATE
  USING (
    bucket_id = 'avatars'
    AND auth.uid()::text = (storage.foldername(name))[1]
  )
  WITH CHECK (
    bucket_id = 'avatars'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

DROP POLICY IF EXISTS own_delete_avatars ON storage.objects;
CREATE POLICY own_delete_avatars ON storage.objects
  FOR DELETE
  USING (
    bucket_id = 'avatars'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );
