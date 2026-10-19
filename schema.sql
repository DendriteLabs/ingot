-- Idempotent; applied by docker-compose on first start.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- metagraph as the chain reader last saw it; swap detection compares against it
CREATE TABLE IF NOT EXISTS neurons (
    uid INT PRIMARY KEY,
    hotkey TEXT NOT NULL,
    coldkey TEXT,
    registration_block BIGINT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- hotkeys that may not submit again: 'eval', 'dedup', 'swap' or 'backfill'
CREATE TABLE IF NOT EXISTS used_hotkeys (
    hotkey TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- one row per verified private upload; `state` is the pipeline (ingot/states.py)
CREATE TABLE IF NOT EXISTS submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotkey TEXT NOT NULL,
    uid INT NOT NULL,
    coldkey TEXT,
    model_uri TEXT NOT NULL,
    model_hash TEXT NOT NULL UNIQUE,
    block BIGINT NOT NULL,
    block_hash TEXT,
    family TEXT NOT NULL,                -- from the registration; validation loads these rules
    quant TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'SUBMITTED',
    attempts INT NOT NULL DEFAULT 0,
    lease_until TIMESTAMPTZ,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- dropped and re-added so a database created before a new state learns it
ALTER TABLE submissions DROP CONSTRAINT IF EXISTS submissions_state_check;
ALTER TABLE submissions ADD CONSTRAINT submissions_state_check CHECK (state IN (
    'SUBMITTED', 'VALIDATING', 'VALID', 'INVALID',
    'DEDUPING', 'UNIQUE', 'DUPLICATE',
    'EVALUATING', 'WON', 'LOST',
    'CROWNING', 'CROWNED', 'FAILED'
));

CREATE INDEX IF NOT EXISTS submissions_state_created_idx ON submissions (state, created_at);

-- every private-store signal seen on chain; one commitment per hotkey per block
CREATE TABLE IF NOT EXISTS commits (
    netuid INT NOT NULL,
    hotkey TEXT NOT NULL,
    block BIGINT NOT NULL,
    uid INT,
    coldkey TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    block_hash TEXT,
    seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (netuid, hotkey, block)
);

-- private R2 upload lifecycle, driven by chain_reader/controller.py
CREATE TABLE IF NOT EXISTS private_registrations (
    id BIGSERIAL PRIMARY KEY,
    netuid INT NOT NULL,
    uid INT,
    hotkey TEXT NOT NULL,
    coldkey TEXT,
    registration_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL DEFAULT 'ACTIVATED' CHECK (state IN (
        'ACTIVATED', 'CREDENTIALED', 'READY', 'REVOKED', 'SUBMITTED', 'FAILED', 'REAPED'
    )),
    activation_block BIGINT NOT NULL,
    submission_pubkey TEXT NOT NULL,
    family TEXT NOT NULL,                -- declared in r2activate; picks the upload limit and rules
    quant TEXT NOT NULL,
    ready_block BIGINT,
    ready_block_hash TEXT,
    model_digest TEXT,
    parent_token_id TEXT,
    credential_expires_at TIMESTAMPTZ,
    fault_message TEXT,
    attempt_count INT NOT NULL DEFAULT 1,
    extra_attempts INT NOT NULL DEFAULT 0,
    model_prefix TEXT,
    submission_id UUID REFERENCES submissions(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT private_registrations_hotkey_uk UNIQUE (netuid, hotkey)
);

ALTER TABLE private_registrations ADD COLUMN IF NOT EXISTS family TEXT;
ALTER TABLE private_registrations ADD COLUMN IF NOT EXISTS quant TEXT;
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS family TEXT;
ALTER TABLE submissions ADD COLUMN IF NOT EXISTS quant TEXT;
ALTER TABLE private_registrations DROP COLUMN IF EXISTS manifest_sha256;

CREATE INDEX IF NOT EXISTS private_registrations_state_updated_idx
    ON private_registrations (state, updated_at);

-- newest row is the model to beat; genesis has no submission
CREATE TABLE IF NOT EXISTS kings (
    id BIGSERIAL PRIMARY KEY,
    submission_id UUID UNIQUE REFERENCES submissions(id),
    hotkey TEXT,
    uid INT,
    model_uri TEXT NOT NULL,
    hf_repo TEXT,
    crowned_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evals (
    id BIGSERIAL PRIMARY KEY,
    submission_id UUID NOT NULL REFERENCES submissions(id),
    king_id BIGINT NOT NULL REFERENCES kings(id),
    teacher TEXT NOT NULL,               -- repo@revision
    dataset_hash TEXT NOT NULL,
    seed TEXT NOT NULL,
    sample_count INT NOT NULL,
    score_challenger DOUBLE PRECISION,   -- mean KL(teacher ‖ model) per answer token
    score_king DOUBLE PRECISION,
    margin DOUBLE PRECISION NOT NULL,    -- KL margin
    dist_challenger DOUBLE PRECISION,    -- mean |h - h_teacher| / |h_teacher| per answer token
    dist_king DOUBLE PRECISION,
    dist_margin DOUBLE PRECISION,
    noise_kl DOUBLE PRECISION,           -- the teacher against its own rerun
    noise_dist DOUBLE PRECISION,
    won BOOLEAN,
    challenger_s REAL,
    king_s REAL,
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

ALTER TABLE evals ADD COLUMN IF NOT EXISTS dist_challenger DOUBLE PRECISION;
ALTER TABLE evals ADD COLUMN IF NOT EXISTS dist_king DOUBLE PRECISION;
ALTER TABLE evals ADD COLUMN IF NOT EXISTS dist_margin DOUBLE PRECISION;
ALTER TABLE evals ADD COLUMN IF NOT EXISTS noise_kl DOUBLE PRECISION;
ALTER TABLE evals ADD COLUMN IF NOT EXISTS noise_dist DOUBLE PRECISION;
ALTER TABLE evals ADD COLUMN IF NOT EXISTS challenger_s REAL;
ALTER TABLE evals ADD COLUMN IF NOT EXISTS king_s REAL;
ALTER TABLE evals ADD COLUMN IF NOT EXISTS samples_done INT NOT NULL DEFAULT 0;  -- while it runs

-- one row per dedup run; the fingerprint itself lives in the OpenSearch bank
CREATE TABLE IF NOT EXISTS dedups (
    id BIGSERIAL PRIMARY KEY,
    submission_id UUID NOT NULL REFERENCES submissions(id),
    duplicate BOOLEAN,
    reason TEXT,                         -- COPY (same tensors) or NEAR-COPY (same error)
    duplicate_of TEXT,                   -- bank id: a submission id or 'genesis'
    corr DOUBLE PRECISION,               -- best correlation of quantization errors in the shortlist
    copied_frac DOUBLE PRECISION,        -- best share of error energy copied tensor by tensor
    rel_to_teacher DOUBLE PRECISION,
    compared INT,                        -- bank entries in the shortlist
    notes TEXT,
    seconds REAL,
    error TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

-- the subnet's registration burn, sampled by the chain reader every PRICE_EVERY_BLOCKS;
-- the day before the reader started is backfilled from an archive node by
-- scripts/backfill_registration_prices.py
CREATE TABLE IF NOT EXISTS registration_prices (
    block BIGINT PRIMARY KEY,
    burn_rao BIGINT NOT NULL,
    at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS registration_prices_at ON registration_prices (at);

CREATE TABLE IF NOT EXISTS weight_sets (
    id BIGSERIAL PRIMARY KEY,
    block BIGINT NOT NULL,
    uids INT[] NOT NULL,
    weights DOUBLE PRECISION[] NOT NULL,
    kings BIGINT[] NOT NULL DEFAULT '{}',
    ok BOOLEAN NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- one row per service instance; only ACTIVE claims. DRAINING finishes its row, OFFLINE is stopped
-- one row per submission state change; keeps every error, submissions.error only the last
CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    submission_id UUID NOT NULL REFERENCES submissions(id),
    state TEXT NOT NULL,
    worker TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS events_submission_idx ON events (submission_id, id);

CREATE TABLE IF NOT EXISTS workers (
    name TEXT PRIMARY KEY,
    service TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (state IN ('ACTIVE', 'DRAINING', 'OFFLINE')),
    busy_with UUID REFERENCES submissions(id),
    queued INT NOT NULL DEFAULT 0,
    last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
