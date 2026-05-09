from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

LOGGER = logging.getLogger("app.db.migrate")

_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    google_id TEXT UNIQUE,
    name TEXT,
    picture_url TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);
CREATE INDEX IF NOT EXISTS ix_users_google_id ON users(google_id);

-- Remove legacy is_admin column (replaced by admin_users table + owner role).
ALTER TABLE users DROP COLUMN IF EXISTS is_admin;

-- Yandex OAuth support.
ALTER TABLE users ADD COLUMN IF NOT EXISTS yandex_id TEXT UNIQUE;
CREATE INDEX IF NOT EXISTS ix_users_yandex_id ON users(yandex_id);

ALTER TABLE cases ADD COLUMN IF NOT EXISTS user_id UUID;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'cases_user_id_fkey'
    ) THEN
        ALTER TABLE cases
            ADD CONSTRAINT cases_user_id_fkey
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS ix_cases_user_id ON cases(user_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_type WHERE typname = 'message_status'
    ) THEN
        CREATE TYPE message_status AS ENUM ('processing', 'done', 'error');
    END IF;
END $$;

ALTER TABLE messages ADD COLUMN IF NOT EXISTS status message_status NOT NULL DEFAULT 'done';
ALTER TABLE messages ADD COLUMN IF NOT EXISTS langgraph_run_id TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS error_text TEXT;
ALTER TABLE messages ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT now();

CREATE INDEX IF NOT EXISTS ix_messages_case_status ON messages(case_id, status);

CREATE TABLE IF NOT EXISTS admin_users (
    id UUID PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_admin_users_username ON admin_users(username);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'npa_status') THEN
        CREATE TYPE npa_status AS ENUM ('uploaded', 'chunking', 'ready', 'indexing', 'indexed', 'failed');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS npa_sources (
    id UUID PRIMARY KEY,
    created_by UUID REFERENCES admin_users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    original_filename TEXT NOT NULL,
    source_name TEXT NOT NULL,
    raw_format TEXT NOT NULL,
    conversion_options JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_txt_s3_key TEXT NOT NULL,
    chunks_json_s3_key TEXT,
    chunks_count INTEGER,
    last_indexed_collection TEXT,
    last_indexed_with_refs BOOLEAN,
    last_indexed_at TIMESTAMPTZ,
    status npa_status NOT NULL DEFAULT 'uploaded',
    error_text TEXT
);

CREATE INDEX IF NOT EXISTS ix_npa_sources_source_name ON npa_sources(source_name);
CREATE INDEX IF NOT EXISTS ix_npa_sources_status ON npa_sources(status);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'subscription_status') THEN
        CREATE TYPE subscription_status AS ENUM ('active', 'expired', 'cancelled');
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'payment_status') THEN
        CREATE TYPE payment_status AS ENUM ('pending', 'succeeded', 'canceled');
    END IF;
END $$;

-- subscription_plans.code is intentionally TEXT (not an enum) so admins can
-- create new plan codes from the UI without DB migrations.
CREATE TABLE IF NOT EXISTS subscription_plans (
    code TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    price_rub NUMERIC(10, 2) NOT NULL DEFAULT 0,
    duration_days INTEGER,
    monthly_generation_limit INTEGER,
    allow_edit BOOLEAN NOT NULL DEFAULT TRUE,
    features_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    description_md TEXT NOT NULL DEFAULT '',
    display_order INTEGER NOT NULL DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS user_subscriptions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan_code TEXT NOT NULL REFERENCES subscription_plans(code) ON DELETE RESTRICT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    status subscription_status NOT NULL DEFAULT 'active',
    yookassa_payment_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_user_subscriptions_user_id ON user_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS ix_user_subscriptions_expires_at ON user_subscriptions(expires_at);

CREATE TABLE IF NOT EXISTS payment_intents (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    plan_code TEXT NOT NULL,
    amount_rub NUMERIC(10, 2) NOT NULL,
    yookassa_payment_id TEXT UNIQUE,
    idempotence_key TEXT NOT NULL UNIQUE,
    status payment_status NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_payment_intents_user_id ON payment_intents(user_id);

-- Convert legacy plan_code enum columns to TEXT so admins can introduce new
-- plan codes. Idempotent: each step checks if it's already done.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'subscription_plans' AND column_name = 'code' AND udt_name = 'plan_code'
    ) THEN
        ALTER TABLE user_subscriptions DROP CONSTRAINT IF EXISTS user_subscriptions_plan_code_fkey;
        ALTER TABLE subscription_plans ALTER COLUMN code TYPE TEXT USING code::text;
        ALTER TABLE user_subscriptions ALTER COLUMN plan_code TYPE TEXT USING plan_code::text;
        ALTER TABLE payment_intents ALTER COLUMN plan_code TYPE TEXT USING plan_code::text;
        ALTER TABLE user_subscriptions
            ADD CONSTRAINT user_subscriptions_plan_code_fkey
            FOREIGN KEY (plan_code) REFERENCES subscription_plans(code) ON DELETE RESTRICT;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'plan_code') THEN
        DROP TYPE plan_code;
    END IF;
END $$;

-- Latecomer columns. Tables predating these go through ADD COLUMN IF NOT EXISTS.
ALTER TABLE subscription_plans ADD COLUMN IF NOT EXISTS description_md TEXT NOT NULL DEFAULT '';
ALTER TABLE subscription_plans ADD COLUMN IF NOT EXISTS display_order INTEGER NOT NULL DEFAULT 0;

-- Backfill description_md from features_json on first run after this column appeared.
UPDATE subscription_plans
SET description_md = (
    SELECT string_agg('- ' || elem, E'\n') FROM jsonb_array_elements_text(features_json) AS elem
)
WHERE description_md = '' AND jsonb_array_length(COALESCE(features_json, '[]'::jsonb)) > 0;

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    contract_generation_policy TEXT NOT NULL DEFAULT 'legal_only',
    ask_personal_data BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- On startup we also clean orphan PROCESSING rows from a previous app crash.
-- That cleanup runs separately so legacy DONE rows are not touched here.
"""

# Default catalog of subscription plans. INSERT ... ON CONFLICT DO NOTHING
# means re-running on a populated DB does not overwrite admin-edited prices.
# Adjust price/limits via the admin UI; new defaults here only apply on a
# pristine DB.
_DEFAULT_PLANS: tuple[dict, ...] = (
    {
        "code": "free",
        "title": "Бесплатный",
        "price_rub": "0",
        "duration_days": None,
        "monthly_generation_limit": 10,
        "allow_edit": False,
        "description_md": (
            "- 10 генераций договора в месяц\n- Followup-вопросы по сгенерированному договору\n- Без правки договора"
        ),
        "display_order": 0,
    },
    {
        "code": "monthly",
        "title": "Месяц",
        "price_rub": "499",
        "duration_days": 30,
        "monthly_generation_limit": None,
        "allow_edit": True,
        "description_md": ("- Без лимита на генерации\n- Правка договора в чате\n- Полный доступ к рекомендациям"),
        "display_order": 1,
    },
    {
        "code": "yearly",
        "title": "Год",
        "price_rub": "4990",
        "duration_days": 365,
        "monthly_generation_limit": None,
        "allow_edit": True,
        "description_md": ("- Всё из месячного тарифа\n- **Экономия** по сравнению с месяцем"),
        "display_order": 2,
    },
    {
        "code": "biennial",
        "title": "2 года",
        "price_rub": "8990",
        "duration_days": 730,
        "monthly_generation_limit": None,
        "allow_edit": True,
        "description_md": ("- Всё из годового тарифа\n- **Максимальная выгода**"),
        "display_order": 3,
    },
)


def run_migrations(engine: Engine, seed_admin_email: str, owner_username: str = "") -> None:
    """Idempotent migrations: schema changes, seed user, legacy backfill."""
    with engine.begin() as conn:
        conn.execute(text(_MIGRATION_SQL))

        # Remove the owner from admin_users (previously seeded via ADMIN_USERNAME).
        # The owner no longer has a DB row — auth is done against env vars only.
        if owner_username:
            result = conn.execute(
                text("DELETE FROM admin_users WHERE username = :u"),
                {"u": owner_username},
            )
            if result.rowcount:
                LOGGER.info("Removed owner from admin_users username=%s", owner_username)

        admin_id_row = conn.execute(
            text("SELECT id FROM users WHERE email = :email"), {"email": seed_admin_email}
        ).first()
        if admin_id_row is None:
            admin_id_row = conn.execute(
                text(
                    "INSERT INTO users (id, email, name) VALUES (gen_random_uuid(), :email, 'Seed Admin') RETURNING id"
                ),
                {"email": seed_admin_email},
            ).first()
            LOGGER.info("Created seed admin user email=%s id=%s", seed_admin_email, admin_id_row[0])
        admin_id = admin_id_row[0]

        result = conn.execute(
            text("UPDATE cases SET user_id = :uid WHERE user_id IS NULL"),
            {"uid": admin_id},
        )
        if result.rowcount:
            LOGGER.info("Backfilled %d legacy cases with seed admin user_id", result.rowcount)


def seed_subscription_plans(engine: Engine) -> None:
    """Idempotently populate subscription_plans with the default catalog.

    Uses ON CONFLICT DO NOTHING so admin-edited prices/limits survive restarts.
    Plan codes other than "free" can be deleted from the admin UI — they will
    NOT be reseeded after that, since admin intent should win. The "free" plan
    is always (re)created if missing because the quota engine assumes it.
    """
    with engine.begin() as conn:
        for plan in _DEFAULT_PLANS:
            conn.execute(
                text(
                    "INSERT INTO subscription_plans "
                    "(code, title, price_rub, duration_days, monthly_generation_limit, "
                    "allow_edit, description_md, display_order, is_active) "
                    "VALUES (:code, :title, :price_rub, :duration_days, :limit, "
                    ":allow_edit, :description_md, :display_order, TRUE) "
                    "ON CONFLICT (code) DO NOTHING"
                ),
                {
                    "code": plan["code"],
                    "title": plan["title"],
                    "price_rub": plan["price_rub"],
                    "duration_days": plan["duration_days"],
                    "limit": plan["monthly_generation_limit"],
                    "allow_edit": plan["allow_edit"],
                    "description_md": plan["description_md"],
                    "display_order": plan["display_order"],
                },
            )
    LOGGER.info("Seeded subscription_plans defaults (idempotent)")


def reset_stuck_npa_jobs(engine: Engine) -> None:
    """Roll back NPA rows stuck mid-task after a previous crash.

    Same idea as cleanup_stuck_processing_messages: chunking/indexing
    background workers don't survive an app restart, so any row left in
    chunking/indexing has no owner and would otherwise hang forever.
    """
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE npa_sources SET status='failed', "
                "error_text=COALESCE(error_text, 'Job interrupted by app restart'), "
                "updated_at=now() "
                "WHERE status IN ('chunking', 'indexing')"
            )
        )
        if result.rowcount:
            LOGGER.warning(
                "Reset %d stuck NPA jobs (chunking/indexing) on startup",
                result.rowcount,
            )


def cleanup_stuck_processing_messages(engine: Engine) -> None:
    """Mark any PROCESSING messages as ERROR.

    These are leftovers from a previous `app` crash/restart: the background
    task that owned them is gone, so they will never reach DONE on their own.
    Called from the FastAPI lifespan so a fresh boot starts in a clean state.
    """
    with engine.begin() as conn:
        result = conn.execute(
            text(
                "UPDATE messages SET status='error', "
                "error_text=COALESCE(error_text, 'Run interrupted by app restart'), "
                "updated_at=now() WHERE status='processing'"
            )
        )
        if result.rowcount:
            LOGGER.warning(
                "Marked %d stuck PROCESSING messages as ERROR on startup",
                result.rowcount,
            )
