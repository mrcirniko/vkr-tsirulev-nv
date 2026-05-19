from __future__ import annotations

from config import settings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from db.migrate import run_migrations
from db.models import Base

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
    Base.metadata.create_all(engine)
    run_migrations(engine, settings.seed_admin_email, settings.owner_username)
