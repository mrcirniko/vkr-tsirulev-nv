"""Test fixtures and env priming.

The app's `config.Settings` uses os.getenv at import time, so we have to seed a
few env vars before any `app.*` import reaches the dataclass. Real DB calls are
mocked in the individual test modules — `conftest` only ensures imports do not
explode in environments without a Postgres or YooKassa set up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Force a deterministic in-test config without touching .env.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test_db")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("SESSION_SECRET", "test-secret-not-real")
# Force-clear billing creds even if docker compose injected real values via
# env_file: tests assume an "unconfigured" baseline and override per-case via
# monkeypatch / object.__setattr__ on settings. setdefault is not enough here
# because the parent process env already has these set.
os.environ["YOOKASSA_SHOP_ID"] = ""
os.environ["YOOKASSA_SECRET_KEY"] = ""

# `pyproject.toml` already sets pythonpath=["app"], but pytest sometimes runs
# from places where it's overridden by another tool. Keep this defensive.
ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
