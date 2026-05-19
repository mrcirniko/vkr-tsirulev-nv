"""User-facing preferences API: /api/preferences {GET, PUT}.

These prefs feed straight into the LangGraph run — see
`server._start_fresh_run` for the bridge and `agent.graph._route_after_written_form`
for the consumer side.
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.session import require_user
from db.crud import get_user_preferences, upsert_user_preferences
from db.models import ALLOWED_CONTRACT_POLICIES, User

LOGGER = logging.getLogger("app.preferences.routes")

router = APIRouter(prefix="/api/preferences", tags=["preferences"])


# Literal pins the API surface; DB layer enforces ALLOWED_CONTRACT_POLICIES independently.
PolicyValue = Literal["legal_only", "always", "always_ask"]
ThemeValue = Literal["dark", "light"]


class PreferencesDTO(BaseModel):
    contract_generation_policy: PolicyValue
    ask_personal_data: bool
    # None means no explicit choice — frontend falls back to prefers-color-scheme.
    theme: ThemeValue | None = None


class PreferencesUpdateRequest(BaseModel):
    contract_generation_policy: PolicyValue | None = Field(default=None)
    ask_personal_data: bool | None = Field(default=None)
    theme: ThemeValue | None = Field(default=None)


def _to_dto(prefs: dict) -> PreferencesDTO:
    return PreferencesDTO(
        contract_generation_policy=prefs["contract_generation_policy"],
        ask_personal_data=prefs["ask_personal_data"],
        theme=prefs.get("theme"),
    )


@router.get("", response_model=PreferencesDTO)
def get_preferences(user: User = Depends(require_user)) -> PreferencesDTO:
    return _to_dto(get_user_preferences(user.id))


@router.put("", response_model=PreferencesDTO)
def put_preferences(
    payload: PreferencesUpdateRequest,
    user: User = Depends(require_user),
) -> PreferencesDTO:
    try:
        updated = upsert_user_preferences(
            user.id,
            contract_generation_policy=payload.contract_generation_policy,
            ask_personal_data=payload.ask_personal_data,
            theme=payload.theme,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    LOGGER.info(
        "Updated preferences user_id=%s policy=%s ask_personal=%s theme=%s",
        user.id,
        updated["contract_generation_policy"],
        updated["ask_personal_data"],
        updated["theme"],
    )
    # Defensive: catalogue mismatch shouldn't reach the frontend.
    if updated["contract_generation_policy"] not in ALLOWED_CONTRACT_POLICIES:
        raise HTTPException(500, "Persisted policy not in allowlist")
    return _to_dto(updated)
