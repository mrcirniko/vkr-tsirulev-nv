from db.crud import (
    add_message,
    create_case,
    get_case,
    get_contract_versions,
    get_latest_version,
    get_messages,
    # list_cases,  # UNUSED — see crud.py
    save_contract_version,
    update_case_status,
)
from db.models import Base, Case, CaseStatus, ContractVersion, Message, MessageRole, MessageStatus
from db.session import SessionLocal, engine, init_db

__all__ = [
    "Base",
    "Case",
    "CaseStatus",
    "ContractVersion",
    "Message",
    "MessageRole",
    "MessageStatus",
    "SessionLocal",
    "engine",
    "init_db",
    "create_case",
    "get_case",
    # "list_cases",  # UNUSED
    "add_message",
    "get_messages",
    "save_contract_version",
    "get_contract_versions",
    "get_latest_version",
    "update_case_status",
]
