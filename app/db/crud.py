from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import (
    ALLOWED_CONTRACT_POLICIES,
    CONTRACT_POLICY_LEGAL_ONLY,
    Case,
    CaseStatus,
    ContractVersion,
    Message,
    MessageRole,
    MessageStatus,
    User,
    UserPreferences,
)
from db.session import SessionLocal


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@contextmanager
def session_scope() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_case(title: str, deal_description: str, owner_id: str | UUID | None = None) -> Case:
    with session_scope() as session:
        owner_uuid = UUID(str(owner_id)) if owner_id is not None else None
        case = Case(title=title, deal_description=deal_description, user_id=owner_uuid)
        session.add(case)
        session.flush()
        session.refresh(case)
        return case


def get_case(case_id: str | UUID) -> Case | None:
    with SessionLocal() as session:
        return session.get(Case, UUID(str(case_id)))


def case_belongs_to_owner(case_id: str | UUID, owner_id: str | UUID) -> bool:
    case = get_case(case_id)
    if case is None or case.user_id is None:
        return False
    return case.user_id == UUID(str(owner_id))


# def list_cases() -> list[Case]:
#     # UNUSED — superseded by list_cases_for_owner; kept here for reference.
#     with SessionLocal() as session:
#         stmt = select(Case).order_by(Case.updated_at.desc(), Case.created_at.desc())
#         return list(session.scalars(stmt).all())


def list_cases_for_owner(owner_id: str | UUID) -> list[Case]:
    owner_uuid = UUID(str(owner_id))
    with SessionLocal() as session:
        stmt = select(Case).where(Case.user_id == owner_uuid).order_by(Case.updated_at.desc(), Case.created_at.desc())
        return list(session.scalars(stmt).all())


def add_message(
    case_id: str | UUID,
    role: str,
    content: str,
    *,
    status: MessageStatus | str = MessageStatus.DONE,
    langgraph_run_id: str | None = None,
) -> Message:
    case_uuid = UUID(str(case_id))
    if isinstance(status, str):
        status = MessageStatus(status)
    with session_scope() as session:
        case = session.get(Case, case_uuid)
        if case is None:
            raise ValueError(f"Case {case_id} not found")

        message = Message(
            case_id=case_uuid,
            role=MessageRole(role),
            content=content,
            status=status,
            langgraph_run_id=langgraph_run_id,
        )
        case.updated_at = _utcnow_naive()
        session.add(message)
        session.flush()
        session.refresh(message)
        return message


def update_message(
    message_id: str | UUID,
    *,
    content: str | None = None,
    status: MessageStatus | str | None = None,
    langgraph_run_id: str | None = None,
    error_text: str | None = None,
) -> Message:
    message_uuid = UUID(str(message_id))
    if isinstance(status, str):
        status = MessageStatus(status)
    with session_scope() as session:
        message = session.get(Message, message_uuid)
        if message is None:
            raise ValueError(f"Message {message_id} not found")

        if content is not None:
            message.content = content
        if status is not None:
            message.status = status
        if langgraph_run_id is not None:
            message.langgraph_run_id = langgraph_run_id
        if error_text is not None:
            message.error_text = error_text
        message.updated_at = _utcnow_naive()
        session.flush()
        session.refresh(message)
        return message


def get_message(message_id: str | UUID) -> Message | None:
    with SessionLocal() as session:
        return session.get(Message, UUID(str(message_id)))


def get_latest_message(case_id: str | UUID) -> Message | None:
    case_uuid = UUID(str(case_id))
    with SessionLocal() as session:
        stmt = (
            select(Message)
            .where(Message.case_id == case_uuid)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(1)
        )
        return session.scalars(stmt).first()


def update_case_description(case_id: str | UUID, deal_description: str) -> Case:
    case_uuid = UUID(str(case_id))
    with session_scope() as session:
        case = session.get(Case, case_uuid)
        if case is None:
            raise ValueError(f"Case {case_id} not found")

        case.deal_description = deal_description
        case.updated_at = _utcnow_naive()
        session.flush()
        session.refresh(case)
        return case


def update_case_title(case_id: str | UUID, title: str) -> Case:
    case_uuid = UUID(str(case_id))
    with session_scope() as session:
        case = session.get(Case, case_uuid)
        if case is None:
            raise ValueError(f"Case {case_id} not found")

        case.title = title
        case.updated_at = _utcnow_naive()
        session.flush()
        session.refresh(case)
        return case


def get_messages(case_id: str | UUID) -> list[Message]:
    case_uuid = UUID(str(case_id))
    with SessionLocal() as session:
        stmt = select(Message).where(Message.case_id == case_uuid).order_by(Message.created_at.asc(), Message.id.asc())
        return list(session.scalars(stmt).all())


def save_contract_version(case_id: str | UUID, content_md: str, docx_path: str | None) -> ContractVersion:
    case_uuid = UUID(str(case_id))
    with session_scope() as session:
        case = session.get(Case, case_uuid)
        if case is None:
            raise ValueError(f"Case {case_id} not found")

        latest_version_number = session.scalar(
            select(func.max(ContractVersion.version_number)).where(ContractVersion.case_id == case_uuid)
        )
        next_version_number = 1 if latest_version_number is None else latest_version_number + 1

        version = ContractVersion(
            case_id=case_uuid,
            version_number=next_version_number,
            content_md=content_md,
            docx_path=docx_path,
        )
        case.updated_at = _utcnow_naive()
        session.add(version)
        session.flush()
        session.refresh(version)
        return version


def get_contract_versions(case_id: str | UUID) -> list[ContractVersion]:
    case_uuid = UUID(str(case_id))
    with SessionLocal() as session:
        stmt = (
            select(ContractVersion)
            .where(ContractVersion.case_id == case_uuid)
            .order_by(ContractVersion.version_number.asc(), ContractVersion.id.asc())
        )
        return list(session.scalars(stmt).all())


def get_latest_version(case_id: str | UUID) -> ContractVersion | None:
    case_uuid = UUID(str(case_id))
    with SessionLocal() as session:
        stmt = (
            select(ContractVersion)
            .where(ContractVersion.case_id == case_uuid)
            .order_by(ContractVersion.version_number.desc(), ContractVersion.id.desc())
            .limit(1)
        )
        return session.scalars(stmt).first()


def update_case_status(case_id: str | UUID, status: str) -> Case:
    case_uuid = UUID(str(case_id))
    with session_scope() as session:
        case = session.get(Case, case_uuid)
        if case is None:
            raise ValueError(f"Case {case_id} not found")

        case.status = CaseStatus(status)
        case.updated_at = _utcnow_naive()
        session.flush()
        session.refresh(case)
        return case


def get_user(user_id: str | UUID) -> User | None:
    with SessionLocal() as session:
        return session.get(User, UUID(str(user_id)))


# def get_user_by_email(email: str) -> User | None:
#     # UNUSED — kept for reference; lookup happens inline inside upsert_user_by_google.
#     with SessionLocal() as session:
#         return session.scalars(select(User).where(User.email == email)).first()


def upsert_user_by_google(
    google_id: str,
    email: str,
    name: str | None = None,
    picture_url: str | None = None,
) -> User:
    with session_scope() as session:
        user = session.scalars(select(User).where(User.google_id == google_id)).first()
        if user is None:
            user = session.scalars(select(User).where(User.email == email)).first()
        if user is None:
            user = User(google_id=google_id, email=email, name=name, picture_url=picture_url)
            session.add(user)
        else:
            if user.google_id is None:
                user.google_id = google_id
            if name and user.name != name:
                user.name = name
            if picture_url and user.picture_url != picture_url:
                user.picture_url = picture_url
        session.flush()
        session.refresh(user)
        return user


def upsert_user_by_yandex(
    yandex_id: str,
    email: str,
    name: str | None = None,
    picture_url: str | None = None,
) -> User:
    with session_scope() as session:
        user = session.scalars(select(User).where(User.yandex_id == yandex_id)).first()
        if user is None:
            user = session.scalars(select(User).where(User.email == email)).first()
        if user is None:
            user = User(yandex_id=yandex_id, email=email, name=name, picture_url=picture_url)
            session.add(user)
        else:
            if user.yandex_id is None:
                user.yandex_id = yandex_id
            if name and user.name != name:
                user.name = name
            if picture_url and user.picture_url != picture_url:
                user.picture_url = picture_url
        session.flush()
        session.refresh(user)
        return user


# ---- User preferences ----------------------------------------------------


def _default_prefs(user_id: UUID) -> dict:
    return {
        "user_id": user_id,
        "contract_generation_policy": CONTRACT_POLICY_LEGAL_ONLY,
        "ask_personal_data": True,
    }


def get_user_preferences(user_id: str | UUID) -> dict:
    """Return user prefs as a plain dict, or sane defaults if no row exists."""
    user_uuid = UUID(str(user_id))
    with SessionLocal() as session:
        row = session.get(UserPreferences, user_uuid)
        if row is None:
            return _default_prefs(user_uuid)
        return {
            "user_id": row.user_id,
            "contract_generation_policy": row.contract_generation_policy,
            "ask_personal_data": row.ask_personal_data,
        }


def upsert_user_preferences(
    user_id: str | UUID,
    *,
    contract_generation_policy: str | None = None,
    ask_personal_data: bool | None = None,
) -> dict:
    """Insert-or-update the prefs row; returns the freshly persisted values.

    None means "leave unchanged" — callers send only the fields the user
    actually flipped.
    """
    user_uuid = UUID(str(user_id))
    if contract_generation_policy is not None and contract_generation_policy not in ALLOWED_CONTRACT_POLICIES:
        raise ValueError(f"contract_generation_policy must be one of {ALLOWED_CONTRACT_POLICIES}")
    with session_scope() as session:
        row = session.get(UserPreferences, user_uuid)
        if row is None:
            row = UserPreferences(
                user_id=user_uuid,
                contract_generation_policy=contract_generation_policy or CONTRACT_POLICY_LEGAL_ONLY,
                ask_personal_data=True if ask_personal_data is None else bool(ask_personal_data),
            )
            session.add(row)
        else:
            if contract_generation_policy is not None:
                row.contract_generation_policy = contract_generation_policy
            if ask_personal_data is not None:
                row.ask_personal_data = bool(ask_personal_data)
        session.flush()
        session.refresh(row)
        return {
            "user_id": row.user_id,
            "contract_generation_policy": row.contract_generation_policy,
            "ask_personal_data": row.ask_personal_data,
        }
