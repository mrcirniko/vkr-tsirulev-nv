from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, Numeric, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class CaseStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(str, enum.Enum):
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


CASE_STATUS_ENUM = Enum(
    CaseStatus,
    name="case_status",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)
MESSAGE_ROLE_ENUM = Enum(
    MessageRole,
    name="message_role",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)
MESSAGE_STATUS_ENUM = Enum(
    MessageStatus,
    name="message_status",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    google_id: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True, index=True)
    yandex_id: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True, index=True)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    picture_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )

    cases: Mapped[list[Case]] = relationship(back_populates="owner")


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[CaseStatus] = mapped_column(
        CASE_STATUS_ENUM,
        nullable=False,
        default=CaseStatus.IN_PROGRESS,
        server_default=CaseStatus.IN_PROGRESS.value,
    )
    deal_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    deal_description: Mapped[str] = mapped_column(Text, nullable=False)

    owner: Mapped[User | None] = relationship(back_populates="cases")
    messages: Mapped[list[Message]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    contract_versions: Mapped[list[ContractVersion]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="ContractVersion.version_number",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    role: Mapped[MessageRole] = mapped_column(MESSAGE_ROLE_ENUM, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[MessageStatus] = mapped_column(
        MESSAGE_STATUS_ENUM,
        nullable=False,
        default=MessageStatus.DONE,
        server_default=MessageStatus.DONE.value,
    )
    langgraph_run_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    case: Mapped[Case] = relationship(back_populates="messages")


class NpaStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    CHUNKING = "chunking"
    READY = "ready"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


NPA_STATUS_ENUM = Enum(
    NpaStatus,
    name="npa_status",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


ALLOWED_THEMES = ("dark", "light")


class AdminUser(Base):
    __tablename__ = "admin_users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(Text, nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    theme: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )


# String column (not enum) so new policies don't need a migration. Validated at API/graph layer.
CONTRACT_POLICY_LEGAL_ONLY = "legal_only"
CONTRACT_POLICY_ALWAYS = "always"
CONTRACT_POLICY_ALWAYS_ASK = "always_ask"
ALLOWED_CONTRACT_POLICIES = (CONTRACT_POLICY_LEGAL_ONLY, CONTRACT_POLICY_ALWAYS, CONTRACT_POLICY_ALWAYS_ASK)


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    contract_generation_policy: Mapped[str] = mapped_column(
        Text, nullable=False, default=CONTRACT_POLICY_LEGAL_ONLY, server_default=CONTRACT_POLICY_LEGAL_ONLY
    )
    ask_personal_data: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    theme: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=_utcnow,
        onupdate=_utcnow,
    )


class NpaSource(Base):
    __tablename__ = "npa_sources"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("admin_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=_utcnow,
        onupdate=_utcnow,
    )
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    raw_format: Mapped[str] = mapped_column(Text, nullable=False)  # 'rtf' | 'txt'
    conversion_options: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict, server_default="{}")
    raw_txt_s3_key: Mapped[str] = mapped_column(Text, nullable=False)
    chunks_json_s3_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunks_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_indexed_collection: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_indexed_with_refs: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[NpaStatus] = mapped_column(
        NPA_STATUS_ENUM,
        nullable=False,
        default=NpaStatus.UPLOADED,
        server_default=NpaStatus.UPLOADED.value,
    )
    error_text: Mapped[str | None] = mapped_column(Text, nullable=True)


# Plan codes are admin-editable strings, not a Postgres enum. FREE_PLAN_CODE is hard-referenced
# by quota fallback and edit-gate logic.
FREE_PLAN_CODE = "free"


class SubscriptionStatus(str, enum.Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class PaymentStatus(str, enum.Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"


SUBSCRIPTION_STATUS_ENUM = Enum(
    SubscriptionStatus,
    name="subscription_status",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)
PAYMENT_STATUS_ENUM = Enum(
    PaymentStatus,
    name="payment_status",
    values_callable=lambda enum_cls: [member.value for member in enum_cls],
)


class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"

    code: Mapped[str] = mapped_column(Text, primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    price_rub: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, default=Decimal("0"))
    duration_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    monthly_generation_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    allow_edit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    features_json: Mapped[list] = mapped_column(JSONB, nullable=False, default=list, server_default="[]")
    description_md: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    display_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="true")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=_utcnow,
        onupdate=_utcnow,
    )


class UserSubscription(Base):
    __tablename__ = "user_subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_code: Mapped[str] = mapped_column(
        Text,
        ForeignKey("subscription_plans.code", ondelete="RESTRICT"),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[SubscriptionStatus] = mapped_column(
        SUBSCRIPTION_STATUS_ENUM,
        nullable=False,
        default=SubscriptionStatus.ACTIVE,
        server_default=SubscriptionStatus.ACTIVE.value,
    )
    yookassa_payment_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )


class PaymentIntent(Base):
    __tablename__ = "payment_intents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_code: Mapped[str] = mapped_column(Text, nullable=False)
    amount_rub: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    yookassa_payment_id: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True)
    idempotence_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    status: Mapped[PaymentStatus] = mapped_column(
        PAYMENT_STATUS_ENUM,
        nullable=False,
        default=PaymentStatus.PENDING,
        server_default=PaymentStatus.PENDING.value,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=_utcnow,
        onupdate=_utcnow,
    )


class ContractVersion(Base):
    __tablename__ = "contract_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow
    )
    content_md: Mapped[str] = mapped_column(Text, nullable=False)
    docx_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    case: Mapped[Case] = relationship(back_populates="contract_versions")
