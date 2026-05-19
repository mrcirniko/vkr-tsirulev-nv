"""Pydantic DTOs for the cases/chat domain.

Kept dependency-free (no DB, no FastAPI) so other layers can import them
without dragging in the runtime stack. Mapping ORM → DTO lives in
`cases.service`, not here.
"""

from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    case_id: str | None = None


class CreateCaseRequest(BaseModel):
    prompt: str


class VersionDTO(BaseModel):
    id: str
    version_number: int
    created_at: str
    content_md: str
    docx_url: str | None


class MessageDTO(BaseModel):
    id: str
    role: str
    content: str
    status: str
    created_at: str
    updated_at: str
    error_text: str | None = None


class CaseSummaryDTO(BaseModel):
    id: str
    title: str
    status: str
    deal_type: str | None
    created_at: str
    updated_at: str
    latest_docx_url: str | None


class CaseDetailDTO(CaseSummaryDTO):
    deal_description: str
    messages: list[MessageDTO]
    versions: list[VersionDTO]
    clarification_needed: bool = False
    clarification_question: str | None = None
    processing_stage: str | None = None
    studio_url: str | None = None


class ChatStartResponseDTO(BaseModel):
    case_id: str
    user_message: MessageDTO
    assistant_message: MessageDTO
