"""Chat session and grounded message routes."""

from fastapi import APIRouter, status

from ...schemas.chats import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatTurnResponse,
)
from ...services import ChatService
from ..dependencies import (
    ApplicationSettings,
    CurrentUser,
    DatabaseSession,
    ModelCatalog,
)

router = APIRouter(prefix="/chat/sessions", tags=["chat"])


def _service(session, settings, catalog) -> ChatService:
    return ChatService(session, settings, catalog)


@router.post(
    "",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a clean session-scoped chat context",
)
async def create_chat_session(
    request: ChatSessionCreate,
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
    catalog: ModelCatalog,
) -> ChatSessionResponse:
    chat = await _service(session, settings, catalog).create_session(user.id, request.title)
    return ChatSessionResponse.model_validate(chat)


@router.get("", response_model=list[ChatSessionResponse], summary="List the current user's chats")
async def list_chat_sessions(
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
    catalog: ModelCatalog,
) -> list[ChatSessionResponse]:
    return await _service(session, settings, catalog).list_sessions(user.id)


@router.get(
    "/{session_id}/messages",
    response_model=list[ChatMessageResponse],
    summary="List messages in one owned chat session",
)
async def list_chat_messages(
    session_id: str,
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
    catalog: ModelCatalog,
) -> list[ChatMessageResponse]:
    return await _service(session, settings, catalog).list_messages(user.id, session_id)


@router.post(
    "/{session_id}/messages",
    response_model=ChatTurnResponse,
    summary="Run one session-scoped LangGraph RAG turn",
)
async def send_chat_message(
    session_id: str,
    request: ChatMessageRequest,
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
    catalog: ModelCatalog,
) -> ChatTurnResponse:
    return await _service(session, settings, catalog).send_message(
        user_id=user.id,
        session_id=session_id,
        request=request,
    )
