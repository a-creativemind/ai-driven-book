"""Chat API endpoints for RAG-based Q&A."""

from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    ChatRequest,
    ChatResponse,
    SelectionChatRequest,
    SourceReference,
)
from ..services.rag import RAGService
from ..dependencies import get_rag_service

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Process a chat message using RAG.

    Retrieves relevant context from the textbook and generates
    an answer using the LLM.

    Args:
        request: Chat request with message and optional history
        rag_service: Injected RAG service

    Returns:
        ChatResponse with answer and source citations
    """
    try:
        answer, sources, tokens_used = await rag_service.chat(
            message=request.message,
            history=request.history,
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            tokens_used=tokens_used,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}",
        )


@router.post("/selection", response_model=ChatResponse)
async def chat_with_selection(
    request: SelectionChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Process a question about user-selected text.

    Uses the selected text as primary context for answering.

    Args:
        request: Selection chat request with message, selected text, and optional history
        rag_service: Injected RAG service

    Returns:
        ChatResponse with answer (no sources since context is the selection)
    """
    try:
        answer, sources, tokens_used = await rag_service.chat_with_selection(
            message=request.message,
            selected_text=request.selected_text,
            chapter_id=request.chapter_id,
            history=request.history,
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            tokens_used=tokens_used,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing selection chat request: {str(e)}",
        )
