"""RAG service for retrieval-augmented generation with multi-provider support."""

import asyncio
from typing import Any, Dict, List, Optional

from ..config import Settings, get_settings
from ..models import ChatMessage, SourceReference
from ..utils.prompts import (
    SELECTION_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_rag_prompt,
    build_selection_prompt,
)
from .database import DatabaseService
from .embedding import EmbeddingService
from .vector_store import VectorStoreService


class RAGService:
    """Service for RAG-based question answering."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        database: DatabaseService,
        settings: Settings | None = None,
    ):
        """Initialize the RAG service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Service for vector similarity search
            database: Service for metadata retrieval
            settings: Application settings
        """
        self.embedding = embedding_service
        self.vector_store = vector_store
        self.database = database
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_provider
        
        if self.provider == "gemini":
            from google import genai
            self.gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
            self.chat_model = self.settings.gemini_chat_model
        else:
            from openai import AsyncOpenAI
            self.openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            self.chat_model = self.settings.openai_chat_model

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks for a query.

        Args:
            query: User's question
            top_k: Number of results to retrieve
            chapter_filter: Optional chapter ID to restrict search

        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = await self.embedding.embed_query(query)

        # Build filter conditions
        filter_conditions = None
        if chapter_filter:
            filter_conditions = {"chapter_id": chapter_filter}

        # Search vector store
        results = await self.vector_store.search(
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=self.settings.similarity_threshold,
            filter_conditions=filter_conditions,
        )

        # Enrich with chapter metadata
        enriched_results = []
        for result in results:
            chapter_id = result.get("chapter_id", "")
            chapter = await self.database.get_chapter(chapter_id) if chapter_id else None

            enriched_results.append(
                {
                    "chunk_id": result["chunk_id"],
                    "chapter_id": chapter_id,
                    "chapter_title": chapter["title"] if chapter else chapter_id,
                    "section_title": result.get("section_title"),
                    "content": result.get("content", ""),
                    "difficulty": result.get("difficulty", "intermediate"),
                    "score": result["score"],
                }
            )

        return enriched_results

    async def _generate_with_gemini(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
    ) -> tuple[str, int]:
        """Generate response using Gemini API."""
        # Convert messages format for Gemini
        # Gemini uses a different format: system instruction + contents
        system_instruction = None
        contents = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Run sync Gemini API in executor
        loop = asyncio.get_event_loop()
        
        def call_gemini():
            return self.gemini_client.models.generate_content(
                model=self.chat_model,
                contents=contents,
                config={
                    "system_instruction": system_instruction,
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7,
                }
            )
        
        response = await loop.run_in_executor(None, call_gemini)
        
        answer = response.text or ""
        # Gemini doesn't always return token counts
        tokens_used = getattr(response, 'usage_metadata', None)
        if tokens_used:
            tokens_used = getattr(tokens_used, 'total_token_count', 0)
        else:
            tokens_used = 0
            
        return answer, tokens_used

    async def _generate_with_openai(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
    ) -> tuple[str, int]:
        """Generate response using OpenAI API."""
        response = await self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,  # type: ignore
            temperature=0.7,
            max_tokens=max_tokens,
        )

        answer = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

        return answer, tokens_used

    async def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        history: List[ChatMessage],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> tuple[str, int]:
        """Generate a response using the LLM with retrieved context.

        Args:
            query: User's question
            context_chunks: Retrieved context from vector search
            history: Conversation history
            system_prompt: System prompt to use

        Returns:
            Tuple of (response text, tokens used)
        """
        # Build the prompt with context
        user_prompt = build_rag_prompt(query, context_chunks)

        # Build messages for the API
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        for msg in history[-6:]:  # Keep last 6 messages for context
            messages.append({"role": msg.role, "content": msg.content})

        # Add current query with context
        messages.append({"role": "user", "content": user_prompt})

        # Generate response based on provider
        if self.provider == "gemini":
            return await self._generate_with_gemini(messages)
        else:
            return await self._generate_with_openai(messages)

    async def chat(
        self,
        message: str,
        history: List[ChatMessage] | None = None,
    ) -> tuple[str, List[SourceReference], int]:
        """Process a chat message with full RAG pipeline.

        Args:
            message: User's question
            history: Optional conversation history

        Returns:
            Tuple of (answer, sources, tokens_used)
        """
        history = history or []

        # Retrieve relevant context
        context_chunks = await self.retrieve_context(
            query=message,
            top_k=self.settings.top_k_results,
        )

        # Generate response
        answer, tokens_used = await self.generate_response(
            query=message,
            context_chunks=context_chunks,
            history=history,
        )

        # Build source references
        sources = [
            SourceReference(
                chunk_id=chunk["chunk_id"],
                chapter_id=chunk["chapter_id"],
                chapter_title=chunk["chapter_title"],
                section_title=chunk.get("section_title"),
                content_preview=chunk["content"][:300] if chunk.get("content") else "",
                similarity_score=chunk["score"],
                difficulty=chunk.get("difficulty", "intermediate"),
            )
            for chunk in context_chunks
        ]

        return answer, sources, tokens_used

    async def chat_with_selection(
        self,
        message: str,
        selected_text: str,
        chapter_id: Optional[str] = None,
        history: List[ChatMessage] | None = None,
    ) -> tuple[str, List[SourceReference], int]:
        """Process a chat message about selected text.

        Args:
            message: User's question
            selected_text: Text selected by the user
            chapter_id: Optional chapter ID for context
            history: Optional conversation history

        Returns:
            Tuple of (answer, sources, tokens_used)
        """
        history = history or []

        # Get chapter info if provided
        chapter_info = None
        if chapter_id:
            chapter_info = await self.database.get_chapter(chapter_id)

        # Build selection-specific prompt
        user_prompt = build_selection_prompt(
            query=message,
            selected_text=selected_text,
            chapter_info=chapter_info,
        )

        # Build messages
        messages = [
            {"role": "system", "content": SELECTION_SYSTEM_PROMPT},
        ]

        for msg in history[-4:]:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_prompt})

        # Generate response based on provider
        if self.provider == "gemini":
            answer, tokens_used = await self._generate_with_gemini(messages, max_tokens=1200)
        else:
            answer, tokens_used = await self._generate_with_openai(messages, max_tokens=1200)

        # For selection-based chat, we don't return RAG sources
        sources: List[SourceReference] = []

        return answer, sources, tokens_used
