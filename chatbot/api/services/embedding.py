"""Embedding service for text vectorization with multi-provider support."""

import asyncio
from typing import List

from ..config import Settings, get_settings


class EmbeddingService:
    """Service for generating text embeddings using OpenAI or Gemini."""

    def __init__(self, settings: Settings | None = None):
        """Initialize the embedding service.

        Args:
            settings: Application settings. If None, loads from environment.
        """
        self.settings = settings or get_settings()
        self.provider = self.settings.llm_provider
        self.dimensions = self.settings.embedding_dimensions
        
        if self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=self.settings.gemini_api_key)
            self.model = self.settings.gemini_embedding_model
        else:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
            self.model = self.settings.openai_embedding_model
            # Initialize tokenizer for OpenAI
            import tiktoken
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Input text

        Returns:
            Number of tokens (approximate for Gemini)
        """
        if self.provider == "gemini":
            # Approximate: ~4 chars per token
            return len(text) // 4
        return len(self._tokenizer.encode(text))

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if self.provider == "gemini":
            # Gemini embeddings API (sync, run in executor)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                )
            )
            return result.embeddings[0].values
        else:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            )
            return response.data[0].embedding

    async def embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each API call

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            if self.provider == "gemini":
                # Gemini batch embedding
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda b=batch: self.client.models.embed_content(
                        model=self.model,
                        contents=b,
                    )
                )
                batch_embeddings = [emb.values for emb in result.embeddings]
            else:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )
                batch_embeddings = [item.embedding for item in response.data]
            
            all_embeddings.extend(batch_embeddings)

            # Small delay to avoid rate limiting
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query.

        This is a convenience wrapper around embed_text for query embeddings.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        return await self.embed_text(query)
