"""FastAPI application for the RAG chatbot."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .dependencies import cleanup_services, get_database_service, get_vector_store_service
from .models import HealthResponse
from .routers import chat_router, chapters_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    settings = get_settings()
    print(f"Starting RAG Chatbot API...")
    print(f"  - OpenAI Model: {settings.openai_chat_model}")
    print(f"  - Embedding Model: {settings.openai_embedding_model}")
    print(f"  - Qdrant Collection: {settings.qdrant_collection_name}")

    # Initialize services
    try:
        database = await get_database_service()
        await database.initialize_tables()
        print("  - PostgreSQL: connected")

        vector_store = await get_vector_store_service()
        await vector_store.initialize_collection()
        print("  - Qdrant: connected")

    except Exception as e:
        print(f"  - Warning: Service initialization error: {e}")

    yield

    # Shutdown
    print("Shutting down RAG Chatbot API...")
    await cleanup_services()


# Create FastAPI app
app = FastAPI(
    title="Physical AI Textbook RAG Chatbot",
    description="RAG-based Q&A system for the Physical AI & Humanoid Robotics textbook",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(chapters_router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Physical AI Textbook RAG Chatbot",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns the health status of the API and connected services.
    """
    # Check service health
    qdrant_healthy = False
    postgres_healthy = False

    try:
        database = await get_database_service()
        postgres_healthy = await database.health_check()
    except Exception:
        pass

    try:
        vector_store = await get_vector_store_service()
        qdrant_healthy = await vector_store.health_check()
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if (qdrant_healthy and postgres_healthy) else "degraded",
        version="1.0.0",
        qdrant_connected=qdrant_healthy,
        postgres_connected=postgres_healthy,
    )


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "chatbot.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
