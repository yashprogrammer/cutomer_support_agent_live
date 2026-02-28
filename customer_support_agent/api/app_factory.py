from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from customer_support_agent.api.routers import (
    drafts_router,
    health_router,
    knowledge_router,
    memory_router,
    tickets_router,
)
from customer_support_agent.core.settings import Settings, ensure_directories, get_settings
from customer_support_agent.repositories.sqlite import init_db



def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        ensure_directories(resolved_settings)
        init_db()
        yield

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)

    app.include_router(health_router)
    app.include_router(tickets_router)
    app.include_router(drafts_router)
    app.include_router(knowledge_router)
    app.include_router(memory_router)

    return app