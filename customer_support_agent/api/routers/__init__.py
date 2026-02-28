from customer_support_agent.api.routers.drafts import router as drafts_router
from customer_support_agent.api.routers.health import router as health_router
from customer_support_agent.api.routers.knowledge import router as knowledge_router
from customer_support_agent.api.routers.memory import router as memory_router
from customer_support_agent.api.routers.tickets import router as tickets_router

__all__ = [
    "health_router",
    "tickets_router",
    "drafts_router",
    "knowledge_router",
    "memory_router",
]
