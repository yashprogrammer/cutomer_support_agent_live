from __future__ import annotations

import uvicorn

from customer_support_agent.api.app_factory import create_app
from customer_support_agent.core.settings import get_settings

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=False)