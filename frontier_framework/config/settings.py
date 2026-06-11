from .main import app
from .extended_routes import extended_router

app.include_router(extended_router)

__all__ = ["app"]