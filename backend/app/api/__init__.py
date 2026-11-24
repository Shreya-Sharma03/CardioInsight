"""
API routers package.
"""
from backend.app.api.routes_predict import router as predict_router
from backend.app.api.routes_insights import router as insights_router

__all__ = ["predict_router", "insights_router"]
