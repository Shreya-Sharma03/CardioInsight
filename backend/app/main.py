"""
CardioInsight FastAPI Backend Application.
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes_predict import router as predict_router
from backend.app.api.routes_insights import router as insights_router
from backend.app.services.model_service import get_model_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cardioinsight_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads model checkpoint and validation feature matrices once on startup.
    """
    logger.info("Initializing CardioInsight ModelService...")
    service = get_model_service()
    service.load()
    logger.info(
        f"ModelService loaded successfully on {service.device}. Total validation records: {service.total_records}"
    )
    yield
    logger.info("Shutting down CardioInsight backend...")


app = FastAPI(
    title="CardioInsight API",
    description="Heart Failure Risk Prediction & Adaptive Reliability Fusion Backend Service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration for local frontend development (Vite / Next.js)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers under /api prefix
app.include_router(predict_router, prefix="/api")
app.include_router(insights_router, prefix="/api")


@app.get("/", summary="Root health redirect")
def root_status():
    return {
        "service": "CardioInsight API",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
