"""
PeakPulse Engine — FastAPI Application Factory.

Registers the traffic controller router and configures CORS,
logging, and lifespan events (NLP model warm-up on startup).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.traffic_controller import router as traffic_router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_level = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("peakpulse")


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    - **Startup**: Pre-load the NLP sentence-transformer model so the first
      request doesn't pay the cold-start penalty.
    - **Shutdown**: Placeholder for cleanup.
    """
    logger.info("🚀 PeakPulse Engine starting up …")
    try:
        from app.services.nlp_service import NLPService
        nlp = NLPService()
        # Trigger lazy model load by running a test mapping
        nlp.map_headers(["Thru"])
        model_info = nlp.get_model_info()
        logger.info(
            "✅ Transformer NLP model loaded: %s (%s)",
            model_info["model_name"],
            model_info["description"],
        )
    except Exception as e:
        logger.warning("⚠️  NLP model pre-load skipped: %s", e)
    yield
    logger.info("🛑 PeakPulse Engine shutting down")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PeakPulse Engine",
    description=(
        "Automated traffic movement count processor. "
        "Upload a Preliminary Excel workbook and receive a Final Excel + PDF report "
        "with peak hour analysis, PHF calculations, and Sankey-style flow diagrams."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS – allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the traffic controller
app.include_router(traffic_router)


# Root redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "PeakPulse Engine v1.0.0", "docs": "/docs"}
