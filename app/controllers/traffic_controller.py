"""
Traffic Controller – API route definitions.

Defines endpoints for health check, file upload (full processing),
and validation (anomaly detection only).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import ProcessingResult, AnomalyAlert
from app.repository.file_repository import FileRepository
from app.services.analytics_service import AnalyticsService
from app.services.anomaly_service import AnomalyService
from app.services.excel_service import ExcelService
from app.services.ingestion_service import IngestionService
from app.services.nlp_service import NLPService
from app.services.pdf_service import PdfService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["traffic"])


# ---------------------------------------------------------------------------
# Dependency injection helpers
# ---------------------------------------------------------------------------

def _get_file_repo() -> FileRepository:
    return FileRepository()


def _get_nlp_service() -> NLPService:
    return NLPService()


def _get_anomaly_service() -> AnomalyService:
    contamination = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))
    return AnomalyService(contamination=contamination)


def _get_ingestion_service(nlp: NLPService = Depends(_get_nlp_service)) -> IngestionService:
    return IngestionService(nlp_service=nlp)


def _get_analytics_service() -> AnalyticsService:
    return AnalyticsService()


def _get_excel_service() -> ExcelService:
    return ExcelService()


def _get_pdf_service() -> PdfService:
    return PdfService()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Health-check endpoint."""
    return {"status": "ok", "service": "PeakPulse Engine"}


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    repo: FileRepository = Depends(_get_file_repo),
    ingestion: IngestionService = Depends(_get_ingestion_service),
    anomaly_svc: AnomalyService = Depends(_get_anomaly_service),
    analytics: AnalyticsService = Depends(_get_analytics_service),
    excel_svc: ExcelService = Depends(_get_excel_service),
    pdf_svc: PdfService = Depends(_get_pdf_service),
):
    """
    Upload a Preliminary Excel file → returns a ZIP with Final Excel + PDF.

    **Full processing pipeline:**
    1. IngestionService   — parse & normalise the Excel
    2. AnomalyService     — detect unusual traffic spikes
    3. AnalyticsService   — compute totals, peaks, PHF, breakdowns
    4. ExcelService       — generate Final Excel workbook
    5. PdfService         — generate Final PDF report
    6. FileRepository     — bundle into ZIP and stream back
    """
    if not file.filename or not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx)")

    session_dir = repo.create_session_dir()

    try:
        # 1. Ingest
        file_bytes = await repo.read_uploaded_file(file)
        metadata, class_dfs = ingestion.parse(file_bytes)
        logger.info("Ingested %d vehicle classes", len(class_dfs))

        # 2. Compute totals
        total_df = analytics.compute_total_volume(class_dfs)

        # 3. Detect anomalies
        anomalies = anomaly_svc.detect_anomalies(total_df)

        # 4. Find peaks
        am_peak = analytics.find_peak_hour(total_df, period="AM")
        pm_peak = analytics.find_peak_hour(total_df, period="PM")

        # 5. Class breakdowns
        total_breakdown = analytics.compute_class_breakdown(class_dfs, total_df)
        am_breakdown = analytics.compute_class_breakdown(class_dfs, total_df, am_peak)
        pm_breakdown = analytics.compute_class_breakdown(class_dfs, total_df, pm_peak)

        # 6. Generate Excel
        excel_path = excel_svc.generate(
            metadata=metadata,
            class_dfs=class_dfs,
            total_df=total_df,
            am_breakdown=am_breakdown,
            pm_breakdown=pm_breakdown,
            total_breakdown=total_breakdown,
            am_peak=am_peak,
            pm_peak=pm_peak,
            output_path=session_dir / "Final_Excel.xlsx",
        )

        # 7. Generate PDF
        pdf_path = pdf_svc.generate(
            metadata=metadata,
            total_df=total_df,
            am_peak=am_peak,
            pm_peak=pm_peak,
            total_breakdown=total_breakdown,
            am_breakdown=am_breakdown,
            pm_breakdown=pm_breakdown,
            anomalies=anomalies,
            output_path=session_dir / "Final_Report.pdf",
        )

        # 8. Bundle into ZIP
        zip_path = repo.create_zip(
            {
                "Final_Excel.xlsx": excel_path,
                "Final_Report.pdf": pdf_path,
            },
            session_dir,
        )

        # Stream the ZIP back
        def _stream():
            with open(zip_path, "rb") as f:
                yield from iter(lambda: f.read(8192), b"")
            # Cleanup after streaming
            repo.cleanup(session_dir)

        return StreamingResponse(
            _stream(),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=PeakPulse_Result.zip"},
        )

    except Exception as e:
        repo.cleanup(session_dir)
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/validate")
async def validate(
    file: UploadFile = File(...),
    repo: FileRepository = Depends(_get_file_repo),
    ingestion: IngestionService = Depends(_get_ingestion_service),
    anomaly_svc: AnomalyService = Depends(_get_anomaly_service),
    analytics: AnalyticsService = Depends(_get_analytics_service),
):
    """
    Validate a Preliminary Excel file — returns anomaly alerts only,
    without generating full reports. Useful for preview / quality check.
    """
    if not file.filename or not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Please upload an Excel file (.xlsx)")

    try:
        file_bytes = await repo.read_uploaded_file(file)
        metadata, class_dfs = ingestion.parse(file_bytes)
        total_df = analytics.compute_total_volume(class_dfs)
        anomalies = anomaly_svc.detect_anomalies(total_df)

        am_peak = analytics.find_peak_hour(total_df, period="AM")
        pm_peak = analytics.find_peak_hour(total_df, period="PM")

        return {
            "metadata": metadata.model_dump(),
            "am_peak": am_peak.model_dump() if am_peak else None,
            "pm_peak": pm_peak.model_dump() if pm_peak else None,
            "anomaly_count": len(anomalies),
            "anomalies": [a.model_dump() for a in anomalies],
        }

    except Exception as e:
        logger.exception("Validation failed")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")
