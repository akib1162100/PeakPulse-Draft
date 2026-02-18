"""
Integration test — end-to-end pipeline validation.

Loads the real Preliminary Excel, runs the full pipeline,
and verifies key outputs match the reference Final Excel.
"""

import os
import pytest
import pandas as pd

from app.services.ingestion_service import IngestionService
from app.services.analytics_service import AnalyticsService
from app.services.nlp_service import NLPService


# Path to the test data — adjust if running outside Docker
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_PRELIMINARY = os.path.join(_DATA_DIR, "Preliminary Excel.xlsx")


@pytest.fixture
def pipeline_result():
    """Run the full ingestion + analytics pipeline."""
    if not os.path.exists(_PRELIMINARY):
        pytest.skip("Preliminary Excel not found in data/ directory")

    nlp = NLPService()
    ingestion = IngestionService(nlp_service=nlp)
    analytics = AnalyticsService()

    with open(_PRELIMINARY, "rb") as f:
        file_bytes = f.read()

    metadata, class_dfs = ingestion.parse(file_bytes)
    total_df = analytics.compute_total_volume(class_dfs)
    am_peak = analytics.find_peak_hour(total_df, period="AM")
    pm_peak = analytics.find_peak_hour(total_df, period="PM")

    return {
        "metadata": metadata,
        "class_dfs": class_dfs,
        "total_df": total_df,
        "am_peak": am_peak,
        "pm_peak": pm_peak,
        "analytics": analytics,
    }


class TestIntegrationPipeline:
    """Verify pipeline output matches reference Final Excel values."""

    def test_metadata_parsed(self, pipeline_result):
        meta = pipeline_result["metadata"]
        assert meta.study_name == "1 - US 301 ALT & SE 53RD ST"
        assert meta.project_code == "26-017"
        assert meta.latitude is not None
        assert abs(meta.latitude - 29.91393544) < 0.001

    def test_class_count(self, pipeline_result):
        """Should have 4 vehicle classes."""
        assert len(pipeline_result["class_dfs"]) == 4

    def test_total_intersection_volume(self, pipeline_result):
        """Total intersection volume across all time bins should be 15128."""
        total_df = pipeline_result["total_df"]
        int_total_col = ("", "", "Int Total")
        grand_total = total_df[int_total_col].sum()
        assert grand_total == 15128, f"Expected 15128, got {grand_total}"

    def test_am_peak_time(self, pipeline_result):
        """AM Peak should be 07:30 - 08:30."""
        am = pipeline_result["am_peak"]
        assert am is not None
        assert am.start_time.hour == 7
        assert am.start_time.minute == 30

    def test_am_peak_phf(self, pipeline_result):
        """AM Peak PHF should be ~0.958."""
        am = pipeline_result["am_peak"]
        assert am is not None
        assert abs(am.phf - 0.958) < 0.01, f"Expected ~0.958, got {am.phf}"

    def test_pm_peak_time(self, pipeline_result):
        """PM Peak should be 16:30 - 17:30."""
        pm = pipeline_result["pm_peak"]
        assert pm is not None
        assert pm.start_time.hour == 16
        assert pm.start_time.minute == 30

    def test_pm_peak_phf(self, pipeline_result):
        """PM Peak PHF should be ~0.993."""
        pm = pipeline_result["pm_peak"]
        assert pm is not None
        assert abs(pm.phf - 0.993) < 0.01, f"Expected ~0.993, got {pm.phf}"
