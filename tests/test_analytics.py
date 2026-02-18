"""
Tests for AnalyticsService — peak hour detection, PHF, class breakdowns.
"""

import pandas as pd
import numpy as np
import pytest

from app.services.analytics_service import AnalyticsService


@pytest.fixture
def analytics():
    return AnalyticsService()


@pytest.fixture
def sample_class_dfs():
    """
    Minimal sample data mimicking the Preliminary Excel structure.
    Two approaches: Southbound (US 301) and Northbound (US 301).
    """
    times = pd.to_datetime([
        "2026-01-28 07:30:00",
        "2026-01-28 07:45:00",
        "2026-01-28 08:00:00",
        "2026-01-28 08:15:00",
    ])

    cols = pd.MultiIndex.from_tuples([
        ("US 301", "Southbound", "Right"),
        ("US 301", "Southbound", "Thru"),
        ("US 301", "Southbound", "Left"),
        ("US 301", "Southbound", "U-Turn"),
        ("US 301", "Southbound", "Peds CW"),
        ("US 301", "Southbound", "Peds CCW"),
        ("US 301", "Northbound", "Right"),
        ("US 301", "Northbound", "Thru"),
        ("US 301", "Northbound", "Left"),
        ("US 301", "Northbound", "U-Turn"),
        ("US 301", "Northbound", "Peds CW"),
        ("US 301", "Northbound", "Peds CCW"),
    ], names=["leg", "direction", "movement"])

    lights_data = np.array([
        [0, 203, 0, 0, 0, 0, 2, 146, 0, 0, 0, 0],
        [0, 196, 0, 0, 0, 0, 0, 181, 0, 0, 0, 0],
        [0, 179, 0, 0, 0, 0, 0, 166, 0, 0, 0, 0],
        [0, 175, 0, 0, 0, 0, 0, 204, 0, 0, 0, 0],
    ])

    heavy_data = np.array([
        [0, 50, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0],
        [0, 57, 0, 0, 0, 0, 0, 43, 0, 0, 0, 0],
        [0, 53, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0],
        [0, 48, 0, 0, 0, 0, 0, 48, 0, 0, 0, 0],
    ])

    lights_df = pd.DataFrame(lights_data, index=times, columns=cols)
    heavy_df = pd.DataFrame(heavy_data, index=times, columns=cols)

    return {
        "Lights and Motorcycles": lights_df,
        "Heavy": heavy_df,
    }


class TestComputeTotalVolume:
    def test_sums_classes(self, analytics, sample_class_dfs):
        total = analytics.compute_total_volume(sample_class_dfs)
        assert not total.empty
        # Check first row Southbound Thru: 203 + 50 = 253
        sb_thru = ("US 301", "Southbound", "Thru")
        assert total.loc[total.index[0], sb_thru] == 253

    def test_app_total(self, analytics, sample_class_dfs):
        total = analytics.compute_total_volume(sample_class_dfs)
        sb_app = ("US 301", "Southbound", "App Total")
        # First row: 0+253+0+0 = 253
        assert total.loc[total.index[0], sb_app] == 253

    def test_int_total(self, analytics, sample_class_dfs):
        total = analytics.compute_total_volume(sample_class_dfs)
        int_total_col = ("", "", "Int Total")
        # First row: SB App 253 + NB App (2+175) = 253+177 = 430
        assert total.loc[total.index[0], int_total_col] == 430


class TestPeakHour:
    def test_am_peak_detection(self, analytics, sample_class_dfs):
        total = analytics.compute_total_volume(sample_class_dfs)
        peak = analytics.find_peak_hour(total, period="AM")
        assert peak is not None
        assert peak.period == "AM"
        assert peak.total_volume > 0

    def test_phf_calculation(self, analytics):
        # PHF = V / (4 * V15)
        bins = pd.Series([100, 120, 110, 130])
        phf = analytics.compute_phf(bins)
        expected = 460 / (4 * 130)  # 0.885
        assert abs(phf - round(expected, 3)) < 0.01


class TestClassBreakdown:
    def test_breakdown_has_grand_total(self, analytics, sample_class_dfs):
        total = analytics.compute_total_volume(sample_class_dfs)
        breakdown = analytics.compute_class_breakdown(sample_class_dfs, total)
        # Should have data rows + Grand Total + % rows
        assert len(breakdown) > len(total)
        assert "Grand Total" in breakdown.index

    def test_breakdown_with_peak(self, analytics, sample_class_dfs):
        total = analytics.compute_total_volume(sample_class_dfs)
        peak = analytics.find_peak_hour(total, period="AM")
        if peak:
            breakdown = analytics.compute_class_breakdown(
                sample_class_dfs, total, peak
            )
            assert "Grand Total" in breakdown.index
