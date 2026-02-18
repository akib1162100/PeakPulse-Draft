"""
Tests for AnomalyService — Isolation Forest anomaly detection.
"""

import pandas as pd
import numpy as np
import pytest

from app.services.anomaly_service import AnomalyService


@pytest.fixture
def anomaly_svc():
    return AnomalyService(contamination=0.1)


@pytest.fixture
def normal_traffic_data():
    """Create a normal-looking traffic DataFrame."""
    times = pd.date_range("2026-01-28 06:30:00", periods=32, freq="15min")
    cols = pd.MultiIndex.from_tuples([
        ("US 301", "Southbound", "Right"),
        ("US 301", "Southbound", "Thru"),
        ("US 301", "Southbound", "Left"),
        ("US 301", "Southbound", "U-Turn"),
    ], names=["leg", "direction", "movement"])

    np.random.seed(42)
    data = np.zeros((32, 4), dtype=int)
    # Normal Thru volumes between 150-250
    data[:, 1] = np.random.randint(150, 250, size=32)
    return pd.DataFrame(data, index=times, columns=cols)


class TestAnomalyDetection:
    def test_returns_list(self, anomaly_svc, normal_traffic_data):
        alerts = anomaly_svc.detect_anomalies(normal_traffic_data)
        assert isinstance(alerts, list)

    def test_detects_spike(self, anomaly_svc, normal_traffic_data):
        """Inject a huge spike — should be flagged."""
        spiked = normal_traffic_data.copy()
        # Inject a 900-vehicle spike in one bin
        spiked.iloc[15, 1] = 900
        alerts = anomaly_svc.detect_anomalies(spiked)
        # There should be at least 1 alert
        assert len(alerts) >= 1
        # The spike should be one of the flagged values
        flagged_values = [a.observed_value for a in alerts]
        assert 900 in flagged_values

    def test_normal_data_few_alerts(self, anomaly_svc, normal_traffic_data):
        """Normal data should produce very few alerts."""
        alerts = anomaly_svc.detect_anomalies(normal_traffic_data)
        # With 10% contamination on 32 rows, expect ~3 max
        assert len(alerts) <= 5
