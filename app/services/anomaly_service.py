"""
Anomaly Detection Service – flags unlikely traffic spikes.

Uses scikit-learn's Isolation Forest trained on the study's own data
distribution to identify outlier 15-minute bins that may indicate sensor
errors or unusual events requiring human review.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.models.schemas import AnomalyAlert, Severity

logger = logging.getLogger(__name__)


class AnomalyService:
    """Detects anomalous traffic volumes using Isolation Forest."""

    def __init__(self, contamination: float = 0.05):
        """
        Parameters
        ----------
        contamination : float
            Expected proportion of anomalies in the data (0..0.5).
        """
        self._contamination = contamination

    def detect_anomalies(
        self, total_df: pd.DataFrame
    ) -> list[AnomalyAlert]:
        """
        Scan the total volume DataFrame for anomalous 15-minute bins.

        Parameters
        ----------
        total_df : DataFrame
            Must have a DatetimeIndex (start_time) and at minimum numeric
            movement count columns. Expected to have multi-level columns
            (leg, direction, movement) with an ``int_total`` column or
            similar aggregate.

        Returns
        -------
        List of AnomalyAlert objects for flagged bins.
        """
        alerts: list[AnomalyAlert] = []

        # ---- Prepare feature matrix per approach ----
        # We look at each (leg, direction) group independently
        if isinstance(total_df.columns, pd.MultiIndex):
            approach_groups = self._extract_approach_totals_multi(total_df)
        else:
            # Flat columns – treat the whole frame as one approach
            approach_groups = {"all": total_df.sum(axis=1).values.reshape(-1, 1)}

        for approach_key, volumes in approach_groups.items():
            if len(volumes) < 8:
                # Too few data points for meaningful anomaly detection
                continue

            new_alerts = self._run_isolation_forest(
                volumes, total_df.index, approach_key
            )
            alerts.extend(new_alerts)

        logger.info("Anomaly detection found %d alerts", len(alerts))
        return alerts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_approach_totals_multi(
        self, df: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """
        From a MultiIndex-column DataFrame, compute approach totals
        as Right+Thru+Left+U-Turn for each (leg, direction).
        """
        groups: dict[str, np.ndarray] = {}
        vehicle_movements = {"Right", "Thru", "Left", "U-Turn"}

        seen_approaches: set[tuple[str, str]] = set()
        for col_tuple in df.columns:
            leg, direction, movement = col_tuple
            if (leg, direction) not in seen_approaches:
                seen_approaches.add((leg, direction))

        for leg, direction in seen_approaches:
            mask = [
                (l == leg and d == direction and m in vehicle_movements)
                for l, d, m in df.columns
            ]
            if not any(mask):
                continue
            approach_total = df.loc[:, mask].sum(axis=1).values.reshape(-1, 1)
            groups[f"{leg} {direction}"] = approach_total

        return groups

    def _run_isolation_forest(
        self,
        volumes: np.ndarray,
        time_index: pd.DatetimeIndex,
        approach_key: str,
    ) -> list[AnomalyAlert]:
        """Run Isolation Forest and generate alerts for outliers."""
        alerts: list[AnomalyAlert] = []

        # Build feature matrix
        X = volumes.copy()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add rolling mean delta as a feature
        series = pd.Series(X.flatten())
        rolling_mean = series.rolling(window=4, min_periods=1).mean()
        delta = (series - rolling_mean).values.reshape(-1, 1)

        # Add hour-of-day encoding
        hours = np.array([t.hour + t.minute / 60 for t in time_index]).reshape(-1, 1)

        features = np.hstack([X, delta, hours])

        # Fit Isolation Forest
        clf = IsolationForest(
            contamination=self._contamination,
            random_state=42,
            n_estimators=100,
        )
        predictions = clf.fit_predict(features)  # -1 = outlier, 1 = normal
        scores = clf.decision_function(features)

        for i, pred in enumerate(predictions):
            if pred == -1:
                value = int(X[i, 0])
                score = float(scores[i])

                # Classify severity by anomaly score
                if score < -0.3:
                    severity = Severity.HIGH
                elif score < -0.1:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW

                mean_val = float(rolling_mean.iloc[i])
                ts = time_index[i]

                alerts.append(
                    AnomalyAlert(
                        timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                        leg=approach_key.split(" ")[0] if " " in approach_key else approach_key,
                        direction=approach_key.split(" ")[-1] if " " in approach_key else "",
                        movement="App Total",
                        observed_value=value,
                        expected_range=f"~{mean_val:.0f} (rolling avg)",
                        severity=severity,
                        message=(
                            f"Unusual volume {value} at {ts.strftime('%H:%M') if hasattr(ts, 'strftime') else ts} "
                            f"for {approach_key} (expected ~{mean_val:.0f})"
                        ),
                    )
                )

        return alerts
