"""
Analytics Service – core traffic analysis calculations.

Computes total volumes, rolling peak hour identification,
Peak Hour Factor (PHF), approach/total percentages, and
class breakdowns matching the Final Excel reference format.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.models.schemas import PeakResult

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Performs all traffic analysis calculations per BRD §2.2."""

    # ------------------------------------------------------------------
    # Total Volume
    # ------------------------------------------------------------------

    def compute_total_volume(
        self, class_dfs: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Sum all vehicle classes into a single total-volume DataFrame
        with Approach Totals and Intersection Totals per time bin.

        Returns a DataFrame with MultiIndex columns:
        (leg, direction, movement) where movement includes the originals
        plus 'App Total', and a final column ('', '', 'Int Total').
        """
        if not class_dfs:
            return pd.DataFrame()

        # Sum all class DataFrames element-wise
        ref = list(class_dfs.values())[0]
        total = ref.copy() * 0  # zero-frame with same structure

        for df in class_dfs.values():
            total = total.add(df, fill_value=0)

        total = total.astype(int)

        # ---- Add App Total and Int Total columns ----
        result_parts: list[pd.DataFrame] = []
        seen_approaches: list[tuple[str, str]] = []

        for col_tuple in total.columns:
            leg, direction, _ = col_tuple
            if (leg, direction) not in seen_approaches:
                seen_approaches.append((leg, direction))

        for leg, direction in seen_approaches:
            # Select vehicle movement columns for this approach
            vehicle_movs = ["Right", "Thru", "Left", "U-Turn"]
            approach_cols = [
                (leg, direction, m)
                for m in vehicle_movs
                if (leg, direction, m) in total.columns
            ]
            # All columns for this approach
            all_approach_cols = [
                c for c in total.columns if c[0] == leg and c[1] == direction
            ]

            approach_data = total[all_approach_cols].copy()

            # Compute App Total
            if approach_cols:
                app_total = total[approach_cols].sum(axis=1).astype(int)
            else:
                app_total = pd.Series(0, index=total.index)

            approach_data[(leg, direction, "App Total")] = app_total
            result_parts.append(approach_data)

        result = pd.concat(result_parts, axis=1)

        # Compute Intersection Total
        app_total_cols = [c for c in result.columns if c[2] == "App Total"]
        result[("", "", "Int Total")] = result[app_total_cols].sum(axis=1).astype(int)

        result.columns = pd.MultiIndex.from_tuples(
            result.columns, names=["leg", "direction", "movement"]
        )

        return result

    # ------------------------------------------------------------------
    # Peak Hour Identification
    # ------------------------------------------------------------------

    def find_peak_hour(
        self, total_df: pd.DataFrame, period: str = "AM"
    ) -> PeakResult | None:
        """
        Identify the peak hour using a rolling 60-minute (4-bin) window.

        Parameters
        ----------
        total_df : DataFrame with an 'Int Total' column (or equivalent)
        period : 'AM' or 'PM' — filters to before/after noon

        Returns
        -------
        PeakResult with start/end time, total volume, and PHF.
        """
        int_total_col = self._find_int_total_col(total_df)
        if int_total_col is None:
            return None

        series = total_df[int_total_col].copy()
        times = series.index

        # Filter by period
        if period.upper() == "AM":
            mask = pd.DatetimeIndex(times).hour < 12
        else:
            mask = pd.DatetimeIndex(times).hour >= 12
        series = series[mask]

        if len(series) < 4:
            return None

        # Rolling 4-bin sum (60 minutes)
        rolling_sum = series.rolling(window=4).sum()
        best_end_idx = rolling_sum.idxmax()
        best_end_pos = series.index.get_loc(best_end_idx)
        best_start_pos = best_end_pos - 3

        if best_start_pos < 0:
            best_start_pos = 0

        peak_bins = series.iloc[best_start_pos : best_end_pos + 1]
        peak_volume = int(peak_bins.sum())

        start_time = peak_bins.index[0]
        end_time = peak_bins.index[-1]

        # End time is the start of the last bin + 15 min
        if hasattr(end_time, "to_pydatetime"):
            end_display = end_time.to_pydatetime() + timedelta(minutes=15)
            start_display = start_time.to_pydatetime()
        else:
            end_display = end_time + timedelta(minutes=15)
            start_display = start_time

        # PHF
        phf = self.compute_phf(peak_bins)

        label = (
            f"{start_display.strftime('%I:%M %p')} - "
            f"{end_display.strftime('%I:%M %p')} ({phf:.3f})"
        )

        return PeakResult(
            period=period.upper(),
            start_time=start_display,
            end_time=end_display,
            total_volume=peak_volume,
            phf=phf,
            label=label,
        )

    # ------------------------------------------------------------------
    # PHF
    # ------------------------------------------------------------------

    @staticmethod
    def compute_phf(peak_bins: pd.Series) -> float:
        """
        Peak Hour Factor = V / (4 × V₁₅)
        where V = total peak-hour volume, V₁₅ = highest 15-min bin.
        """
        total = peak_bins.sum()
        max_bin = peak_bins.max()
        if max_bin == 0:
            return 0.0
        return round(float(total / (4 * max_bin)), 3)

    # ------------------------------------------------------------------
    # Class Breakdown
    # ------------------------------------------------------------------

    def compute_class_breakdown(
        self,
        class_dfs: dict[str, pd.DataFrame],
        total_df: pd.DataFrame,
        peak: PeakResult | None = None,
    ) -> pd.DataFrame:
        """
        Build the breakdown table (Total Volume, AM Peak, or PM Peak).

        If *peak* is provided, filters data to the peak hour bins only.
        Appends Grand Total, % Approach, % Total, PHF (for peaks),
        and per-class count/percentage rows.
        """
        # Filter to peak window if applicable
        if peak:
            mask = (total_df.index >= peak.start_time) & (
                total_df.index < peak.end_time
            )
            filtered = total_df.loc[mask].copy()
        else:
            filtered = total_df.copy()

        rows: list[pd.Series] = []
        for idx in filtered.index:
            rows.append(filtered.loc[idx])

        # ---- Grand Total row ----
        grand_total = filtered.select_dtypes(include=[np.number]).sum().astype(int)
        grand_total.name = "Grand Total"

        # ---- % Approach row ----
        pct_approach = pd.Series(index=filtered.columns, dtype=float, name="% Approach")
        for col in filtered.columns:
            leg, direction, movement = col
            if movement in ("App Total", "Int Total", "Peds CW", "Peds CCW"):
                pct_approach[col] = np.nan
            else:
                app_total_col = (leg, direction, "App Total")
                if app_total_col in grand_total.index and grand_total[app_total_col] > 0:
                    pct_approach[col] = grand_total[col] / grand_total[app_total_col]
                else:
                    pct_approach[col] = 0.0

        # ---- % Total row ----
        pct_total = pd.Series(index=filtered.columns, dtype=float, name="% Total")
        int_total_col = self._find_int_total_col(filtered)
        int_total_val = grand_total[int_total_col] if int_total_col else 1
        for col in filtered.columns:
            leg, direction, movement = col
            if movement in ("Peds CW", "Peds CCW"):
                pct_total[col] = np.nan
            elif movement == "Int Total":
                pct_total[col] = np.nan
            elif int_total_val > 0:
                pct_total[col] = grand_total[col] / int_total_val
            else:
                pct_total[col] = 0.0

        # ---- Assemble summary rows ----
        summary = pd.DataFrame([grand_total, pct_approach, pct_total])

        # ---- PHF row (only for peak breakdowns) ----
        if peak:
            phf_row = pd.Series(index=filtered.columns, dtype=float, name=f"PHF ({peak.label.split('(')[0].strip()})")
            for col in filtered.columns:
                leg, direction, movement = col
                if movement in ("Peds CW", "Peds CCW"):
                    phf_row[col] = np.nan
                else:
                    bins = filtered[col]
                    max_bin = bins.max()
                    total_col = bins.sum()
                    if max_bin > 0:
                        phf_row[col] = round(float(total_col / (4 * max_bin)), 15)
                    else:
                        phf_row[col] = 0.0
            summary = pd.concat([summary, phf_row.to_frame().T])

        # ---- Per-class breakdown rows ----
        for class_name, class_df in class_dfs.items():
            if peak:
                mask = (class_df.index >= peak.start_time) & (
                    class_df.index < peak.end_time
                )
                cls_filtered = class_df.loc[mask]
            else:
                cls_filtered = class_df

            # Sum class data, add App Total, match total_df column structure
            cls_total = self._sum_class_to_total_structure(cls_filtered, filtered)
            cls_total.name = class_name

            # % class
            pct_cls = pd.Series(index=filtered.columns, dtype=float, name=f"% {class_name}")
            for col in filtered.columns:
                leg, direction, movement = col
                if movement in ("Peds CW", "Peds CCW"):
                    # Peds/Bicycles fill CW/CCW columns directly
                    if class_name in ("Pedestrians", "Bicycles on Crosswalk"):
                        if grand_total.get(col, 0) > 0:
                            pct_cls[col] = cls_total.get(col, 0) / grand_total[col]
                        else:
                            pct_cls[col] = 0.0
                    else:
                        pct_cls[col] = np.nan
                elif movement == "Int Total":
                    if int_total_val > 0:
                        pct_cls[col] = cls_total.get(col, 0) / int_total_val
                    else:
                        pct_cls[col] = 0.0
                elif movement == "App Total":
                    app_total_gt = grand_total.get(col, 0)
                    if app_total_gt > 0:
                        pct_cls[col] = cls_total.get(col, 0) / app_total_gt
                    else:
                        pct_cls[col] = 0.0
                else:
                    gt_val = grand_total.get(col, 0)
                    if gt_val > 0:
                        pct_cls[col] = cls_total.get(col, 0) / gt_val
                    else:
                        pct_cls[col] = 0.0

            summary = pd.concat(
                [summary, cls_total.to_frame().T, pct_cls.to_frame().T]
            )

        result = pd.concat([filtered, summary])
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_int_total_col(df: pd.DataFrame):
        """Find the Int Total column in a MultiIndex DataFrame."""
        for col in df.columns:
            if isinstance(col, tuple) and col[-1] == "Int Total":
                return col
        if "Int Total" in df.columns:
            return "Int Total"
        return None

    def _sum_class_to_total_structure(
        self, cls_df: pd.DataFrame, total_df: pd.DataFrame
    ) -> pd.Series:
        """
        Sum a class DataFrame's columns and map them into the total_df
        column structure (including App Total, Int Total).
        """
        result = pd.Series(0, index=total_df.columns, dtype=float)

        for col in total_df.columns:
            leg, direction, movement = col
            if movement == "App Total":
                # Sum vehicle movements from class
                vehicle_movs = ["Right", "Thru", "Left", "U-Turn"]
                total = 0
                for m in vehicle_movs:
                    cls_col = (leg, direction, m)
                    if cls_col in cls_df.columns:
                        total += cls_df[cls_col].sum()
                result[col] = int(total)
            elif movement == "Int Total":
                # Will be computed after App Totals are set
                pass
            elif col in cls_df.columns:
                result[col] = int(cls_df[col].sum())

        # Compute Int Total as sum of App Totals
        int_total_col = self._find_int_total_col(total_df)
        if int_total_col:
            app_cols = [c for c in total_df.columns if c[2] == "App Total"]
            result[int_total_col] = int(sum(result.get(c, 0) for c in app_cols))

        return result
