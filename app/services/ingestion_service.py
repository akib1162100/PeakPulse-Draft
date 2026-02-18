"""
Ingestion Service – parses Preliminary Excel files into normalised DataFrames.

Reads the Summary sheet for metadata and each vehicle-class sheet into a
DataFrame, normalises timestamps to clean 15-minute bins, aligns all
class DataFrames to a unified time index, and fills missing values with 0.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.models.schemas import StudyMetadata


# Canonical vehicle-class sheet names we look for
_CLASS_SHEETS = [
    "Lights and Motorcycles",
    "Heavy",
    "Pedestrians",
    "Bicycles on Crosswalk",
]

# Standard movement column names
_MOVEMENT_COLS = ["Right", "Thru", "Left", "U-Turn", "Peds CW", "Peds CCW"]


class IngestionService:
    """Parses a Preliminary Excel workbook into structured DataFrames."""

    def __init__(self, nlp_service=None):
        """
        Parameters
        ----------
        nlp_service : NLPService, optional
            If provided, used to map non-standard column headers to the
            canonical movement names.
        """
        self._nlp = nlp_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self, file_bytes: bytes
    ) -> tuple[StudyMetadata, dict[str, pd.DataFrame]]:
        """
        Parse a Preliminary Excel workbook.

        Returns
        -------
        metadata : StudyMetadata
        class_dfs : dict mapping class name -> DataFrame
            Each DataFrame is indexed by rounded ``start_time`` and has a
            MultiIndex of columns: (leg, direction, movement).
        """
        wb = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")

        metadata = self._parse_summary(wb)
        class_dfs: dict[str, pd.DataFrame] = {}

        for sheet_name in _CLASS_SHEETS:
            if sheet_name in wb.sheet_names:
                df = self._parse_class_sheet(wb, sheet_name)
                if df is not None and not df.empty:
                    class_dfs[sheet_name] = df

        # Align all dataframes to unified time index
        class_dfs = self._align_time_index(class_dfs)

        return metadata, class_dfs

    # ------------------------------------------------------------------
    # Summary sheet
    # ------------------------------------------------------------------

    def _parse_summary(self, wb: pd.ExcelFile) -> StudyMetadata:
        """Extract study metadata from the Summary sheet."""
        raw = pd.read_excel(wb, sheet_name="Summary", header=None)
        kv: dict[str, Any] = {}
        for _, row in raw.iterrows():
            key = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            val = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else ""
            if key:
                kv[key] = val

        lat, lon = None, None
        ll_str = str(kv.get("Latitude and Longitude", ""))
        if "," in ll_str:
            parts = ll_str.split(",")
            try:
                lat, lon = float(parts[0].strip()), float(parts[1].strip())
            except ValueError:
                pass

        start_time = self._to_datetime(kv.get("Start Time"))
        end_time = self._to_datetime(kv.get("End Time"))

        return StudyMetadata(
            study_name=str(kv.get("Study Name", "")),
            project=str(kv.get("Project", "")),
            project_code=str(kv.get("Project Code", "")),
            legs_and_movements=str(kv.get("Legs and Movements", "")),
            bin_size=str(kv.get("Bin Size", "15 minutes")),
            timezone=str(kv.get("Time Zone", "America/New_York")),
            start_time=start_time,
            end_time=end_time,
            location=str(kv.get("Location", "")),
            latitude=lat,
            longitude=lon,
        )

    # ------------------------------------------------------------------
    # Class sheets
    # ------------------------------------------------------------------

    def _parse_class_sheet(
        self, wb: pd.ExcelFile, sheet_name: str
    ) -> pd.DataFrame | None:
        """
        Parse a vehicle-class sheet (row 0=Leg, row 1=Direction, row 2=Header,
        rows 3+ = data).
        """
        raw = pd.read_excel(wb, sheet_name=sheet_name, header=None)

        if raw.shape[0] < 4:
            return None

        # --- Extract leg and direction labels from rows 0-1 ----------------
        legs_row = raw.iloc[0, 1:].tolist()
        dirs_row = raw.iloc[1, 1:].tolist()
        hdrs_row = raw.iloc[2, 1:].tolist()

        # Forward-fill leg and direction across merged cells
        legs = self._forward_fill(legs_row)
        dirs = self._forward_fill(dirs_row)

        # Map headers through NLP if available
        if self._nlp:
            header_map = self._nlp.map_headers(
                [str(h) for h in hdrs_row if pd.notna(h) and str(h).strip()]
            )
        else:
            header_map = {}

        # Build multi-level column index: (leg, direction, movement)
        columns = []
        for leg, direction, hdr in zip(legs, dirs, hdrs_row):
            mov = str(hdr).strip() if pd.notna(hdr) else ""
            mov = header_map.get(mov, mov)  # apply NLP mapping
            columns.append((str(leg).strip(), str(direction).strip(), mov))

        multi_cols = pd.MultiIndex.from_tuples(
            columns, names=["leg", "direction", "movement"]
        )

        # --- Extract data rows (skip header rows, skip empty tails) --------
        data = raw.iloc[3:].copy()
        data = data.dropna(subset=[data.columns[0]])  # drop rows with no time

        timestamps = data.iloc[:, 0].apply(self._normalise_timestamp)
        valid_mask = timestamps.notna()
        data = data.loc[valid_mask].copy()
        timestamps = timestamps.loc[valid_mask]

        values = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        values.columns = multi_cols
        values.index = timestamps
        values.index.name = "start_time"

        return values

    # ------------------------------------------------------------------
    # Time alignment
    # ------------------------------------------------------------------

    def _align_time_index(
        self, class_dfs: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """
        Build a unified 15-minute time index from the union of all class
        DataFrames and re-index each one (filling with 0).
        """
        if not class_dfs:
            return class_dfs

        all_times: set[datetime] = set()
        for df in class_dfs.values():
            all_times.update(df.index.tolist())

        unified_index = pd.DatetimeIndex(sorted(all_times), name="start_time")

        aligned: dict[str, pd.DataFrame] = {}
        for name, df in class_dfs.items():
            aligned[name] = df.reindex(unified_index, fill_value=0)

        return aligned

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_timestamp(val) -> datetime | None:
        """
        Convert a cell value to a clean 15-minute-aligned datetime.
        Handles the fractional-second timestamps like '2026-01-28 06:29:59.870000'
        by rounding to the nearest 15-minute bin start.
        """
        if pd.isna(val):
            return None

        if isinstance(val, datetime):
            ts = val
        else:
            try:
                ts = pd.Timestamp(val).to_pydatetime()
            except Exception:
                return None

        # Round to nearest 15-minute bin
        # e.g. 06:29:59.870 → 06:30:00, 06:44:59.865 → 06:45:00
        total_seconds = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6
        bin_seconds = 15 * 60  # 900
        rounded_bin = round(total_seconds / bin_seconds) * bin_seconds

        rounded_hour = int(rounded_bin // 3600)
        rounded_min = int((rounded_bin % 3600) // 60)

        return ts.replace(hour=rounded_hour, minute=rounded_min, second=0, microsecond=0)

    @staticmethod
    def _forward_fill(values: list) -> list:
        """Forward-fill None/NaN values in a list."""
        result = []
        last = ""
        for v in values:
            if pd.notna(v) and str(v).strip():
                last = str(v).strip()
            result.append(last)
        return result

    @staticmethod
    def _to_datetime(val) -> datetime | None:
        if val is None or (isinstance(val, str) and not val.strip()):
            return None
        if isinstance(val, datetime):
            return val
        try:
            return pd.Timestamp(val).to_pydatetime()
        except Exception:
            return None
