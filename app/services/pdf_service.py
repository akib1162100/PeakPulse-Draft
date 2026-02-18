"""
PDF Service – generates the Final PDF report.

Produces a multi-page PDF with:
  - Header page with study metadata and peak hour summary
  - Movement tables for Total, AM Peak, and PM Peak (merged Leg–Direction headers)
  - Modern intersection flow diagrams with logarithmic line thickness scaling
"""

from __future__ import annotations

import io
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)

from app.models.schemas import AnomalyAlert, PeakResult, StudyMetadata
from app.services.layout_service import LayoutService

logger = logging.getLogger(__name__)

# ── Modern Color Palette ──────────────────────────────────────────────
_PAL = {
    "bg":       "#F5F5F4",   # off-white / light gray
    "surface":  "#FFFFFF",   # white cards
    "card":     "#FFFFFF",
    "accent1":  "#2563EB",   # blue-600   (South)
    "accent2":  "#059669",   # emerald-600 (North)
    "accent3":  "#D97706",   # amber-600  (West)
    "accent4":  "#7C3AED",   # violet-600 (East)
    "white":    "#1E293B",   # dark text on light bg
    "muted":    "#64748B",
    "grid":     "#CBD5E1",
    "glow":     "#2563EB18",
}


class PdfService:
    """Generates the Final PDF report with tables and flow diagrams."""

    def __init__(self):
        self._layout = LayoutService()
        self._styles = getSampleStyleSheet()
        self._title_style = ParagraphStyle(
            "CustomTitle",
            parent=self._styles["Title"],
            fontSize=18,
            spaceAfter=12,
        )
        self._heading_style = ParagraphStyle(
            "CustomHeading",
            parent=self._styles["Heading2"],
            fontSize=14,
            spaceAfter=8,
        )
        self._normal_style = self._styles["Normal"]
        self._wrap_style = ParagraphStyle(
            "WrapCell",
            parent=self._styles["Normal"],
            fontSize=7,
            leading=8,
            wordWrap="CJK",
        )
        self._wrap_style_bold = ParagraphStyle(
            "WrapCellBold",
            parent=self._wrap_style,
            fontName="Helvetica-Bold",
        )

    def generate(
        self,
        metadata: StudyMetadata,
        total_df: pd.DataFrame,
        am_peak: PeakResult | None,
        pm_peak: PeakResult | None,
        total_breakdown: pd.DataFrame,
        am_breakdown: pd.DataFrame,
        pm_breakdown: pd.DataFrame,
        anomalies: list[AnomalyAlert],
        output_path: Path,
    ) -> Path:
        """Build the full PDF and save to *output_path*."""
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=landscape(A4),
            leftMargin=30,
            rightMargin=30,
            topMargin=30,
            bottomMargin=30,
        )
        story: list = []

        # --- Page 1: Header & Summary ---
        story.extend(self._build_header(metadata, am_peak, pm_peak))
        story.append(PageBreak())

        # --- Page 2: Total Volume Table ---
        story.append(Paragraph("Total Volume Breakdown", self._heading_style))
        story.append(Spacer(1, 6))
        story.append(self._build_summary_table(total_breakdown, "Total"))
        story.append(PageBreak())

        # --- Page 3: AM Peak Table ---
        if am_peak:
            story.append(Paragraph(f"AM Peak: {am_peak.label}", self._heading_style))
            story.append(Spacer(1, 6))
            story.append(self._build_summary_table(am_breakdown, "AM Peak"))
            story.append(PageBreak())

        # --- Page 4: PM Peak Table ---
        if pm_peak:
            story.append(Paragraph(f"PM Peak: {pm_peak.label}", self._heading_style))
            story.append(Spacer(1, 6))
            story.append(self._build_summary_table(pm_breakdown, "PM Peak"))
            story.append(PageBreak())

        # --- Pages 5-7: Modern flow diagrams ---
        for label, df in [
            ("Total Period", total_df),
            ("AM Peak", self._filter_peak(total_df, am_peak)),
            ("PM Peak", self._filter_peak(total_df, pm_peak)),
        ]:
            if df is not None and not df.empty:
                story.append(Paragraph(f"Traffic Flow — {label}", self._heading_style))
                story.append(Spacer(1, 6))
                img = self._generate_flow_diagram(df, label)
                if img:
                    story.append(img)
                story.append(PageBreak())

        # --- Anomaly alerts ---
        if anomalies:
            story.append(Paragraph("⚠ Anomaly Alerts", self._heading_style))
            story.append(Spacer(1, 6))
            story.append(self._build_anomaly_table(anomalies))

        doc.build(story)
        logger.info("Final PDF written to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Header page
    # ------------------------------------------------------------------

    def _build_header(self, metadata, am_peak, pm_peak) -> list:
        elements = []
        elements.append(Paragraph("PeakPulse Traffic Report", self._title_style))
        elements.append(Spacer(1, 12))

        info = [
            ["Study Name", metadata.study_name],
            ["Project", metadata.project],
            ["Project Code", metadata.project_code],
            ["Location", metadata.location],
            [
                "Coordinates",
                f"{metadata.latitude}, {metadata.longitude}"
                if metadata.latitude
                else "N/A",
            ],
            [
                "Study Period",
                f"{metadata.start_time} — {metadata.end_time}"
                if metadata.start_time
                else "N/A",
            ],
            ["AM Peak", am_peak.label if am_peak else "N/A"],
            ["PM Peak (Overall)", pm_peak.label if pm_peak else "N/A"],
        ]

        table = Table(info, colWidths=[150, 400])
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.Color(0.85, 0.89, 0.95)),
                ]
            )
        )
        elements.append(table)
        return elements

    # ------------------------------------------------------------------
    # Summary / breakdown tables — word-wrapped cells
    # ------------------------------------------------------------------

    def _build_summary_table(self, df: pd.DataFrame, label: str):
        """Build table with merged approach headers and word-wrapped cells."""
        if df.empty:
            return Paragraph("No data available", self._normal_style)

        cols = list(df.columns)
        legs = [c[0] if isinstance(c, tuple) else "" for c in cols]
        dirs = [c[1] if isinstance(c, tuple) else "" for c in cols]
        movs = [c[2] if isinstance(c, tuple) else str(c) for c in cols]

        # Build approach labels (merged Leg — Direction)
        approach_labels = []
        for leg, direction in zip(legs, dirs):
            if leg and direction:
                approach_labels.append(f"{leg} — {direction}")
            else:
                approach_labels.append(leg or direction or "")

        # Use Paragraph objects for word-wrapping in header cells
        header_row1 = [Paragraph("<b>Approach</b>", self._wrap_style_bold)]
        for lbl in approach_labels:
            header_row1.append(Paragraph(f"<b>{lbl}</b>", self._wrap_style_bold))

        header_row2 = [Paragraph("<b>Start Time</b>", self._wrap_style_bold)]
        for m in movs:
            header_row2.append(Paragraph(f"<b>{m}</b>", self._wrap_style_bold))

        data_rows = [header_row1, header_row2]

        for idx, row in df.iterrows():
            label_cell = str(idx)
            if hasattr(idx, "strftime"):
                label_cell = idx.strftime("%H:%M")
            values = [Paragraph(label_cell, self._wrap_style)]
            for v in row:
                if pd.isna(v):
                    values.append(Paragraph("", self._wrap_style))
                elif isinstance(v, float):
                    txt = f"{v:.4f}" if abs(v) < 1 else f"{v:.3f}"
                    values.append(Paragraph(txt, self._wrap_style))
                else:
                    txt = str(int(v)) if v == int(v) else str(v)
                    values.append(Paragraph(txt, self._wrap_style))
            data_rows.append(values)

        col_count = len(data_rows[0])
        col_w = min(55, (landscape(A4)[0] - 60) / col_count)
        col_widths = [80] + [col_w] * (col_count - 1)

        table = Table(data_rows, colWidths=col_widths, repeatRows=2)

        style_cmds = [
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.184, 0.333, 0.592)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, 1), colors.Color(0.84, 0.89, 0.94)),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]

        # Zebra striping
        for r in range(2, len(data_rows)):
            if r % 2 == 0:
                style_cmds.append(
                    ("BACKGROUND", (0, r), (-1, r), colors.Color(0.95, 0.97, 0.98))
                )

        # Merge approach header cells
        merge_spans = self._compute_merge_spans(legs, dirs, start_col=1)
        for start_c, end_c in merge_spans:
            if end_c > start_c:
                style_cmds.append(("SPAN", (start_c, 0), (end_c, 0)))

        table.setStyle(TableStyle(style_cmds))
        return table

    def _compute_merge_spans(self, legs, dirs, start_col=1):
        """Compute column spans for identical Leg+Direction groups."""
        if not legs:
            return []
        spans = []
        merge_start = start_col
        prev_key = (legs[0], dirs[0])
        for i in range(1, len(legs)):
            curr_key = (legs[i], dirs[i])
            if curr_key != prev_key:
                spans.append((merge_start, start_col + i - 1))
                merge_start = start_col + i
                prev_key = curr_key
        spans.append((merge_start, start_col + len(legs) - 1))
        return spans

    # ------------------------------------------------------------------
    # Modern intersection flow diagram
    # ------------------------------------------------------------------

    def _generate_flow_diagram(self, df: pd.DataFrame, title: str):
        """
        Modern dark-themed intersection flow diagram.

        • Logarithmic scaling for line thickness
        • Direction legend in top-right corner
        • Glowing neon-style arrows on dark background
        • Non-overlapping labels in rounded pill badges
        """
        try:
            approach_data = self._extract_approach_movements(df)
            if not approach_data:
                return None

            fig, ax = plt.subplots(1, 1, figsize=(11, 8), facecolor=_PAL["bg"])
            ax.set_xlim(-6.5, 6.5)
            ax.set_ylim(-6.5, 6.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_facecolor(_PAL["bg"])

            # ── Title ──
            ax.text(
                0, 6.0, f"Traffic Flow — {title}",
                ha="center", va="center", fontsize=17, fontweight="bold",
                color=_PAL["white"], fontfamily="sans-serif",
            )

            # ── Intersection center (glowing box) ──
            # Subtle shadow
            for offset, alpha in [(0.12, 0.02), (0.08, 0.04), (0.04, 0.06)]:
                glow = mpatches.FancyBboxPatch(
                    (-1.2 - offset, -1.2 - offset),
                    2.4 + 2 * offset, 2.4 + 2 * offset,
                    boxstyle="round,pad=0.2",
                    facecolor="#94A3B8", edgecolor="none",
                    alpha=alpha, zorder=1,
                )
                ax.add_patch(glow)

            intersection = mpatches.FancyBboxPatch(
                (-1.2, -1.2), 2.4, 2.4,
                boxstyle="round,pad=0.2", facecolor="white",
                edgecolor="#94A3B8", linewidth=1.5, zorder=2,
            )
            ax.add_patch(intersection)

            # Cross-hair lines inside intersection
            ax.plot([0, 0], [-1.2, 1.2], color=_PAL["grid"], lw=0.8, ls="--", zorder=3)
            ax.plot([-1.2, 1.2], [0, 0], color=_PAL["grid"], lw=0.8, ls="--", zorder=3)

            # ── Approach configs ──
            # Directions follow the DRIVER's perspective looking outward:
            #   Southbound (driver faces south/down):  Right→West(-x)  Left→East(+x)
            #   Northbound (driver faces north/up):    Right→East(+x)  Left→West(-x)
            #   Westbound  (driver faces west/left):   Right→North(+y) Left→South(-y)
            #   Eastbound  (driver faces east/right):  Right→South(-y) Left→North(+y)
            cfgs = {
                "Southbound": {
                    "pos": (0, 4.5), "color": _PAL["accent1"],
                    "arr_from": (0, 3.3), "arr_to": (0, 1.4),
                    # Right → West (screen left, -x)
                    "right_end": (-2.6, 0.8), "r_lbl": (-3.4, 1.6),
                    # Left → East (screen right, +x)
                    "left_end": (2.6, 0.8), "l_lbl": (3.4, 1.6),
                    "t_lbl": (0.8, 2.4),
                    # U-turn: back north (up)
                    "u_end": (0.8, 3.3), "u_lbl": (1.6, 3.6),
                    "r_rad": -0.4, "l_rad": 0.4, "u_rad": 1.0,
                },
                "Northbound": {
                    "pos": (0, -4.5), "color": _PAL["accent2"],
                    "arr_from": (0, -3.3), "arr_to": (0, -1.4),
                    # Right → East (screen right, +x)
                    "right_end": (2.6, -0.8), "r_lbl": (3.4, -1.6),
                    # Left → West (screen left, -x)
                    "left_end": (-2.6, -0.8), "l_lbl": (-3.4, -1.6),
                    "t_lbl": (-0.8, -2.4),
                    # U-turn: back south (down)
                    "u_end": (-0.8, -3.3), "u_lbl": (-1.6, -3.6),
                    "r_rad": -0.4, "l_rad": 0.4, "u_rad": 1.0,
                },
                "Westbound": {
                    "pos": (4.8, 0), "color": _PAL["accent3"],
                    "arr_from": (3.5, 0), "arr_to": (1.4, 0),
                    # Right → North (screen up, +y)
                    "right_end": (0.8, 2.6), "r_lbl": (1.6, 3.4),
                    # Left → South (screen down, -y)
                    "left_end": (0.8, -2.6), "l_lbl": (1.6, -3.4),
                    "t_lbl": (2.5, 0.7),
                    # U-turn: back east (right)
                    "u_end": (3.5, 0.8), "u_lbl": (3.9, 1.6),
                    "r_rad": -0.4, "l_rad": 0.4, "u_rad": 1.0,
                },
                "Eastbound": {
                    "pos": (-4.8, 0), "color": _PAL["accent4"],
                    "arr_from": (-3.5, 0), "arr_to": (-1.4, 0),
                    # Right → South (screen down, -y)
                    "right_end": (-0.8, -2.6), "r_lbl": (-1.6, -3.4),
                    # Left → North (screen up, +y)
                    "left_end": (-0.8, 2.6), "l_lbl": (-1.6, 3.4),
                    "t_lbl": (-2.5, -0.7),
                    # U-turn: back west (left)
                    "u_end": (-3.5, -0.8), "u_lbl": (-3.9, -1.6),
                    "r_rad": -0.4, "l_rad": 0.4, "u_rad": 1.0,
                },
            }

            # Collect all volumes for log scaling
            all_vols = []
            for d in approach_data.values():
                for k in ("thru", "right", "left", "u_turn"):
                    v = d.get(k, 0)
                    if v > 0:
                        all_vols.append(v)

            if not all_vols:
                plt.close(fig)
                return None

            min_vol = min(all_vols)
            max_vol = max(all_vols)

            def _log_width(vol):
                """Logarithmic scaling: maps volume → line width [1.5, 9.0]."""
                if vol <= 0:
                    return 0
                if max_vol == min_vol:
                    return 5.0
                log_min = math.log1p(min_vol)
                log_max = math.log1p(max_vol)
                t = (math.log1p(vol) - log_min) / (log_max - log_min)
                return 1.5 + t * 7.5

            # ── Draw each approach ──
            for direction, data in approach_data.items():
                if direction not in cfgs:
                    continue
                cfg = cfgs[direction]
                total = (data.get("thru", 0) + data.get("right", 0)
                         + data.get("left", 0) + data.get("u_turn", 0))
                if total == 0:
                    continue

                clr = cfg["color"]
                px, py = cfg["pos"]

                # Approach badge
                ax.text(
                    px, py, f"{direction}\n{total:,} veh",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white", zorder=10,
                    bbox=dict(
                        boxstyle="round,pad=0.5", facecolor=clr,
                        edgecolor="white", linewidth=1.5, alpha=0.95,
                    ),
                    path_effects=[
                        path_effects.withStroke(linewidth=0, foreground="white"),
                    ],
                )

                # Arrow origin: slightly inward from badge
                origin = (px * 0.65, py * 0.65)

                # ─ Thru arrow (straight through intersection) ─
                thru_vol = data.get("thru", 0)
                if thru_vol > 0:
                    w = _log_width(thru_vol)
                    ax.annotate(
                        "", xy=cfg["arr_to"], xytext=cfg["arr_from"],
                        arrowprops=dict(
                            arrowstyle=f"->,head_width={w * 0.08:.2f},head_length=0.30",
                            color=clr, lw=w, alpha=0.8,
                            shrinkA=0, shrinkB=0,
                        ),
                        zorder=5,
                    )
                    tx, ty = cfg["t_lbl"]
                    ax.text(
                        tx, ty, f"T: {thru_vol:,}",
                        fontsize=7, color=clr, fontweight="bold", zorder=10,
                        bbox=dict(
                            boxstyle="round,pad=0.15", facecolor=_PAL["surface"],
                            edgecolor=clr, linewidth=0.7, alpha=0.9,
                        ),
                    )

                # ─ Right turn (curves to driver's right) ─
                right_vol = data.get("right", 0)
                if right_vol > 0:
                    w = _log_width(right_vol)
                    rx, ry = cfg["right_end"]
                    ax.annotate(
                        "", xy=(rx, ry), xytext=origin,
                        arrowprops=dict(
                            arrowstyle=f"->,head_width={w * 0.07:.2f},head_length=0.25",
                            color=clr, lw=w, alpha=0.65,
                            connectionstyle=f"arc3,rad={cfg['r_rad']}",
                            shrinkA=2, shrinkB=2,
                        ),
                        zorder=4,
                    )
                    lx, ly = cfg["r_lbl"]
                    ax.text(
                        lx, ly, f"R: {right_vol:,}",
                        fontsize=6.5, color=clr, fontweight="bold", zorder=10,
                        bbox=dict(
                            boxstyle="round,pad=0.12", facecolor=_PAL["surface"],
                            edgecolor=clr, linewidth=0.5, alpha=0.85,
                        ),
                    )

                # ─ Left turn (curves to driver's left) ─
                left_vol = data.get("left", 0)
                if left_vol > 0:
                    w = _log_width(left_vol)
                    lx_c, ly_c = cfg["left_end"]
                    ax.annotate(
                        "", xy=(lx_c, ly_c), xytext=origin,
                        arrowprops=dict(
                            arrowstyle=f"->,head_width={w * 0.07:.2f},head_length=0.25",
                            color=clr, lw=w, alpha=0.65,
                            connectionstyle=f"arc3,rad={cfg['l_rad']}",
                            shrinkA=2, shrinkB=2,
                        ),
                        zorder=4,
                    )
                    llx, lly = cfg["l_lbl"]
                    ax.text(
                        llx, lly, f"L: {left_vol:,}",
                        fontsize=6.5, color=clr, fontweight="bold", zorder=10,
                        bbox=dict(
                            boxstyle="round,pad=0.12", facecolor=_PAL["surface"],
                            edgecolor=clr, linewidth=0.5, alpha=0.85,
                        ),
                    )

                # ─ U-turn (tight 180° curve back) ─
                u_vol = data.get("u_turn", 0)
                if u_vol > 0:
                    w = _log_width(u_vol)
                    ux, uy = cfg["u_end"]
                    ax.annotate(
                        "", xy=(ux, uy), xytext=origin,
                        arrowprops=dict(
                            arrowstyle=f"->,head_width={w * 0.06:.2f},head_length=0.20",
                            color=clr, lw=w, alpha=0.5,
                            connectionstyle=f"arc3,rad={cfg['u_rad']}",
                            shrinkA=2, shrinkB=2,
                        ),
                        zorder=4,
                    )
                    ulx, uly = cfg["u_lbl"]
                    ax.text(
                        ulx, uly, f"U: {u_vol:,}",
                        fontsize=6, color=clr, fontweight="bold", zorder=10,
                        bbox=dict(
                            boxstyle="round,pad=0.10", facecolor=_PAL["surface"],
                            edgecolor=clr, linewidth=0.4, alpha=0.8,
                        ),
                    )

            # ── Direction compass (top-right corner) ──
            cx, cy = 5.5, 5.0
            ax.text(cx, cy, "N", ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=_PAL["white"], zorder=10)
            ax.annotate("", xy=(cx, cy - 0.05), xytext=(cx, cy - 0.8),
                        arrowprops=dict(arrowstyle="->", color=_PAL["white"], lw=1.5),
                        zorder=10)
            ax.text(cx, cy - 1.1, "S", ha="center", va="top", fontsize=8,
                    color=_PAL["muted"], zorder=10)
            ax.text(cx + 0.6, cy - 0.5, "E", ha="left", va="center", fontsize=8,
                    color=_PAL["muted"], zorder=10)
            ax.text(cx - 0.6, cy - 0.5, "W", ha="right", va="center", fontsize=8,
                    color=_PAL["muted"], zorder=10)
            # Circle around compass
            compass = plt.Circle((cx, cy - 0.45), 0.75, fill=False,
                                 edgecolor=_PAL["muted"], linewidth=0.8,
                                 linestyle="--", zorder=9)
            ax.add_patch(compass)

            # ── Legend (top-right, below compass) ──
            legend_x, legend_y_start = 5.8, 3.6
            legend_items = [
                (d, cfgs[d]["color"])
                for d in ["Southbound", "Northbound", "Westbound", "Eastbound"]
                if d in approach_data and d in cfgs
            ]
            for i, (name, clr) in enumerate(legend_items):
                y = legend_y_start - i * 0.55
                ax.plot([legend_x - 0.5, legend_x], [y, y], color=clr, lw=3, alpha=0.85)
                ax.text(legend_x - 0.6, y, name, fontsize=7, va="center",
                        ha="right", color=_PAL["white"], fontweight="bold")

            # ── Watermark ──
            ax.text(
                6.0, -6.0, "PeakPulse Engine",
                ha="right", va="bottom", fontsize=7,
                color=_PAL["muted"], alpha=0.3, fontstyle="italic",
            )

            plt.tight_layout(pad=0.5)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                        facecolor=_PAL["bg"], edgecolor="none")
            plt.close(fig)
            buf.seek(0)

            return Image(buf, width=560, height=410)

        except Exception as e:
            logger.error("Failed to generate flow diagram: %s", e, exc_info=True)
            return None

    def _extract_approach_movements(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, int]]:
        """Extract total movement volumes per approach direction."""
        result: dict[str, dict[str, int]] = {}

        if not isinstance(df.columns, pd.MultiIndex):
            return result

        for col in df.columns:
            leg, direction, movement = col
            if movement in ("App Total", "Int Total", "Peds CW", "Peds CCW"):
                continue
            if direction not in result:
                result[direction] = {"right": 0, "thru": 0, "left": 0, "u_turn": 0}

            mov_key = movement.lower().replace("-", "_").replace(" ", "_")
            if mov_key in result[direction]:
                result[direction][mov_key] += int(df[col].sum())

        return result

    # ------------------------------------------------------------------
    # Anomaly table
    # ------------------------------------------------------------------

    def _build_anomaly_table(self, anomalies: list[AnomalyAlert]):
        data = [["Time", "Approach", "Movement", "Value", "Expected", "Severity", "Message"]]
        for a in anomalies:
            data.append([
                a.timestamp.strftime("%H:%M") if hasattr(a.timestamp, "strftime") else str(a.timestamp),
                f"{a.leg} — {a.direction}",
                a.movement,
                str(a.observed_value),
                a.expected_range,
                a.severity.value,
                a.message[:60],
            ])

        table = Table(data, colWidths=[50, 100, 60, 50, 80, 50, 180])
        style_cmds = [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(1, 0.9, 0.8)),
        ]
        table.setStyle(TableStyle(style_cmds))
        return table

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_peak(df: pd.DataFrame, peak: PeakResult | None) -> pd.DataFrame | None:
        if peak is None or df.empty:
            return None
        mask = (df.index >= peak.start_time) & (df.index < peak.end_time)
        return df.loc[mask]
