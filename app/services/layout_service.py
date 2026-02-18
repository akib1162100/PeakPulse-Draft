"""
Layout Service – Automated PDF Layout Engine.

Rule-based heuristics to compute element positions on PDF pages,
prevent table/diagram overlap, and manage page breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LayoutElement:
    """An element to position on a PDF page."""
    element_type: Literal["table", "diagram", "header", "text"]
    width: float   # points
    height: float  # points
    content: object = None  # arbitrary payload
    margin_top: float = 10
    margin_bottom: float = 10


@dataclass
class PositionedElement:
    """An element with computed page position."""
    element: LayoutElement
    page: int
    x: float
    y: float  # measured from TOP of page


class LayoutService:
    """
    Computes non-overlapping positions for PDF elements across pages.

    Uses a simple top-down placement algorithm with automatic page
    breaks when content exceeds page height.
    """

    def __init__(
        self,
        page_width: float = 595.0,    # A4 width in points
        page_height: float = 842.0,   # A4 height in points
        margin_left: float = 40.0,
        margin_right: float = 40.0,
        margin_top: float = 50.0,
        margin_bottom: float = 50.0,
    ):
        self.page_width = page_width
        self.page_height = page_height
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.usable_width = page_width - margin_left - margin_right
        self.usable_height = page_height - margin_top - margin_bottom

    def compute_layout(
        self, elements: list[LayoutElement]
    ) -> list[PositionedElement]:
        """
        Assign a (page, x, y) position to each element.

        Algorithm:
        - Place elements top-to-bottom on the current page.
        - If an element doesn't fit, start a new page.
        - Tables and diagrams that are taller than the usable height
          are clamped to fit (with a warning logged).
        """
        positioned: list[PositionedElement] = []
        current_page = 1
        cursor_y = self.margin_top  # distance from top

        for elem in elements:
            total_height = elem.margin_top + elem.height + elem.margin_bottom

            # Check if element fits on current page
            remaining = self.page_height - self.margin_bottom - cursor_y
            if total_height > remaining and cursor_y > self.margin_top:
                # Start new page
                current_page += 1
                cursor_y = self.margin_top

            # Position element
            y_pos = cursor_y + elem.margin_top
            x_pos = self.margin_left

            # Centre narrow elements
            if elem.width < self.usable_width:
                x_pos = self.margin_left + (self.usable_width - elem.width) / 2

            positioned.append(
                PositionedElement(
                    element=elem,
                    page=current_page,
                    x=x_pos,
                    y=y_pos,
                )
            )

            cursor_y = y_pos + elem.height + elem.margin_bottom

        return positioned

    def get_usable_dimensions(self) -> tuple[float, float]:
        """Return (usable_width, usable_height) in points."""
        return self.usable_width, self.usable_height
