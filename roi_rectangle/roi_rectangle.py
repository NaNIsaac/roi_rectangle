from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy.typing as npt


@dataclass
class RoiRectangle:
    """
    A rectangular region of interest.

    Intervals are half-open: rows [y1, y2) and columns [x1, x2), so y2 and x2
    are not part of the region.

    Args:
        y1 (int): First row.
        y2 (int | None): End row (exclusive). None extends the region to the
            end of the axis.
        x1 (int): First column.
        x2 (int | None): End column (exclusive). None extends the region to
            the end of the axis.
    """

    y1: int
    y2: int | None
    x1: int
    x2: int | None

    def __post_init__(self) -> None:
        if self.x2 is not None and self.x2 < self.x1:
            raise ValueError("x2 must be >= x1")
        if self.y2 is not None and self.y2 < self.y1:
            raise ValueError("y2 must be >= y1")

    @property
    def width(self) -> int | None:
        return None if self.x2 is None else self.x2 - self.x1

    @property
    def height(self) -> int | None:
        return None if self.y2 is None else self.y2 - self.y1

    @property
    def center(self) -> tuple[int, int] | None:
        if self.x2 is None or self.y2 is None:
            return None
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    def _anchor(self) -> tuple[int, int]:
        """center, or with an unbounded axis, the start of that axis."""
        if self.center is not None:
            return self.center
        return self.x1 + (self.width or 0) // 2, self.y1 + (self.height or 0) // 2

    def move(self, dx: int, dy: int) -> RoiRectangle:
        """Return a new ROI shifted by (dx, dy)."""
        return RoiRectangle(
            self.y1 + dy,
            self.y2 + dy if self.y2 is not None else None,
            self.x1 + dx,
            self.x2 + dx if self.x2 is not None else None,
        )

    def recenter(self, new_center: tuple[int, int]) -> RoiRectangle:
        """Return a new ROI of the same size centered at new_center (x, y)."""
        x_i, y_i = self._anchor()
        x_f, y_f = new_center
        return self.move(x_f - x_i, y_f - y_i)

    def resize(self, new_width: int, new_height: int) -> RoiRectangle:
        """Return a new ROI of the given size around the same center."""
        if self.x2 is None and self.y2 is None:
            return self
        cx, cy = self._anchor()

        half_w, half_h = new_width // 2, new_height // 2
        new_x1, new_y1 = cx - half_w, cy - half_h
        new_x2, new_y2 = cx + (new_width - half_w), cy + (new_height - half_h)

        # An unbounded (None) axis has no size to change: keep its start.
        return RoiRectangle(
            y1=new_y1 if self.y2 is not None else self.y1,
            y2=new_y2 if self.y2 is not None else None,
            x1=new_x1 if self.x2 is not None else self.x1,
            x2=new_x2 if self.x2 is not None else None,
        )

    def to_tuple(self) -> tuple[int, int | None, int, int | None]:
        """Return the coordinates as (y1, y2, x1, x2)."""
        return self.y1, self.y2, self.x1, self.x2

    @property
    def area(self) -> int | None:
        """Number of pixels in the ROI, or None if an axis is unbounded."""
        if self.width is None or self.height is None:
            return None
        return self.width * self.height

    @property
    def shape(self) -> tuple[int | None, int | None]:
        """(height, width) of the ROI."""
        return (self.height, self.width)

    def get_slices(self) -> tuple[slice, slice]:
        """
        Return slices for indexing an image with the ROI.

        Returns:
            tuple: Slices for the y and x axes, in that order.
        """
        return slice(self.y1, self.y2), slice(self.x1, self.x2)

    def slice(self, image: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Select the ROI from the last two axes of the image.

        Args:
            image (np.ndarray): Input image or stack of images.

        Returns:
            np.ndarray: A view of the region, not a copy.
        """
        return image[..., self.y1 : self.y2, self.x1 : self.x2]

    @classmethod
    def from_tuple(cls, coords: tuple[int, int | None, int, int | None]) -> RoiRectangle:
        """Create an ROI from a (y1, y2, x1, x2) tuple."""
        y1, y2, x1, x2 = coords
        return cls(y1=y1, y2=y2, x1=x1, x2=x2)


RoiRectangle.__module__ = "roi_rectangle"
