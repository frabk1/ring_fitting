# ============================
# analysis.py
# ============================

"""
====================================

* **Filename**:          analysis.py
* **Author**:            Frank Myhre
* **Description**:       High-level analysis container for ring feature extraction.

====================================

**Notes**
* Provides an AnalysisObject wrapper to manage an image plus defaults for ring
  analysis routines.
* Currently includes bright-point detection with optional masking.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class AnalysisObject:
    """
    Container for image-derived analysis and feature extraction.
    """

    def __init__(
        self,
        image,
        *,
        default_radius: Optional[float] = None,
        default_mask_radius: Optional[float] = None,
    ) -> None:
        """
        Initialize an AnalysisObject.

        **Args**:
        * image (ehtim.image.Image or array-like): source image.
        * default_radius (float, optional): default neighborhood/sample radius
          (pixels) for analysis functions (e.g., local maxima / ring sampling).
          If None, per-function fallbacks are used.
        * default_mask_radius (float, optional): default masking radius to suppress
          the bright core / center. If < 1.0, interpreted as a fraction of the
          image half-size. If >= 1.0, interpreted as pixels. If None, masking is
          disabled by default unless explicitly provided per call.

        **Returns**:
        * None

        """
        self.image = image
        self.default_radius = default_radius
        self.default_mask_radius = default_mask_radius

        # Results
        self.bright_points: List[Tuple[float, float]] = []
        self.center: Optional[Tuple[float, float]] = None

    # ---------- helpers ----------

    def _image_shape(self) -> Tuple[int, int]:
        """
        Returns (H, W) for numpy-like access. Tries eht-imaging Image first.

        **Args**:
        * None

        **Returns**:
        * shape (tuple[int, int]): (H, W) image shape.

        """
        im = self.image
        for attr in ("imarr", "arr", "image", "data"):
            if hasattr(im, attr):
                arr = getattr(im, attr)
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 2:
                        return arr.shape
                    if arr.ndim > 2:
                        return arr.shape[-2], arr.shape[-1]
        if isinstance(im, np.ndarray):
            if im.ndim == 2:
                return im.shape
            if im.ndim > 2:
                return im.shape[-2], im.shape[-1]
        raise ValueError("Unable to infer image shape from provided image.")

    def _image_array(self) -> np.ndarray:
        """
        Get a clean 2D numpy array from the image.

        **Args**:
        * None

        **Returns**:
        * arr (np.ndarray): 2D float array (H, W) representing image data.

        """
        im = self.image
        for attr in ("imarr", "arr", "image", "data"):
            if hasattr(im, attr):
                arr = getattr(im, attr)
                if isinstance(arr, np.ndarray):
                    return arr if arr.ndim == 2 else arr.squeeze()
        if isinstance(im, np.ndarray):
            return im if im.ndim == 2 else im.squeeze()
        raise ValueError("Unable to obtain ndarray from image.")

    def _center_xy(self) -> Tuple[float, float]:
        """
        Returns analysis center. If self.center is set, use it;
        otherwise use geometric image center.

        **Args**:
        * None

        **Returns**:
        * center (tuple[float, float]): (xc, yc) pixel center.

        """
        if self.center is not None:
            return self.center
        H, W = self._image_shape()
        return (W - 1) / 2.0, (H - 1) / 2.0

    def _mask_radius_to_pixels(self, mask_radius: Optional[float]) -> Optional[float]:
        """
        Convert a user mask_radius (fraction or pixels) to pixels.
        <1.0 => fraction of half-size (min(H,W)/2), >=1.0 => pixels.

        **Args**:
        * mask_radius (float or None): mask radius in fraction or pixels.

        **Returns**:
        * mask_px (float or None): mask radius in pixels.

        """
        if mask_radius is None:
            return None
        H, W = self._image_shape()
        half = min(H, W) / 2.0
        return mask_radius * half if (0 < mask_radius < 1.0) else mask_radius

    # ---------- primary API ----------

    def find_bright_points(
        self,
        *,
        threshold: Optional[float] = None,
        radius: Optional[float] = None,
        mask_radius: Optional[float] = None,
        max_points: Optional[int] = None,
        normalize_brightness: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Detect bright points on the ring with optional central masking.

        **Args**:
        * threshold (float, optional): absolute intensity threshold (image units).
          If None, an automatic heuristic is used (percentile-based).
        * radius (float, optional): neighborhood/sample radius in pixels.
          Overrides `default_radius`. If None and default is None, uses 6.0.
        * mask_radius (float, optional): center mask radius. If <1.0, treated as
          fraction of image half-size; if >=1.0, treated as pixels. Overrides
          `default_mask_radius`. If None and default is None, no masking.
        * max_points (int, optional): cap on number of returned points after
          sorting by brightness.
        * normalize_brightness (bool): if True, normalize image to [0, 1]
          before thresholding.

        **Returns**:
        * points (list[tuple[float, float]]): bright point coordinates (x, y).

        """
        arr = self._image_array().astype(float)

        if normalize_brightness:
            lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
            if hi > lo:
                arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        R = radius if radius is not None else (
            self.default_radius if self.default_radius is not None else 6.0
        )

        mr_user = mask_radius if mask_radius is not None else self.default_mask_radius
        mr_px = self._mask_radius_to_pixels(mr_user) if mr_user is not None else R

        if threshold is None:
            threshold = float(np.percentile(arr, 90))

        H, W = arr.shape
        active = np.ones((H, W), dtype=bool)
        limit = max_points if (max_points is not None and max_points > 0) else 999
        pts: List[Tuple[float, float]] = []

        for _ in range(limit):
            masked = arr * active
            peak = float(masked.max())
            if peak < threshold:
                break
            y, x = np.unravel_index(np.argmax(masked), arr.shape)
            pts.append((float(x), float(y)))

            # Blank a circle of mr_px around this point
            y0, y1 = max(0, y - int(mr_px)), min(H, y + int(mr_px) + 1)
            x0, x1 = max(0, x - int(mr_px)), min(W, x + int(mr_px) + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            active[y0:y1, x0:x1][dist <= mr_px] = False

        self.bright_points = pts
        return self.bright_points
