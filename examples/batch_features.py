# ============================
# batch_features.py
# ============================

"""
====================================

* **Filename**:          batch_features.py
* **Author**:            Frank Myhre
* **Description**:       Batch ring feature extraction and pooled summary statistics.

====================================

**Notes**
* Loads eht-imaging images from disk, extracts ring samples via RBP bright points,
  and computes per-image ring features.
* Falls back to a simple radial scan if too few bright points are found.
* Provides pooled summary stats across a batch of images.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import ehtim as eh
import numpy as np

from ringfit.extraction import compute_ring_features, rbp_find_bright_points


def _im_to_array(im) -> np.ndarray:
    """
    Return a 2D float image array from an eht-imaging Image.

    **Args**:
    * im (ehtim.Image or similar): image object exposing `imarr()` or `imarr`.

    **Returns**:
    * arr (np.ndarray): 2D float array (H, W).

    """
    arr = im.imarr() if callable(getattr(im, "imarr", None)) else im.imarr
    a = np.asarray(arr, float)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {a.shape}")
    return a


def _flux_center(data: np.ndarray) -> Tuple[float, float]:
    """
    Brightness centroid (cx, cy) in pixels.

    **Args**:
    * data (np.ndarray): 2D image array (H, W).

    **Returns**:
    * center (tuple[float, float]): (cx, cy) flux-weighted centroid.

    """
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tot = float(np.nansum(data))
    if not np.isfinite(tot) or tot <= 0:
        raise ValueError("Non-positive total flux.")
    return float((xx * data).sum() / tot), float((yy * data).sum() / tot)


def _fallback_radial_scan(
    data: np.ndarray,
    cx: float,
    cy: float,
    n_th: int = 360,
    n_r: int = 512,
    rfrac: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple per-angle max along a ray.

    **Args**:
    * data (np.ndarray): 2D image array (H, W).
    * cx (float): x-center for radial scan.
    * cy (float): y-center for radial scan.
    * n_th (int): number of angular rays.
    * n_r (int): number of radial samples per ray.
    * rfrac (float): max radius as a fraction of min(H, W).

    **Returns**:
    * angles (np.ndarray): angles in radians, shape (n_th,).
    * radii (np.ndarray): peak radius per angle, shape (n_th,).
    * brightness (np.ndarray): peak brightness per angle, shape (n_th,).

    """
    h, w = data.shape
    tau = 2 * np.pi
    th = np.linspace(0, tau, n_th, endpoint=False)
    rmax = rfrac * min(w, h)
    rg = np.linspace(0.0, rmax, n_r)

    radii = np.empty(n_th, float)
    bright = np.empty(n_th, float)

    for i, t in enumerate(th):
        xr = cx + rg * np.cos(t)
        yr = cy + rg * np.sin(t)
        xi = np.clip(np.rint(xr).astype(int), 0, w - 1)
        yi = np.clip(np.rint(yr).astype(int), 0, h - 1)
        prof = data[yi, xi]
        k = int(np.argmax(prof))
        radii[i] = rg[k]
        bright[i] = float(prof[k])

    return th, radii, bright


def features_for_image(path: str, *, bins: int = 360) -> Dict[str, float]:
    """
    One image -> one feature dict.

    Uses RBP bright points; if too few points are found,
    falls back to a radial scan.

    **Args**:
    * path (str): filesystem path to an eht-imaging-readable image.
    * bins (int): number of angular bins for feature computation (even).

    **Returns**:
    * features (dict[str, float]): ring feature dictionary.

    """
    im = eh.image.load_image(path)
    data = _im_to_array(im)
    cx, cy = _flux_center(data)

    pts = rbp_find_bright_points(im, threshold=None, radius=None)

    if pts is None or len(pts) < 8:
        angles, radii, brightness = _fallback_radial_scan(
            data, cx, cy, n_th=max(360, bins)
        )
    else:
        x, y = pts[:, 0], pts[:, 1]
        angles = (np.arctan2(y - cy, x - cx)) % (2 * np.pi)
        radii = np.hypot(x - cx, y - cy)

        h, w = data.shape
        xi = np.clip(np.rint(x).astype(int), 0, w - 1)
        yi = np.clip(np.rint(y).astype(int), 0, h - 1)
        brightness = data[yi, xi]

    return compute_ring_features(angles, radii, brightness, bins=bins)


def _summarize(col: np.ndarray) -> Dict[str, float]:
    """
    Basic summary stats on a 1D float array (ignores NaNs).

    **Args**:
    * col (np.ndarray): 1D array of values.

    **Returns**:
    * stats (dict[str, float]): mean/std/median/iqr/min/max/n.

    """
    x = np.asarray(col, float)
    m = np.isfinite(x)
    x = x[m]

    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n": 0,
        }

    q25, q75 = np.percentile(x, [25, 75])
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=0)),
        "median": float(np.median(x)),
        "iqr": float(q75 - q25),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def compute_batch_ring_stats(paths: Sequence[str], *, bins: int = 360) -> Dict[str, Any]:
    """
    Many images -> per-image features + pooled stats.

    **Args**:
    * paths (Sequence[str]): list/sequence of image paths.
    * bins (int): number of angular bins for feature computation (even).

    **Returns**:
    * out (dict): {
        "per_image": [{"path": ..., <features>}, ...],
        "summary": {feature_key: {mean,std,median,iqr,min,max,n}, ...}
      }

    """
    per = []
    for p in paths:
        f = features_for_image(p, bins=bins)
        row = {"path": p}
        row.update(f)
        per.append(row)

    if not per:
        return {"per_image": [], "summary": {}}

    keys = [k for k in per[0].keys() if k != "path"]
    summary = {
        k: _summarize(np.array([d.get(k, np.nan) for d in per], float))
        for k in keys
    }
    return {"per_image": per, "summary": summary}
