# ============================
# centroids.py
# ============================

"""
====================================

* **Filename**:          centroids.py
* **Author**:            Frank Myhre
* **Description**:       Simple center-finding utilities for 2D images.

====================================

**Notes**
* Provides geometric and thresholded centroid estimators for numpy arrays.
"""

import numpy as np


def geometric_centroid(data):
    """
    Geometric centroid: average weighted by pixel values.

    **Args**:
    * data (np.ndarray): 2D image array.

    **Returns**:
    * center (tuple[float, float]): (xc, yc) centroid in pixel coordinates.

    """
    yy, xx = np.indices(data.shape)
    tot = data.sum()
    return (xx * data).sum() / tot, (yy * data).sum() / tot


def threshold_center(data, q=25):
    """
    Center of pixels above a percentile threshold.

    **Args**:
    * data (np.ndarray): 2D image array.
    * q (float): percentile threshold (0–100). Pixels >= percentile are used.

    **Returns**:
    * center (tuple[float, float]): (xc, yc) of thresholded mask.

    """
    thresh = np.percentile(data, q)
    mask = data >= thresh
    yy, xx = np.indices(data.shape)
    tot = mask.sum()
    return xx[mask].sum() / tot, yy[mask].sum() / tot
