import os
import numpy as np
from PIL import Image
from extraction import (rbp_find_bright_points, find_true_center, compute_centers, measure_thickness_fwhm,)
IMG = "im1.jpg"
arr = np.array(Image.open(IMG).convert("L"), dtype=np.float32)
if arr.max() > 0:
    arr /= arr.max()
    pts = rbp_find_bright_points(arr.copy(), threshold=0.02, radius=30, margin=10)
    xs, ys = find_true_center(pts)
    (cx_geo, cy_geo), (cx_flux, cy_flux), (cx_thr, cy_thr) = compute_centers(arr)

    print(f"points: {len(pts)}")
    print(f"RBP center : ({xs:.1f}, {ys:.1f})")
    print(f"geom : ({cx_geo:.1f}, {cy_geo:.1f})")
    print(f"flux : ({cx_flux:.1f}, {cy_flux:.1f})")
    print(f"25%  : ({cx_thr:.1f}, {cy_thr:.1f})")

    if pts:
        angs, thk, _, _, _ = measure_thickness_fwhm(
            arr, xs, ys, pts, rmin=20, rmax=100, nr=400
        )
        print(f"mean FWHM: {np.nanmean(thk):.2f} px")