# ============================
# extraction.py
# ============================

"""
====================================

* **Filename**:          extraction.py
* **Author**:            Frank Myhre
* **Description**:  

====================================

**Notes**
* Utilities for extracting ring points and single-number ring features from images.
"""

import numpy as np

def _img_to_array(img):
    """
    Get a clean 2D numpy array from the image.

    **Args**:
    * img (ehtim.Image or similar): image object exposing `imarr()` or `imarr`.

    **Returns**:
    * arr (np.ndarray): 2D float array (H, W) representing image data.

    """
    arr = img.imarr()
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image array, got shape {arr.shape}")
    return arr

def _flux_center(data):
    """
    Flux-weighted center (x, y) using nonnegative weights.

    **Args**:
    * data (np.ndarray): 2D image array (H, W), nonnegative preferred.

    **Returns**:
    * center (tuple[float, float]): (xc, yc) in pixel coordinates.

    """
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    wts = np.maximum(data, 0.0)
    tot = wts.sum()
    if tot <= 0:
        return (w - 1) / 2.0, (h - 1) / 2.0
    xc = (wts * xx).sum() / tot
    yc = (wts * yy).sum() / tot
    return float(xc), float(yc)

def _estimate_background(data, patch_frac=0.12):
    """
    Quick background estimate from the darkest corner patches.

    **Args**:
    * data (np.ndarray): 2D image array (H, W).
    * patch_frac (float): fractional patch size based on min(H, W).

    **Returns**:
    * background (float): estimated background level.

    """
    h, w = data.shape
    k = max(1, int(patch_frac * min(h, w)))
    patches = [
        data[0:k, 0:k],
        data[0:k, w - k:w],
        data[h - k:h, 0:k],
        data[h - k:h, w - k:w],
    ]
    means = [p.mean() for p in patches]
    return float(min(means))

def _radial_profile(data, xc, yc, bin_size=1.0):
    """
    Azimuthal average vs radius around (xc, yc).

    **Args**:
    * data (np.ndarray): 2D image array (H, W).
    * xc (float): x-coordinate of center.
    * yc (float): y-coordinate of center.
    * bin_size (float): radial bin size in pixels.

    **Returns**:
    * r_centers (np.ndarray): radius centers for bins.
    * prof (np.ndarray): azimuthally averaged profile at r_centers.

    """
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2).ravel()
    vv = data.ravel()
    rmax = rr.max()
    nbins = max(1, int(np.ceil((rmax + 1e-9) / bin_size)))
    idx = np.clip((rr / bin_size).astype(int), 0, nbins - 1)
    sum_v = np.bincount(idx, weights=vv, minlength=nbins)
    cnt_v = np.bincount(idx, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = sum_v / np.maximum(cnt_v, 1)
    r_centers = (np.arange(nbins) + 0.5) * bin_size
    return r_centers, prof

def _smooth_profile(prof):
    """
    Tiny 1D smooth to quiet pixel noise.

    **Args**:
    * prof (np.ndarray): 1D profile array.

    **Returns**:
    * smoothed (np.ndarray): smoothed profile, same shape as input.

    """
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    k /= k.sum()
    return np.convolve(prof, k, mode="same")

def _fwhm_width(r, prof, peak_idx, background):
    """
    FWHM-style width around a peak; falls back to moment width.

    **Args**:
    * r (np.ndarray): radius array (same length as prof).
    * prof (np.ndarray): profile values over r.
    * peak_idx (int): index of the peak in prof.
    * background (float): background level for half-max.

    **Returns**:
    * width (float): estimated FWHM-like width in pixel units.

    """
    p_peak = prof[peak_idx]
    half = background + 0.5 * (p_peak - background)
    i_left = None
    for i in range(peak_idx, -1, -1):
        if prof[i] <= half:
            i_left = i
            break
    i_right = None
    for i in range(peak_idx, len(prof)):
        if prof[i] <= half:
            i_right = i
            break
    if i_left is not None and i_right is not None and i_right > i_left:
        def interp_x(i0, i1):
            y0, y1 = prof[i0], prof[i1]
            x0, x1 = r[i0], r[i1]
            if y1 == y0:
                return x0
            t = (half - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
        rL = interp_x(i_left - 1 if i_left > 0 else i_left, i_left)
        rR = interp_x(i_right, i_right + 1 if i_right + 1 < len(prof) else i_right)
        return float(max(1e-6, rR - rL))
    wL = max(0, peak_idx - 3)
    wR = min(len(prof), peak_idx + 4)
    rw = r[wL:wR]
    pw = np.maximum(prof[wL:wR] - background, 0.0)
    sw = pw.sum()
    if sw <= 0:
        return 2.0
    mu = (rw * pw).sum() / sw
    var = (pw * (rw - mu) ** 2).sum() / sw
    return float(max(2.0, 2.355 * np.sqrt(max(var, 1e-12))))

def estimate_ring_parameters(img, bin_size=1.0, patch_frac=0.12, n_theta=180, step=0.5):
    """
    Fast ring guesses: radius, width, peak, background, and center.

    **Args**:
    * img (ehtim.Image or similar): input image object.
    * bin_size (float): radial bin size for fallback profile.
    * patch_frac (float): corner patch fraction for background estimate.
    * n_theta (int): number of rays for radial scans.
    * step (float): step size along each ray (pixels).

    **Returns**:
    * radius (float): median peak radius across rays or from fallback profile.
    * width (float): median FWHM-like width across rays or from fallback.
    * peak_value (float): maximum peak level found.
    * background (float): estimated background level.
    * center (tuple[float, float]): (xc, yc) using image midpoint.

    """
    data = _img_to_array(img)
    h, w = data.shape
    bkg = _estimate_background(data, patch_frac=patch_frac)

    # center = image middle (as requested)
    xc, yc = (w - 1) / 2.0, (h - 1) / 2.0

    rmax = float(np.hypot(h, w))
    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_theta), endpoint=False)

    radii = []
    widths = []
    peak_vals = []

    k = np.array([1, 2, 3, 2, 1], float)  # light smoothing for 1D ray profiles
    k /= k.sum()

    for th in thetas:
        rs = np.arange(0.0, rmax, float(step))
        xs = xc + rs * np.cos(th)
        ys = yc + rs * np.sin(th)

        m = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
        if not np.any(m):
            continue
        xs, ys, rs = xs[m], ys[m], rs[m]
        if rs.size < 3:
            continue

        ix = np.rint(xs).astype(int)
        iy = np.rint(ys).astype(int)
        prof = data[iy, ix]
        prof_s = np.convolve(prof, k, mode="same")

        j = int(np.nanargmax(prof_s))
        vmax = float(prof_s[j])
        peak_vals.append(vmax)
        radii.append(float(rs[j]))

        half = bkg + 0.5 * (vmax - bkg)

        jl = j
        while jl > 0 and prof_s[jl] > half:
            jl -= 1
        jr = j
        L = len(prof_s)
        while jr < L - 1 and prof_s[jr] > half:
            jr += 1

        def _interp_r(i0, i1):
            y0, y1 = prof_s[i0], prof_s[i1]
            r0, r1 = rs[i0], rs[i1]
            if y1 == y0:
                return r0
            t = (half - y0) / (y1 - y0)
            return r0 + t * (r1 - r0)

        if jl > 0 and jr < L - 1:
            rL = _interp_r(jl, jl + 1)
            rR = _interp_r(jr - 1, jr)
            widths.append(float(max(1e-6, rR - rL)))

    if len(radii) == 0:
        """
        Fallback: use global radial profile if rays fail.
        """
        r, prof = _radial_profile(data, xc, yc, bin_size=bin_size)
        ps = _smooth_profile(prof)
        idx = int(np.nanargmax(ps))
        radius = float(r[idx])
        width = _fwhm_width(r, ps, idx, bkg)
        peak_value = float(ps[idx])
    else:
        radius = float(np.median(radii))
        width = float(np.median(widths)) if widths else max(2.0, 0.1 * radius)
        peak_value = float(np.max(peak_vals))

    return radius, width, peak_value, float(bkg), (xc, yc)

def rbp_find_bright_points(img, threshold=None, radius=None, margin=None, max_it=999):
    """
    Recursive brightest-point finder with circular blanking.

    **Args**:
    * img (ehtim.Image or similar): input image object.
    * threshold (float or None): minimum brightness to accept a point; if None, auto.
    * radius (float or None): blanking radius in pixels; if None, auto from width.
    * margin (int or None): border margin to ignore; if None, derived from radius.
    * max_it (int): maximum number of points to extract.

    **Returns**:
    * points (np.ndarray): shape (N, 2) array of (x, y) pixel coordinates.

    """
    data = _img_to_array(img)
    if threshold is None or radius is None:
        r0, w0, pmax, bkg, _ = estimate_ring_parameters(img)
        if threshold is None:
            threshold = bkg + 0.5 * (pmax - bkg)
        if radius is None:
            radius = max(1.0, 3.0 * w0)
    h, w = data.shape
    if margin is None:
        margin = int(np.ceil(radius + 1))
    mask = np.ones_like(data, dtype=bool)
    points = []
    for _ in range(max_it):
        masked = data * mask
        peak = masked.max()
        if peak < threshold:
            break
        y, x = np.unravel_index(np.argmax(masked), data.shape)
        if x < margin or x >= (w - margin) or y < margin or y >= (h - margin):
            mask[y, x] = False
            continue
        points.append((x, y))
        y0, y1 = max(0, y - int(radius)), min(h, y + int(radius) + 1)
        x0, x1 = max(0, x - int(radius)), min(w, x + int(radius) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        mask[y0:y1, x0:x1][dist <= radius] = False
    return np.array(points)

import numpy as np
TAU = 2*np.pi

def compute_ring_features(angles, radii, brightness, bins=360):
    """
    Compute single-number diagnostics from per-angle ring samples.

    **Args**:
    * angles (array-like): angles in radians for each sample.
    * radii (array-like): radius at each angle (pixels).
    * brightness (array-like): brightness/intensity per sample.
    * bins (int): even number of angular bins for pairing and widths.

    **Returns**:
    * features (dict): {
        mean_radius, std_radius,
        radial_asymmetry, brightness_asymmetry,
        angular_brightness_R, angular_brightness_circstd_rad,
        angular_brightness_circstd_deg, angular_brightness_FWHM_deg,
        samples_used, bins
      }

    """
    a = np.asarray(angles, float).ravel()
    r = np.asarray(radii, float).ravel()
    b = np.asarray(brightness, float).ravel()
    m = np.isfinite(a) & np.isfinite(r) & np.isfinite(b)
    a, r, b = a[m] % TAU, r[m], b[m]
    if a.size < 8 or (bins % 2): 
        raise ValueError("Need >=8 samples and even `bins`.")
    mean_radius, std_radius = float(r.mean()), float(r.std())

    # Uniform angular bins
    edges = np.linspace(0, TAU, bins+1)
    idx = np.digitize(a, edges, right=False) - 1
    idx[idx == bins] = 0
    cnt = np.bincount(idx, minlength=bins)
    rsum = np.bincount(idx, weights=r, minlength=bins)
    bsum = np.bincount(idx, weights=b, minlength=bins)
    r_u = rsum / np.maximum(1, cnt)
    b_u = bsum.astype(float)
    th = 0.5*(edges[:-1] + edges[1:])

    # Opposite-angle asymmetries
    def _opp_asym(x, norm):
        d = np.abs(x - np.roll(x, x.size//2)).mean()
        return float(d / (norm if norm > 0 else 1.0))
    radial_asym = _opp_asym(r_u, mean_radius)
    mean_b = float(b_u.mean())
    brightness_asym = _opp_asym(b_u, mean_b)

    # Circular concentration & circular std (brightness-weighted)
    W = b_u.sum()
    if W <= 0 or not np.isfinite(W):
        R = 0.0; circstd_rad = np.nan
    else:
        C = float((b_u*np.cos(th)).sum()/W)
        S = float((b_u*np.sin(th)).sum()/W)
        R = float(np.hypot(C, S))
        circstd_rad = (np.sqrt(max(0.0, -2*np.log(R))) if R > 0 else np.inf)
    circstd_deg = float(np.degrees(circstd_rad)) if np.isfinite(circstd_rad) else np.nan

    # FWHM (deg) of largest brightness peak (simple half-max crossing)
    def _fwhm_deg(y):
        n = y.size
        k = int(np.argmax(y)); ymax = float(y[k])
        if not np.isfinite(ymax) or ymax <= 0: return np.nan
        half = 0.5*ymax
        def cross(start, step):
            i = start
            for _ in range(n):
                j = (i + step) % n
                yi, yj = y[i], y[j]
                if (yi - half)*(yj - half) <= 0 and yi != yj:
                    frac = (half - yi)/(yj - yi)
                    return (i + frac*step) % n
                i = j
            return None
        L, Rr = cross(k, -1), cross(k, +1)
        if L is None or Rr is None: return np.nan
        dth = TAU/n
        w = ((Rr - L) % n)*dth
        if w > np.pi: w = TAU - w
        return float(np.degrees(w))
    fwhm = _fwhm_deg(b_u)

    return {
        "mean_radius": mean_radius,
        "std_radius": std_radius,
        "radial_asymmetry": radial_asym,
        "brightness_asymmetry": brightness_asym,
        "angular_brightness_R": R,
        "angular_brightness_circstd_rad": float(circstd_rad),
        "angular_brightness_circstd_deg": float(circstd_deg),
        "angular_brightness_FWHM_deg": float(fwhm),
        "samples_used": int(a.size),
        "bins": int(bins),
    }
