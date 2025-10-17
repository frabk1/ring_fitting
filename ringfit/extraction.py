import numpy as np

def _img_to_array(img):
    """Get a clean 2D numpy array from the image."""
    arr = img.imarr()
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image array, got shape {arr.shape}")
    return arr

def _flux_center(data):
    """Flux-weighted center (x, y) using nonnegative weights."""
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
    """Quick background estimate from the darkest corner patches."""
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
    """Azimuthal average vs radius around (xc, yc)."""
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
    """Tiny 1D smooth to quiet pixel noise."""
    k = np.array([1, 2, 3, 2, 1], dtype=float)
    k /= k.sum()
    return np.convolve(prof, k, mode="same")

def _fwhm_width(r, prof, peak_idx, background):
    """FWHM-style width around a peak; falls back to moment width."""
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
    """Fast ring guesses: radius, width, peak, background, and center."""
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
        """Fallback: use global radial profile if rays fail."""
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
    """Recursive brightest-point finder with circular blanking."""
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
