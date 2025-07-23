# extraction.py – ring-analysis helpers (NumPy / PIL / ehtim compatible)
import os, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Circle
import ehtim as eh

RADPERUAS = eh.RADPERUAS          # rad / µas

# ------------------------------------------------------------------ I/O
def _to_array(im, pol=None):
    if isinstance(im, eh.image.Image):                 # ehtim image
        arr = im.imarr(pol)
    elif isinstance(im, Image.Image):                  # PIL image
        arr = np.asarray(im.convert("L"), dtype=np.float32)
    elif isinstance(im, str):                          # filename
        arr = np.asarray(Image.open(im).convert("L"), dtype=np.float32)
    else:                                              # ndarray
        arr = np.asarray(im, dtype=np.float32).copy()
    m = arr.max()
    return arr / m if m > 0 else arr

def blur_eht_image(im, fwhm_uas):
    return im.blur_circ(fwhm_uas * RADPERUAS)

# ------------------------------------------------------ bright points
def rbp_find_bright_points(img, th, rad, pts=None, it=0, max_it=999, margin=0):
    if pts is None: pts = []
    if it > max_it: return pts
    ny, nx = img.shape
    sub = img[margin:ny-margin, margin:nx-margin]
    if sub.size == 0: return pts
    y, x = np.unravel_index(np.argmax(sub), sub.shape)
    y += margin; x += margin
    if img[y, x] <= th: return pts
    pts.append((y, x, img[y, x]))
    r = int(np.ceil(rad))
    yy, xx = np.ogrid[max(0, y-r):min(ny, y+r+1),
                      max(0, x-r):min(nx, x+r+1)]
    mask = (xx-x)**2 + (yy-y)**2 <= rad**2
    img[max(0, y-r):min(ny, y+r+1),
        max(0, x-r):min(nx, x+r+1)][mask] = 0
    return rbp_find_bright_points(img, th, rad, pts, it+1, max_it, margin)

# ----------------------------------------------------------- centers
def circumcenter(p1, p2, p3, tol=1e-6):
    y1, x1 = p1[:2]; y2, x2 = p2[:2]; y3, x3 = p3[:2]
    d = 2*(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    if abs(d) < tol: return None
    a1, a2, a3 = x1**2+y1**2, x2**2+y2**2, x3**2+y3**2
    cx = (a1*(y2-y3) + a2*(y3-y1) + a3*(y1-y2)) / d
    cy = (a1*(x3-x2) + a2*(x1-x3) + a3*(x2-x1)) / d
    return cx, cy

def find_true_center(pts, n=10):
    if len(pts) < 3:
        xs = np.mean([p[1] for p in pts]) if pts else 0
        ys = np.mean([p[0] for p in pts]) if pts else 0
        return xs, ys
    centers = [circumcenter(*random.sample(pts, 3)) for _ in range(n)]
    centers = np.array([c for c in centers if c is not None])
    return (centers[:, 0].mean(), centers[:, 1].mean()) if len(centers) else (0, 0)

def compute_centers(arr):
    ny, nx = arr.shape
    Y, X = np.indices(arr.shape)
    mask = arr > 0
    cg = (X[mask].mean(), Y[mask].mean())
    w = arr[mask]; wtot = w.sum()
    cf = ((X[mask]*w).sum()/wtot, (Y[mask]*w).sum()/wtot)
    thresh = np.percentile(w, 25); mask2 = arr >= thresh
    ct = (X[mask2].mean(), Y[mask2].mean())
    return cg, cf, ct

# -------------------------------------------- radial sampling & FWHM
def _sample(img, xs, ys, ang, rs):
    ny, nx = img.shape; vals = []
    for r in rs:
        xf, yf = xs + r*np.cos(ang), ys + r*np.sin(ang)
        if not (0 <= xf < nx-1 and 0 <= yf < ny-1):
            vals.append(0); continue
        x1, y1 = int(xf), int(yf)
        dx, dy = xf-x1, yf-y1
        x2, y2 = x1+1, y1+1
        I00, I10 = img[y1, x1], img[y1, x2]
        I01, I11 = img[y2, x1], img[y2, x2]
        vals.append(I00*(1-dx)*(1-dy) + I10*dx*(1-dy) +
                    I01*(1-dx)*dy     + I11*dx*dy)
    return np.array(vals)

def _fwhm(rs, prof):
    m = prof.max()
    if m <= 0: return 0, np.nan, np.nan
    half = m/2; im = np.argmax(prof)
    l = next((rs[i] + (half-prof[i])*(rs[i+1]-rs[i])/(prof[i+1]-prof[i])
              for i in range(im-1, -1, -1) if prof[i] < half), np.nan)
    r = next((rs[i-1] + (half-prof[i-1])*(rs[i]-rs[i-1])/(prof[i]-prof[i-1])
              for i in range(im+1, len(prof)) if prof[i] < half), np.nan)
    return r - l, l, r

# --- per-ray profile metrics ----------------------------------------------
def _measure_profiles(arr, xs, ys, pts, halfwin=40, nr=400):
    ang, flux, width, r_bp = [], [], [], []
    for yb, xb, _ in pts:
        r0  = np.hypot(yb - ys, xb - xs)
        rs  = np.linspace(max(0, r0 - halfwin), r0 + halfwin, nr)
        ang.append(np.degrees(np.arctan2(yb - ys, xb - xs)))
        prof = _sample(arr, xs, ys, np.arctan2(yb - ys, xb - xs), rs)
        flux.append(prof.sum())
        w, _, _ = _fwhm(rs, prof)
        width.append(w if w > 0 else np.nan)
        r_bp.append(r0)
    s = np.argsort(ang)
    return (np.array(ang)[s], np.array(r_bp)[s],
            np.array(flux)[s], np.array(width)[s])

# --- public entry ---------------------------------------------------------
def get_quantities(im, qlist=('angle', 'radius', 'brightness', 'width'),
                   bp_th=0.005, bp_rad=30, margin=15, halfwin=40):
    arr = _to_array(im)
    pts = rbp_find_bright_points(arr.copy(), bp_th, bp_rad, margin=margin)
    xs, ys = find_true_center(pts)
    ang, rad, flux, width = _measure_profiles(arr, xs, ys, pts, halfwin=halfwin)
    q = {'angle': ang, 'radius': rad, 'brightness': flux, 'width': width}
    return tuple(q[k] for k in qlist)


# ----------------------------------------------------------- viz
def show_centers(im, pts=None, ax=None):
    arr = _to_array(im)
    if pts is None:
        pts = rbp_find_bright_points(arr.copy(), 0.02, 30, margin=15)
    xs, ys = find_true_center(pts)
    (cgx, cgy), (cfx, cfy), (ctx, cty) = compute_centers(arr)
    if ax is None: _, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(arr, cmap='gray', origin='lower')
    ax.scatter([xs], [ys], c='r', marker='+', label='RBP')
    ax.scatter([cgx], [cgy], c='y', marker='x', label='Geom')
    ax.scatter([cfx], [cfy], c='c', marker='*', label='Flux')
    ax.scatter([ctx], [cty], c='g', marker='^', label='25%')
    for y, x, _ in pts:
        ax.add_patch(Circle((x, y), 3, fill=False, ec='r', lw=0.5))
    ax.set_xticks([]); ax.set_yticks([]); ax.legend()
    return ax

# ---------------------------------------------------- self-test
def im1():
    if not os.path.exists("im1.jpg"):
        print("im1.jpg not found"); return
    arr = _to_array("im1.jpg")
    pts = rbp_find_bright_points(arr.copy(), 0.02, 30, margin=15)
    xs, ys = find_true_center(pts)
    (cgx, cgy), (cfx, cfy), (ctx, cty) = compute_centers(arr)
    print(f"RBP centre : {xs:.1f}, {ys:.1f}")
    print(f"Geom       : {cgx:.1f}, {cgy:.1f}")
    print(f"Flux COM   : {cfx:.1f}, {cfy:.1f}")
    print(f"25% COM    : {ctx:.1f}, {cty:.1f}")
    ang, width, *_ = _measure_profiles(arr, xs, ys, pts)
    print(f"Mean FWHM  : {np.nanmean(width):.2f}px")
    show_centers(arr, pts); plt.show()

if __name__ == "__main__":
    im1()
