# --- Minimal usage with a simple path fallback (paste & run) ---
import os, numpy as np, ehtim as eh
from ringfit.extraction import rbp_find_bright_points, compute_ring_features

# Pick a FITS path: prefer ./test.fits else ~/Downloads/test.fits
fits_path = "test.fits"
if not os.path.exists(fits_path):
    alt = os.path.expanduser("~/Downloads/test.fits")
    if os.path.exists(alt): fits_path = alt
    else: raise FileNotFoundError(f"Could not find 'test.fits'. CWD={os.getcwd()} — put it here or in ~/Downloads/")

im   = eh.image.load_image(fits_path)
data = np.asarray(im.imarr() if callable(getattr(im, "imarr", None)) else im.imarr, float)
H, W = data.shape
yy, xx = np.mgrid[0:H, 0:W]
tot = data.sum(); cx = (xx*data).sum()/tot; cy = (yy*data).sum()/tot

pts = rbp_find_bright_points(im, threshold=None, radius=None)
if pts is None or len(pts) < 8:
    TAU = 2*np.pi; th = np.linspace(0, TAU, 360, endpoint=False)
    rmax = 0.45*min(W, H); rg = np.linspace(0, rmax, 512)
    radii = np.empty_like(th); bright = np.empty_like(th)
    for i, t in enumerate(th):
        xr = cx + rg*np.cos(t); yr = cy + rg*np.sin(t)
        xi = np.clip(np.rint(xr).astype(int), 0, W-1); yi = np.clip(np.rint(yr).astype(int), 0, H-1)
        prof = data[yi, xi]; k = int(np.argmax(prof))
        radii[i] = rg[k]; bright[i] = prof[k]
    angles = th
else:
    x, y = pts[:,0], pts[:,1]
    angles = (np.arctan2(y - cy, x - cx)) % (2*np.pi)
    radii  = np.hypot(x - cx, y - cy)
    xi = np.clip(np.rint(x).astype(int), 0, W-1); yi = np.clip(np.rint(y).astype(int), 0, H-1)
    bright = data[yi, xi]

features = compute_ring_features(angles, radii, bright, bins=360)
for k in ["mean_radius","std_radius","radial_asymmetry","brightness_asymmetry",
          "angular_brightness_R","angular_brightness_circstd_deg","angular_brightness_FWHM_deg",
          "samples_used","bins"]:
    print(f"{k:>32s}: {features[k]}")
# --- end ---
