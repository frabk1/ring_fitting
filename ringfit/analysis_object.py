import numpy as np
import matplotlib.pyplot as plt
from . import extraction as ex
from . import utils

class AnalysisObject:
    """Small, simple holder for image + quick ring diagnostics."""

    def __init__(self, im):
        """Grab image + basic meta once."""
        self.image = im
        self.data = im.imarr()
        self.xdim = im.xdim
        self.ydim = im.ydim
        self.cell = im.psize
        self.ra = im.ra
        self.dec = im.dec
        self.peak = getattr(im, "peak", None)
        self.total_flux = im.total_flux()
        self.compute_centers()
        self.bright_points = None

    def compute_centers(self):
        """Just the simple ones we actually use."""
        self.geo_c = utils.geometric_centroid(self.data)
        self.q25_c = utils.threshold_center(self.data, q=25)

    def find_bright_points(self, threshold=0.5, radius=5.0, margin=None, max_it=999):
        """Primary bright-point finder (recursive blanking)."""
        self.bright_points = ex.rbp_find_bright_points(self.image, threshold, radius, margin, max_it)
        return self.bright_points

    def plot_centers(self):
        """Show image with our reference centers."""
        fig, ax = plt.subplots(figsize=(6, 6))
        extent = [0, self.xdim * self.cell, 0, self.ydim * self.cell]
        ax.imshow(self.data, origin="lower", cmap="afmhot", extent=extent)
        gx, gy = self.geo_c
        tx, ty = self.q25_c
        ax.plot(gx * self.cell, gy * self.cell, "wo", label="Geometric")
        ax.plot(tx * self.cell, ty * self.cell, "bo", label="Threshold")
        ax.legend(loc="upper right")
        ax.set_xlabel("x [radian]")
        ax.set_ylabel("y [radian]")
        return fig

    def plot_angle_profiles(
        self,
        n_angles=20,
        center="q25",            # keep it simple: default to threshold center
        normalize=True,          # <- now normalize brightness (0..1)
        *,
        threshold_factor=0.01,   # used to seed bright-point extraction
        mask_mult=5.0,
        reuse_points=True,
    ):
        """
        Super-simple version:
        - Pick center (default: threshold center).
        - Use bright points only.
        - Clock-style angle: 12→0°, 3→90°, 6→180°, 9→270°.
        - Bin 0..360° into `n_angles`, keep *brightest* point per bin.
        - Plots: image+points | Radius vs Angle | Brightness vs Angle (optionally normed).
        """
        data = self.data
        H, W = data.shape

        # ring stats to seed bright-point search and (optionally) normalize brightness
        r_est, w_est, pmax, bkg, (xc_est, yc_est) = ex.estimate_ring_parameters(self.image)

        # ensure we have plenty of bright points
        if (self.bright_points is None) or (not reuse_points) or (len(self.bright_points) == 0):
            thr = bkg + float(threshold_factor) * (pmax - bkg)
            mask_r = max(1.0, float(mask_mult) * w_est)
            self.find_bright_points(threshold=thr, radius=mask_r)
        if self.bright_points is None or len(self.bright_points) == 0:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.text(0.5, 0.5, "No bright points found", ha="center", va="center")
            ax.set_axis_off()
            return fig

        pts = np.asarray(self.bright_points, float)
        x = pts[:, 0]
        y = pts[:, 1]

        # choose center (no fitting, no extras)
        if isinstance(center, (tuple, list, np.ndarray)) and len(center) == 2:
            xs, ys = float(center[0]), float(center[1])
        else:
            key = str(center).lower()
            if key in ("geo", "geometric"):
                xs, ys = self.geo_c
            elif key in ("ring", "est", "mid", "image"):
                xs, ys = float(xc_est), float(yc_est)
            else:  # "q25"/"threshold"/anything else → threshold center
                xs, ys = self.q25_c

        # angles (clock) and radii w.r.t. chosen center
        dx = x - xs
        dy = y - ys  # origin='lower' → y is up in display
        angles = (90.0 - np.degrees(np.arctan2(dy, dx))) % 360.0
        radii = np.hypot(dx, dy)

        # brightness at bright-point pixels
        xi = np.clip(np.rint(x).astype(int), 0, W - 1)
        yi = np.clip(np.rint(y).astype(int), 0, H - 1)
        bright = data[yi, xi].astype(float)

        # pick brightest in each bin to spread samples around ring
        n_bins = int(max(1, n_angles))
        edges = np.linspace(0.0, 360.0, n_bins + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])

        pick_ang, pick_rad, pick_bri, pick_xy = [], [], [], []
        for i in range(n_bins):
            a0, a1 = edges[i], edges[i + 1]
            sel = (angles >= a0) & (angles < a1) if i < n_bins - 1 else (angles >= a0) & (angles <= a1)
            if not np.any(sel):
                continue
            jloc = np.argmax(bright[sel])
            j = np.flatnonzero(sel)[jloc]
            pick_ang.append(mids[i])
            pick_rad.append(radii[j])
            pick_bri.append(bright[j])
            pick_xy.append((x[j], y[j]))

        # normalize brightness to 0..1 if requested (simple background/peak scale)
        if normalize and pick_bri:
            denom = max(pmax - bkg, 1e-12)
            pick_bri = np.clip((np.asarray(pick_bri) - bkg) / denom, 0.0, 1.0).tolist()

        # figure
        fig, (ax_img, ax_r, ax_b) = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        # left panel: image + all bright points + selected per-bin + center used
        ax_img.imshow(data, origin="lower", cmap="afmhot")
        ax_img.scatter(x, y, s=10, c="c")
        if pick_xy:
            sel = np.asarray(pick_xy)
            ax_img.scatter(sel[:, 0], sel[:, 1], s=30, facecolors="none", edgecolors="w")
        ax_img.plot([xs], [ys], "wx", ms=12, mew=2)
        ax_img.set_xticks([]); ax_img.set_yticks([])

        # middle: radius vs angle (scatter only)
        if pick_ang:
            ax_r.scatter(pick_ang, pick_rad, s=30)
        ax_r.set_xlim(0, 360)
        ax_r.set_xticks([0, 90, 180, 270, 360])
        ax_r.set_xlabel("Angle [deg]")
        ax_r.set_ylabel("Radius [px]")

        # right: brightness vs angle (normalized if normalize=True)
        if pick_ang:
            ax_b.scatter(pick_ang, pick_bri, s=30)
        ax_b.set_xlim(0, 360)
        ax_b.set_xticks([0, 90, 180, 270, 360])
        ax_b.set_xlabel("Angle [deg]")
        ax_b.set_ylabel("Brightness" + (" [norm]" if normalize else ""))

        return fig
