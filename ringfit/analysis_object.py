import numpy as np
import matplotlib.pyplot as plt

from .extraction import rbp_find_bright_points
from .fitting import fit_circle, fit_ellipse, fit_limacon
from .utils import (
    estimate_center_via_triangles,
    geometric_centroid,
    flux_center,
    thresholded_flux_center,
    radial_fwhm_profile
)

class AnalysisObject:
    def __init__(self, image):
        self.im = image
        self.data = np.array(image)

    def run(self, thresh=0.5, rad=5.0, plot=True):
        # 1) find bright points
        pts = rbp_find_bright_points(self.im, thresh, rad)
        # 2) several center estimates
        geo_c = geometric_centroid(pts)
        flux_c = flux_center(self.im)
        th_c = thresholded_flux_center(self.im)
        tri_c = estimate_center_via_triangles(pts)
        # 3) fits
        r_circ = fit_circle(pts, *tri_c)
        e_w, e_h = fit_ellipse(pts, *tri_c)
        lc_c, lc_L2, lc_phi = fit_limacon(pts, *tri_c)
        # 4) width profile
        angs = np.arctan2(pts[:,1]-tri_c[1], pts[:,0]-tri_c[0])
        widths = radial_fwhm_profile(self.im, tri_c, angs)

        if plot:
            fig, axs = plt.subplots(2,2, figsize=(8,8))
            ax = axs[0,0]
            ax.imshow(self.data, origin='lower', cmap='gray')
            ax.scatter(pts[:,0], pts[:,1], s=20, c='r')
            ax.set_title("Bright points")

            ax = axs[0,1]
            ax.bar(['circ','ellipse h','ellipse w','limacon L2'], [r_circ, e_h/2, e_w/2, lc_L2])
            ax.set_title("Shape params")

            ax = axs[1,0]
            centers = np.vstack([geo_c, flux_c, th_c, tri_c])
            labels = ['geo','flux','thresh','tri']
            ax.scatter(centers[:,0], centers[:,1], c='k')
            for lab,(x,y) in zip(labels, centers):
                ax.text(x,y,lab)
            ax.set_title("Center estimates")

            ax = axs[1,1]
            ax.plot(np.sort(angs), np.sort(widths), '.-')
            ax.set_title("Radial FWHM vs angle")

            plt.tight_layout()
            plt.show()

        return {
            'points': pts,
            'centers': dict(geo=geo_c, flux=flux_c, thresh=th_c, tri=tri_c),
            'circle_radius': r_circ,
            'ellipse_dims': (e_w, e_h),
            'limacon': (lc_c, lc_L2, lc_phi),
            'widths': widths,
        }
