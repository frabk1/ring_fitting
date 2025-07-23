import numpy as np
import matplotlib.pyplot as plt
from extraction import (rbp_find_bright_points, find_true_center,sample_along_ray, measure_thickness_fwhm)

def angular_profile(img, threshold, radius, rmin, rmax, nr=500, num_samples=10, show=False):
    """
    From a grayscale image array, extract at each angle:
      - angle (deg)
      - radius of peak intensity
      - FWHM thickness
      - brightness (peak)
    If show=True, plots results.
    """
    pts = rbp_find_bright_points(img.copy(), threshold, radius, margin=15)
    xs, ys = find_true_center(pts, num_samples)
    angs, widths, r_in, r_out, r_peak = measure_thickness_fwhm(img, xs, ys, pts, rmin, rmax, nr)
    # Brightness at each peak radius
    brightness = []
    for ang_rad, r in zip(np.radians(angs), r_peak):
        val = sample_along_ray(img, xs, ys, ang_rad, np.array([r]))[0]
        brightness.append(val)
    brightness = np.array(brightness)

    if show:
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        # 1) Image + points + ring center
        ax[0,0].imshow(img, cmap='gray', origin='upper')
        ax[0,0].scatter([p[1] for p in pts], [p[0] for p in pts], c='cyan', marker='x')
        ax[0,0].plot(xs, ys, 'r+', ms=12)
        ax[0,0].set_title('Bright Points & Center')
        ax[0,0].axis('off')
        # 2) FWHM vs angle
        ax[0,1].plot(angs, widths, '.-')
        ax[0,1].set_title('FWHM vs Angle')
        ax[0,1].set_xlabel('Angle (deg)'); ax[0,1].set_ylabel('FWHM')
        # 3) Radius vs angle
        ax[1,0].plot(angs, r_peak, '.-')
        ax[1,0].set_title('Radius vs Angle')
        ax[1,0].set_xlabel('Angle (deg)'); ax[1,0].set_ylabel('Radius')
        # 4) Brightness vs angle
        ax[1,1].plot(angs, brightness, '.-')
        ax[1,1].set_title('Brightness vs Angle')
        ax[1,1].set_xlabel('Angle (deg)'); ax[1,1].set_ylabel('Brightness')
        plt.tight_layout()
        plt.show()

    return np.array(angs), np.array(r_peak), np.array(widths), brightness

