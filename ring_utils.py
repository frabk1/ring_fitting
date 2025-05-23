import numpy as np
import random


def rbp_find_bright_points(img, threshold, radius, pts=None, it=0, max_it=999, margin=0):
    if pts is None:
        pts = []
    if it > max_it:
        return pts
    ny, nx = img.shape
    sub = img[margin:ny-margin, margin:nx-margin]
    if sub.size == 0:
        return pts
    ysub, xsub = np.unravel_index(np.argmax(sub), sub.shape)
    val = sub[ysub, xsub]
    yB, xB = ysub + margin, xsub + margin
    if val <= threshold:
        return pts
    pts.append((yB, xB, val))
    r_int = int(np.ceil(radius))
    y0, y1 = max(0, yB - r_int), min(ny, yB + r_int + 1)
    x0, x1 = max(0, xB - r_int), min(nx, xB + r_int + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - xB)**2 + (yy - yB)**2 <= radius**2
    img[y0:y1, x0:x1][mask] = 0.0
    return rbp_find_bright_points(img, threshold, radius, pts, it+1, max_it, margin)




def polygon_for_ring(xs, ys, r_in, r_out, angs):
    angs = np.radians(angs)
    x_out, y_out = xs + r_out*np.cos(angs), ys + r_out*np.sin(angs)
    x_in,  y_in  = xs + r_in*np.cos(angs),  ys + r_in*np.sin(angs)
    return np.concatenate([x_out, x_in[::-1]]), np.concatenate([y_out, y_in[::-1]])

