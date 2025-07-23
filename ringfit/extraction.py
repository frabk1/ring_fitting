import numpy as np

def rbp_find_bright_points(img, threshold, radius, max_it=999):
    """
    Finds bright points above `threshold`, zeroing out a circular region
    of `radius` around each found peak. Returns an (N,2) array of (x, y).
    """
    data = np.array(img)
    h, w = data.shape
    mask = np.ones_like(data, bool)
    pts = []

    for _ in range(max_it):
        m = data * mask
        y, x = np.unravel_index(np.argmax(m), m.shape)
        if m[y, x] < threshold:
            break
        pts.append((x, y))
        yy, xx = np.ogrid[:h, :w]
        mask[(yy - y)**2 + (xx - x)**2 <= radius**2] = False

    return np.array(pts)


def polygon_for_ring(xs, ys, r_in, r_out, angs):
    """
    Build a closed polygon following two arcs (inner & outer).
    - xs, ys: center
    - r_in, r_out: inner/outer radii
    - angs: 1D array of angles in radians
    Returns an (2*len(angs), 2) array of vertices.
    """
    xi = xs + r_in * np.cos(angs)
    yi = ys + r_in * np.sin(angs)
    xo = xs + r_out * np.cos(angs[::-1])
    yo = ys + r_out * np.sin(angs[::-1])
    x = np.concatenate([xi, xo])
    y = np.concatenate([yi, yo])
    return np.vstack([x, y]).T
