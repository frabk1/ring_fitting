import numpy as np
import itertools
from scipy.interpolate import interp1d

def geometric_centroid(points):
    """Mean of (x,y) points."""
    return points[:,0].mean(), points[:,1].mean()

def flux_center(img):
    """Center of mass weighted by pixel intensity over entire image."""
    data = np.array(img, float)
    h, w = data.shape
    yy, xx = np.indices((h, w))
    total = data.sum()
    return (xx*data).sum()/total, (yy*data).sum()/total

def thresholded_flux_center(img, percentile=25):
    """
    Threshold at given percentile, binarize, then COM of mask.
    """
    data = np.array(img, float)
    thresh = np.percentile(data, percentile)
    mask = data >= thresh
    yy, xx = np.indices(data.shape)
    total = mask.sum()
    return xx[mask].sum()/total, yy[mask].sum()/total

def estimate_center_via_triangles(points, trials=10):
    """
    Pick random triplets of points, fit circle through each,
    then average centers.
    """
    def circumcircle(pt1, pt2, pt3):
        x1,y1=pt1; x2,y2=pt2; x3,y3=pt3
        d = 2*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
        if abs(d)<1e-6: return None
        ux = ((x1**2+y1**2)*(y2-y3)+(x2**2+y2**2)*(y3-y1)+(x3**2+y3**2)*(y1-y2))/d
        uy = ((x1**2+y1**2)*(x3-x2)+(x2**2+y2**2)*(x1-x3)+(x3**2+y3**2)*(x2-x1))/d
        return ux, uy

    centers = []
    pts = points.tolist()
    for tri in itertools.islice(itertools.permutations(pts,3), 0, trials):
        c = circumcircle(*tri)
        if c is not None:
            centers.append(c)
    xs, ys = zip(*centers)
    return np.mean(xs), np.mean(ys)

def radial_fwhm_profile(img, center, angs, num=500):
    """
    For each angle in `angs`, sample along the ray from center,
    compute FWHM of intensity profile, return widths array.
    """
    data = np.array(img, float)
    h, w = data.shape
    yc, xc = center[1], center[0]
    max_r = np.hypot(max(xc, w-xc), max(yc, h-yc))
    widths = []
    for theta in angs:
        xs = xc + np.cos(theta)*np.linspace(-max_r, max_r, num)
        ys = yc + np.sin(theta)*np.linspace(-max_r, max_r, num)
        # clip
        valid = (
            (xs>=0)&(xs<w)&
            (ys>=0)&(ys<h)
        )
        xs2, ys2 = xs[valid], ys[valid]
        vals = interp1d(np.arange(len(xs2)), data[ys2.astype(int), xs2.astype(int)], kind='linear')(np.arange(len(xs2)))
        half = (vals.max()+vals.min())/2
        above = np.where(vals>=half)[0]
        if len(above)>1:
            width = (above.max()-above.min())*(2*max_r/num)
        else:
            width = 0.0
        widths.append(width)
    return np.array(widths)
