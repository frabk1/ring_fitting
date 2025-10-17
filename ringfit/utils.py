import numpy as np

def geometric_centroid(data):
    """Geometric centroid: average weighted by pixel values."""
    yy, xx = np.indices(data.shape)
    tot = data.sum()
    return (xx * data).sum() / tot, (yy * data).sum() / tot

def threshold_center(data, q=25):
    """Center of pixels above a percentile threshold."""
    thresh = np.percentile(data, q)
    mask = data >= thresh
    yy, xx = np.indices(data.shape)
    tot = mask.sum()
    return xx[mask].sum() / tot, yy[mask].sum() / tot
