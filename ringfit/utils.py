import numpy as np

def geometric_centroid(data):
    yy, xx = np.indices(data.shape)
    tot = data.sum()
    return (xx*data).sum()/tot, (yy*data).sum()/tot

def flux_center(data):
    yy, xx = np.indices(data.shape)
    tot = data.sum()
    return (xx*data).sum()/tot, (yy*data).sum()/tot

def threshold_center(data, q=25):
    thresh = np.percentile(data, q)
    mask = data >= thresh
    yy, xx = np.indices(data.shape)
    tot = mask.sum()
    return xx[mask].sum()/tot, yy[mask].sum()/tot
