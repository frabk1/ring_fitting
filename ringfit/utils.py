import numpy as np
import itertools

def geometric_centroid(points):
    return points[:,0].mean(), points[:,1].mean()

def flux_center(data):
    h, w = data.shape
    yy, xx = np.indices((h, w))
    total = data.sum()
    return (xx*data).sum()/total, (yy*data).sum()/total

def threshold_center(data, q=25):
    thresh = np.percentile(data, q)
    mask = data >= thresh
    yy, xx = np.indices(data.shape)
    total = mask.sum()
    return xx[mask].sum()/total, yy[mask].sum()/total

def estimate_center_via_triangles(points, trials=10):
    def circumcircle(pt1, pt2, pt3):
        x1,y1=pt1; x2,y2=pt2; x3,y3=pt3
        d = 2*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))
        if abs(d)<1e-6: return None
        ux = ((x1**2+y1**2)*(y2-y3)+(x2**2+y2**2)*(y3-y1)+(x3**2+y3**2)*(y1-y2))/d
        uy = ((x1**2+y1**2)*(x3-x2)+(x2**2+y2**2)*(x1-x3)+(x3**2+y3**2)*(x2-x1))/d
        return ux, uy
    centers = []
    for _ in range(trials):
        pts = np.array(list(itertools.islice(itertools.cycle(points), 3)))
        circ = circumcircle(*pts)
        if circ:
            centers.append(circ)
    return np.mean(centers, axis=0)
