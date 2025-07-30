import numpy as np
from scipy.optimize import minimize

def fit_circle(points, xs, ys):
    r = np.sqrt((points[:,0] - xs)**2 + (points[:,1] - ys)**2)
    return r.mean()

def fit_ellipse(points, xs, ys):
    x, y = points[:,0] - xs, points[:,1] - ys
    def cost(ab):
        a, b = ab
        if a <= 0 or b <= 0: return 1e12
        return np.sum((x/a)**2 + (y/b)**2 - 1)**2
    a0 = np.std(x)
    b0 = np.std(y)
    res = minimize(cost, [a0, b0], method='Powell')
    a, b = res.x if res.success else (a0, b0)
    return 2*a, 2*b

def fit_limacon(points, xs, ys):
    dx, dy = points[:,0] - xs, points[:,1] - ys
    r_obs = np.sqrt(dx*dx + dy*dy)
    th_obs = np.arctan2(dy, dx)
    def cost(params):
        c, L2, phi = params
        if c <= 0 or abs(L2) >= 1: return 1e12
        r_pred = c*(1 + L2*np.cos(th_obs - phi))
        return np.sum((r_obs - r_pred)**2)
    c0 = r_obs.mean()
    L20 = 0.1
    phi0 = 0.0
    res = minimize(cost, [c0, L20, phi0], method='Powell')
    return res.x if res.success else (c0, L20, phi0)
