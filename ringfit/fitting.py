import numpy as np
from scipy.optimize import minimize

def fit_circle(points, xs, ys):
    """Quick circle fit: mean radius from center."""
    r = np.sqrt((points[:, 0] - xs) ** 2 + (points[:, 1] - ys) ** 2)
    return r.mean()

def fit_ellipse(points, xs, ys):
    """Axis-only ellipse fit around (xs, ys)."""
    x, y = points[:, 0] - xs, points[:, 1] - ys
    def cost(ab):
        a, b = ab
        if a <= 0 or b <= 0:
            return 1e12
        return np.sum(((x / a) ** 2 + (y / b) ** 2 - 1) ** 2)
    a0 = np.std(x)
    b0 = np.std(y)
    res = minimize(cost, [a0, b0], method="Powell")
    a, b = res.x if res.success else (a0, b0)
    return 2 * a, 2 * b

def fit_limacon(points, xs, ys):
    """Limacon fit: r = c*(1 + L2*cos(theta - phi))."""
    dx, dy = points[:, 0] - xs, points[:, 1] - ys
    r_obs = np.sqrt(dx * dx + dy * dy)
    th_obs = np.arctan2(dy, dx)
    def cost(params):
        c, L2, phi = params
        if c <= 0 or abs(L2) >= 1:
            return 1e12
        r_pred = c * (1 + L2 * np.cos(th_obs - phi))
        return np.sum((r_obs - r_pred) ** 2)
    c0 = r_obs.mean()
    L20 = 0.1
    phi0 = 0.0
    res = minimize(cost, [c0, L20, phi0], method="Powell")
    c, L2, phi = (res.x if res.success else (c0, L20, phi0))
    if L2 < 0:
        L2 = -L2
        phi = _wrap_angle_pi(phi + np.pi)
    return float(c), float(L2), float(phi)

def _wrap_angle_pi(phi):
    """Wrap angle to [-pi, pi]."""
    return (phi + np.pi) % (2 * np.pi) - np.pi

def _default_desired(shape, params, xs, ys):
    """Tiny helper: standard fields for outputs."""
    out = {"center": (float(xs), float(ys))}
    if shape == "circle":
        (r,) = params
        out.update({
            "radius": float(r),
            "diameter": float(2 * r),
            "area": float(np.pi * r * r),
            "circumference": float(2 * np.pi * r),
        })
    elif shape == "ellipse":
        d2a, d2b = params
        a = 0.5 * float(d2a)
        b = 0.5 * float(d2b)
        h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) != 0 else 0.0
        perim = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h))) if (a + b) > 0 else 0.0
        ecc = np.sqrt(1 - (min(a, b) / max(a, b)) ** 2) if max(a, b) > 0 else 0.0
        out.update({
            "a": a,
            "b": b,
            "diameters": (float(d2a), float(d2b)),
            "area": float(np.pi * a * b),
            "perimeter_approx": float(perim),
            "eccentricity_approx": float(ecc),
        })
    elif shape == "limacon":
        c, L2, phi = params
        out.update({
            "c": float(c),
            "L2": float(L2),
            "phi": float(_wrap_angle_pi(phi)),
            "r_max": float(c * (1 + abs(L2))),
            "r_min": float(c * (1 - abs(L2))),
        })
    return out

def general_fit(
    points,
    xs,
    ys,
    *,
    shape="circle",
    method="Powell",
    options=None,
    bounds=None,
    return_desired=False,
    desired=None,
):
    """Unified fitter for circle/ellipse/limacon around (xs, ys)."""
    shape = str(shape).lower()
    points = np.asarray(points, dtype=float)
    xs = float(xs)
    ys = float(ys)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array.")

    if shape == "circle":
        r = fit_circle(points, xs, ys)
        params = (float(r),)

    elif shape == "ellipse":
        x, y = points[:, 0] - xs, points[:, 1] - ys
        def ell_cost(ab):
            a, b = ab
            if a <= 0 or b <= 0:
                return 1e12
            return np.sum(((x / a) ** 2 + (y / b) ** 2 - 1) ** 2)
        a0 = np.std(x)
        b0 = np.std(y)
        x0 = np.array([a0, b0], dtype=float)
        res = minimize(ell_cost, x0, method=method, bounds=bounds, options=options if options is not None else None)
        if res.success:
            a, b = res.x
        else:
            a, b = a0, b0
        params = (float(2 * a), float(2 * b))

    elif shape == "limacon":
        dx, dy = points[:, 0] - xs, points[:, 1] - ys
        r_obs = np.sqrt(dx * dx + dy * dy)
        th_obs = np.arctan2(dy, dx)
        def lim_cost(params_vec):
            c, L2, phi = params_vec
            if c <= 0 or abs(L2) >= 1:
                return 1e12
            r_pred = c * (1 + L2 * np.cos(th_obs - phi))
            return np.sum((r_obs - r_pred) ** 2)
        c0 = float(np.mean(r_obs))
        L20 = 0.1
        phi0 = 0.0
        x0 = np.array([c0, L20, phi0], dtype=float)
        default_bounds = [(1e-12, None), (-0.999, 0.999), (-2 * np.pi, 2 * np.pi)]
        use_bounds = bounds if bounds is not None else default_bounds
        res = minimize(lim_cost, x0, method=method, bounds=use_bounds, options=options if options is not None else None)
        if res.success:
            c, L2, phi = res.x
        else:
            c, L2, phi = x0
        if L2 < 0:
            L2 = -L2
            phi = _wrap_angle_pi(phi + np.pi)
        params = (float(c), float(L2), float(phi))

    else:
        raise ValueError("shape must be one of: 'circle', 'ellipse', 'limacon'.")

    return {"shape": shape, "params": params, "center": (xs, ys)}
