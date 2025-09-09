# tests/test_fitting.py
import numpy as np
import pytest

from ringfit.fitting import (
    fit_circle,
    fit_ellipse,
    fit_limacon,
    general_fit,
)

def _angle_wrap_pi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi

def _angle_diff(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def _gen_circle_points(r, xs, ys, n=360, noise=0.0, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr = r + rng.normal(0.0, noise, size=n)
    x = xs + rr * np.cos(th)
    y = ys + rr * np.sin(th)
    return np.column_stack([x, y])

def _gen_ellipse_points(a, b, xs, ys, n=360, noise=0.0, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = xs + a * np.cos(th) + rng.normal(0.0, noise, size=n)
    y = ys + b * np.sin(th) + rng.normal(0.0, noise, size=n)
    return np.column_stack([x, y])

def _gen_limacon_points(c, L2, phi, xs, ys, n=720, noise=0.0, rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = c * (1 + L2 * np.cos(th - phi))
    r_noisy = r + rng.normal(0.0, noise, size=n)
    x = xs + r_noisy * np.cos(th)
    y = ys + r_noisy * np.sin(th)
    return np.column_stack([x, y])

@pytest.fixture(scope="module")
def rng_seed():
    return 12345

def test_fit_circle_basic(rng_seed):
    xs, ys, r_true = 2.0, -1.0, 5.0
    pts = _gen_circle_points(r_true, xs, ys, n=720, noise=0.05, rng=rng_seed)
    r_est = fit_circle(pts, xs, ys)
    assert np.isfinite(r_est)
    assert abs(r_est - r_true) < 0.1

def test_fit_ellipse_basic(rng_seed):
    xs, ys, a_true, b_true = -0.5, 0.75, 4.0, 2.5
    pts = _gen_ellipse_points(a_true, b_true, xs, ys, n=720, noise=0.03, rng=rng_seed)
    d2a_est, d2b_est = fit_ellipse(pts, xs, ys)
    assert np.isfinite(d2a_est) and np.isfinite(d2b_est)
    est_sorted = np.sort([d2a_est, d2b_est])
    true_sorted = np.sort([2 * a_true, 2 * b_true])
    np.testing.assert_allclose(est_sorted, true_sorted, rtol=0.08, atol=0.1)

def test_fit_limacon_basic(rng_seed):
    xs, ys = 0.0, 0.0
    c_true, L2_true, phi_true = 8.0, 0.22, 0.7
    pts = _gen_limacon_points(c_true, L2_true, phi_true, xs, ys, n=720, noise=0.03, rng=rng_seed)
    c_est, L2_est, phi_est = fit_limacon(pts, xs, ys)
    phi_est = _angle_wrap_pi(phi_est)
    phi_wrapped_true = _angle_wrap_pi(phi_true)
    assert np.isfinite(c_est) and np.isfinite(L2_est) and np.isfinite(phi_est)
    assert abs(c_est - c_true) < 0.2
    assert abs(L2_est - L2_true) < 0.03
    assert abs(_angle_diff(phi_est, phi_wrapped_true)) < 0.05

def test_general_fit_circle_desired(rng_seed):
    xs, ys, r_true = 3.0, -2.0, 6.0
    pts = _gen_circle_points(r_true, xs, ys, n=720, noise=0.04, rng=rng_seed)
    res = general_fit(pts, xs, ys, shape="circle", return_desired=True)
    assert res["shape"] == "circle"
    assert res["success"] is True
    assert res["cost"] >= 0.0
    r_est = res["params"][0]
    assert abs(r_est - r_true) < 0.15
    assert "desired" in res
    assert "radius" in res["desired"]
    np.testing.assert_allclose(res["desired"]["radius"], r_est, rtol=1e-12, atol=1e-12)

def test_general_fit_ellipse_method_and_bounds(rng_seed):
    xs, ys, a_true, b_true = 1.0, 1.5, 3.5, 2.0
    pts = _gen_ellipse_points(a_true, b_true, xs, ys, n=600, noise=0.02, rng=rng_seed)
    bounds = [(0.5 * a_true, 1.5 * a_true), (0.5 * b_true, 1.5 * b_true)]
    res = general_fit(pts, xs, ys, shape="ellipse", method="L-BFGS-B", bounds=bounds, return_desired=True)
    assert res["shape"] == "ellipse"
    assert res["success"] is True
    d2a_est, d2b_est = res["params"]
    est_sorted = np.sort([d2a_est, d2b_est])
    true_sorted = np.sort([2 * a_true, 2 * b_true])
    np.testing.assert_allclose(est_sorted, true_sorted, rtol=0.08, atol=0.1)
    assert "desired" in res and "area" in res["desired"]

def test_general_fit_limacon_phi_wrapped_and_desired(rng_seed):
    xs, ys = -1.0, 0.5
    c_true, L2_true, phi_true = 10.0, 0.18, 3.2
    pts = _gen_limacon_points(c_true, L2_true, phi_true, xs, ys, n=900, noise=0.04, rng=rng_seed)
    res = general_fit(pts, xs, ys, shape="limacon", method="Powell", return_desired=True)
    assert res["shape"] == "limacon"
    assert res["success"] is True
    c_est, L2_est, phi_est = res["params"]
    phi_wrapped_true = _angle_wrap_pi(phi_true)
    assert abs(c_est - c_true) < 0.25
    assert abs(L2_est - L2_true) < 0.035
    assert abs(_angle_diff(phi_est, phi_wrapped_true)) < 0.06
    assert "desired" in res and "c" in res["desired"] and "phi" in res["desired"]

def test_general_fit_invalid_points_shape_raises():
    with pytest.raises(ValueError):
        general_fit(np.array([1.0, 2.0, 3.0]), 0.0, 0.0, shape="circle")

def test_general_fit_unknown_shape_raises():
    pts = _gen_circle_points(3.0, 0.0, 0.0, n=100, noise=0.0, rng=0)
    with pytest.raises(ValueError):
        general_fit(pts, 0.0, 0.0, shape="square")

def test_general_fit_custom_desired(rng_seed):
    xs, ys, r_true = 0.0, 0.0, 4.0
    pts = _gen_circle_points(r_true, xs, ys, n=360, noise=0.03, rng=rng_seed)
    custom = {
        "diameter_check": lambda p, xs, ys: 2.0 * p[0],
        "center_str": lambda p, xs, ys: f"({xs:.1f},{ys:.1f})",
    }
    res = general_fit(pts, xs, ys, shape="circle", return_desired=True, desired=custom)
    assert "desired" in res
    assert np.isclose(res["desired"]["diameter_check"], 2 * res["params"][0], rtol=1e-12, atol=1e-12)
    assert res["desired"]["center_str"] == "(0.0,0.0)"
