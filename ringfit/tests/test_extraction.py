# ringfit/tests/test_extraction.py
import numpy as np
import pytest

from ringfit.extraction import _img_to_array, rbp_find_bright_points


class MockImg:
    def __init__(self, arr):
        self._arr = arr
    def imarr(self):
        return self._arr


@pytest.mark.parametrize("shape", [(32, 48), (1, 32, 48), (32, 48, 1)])
def test_img_to_array_squeezes_to_2d(shape):
    arr = np.zeros(shape, dtype=float)
    img = MockImg(arr)
    out = _img_to_array(img)
    assert out.ndim == 2
    assert out.shape == (32, 48)


def test_img_to_array_raises_for_non_2d_after_squeeze():
    arr = np.zeros((32, 48, 3), dtype=float)
    img = MockImg(arr)
    with pytest.raises(ValueError):
        _img_to_array(img)


def _make_image(h, w, peaks):
    """
    peaks: iterable of (x, y, value); x is column, y is row.
    """
    img = np.zeros((h, w), dtype=float)
    for x, y, v in peaks:
        img[int(y), int(x)] = float(v)
    return img


def test_rbp_two_interior_peaks_found():
    h, w = 64, 64
    radius = 3.0
    threshold = 5.0
    peaks = [(10, 12, 9.0), (40, 40, 8.0)]
    img = MockImg(_make_image(h, w, peaks))
    pts = rbp_find_bright_points(img, threshold=threshold, radius=radius)
    pts_set = {tuple(map(int, p)) for p in pts}
    assert (10, 12) in pts_set
    assert (40, 40) in pts_set
    assert len(pts_set) == 2


def test_rbp_skips_edge_peaks_by_default_margin():
    h, w = 64, 64
    radius = 3.0  # default margin = ceil(radius+1) = 4
    threshold = 5.0
    peaks = [
        (1, 1, 20.0),     # near corner, should be skipped
        (30, 30, 10.0),   # interior, should be kept
    ]
    img = MockImg(_make_image(h, w, peaks))
    pts = rbp_find_bright_points(img, threshold=threshold, radius=radius)
    pts_set = {tuple(map(int, p)) for p in pts}
    assert (30, 30) in pts_set
    assert (1, 1) not in pts_set
    assert len(pts_set) == 1


def test_rbp_custom_margin_includes_near_edge():
    h, w = 64, 64
    radius = 3.0
    threshold = 5.0
    # With default margin=4 this would be skipped; margin=1 should allow it.
    peaks = [(2, 5, 12.0)]
    img = MockImg(_make_image(h, w, peaks))
    pts = rbp_find_bright_points(img, threshold=threshold, radius=radius, margin=1)
    pts_set = {tuple(map(int, p)) for p in pts}
    assert (2, 5) in pts_set


def test_rbp_radius_blanking_keeps_only_strongest_when_close():
    h, w = 64, 64
    radius = 5.0
    threshold = 1.0
    # Two peaks within 'radius' — only the strongest should remain.
    peaks = [(20, 20, 10.0), (22, 20, 8.0)]
    img = MockImg(_make_image(h, w, peaks))
    pts = rbp_find_bright_points(img, threshold=threshold, radius=radius)
    pts_set = {tuple(map(int, p)) for p in pts}
    assert (20, 20) in pts_set
    assert (22, 20) not in pts_set
    assert len(pts_set) == 1


def test_rbp_respects_threshold():
    h, w = 32, 32
    radius = 3.0
    threshold = 50.0
    peaks = [(10, 10, 9.0), (20, 21, 12.0)]
    img = MockImg(_make_image(h, w, peaks))
    pts = rbp_find_bright_points(img, threshold=threshold, radius=radius)
    assert pts.size == 0


def test_rbp_max_it_caps_number_of_iterations():
    h, w = 80, 80
    radius = 3.0
    threshold = 1.0
    # Place many interior peaks spaced apart > radius.
    peaks = []
    for y in range(10, 70, 10):
        for x in range(10, 70, 10):
            peaks.append((x, y, 10.0 + x + y))
    img = MockImg(_make_image(h, w, peaks))
    pts = rbp_find_bright_points(img, threshold=threshold, radius=radius, max_it=3)
    assert pts.shape[0] <= 3
