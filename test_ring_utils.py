if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from ring_utils import rbp_find_bright_points, polygon_for_ring
    img = np.array(Image.open("im1.jpg").convert("L"), dtype=float)
    pts = rbp_find_bright_points(img.copy(), threshold=5, radius=30, margin = 15)
    ys = np.array([p[0] for p in pts])
    xs = np.array([p[1] for p in pts])
    centre_x, centre_y = xs.mean(), ys.mean()
    angles = np.linspace(0, 360, num=32, endpoint=False)
    r_in   = np.full_like(angles, 80.0)
    r_out  = np.full_like(angles, 100.0)
    poly_x, poly_y = polygon_for_ring(centre_x, centre_y, r_in, r_out, angles)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", origin="upper")
    ax.scatter(xs, ys, c="cyan", marker="x", label="Bright points")
    ax.plot(centre_x, centre_y, "r+", ms=12, mew=2, label="Centroid")
    ax.fill(poly_x, poly_y, color="yellow", alpha=0.3, label="Ring polygon")
    ax.set_axis_off()
    ax.legend()
    plt.show()