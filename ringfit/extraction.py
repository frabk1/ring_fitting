import numpy as np

def _img_to_array(img):
    if hasattr(img, "imarr"):
        arr = img.imarr
    else:
        arr = np.array(img)
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image array, got shape {arr.shape}")
    return arr

def rbp_find_bright_points(img, threshold, radius, margin=None, max_it=999):
    data = _img_to_array(img)
    h, w = data.shape
    if margin is None:
        margin = int(np.ceil(radius + 1))
    mask = np.ones_like(data, dtype=bool)
    points = []
    for _ in range(max_it):
        masked_data = data * mask
        peak = masked_data.max()
        if peak < threshold:
            break
        y, x = np.unravel_index(np.argmax(masked_data), data.shape)
        if x < margin or x >= (w - margin) or y < margin or y >= (h - margin):
            mask[y, x] = False
            continue
        points.append((x, y))
        y0, y1 = max(0, y - int(radius)), min(h, y + int(radius) + 1)
        x0, x1 = max(0, x - int(radius)), min(w, x + int(radius) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - x)**2 + (yy - y)**2)
        mask[y0:y1, x0:x1][dist <= radius] = False
    return np.array(points)
