import os, numpy as np
from PIL import Image
import extraction as ex         
import fitting   as ft           
IMG = "im1.jpg"
arr = np.array(Image.open(IMG).convert("L"), dtype=np.float32)
arr /= arr.max()

pts = ex.rbp_find_bright_points(arr.copy(), threshold=0.02, radius=30, margin=10)
if not pts:
    raise RuntimeError("no bright points found")

xs, ys = ex.find_true_center(pts)
P = np.array([[p[1], p[0]] for p in pts])   # (x,y) array

R        = ft.fit_circle(P, xs, ys)
ew, eh   = ft.fit_ellipse(P, xs, ys)
lima     = ft.fit_limacon(P, xs, ys)        # (xs, ys, L1, L2, φ)

print(f"points: {len(pts)}")
print(f"RBP center : ({xs:.1f}, {ys:.1f})")
print(f"circle rad : {R:.2f}")
print(f"ellipse w,h: ({ew:.2f}, {eh:.2f})")
print(f"lima  L1={lima[2]:.2f}  L2={lima[3]:.2f}  φ={lima[4]:.2f}")