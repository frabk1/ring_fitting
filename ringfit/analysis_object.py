import numpy as np
import matplotlib.pyplot as plt
from . import extraction as ex
from . import fitting as fit
from . import utils

class AnalysisObject:
    def __init__(self, im):
        self.image = im
        self.data = im.imarr
        self.npix = im.npix
        self.cell = im.psize
        self.ra = im.ra
        self.dec = im.dec
        self.peak = im.peak
        self.total_flux = im.total
        self.xarr = im.x
        self.yarr = im.y
        self.compute_centers()
        self.bright_points = None

    def compute_centers(self):
        self.geo_c = utils.geometric_centroid(self.data)
        self.flux_c = utils.flux_center(self.data)
        self.q25_c = utils.threshold_center(self.data, q=25)

    def find_bright_points(self, threshold=0.5, radius=5.0, margin=None, max_it=999):
        self.bright_points = ex.rbp_find_bright_points(self.image, threshold, radius, margin, max_it)
        return self.bright_points

    def plot_centers(self):
        fig, ax = plt.subplots(figsize=(6,6))
        extent = [self.image.x[0,0], self.image.x[0,-1], self.image.y[0,0], self.image.y[-1,0]]
        ax.imshow(self.data, origin='lower', cmap='afmhot', extent=extent)
        gx, gy = self.geo_c
        fx, fy = self.flux_c
        tx, ty = self.q25_c
        gx, gy = self.xarr[int(round(gy)), int(round(gx))], self.yarr[int(round(gy)), int(round(gx))]
        fx, fy = self.xarr[int(round(fy)), int(round(fx))], self.yarr[int(round(fy)), int(round(fx))]
        tx, ty = self.xarr[int(round(ty)), int(round(tx))], self.yarr[int(round(ty)), int(round(tx))]
        ax.plot(gx, gy, 'wo', label='Geometric')
        ax.plot(fx, fy, 'go', label='Flux Center')
        ax.plot(tx, ty, 'bo', label='Threshold Center')
        ax.legend()
        ax.set_title("Centers Overlaid on Image")
        ax.set_xlabel('x [μas]')
        ax.set_ylabel('y [μas]')
        return fig
