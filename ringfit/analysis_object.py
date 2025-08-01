import numpy as np
import matplotlib.pyplot as plt
from . import extraction as ex
from . import utils

class AnalysisObject:
    def __init__(self, im):
        self.image = im
        self.data = im.imarr()
        self.xdim = im.xdim
        self.ydim = im.ydim
        self.cell = im.psize
        self.ra = im.ra
        self.dec = im.dec
        self.peak = getattr(im, 'peak', None)
        self.total_flux = im.total_flux()
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
        extent = [0, self.xdim*self.cell, 0, self.ydim*self.cell]
        ax.imshow(self.data, origin='lower', cmap='afmhot', extent=extent)

        gx, gy = self.geo_c
        fx, fy = self.flux_c
        tx, ty = self.q25_c
        ax.plot(gx*self.cell, gy*self.cell, 'wo', label='Geometric')
        ax.plot(fx*self.cell, fy*self.cell, 'go', label='Flux Center')
        ax.plot(tx*self.cell, ty*self.cell, 'bo', label='Threshold Center')
        ax.legend()
        ax.set_title("Centers Overlaid")
        ax.set_xlabel('x [radian]')
        ax.set_ylabel('y [radian]')
        return fig
