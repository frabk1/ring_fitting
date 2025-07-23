import numpy as np, matplotlib.pyplot as plt, extraction as ex

def plot_profiles(im='im1.jpg', bp_th=0.005, bp_rad=30, margin=15, polar=False):
    ang, rad, flux, width = ex.get_quantities(
    im, ('angle', 'radius', 'brightness', 'width'),
    bp_th=0.005, bp_rad=30, margin=15      # keywords now match extraction.py
)

    ang = (ang+360)%360; s = np.argsort(ang)
    ang, rad, flux, width = ang[s], rad[s], flux[s], width[s]
    wmask = ~np.isnan(width)
    rows = 4 if polar else 3
    fig, ax = plt.subplots(rows, 1, figsize=(7,2.6*rows), sharex=not polar)
    ax[0].plot(ang, flux);             ax[0].scatter(ang, flux, s=12);          ax[0].set_ylabel('Flux');   ax[0].grid(True)
    ax[1].plot(ang, rad);              ax[1].scatter(ang, rad, s=12);           ax[1].set_ylabel('Radius'); ax[1].grid(True)
    ax[2].plot(ang[wmask], width[wmask]); ax[2].scatter(ang[wmask], width[wmask], s=12)
    ax[2].set_ylabel('Width'); ax[2].set_xlabel('Angle'); ax[2].grid(True)
    if polar:
        p = fig.add_subplot(rows,1,rows,projection='polar')
        p.plot(np.deg2rad(ang), flux); p.scatter(np.deg2rad(ang), flux, s=12)
        p.set_theta_zero_location('E'); p.set_theta_direction(-1); p.grid(True)
    fig.tight_layout(); plt.show()

if __name__=='__main__':
    plot_profiles()
