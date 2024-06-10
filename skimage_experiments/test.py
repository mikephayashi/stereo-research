import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from skimage import color, data, restoration, filters
from scipy import signal

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

astro = io.imread(f"./data/input/Adirondack/im0.png", as_gray=True)
astro = filters.gaussian(astro)

psf = np.ones((5, 5)) / 25
deconvolved, _ = restoration.unsupervised_wiener(astro, gkern(5, 2))

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharex=True, sharey=True)

plt.gray()

ax[0].imshow(astro, vmin=deconvolved.min(), vmax=deconvolved.max())
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()