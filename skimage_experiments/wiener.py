import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from skimage import color, data, restoration

rng = np.random.default_rng()

astro = io.imread(f"./data/input/Adirondack/im0.png", as_gray=True)
from scipy.signal import convolve2d as conv2

psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')
astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

deconvolved, _ = restoration.unsupervised_wiener(astro, psf)

import pdb; pdb.set_trace()

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