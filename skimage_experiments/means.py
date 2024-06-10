import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import rank
from skimage import io, util



image = io.imread(f"./data/input/Adirondack/im0.png", as_gray=True)
image = util.img_as_ubyte(image)
footprint = disk(20)

# import pdb; pdb.set_trace()

percentile_result = rank.mean_percentile(image, footprint=footprint, p0=0.1, p1=0.9)
bilateral_result = rank.mean_bilateral(image, footprint=footprint, s0=500, s1=500)
normal_result = rank.mean(image, footprint=footprint)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
ax = axes.ravel()

titles = ['Original', 'Percentile mean', 'Bilateral mean', 'Local mean']
imgs = [image, percentile_result, bilateral_result, normal_result]
for n in range(0, len(imgs)):
    ax[n].imshow(imgs[n], cmap=plt.cm.gray)
    ax[n].set_title(titles[n])
    ax[n].axis('off')

plt.tight_layout()
plt.show()