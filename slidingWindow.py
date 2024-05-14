from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view

image1 = io.imread("data/im0.png", as_gray=True)
image2 = io.imread("data/im1.png", as_gray=True)
height, width = image1.shape
max_disparity = 4
half_window = 4
window_length = 9
window_size = (window_length, window_length)
# cache = np.zeros(((width - 2 * half_window, height - 2 * half_window - max_disparity, max_disparity)))
cache = np.zeros(((width, height, max_disparity)))

'''
Einsum
'''
start = time.time()
image1_windows = sliding_window_view(image1, window_size)
elaped = time.time() - start
print(f"time: {elaped}")
image2_windows = sliding_window_view(image2, window_size)
elaped = time.time() - start
print(f"time: {elaped}")
c = np.einsum('ijkl,ijkl->ij', image1_windows, image2_windows)
elaped = time.time() - start
print(f"time: {elaped}")
import pdb; pdb.set_trace()
# disparity_values = np.argmax(c, axis=2)             
# io.imshow(disparity_values)
# plt.show()

'''
Reshapes + sum along axes
'''
# c = image1_windows.reshape(-1, window_length, window_length)
# d = image2_windows.reshape(-1, window_length, window_length)
# elaped = time.time() - start
# print(f"time: {elaped}")
# c = np.sum(image1_windows * image2_windows, axis = (2,3))
# elaped = time.time() - start
# print(f"time: {elaped}")
# d = (image1_windows * image2_windows).reshape((image1_windows.shape[0], image1_windows.shape[1], window_length * window_length))
# e = np.sum(d, axis = 2)
# elaped = time.time() - start
# print(f"time: {elaped}")