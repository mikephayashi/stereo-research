from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from justpfm import justpfm

def print_time(start):
    elaped = time.time() - start
    print(f"time: {elaped}")
    
def shiftOne(image1, image1_windows):
    image1 = np.delete(image1, 0, 1)
    image1 = np.append(image1, np.zeros((image1.shape[0], 1)), axis=1)
    image1_windows = sliding_window_view(image1, window_size)
    return image1, image1_windows

def getWindowDotProduct(image1_windows, image2_windows, cache, disparity):
    windowDotProduct = np.einsum('ijkl,ijkl->ij', image1_windows, image2_windows)
    cache[:,:,disparity] = windowDotProduct
    return windowDotProduct

start = time.time()
image1 = io.imread("data/im0.png", as_gray=True)
image2 = io.imread("data/im1.png", as_gray=True)
height, width = image1.shape
max_disparity = 9
half_window = 4
window_length = 9
window_size = (window_length, window_length)
# cache = np.zeros(((width - 2 * half_window, height - 2 * half_window - max_disparity, max_disparity)))
cache = np.zeros(((height - max_disparity + 1, width - max_disparity + 1, max_disparity)))
print_time(start)

'''
Einsum
'''
image1_windows = sliding_window_view(image1, window_size)
print_time(start)
image2_windows = sliding_window_view(image2, window_size)
print_time(start)
res = getWindowDotProduct(image1_windows, image2_windows, cache, 0)
for disparity in range(max_disparity):
    if disparity == 0:
        continue
    image1, image1_windows = shiftOne(image1, image1_windows)
    image2, image2_windows = shiftOne(image2, image2_windows)
    res = getWindowDotProduct(image1_windows, image2_windows, cache, disparity)
    print_time(start)
disparity_values = np.argmax(cache, axis=2) 
disparity_values = disparity_values / max_disparity
disparity_float32 = np.float32(disparity_values)      
justpfm.write_pfm(file_name="test.pfm", data=disparity_values)
io.imshow(disparity_values)
plt.show()
# image1_windows = np.delete(image1_windows, 0, 1)
# elaped = time.time() - start
# print(f"time: {elaped}")
# image1_windows = np.append(image1_windows, np.zeros((image1_windows.shape[0], 1, window_length, window_length)), axis=1)
# image1_windows = np.concatenate((image1_windows, np.zeros((image1_windows.shape[0], 1, window_length, window_length))), axis=1)
# elaped = time.time() - start
# print(f"time: {elaped}")
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