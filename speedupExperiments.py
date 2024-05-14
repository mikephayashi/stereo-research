from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import time

image1 = io.imread("data/im0.png", as_gray=True)
image2 = io.imread("data/im1.png", as_gray=True)
height, width = image1.shape
max_disparity = 4
half_window = 4
# cache = np.zeros(((width, height, max_disparity)))

'''
Triple nested for loop
'''
# start_time = time.time()
# for i in range(half_window, 200):
#     print(i)
#     for j in range(half_window + max_disparity, height - half_window):
#         for d in range(max_disparity):
#             try:
#                 cache[i][j][d] = np.max(np.dot(image1[i - half_window: i + half_window, j - half_window: j + half_window], 
#                                         image2[i - half_window: i + half_window, j - d - half_window: j - d + half_window]))
#             except:
#                 import pdb; pdb.set_trace()        
# elapsed_time = time.time() - start_time
# print(f"{elapsed_time}")

'''
Numpy iterator
'''
# cache = np.zeros(((200, height, max_disparity)))
# it = np.nditer(cache, flags=['multi_index'])
# start_time = time.time()
# for x in it:
#     i = it.multi_index[0] + half_window
#     j = it.multi_index[1] + half_window + max_disparity
#     d = it.multi_index[2]
#     try:
#         cache[i][j][d] = np.max(np.dot(image1[i - half_window: i + half_window, j - half_window: j + half_window], 
#                                 image2[i - half_window: i + half_window, j - d - half_window: j - d + half_window]))
#     except:
#         # import pdb; pdb.set_trace()        
#         pass
# elapsed_time = time.time() - start_time
# print(f"{elapsed_time}")

'''
Cython
'''
# import cython
# import pyximport; pyximport.install()
# @cython.boundscheck(False)
# def cythonized():
#     cache = np.zeros(((200, height, max_disparity)))
#     # it = np.nditer(cache, flags=['multi_index'])
#     it = np.nditer(cache, flags=['reduce_ok', 'external_loop', 'buffered', 'delay_bufalloc', 'multi_index'],
#                 # op_flags=[['readonly'], ['readwrite', 'allocate']],
#                 # op_axes=[None, axeslist],
#                 # op_dtypes=['float64', 'float64'])
#     )
#     it.reset()
#     start_time = time.time()
#     for x in it:
#         i = it.multi_index[0] + half_window
#         j = it.multi_index[1] + half_window + max_disparity
#         d = it.multi_index[2]
#         try:
#             cache[i][j][d] = np.max(np.dot(image1[i - half_window: i + half_window, j - half_window: j + half_window], 
#                                     image2[i - half_window: i + half_window, j - d - half_window: j - d + half_window]))
#         except:
#             # import pdb; pdb.set_trace()        
#             pass
#     elapsed_time = time.time() - start_time
#     print(f"{elapsed_time}")
# cythonized()
                
# disparity_values = np.argmin(cache, axis=2)             
# io.imshow(disparity_values)
# plt.show()

'''
Numa
'''
from numba import jit
@jit
def jitized(cache):
    # for i in range(half_window, width - half_window):
    for i in range(half_window, height - half_window):
        # print(i)
        for j in range(half_window + max_disparity, width - half_window):
            for d in range(max_disparity):
                res = np.max(np.dot(image1[i - half_window: i + half_window, j - half_window: j + half_window], 
                                        image2[i - half_window: i + half_window, j - d - half_window: j - d + half_window]))  
                # print(res)
                cache[i][j][d] = res
start_time = time.time()  
cache = np.zeros(((height, width, max_disparity)))
jitized(cache)
elapsed_time = time.time() - start_time
print(f"{elapsed_time}")
disparity_values = np.argmax(cache, axis=2)     
io.imshow(disparity_values)
plt.show()