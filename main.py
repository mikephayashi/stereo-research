from skimage import io
import matplotlib.pyplot as plt
import numpy as np

# gt_image = read_pfm("data/disp0GT.pfm")

image1 = io.imread("data/im0.png", as_gray=True)
image2 = io.imread("data/im1.png", as_gray=True)
height, width = image1.shape
max_disparity = 4
half_window = 4
# cache = np.zeros(((width - 2 * half_window, height - 2 * half_window - max_disparity, max_disparity)))
cache = np.zeros(((width, height, max_disparity)))

# for i in range(window_size, width - window_size):
#     for j in range(window_size, height - window_size):
#         for i_w in range(-window_size, window_size):
#             for j_w in range(-window_size, window_size):
#                 for d in range(max_disparity):
#                     cache[i][j][d] = np.abs(image1[i][j] - image2[i + i_w][j + j_w - d])

# for i in range(half_window, width - half_window, 2):
for i in range(half_window, 200, 2):
    # print(i)
    for j in range(half_window + max_disparity, height - half_window):
        for d in range(max_disparity):
            try:
                cache[i][j][d] = np.max(np.dot(image1[i - half_window: i + half_window, j - half_window: j + half_window], 
                                        image2[i - half_window: i + half_window, j - d - half_window: j - d + half_window]))
            except:
                import pdb; pdb.set_trace()
                
        
                
disparity_values = np.argmax(cache, axis=2)             
io.imshow(disparity_values)
plt.show()