from skimage import io
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from justpfm import justpfm

def print_time(start):
    elaped = time.time() - start
    print(f"time: {elaped}")
    
def shiftOne(image1, image1_windows):
    # image1 = np.delete(image1, 0, 1)
    # image1 = np.append(image1, np.zeros((image1.shape[0], 1)), axis=1)
    image1 = np.delete(image1, 0, 0)
    image1 = np.append(image1, np.zeros((1, image1.shape[1])), axis=0)
    image1_windows = sliding_window_view(image1, window_size)
    return image1, image1_windows

def getWindowDotProduct(image1_windows, image2_windows, cache, disparity):
    windowDotProduct = np.einsum('ijkl,ijkl->ij', image1_windows, image2_windows)
    cache[:,:,disparity] = windowDotProduct
    return windowDotProduct

def getAbsoluteDifference(image1_windows, image2_windows, cache, disparity):
    absolute = np.abs(image1_windows - image2_windows)
    mean = np.mean(absolute, axis = (2,3))
    cache[:,:,disparity] = mean
    return mean

def getSquaredDifference(image1_windows, image2_windows, cache, disparity):
    squared_diff = np.square(image1_windows - image2_windows)
    mean = np.mean(squared_diff, axis = (2,3))
    cache[:,:,disparity] = mean
    return mean

if __name__ == '__main__':
    max_disparity = 50
    window_length = 9
    window_size = (window_length, window_length)
    # image_names = ["Adirondack", "ArtL", "Jadeplant", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"]
    image_names = ["Adirondack"]

    for image_name in image_names:
        image1 = io.imread(f"data/input/{image_name}/im0.png", as_gray=True)
        image2 = io.imread(f"data/input/{image_name}/im1.png", as_gray=True)
        gt_pfm = justpfm.read_pfm(file_name=f"./data/gt/{image_name}/disp0GT.pfm")
        height, width = image1.shape
        cache = np.zeros(((height - window_length + 1, width - window_length + 1, max_disparity)))

        '''
        Einsum
        '''
        start = time.time()
        print_time(start)
        disparity_values = None
        image1_windows = sliding_window_view(image1, window_size)
        print_time(start)
        image2_windows = sliding_window_view(image2, window_size)
        print_time(start)
        res = getWindowDotProduct(image1_windows, image2_windows, cache, 0)
        print_time(start)
        for disparity in range(max_disparity):
            if disparity == 0:
                continue
            image1, image1_windows = shiftOne(image1, image1_windows)
            image2, image2_windows = shiftOne(image2, image2_windows)
            res = getWindowDotProduct(image1_windows, image2_windows, cache, disparity)
            print_time(start)
        for disparity in range(2, max_disparity + 1):
            temp_cache = cache[:,:,:disparity]
            disparity_values = np.argmin(cache, axis=2) 
            disparity_values = disparity_values / (disparity - 1)
            disparity_float32 = np.float32(disparity_values)      
            pfm_path = f"./results/slidingWindow-dotProduct/pfm/{image_name}-{disparity}.pfm"
            justpfm.write_pfm(file_name=pfm_path, data=disparity_float32)

'''
Append and concatenate
'''
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