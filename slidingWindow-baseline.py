from skimage import io, util
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from justpfm import justpfm
from pathlib import Path
from filters import *
import matplotlib.pyplot as plt
from scipy import signal

def print_time(start):
    elaped = time.time() - start
    print(f"time: {elaped}")
    
def shiftOne(image1, image1_windows):
    # image1 = np.delete(import matplotlib.pyplot as pltimage1, 0, 1)
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

def display(image):
    io.imshow(image)
    plt.show()
    
def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def generate_filters():
    return ["baseline"]

if __name__ == '__main__':
    max_disparity = 30
    window_length = 9
    window_size = (window_length, window_length)
    # image_names = ["Adirondack", "ArtL", "Jadeplant", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"]
    image_names = ["Adirondack", "Jadeplant", "Piano", "Playroom"]
    
    start = time.time()
    for filterCombo in generate_filters():
        for image_name in image_names:
            print(image_name)
            image1_raw = io.imread(f"data/input/{image_name}/im0.png", as_gray=True)
            image2_raw = io.imread(f"data/input/{image_name}/im1.png", as_gray=True)
            
            image1_raw = util.img_as_float(image1_raw)
            image2_raw = util.img_as_float(image2_raw)
            
            image1 = image1_raw
            image2 = image2_raw
            
            gt_pfm = justpfm.read_pfm(file_name=f"./data/gt/{image_name}/disp0GT.pfm")
            height, width = image1.shape
            cache = np.zeros(((height - window_length + 1, width - window_length + 1, max_disparity)))

            '''
            Einsum
            '''
            
            disparity_values = None
            image1_windows = sliding_window_view(image1, window_size)
            image2_windows = sliding_window_view(image2, window_size)
            res = getWindowDotProduct(image1_windows, image2_windows, cache, 0)
            for disparity in range(max_disparity):
                if disparity == 0:
                    continue
                image1, image1_windows = shiftOne(image1, image1_windows)
                image2, image2_windows = shiftOne(image2, image2_windows)
                res = getWindowDotProduct(image1_windows, image2_windows, cache, disparity)
            temp_cache = cache[:,:,:max_disparity]
            disparity_values = np.argmin(cache, axis=2) 
            disparity_values = disparity_values / (max_disparity - 1)
            disparity_float32 = np.float32(disparity_values)   
            filterComboName = "baseline"
            directory = f"./results/{filterComboName}/pfm"
            Path(directory).mkdir(parents=True, exist_ok=True) 
            pfm_path = f"{directory}/{image_name}-{max_disparity}.pfm"
            justpfm.write_pfm(file_name=pfm_path, data=disparity_float32)
            print("Total time:")
            print_time(start)
            print("----")