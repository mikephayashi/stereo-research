from skimage import io, util
import numpy as np
import time
from numpy.lib.stride_tricks import sliding_window_view
from justpfm import justpfm
from pathlib import Path
from filters import *
import matplotlib.pyplot as plt

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

def generate_filters():
    combos = []
    smoothings = [
        None, 
        {"name": "gaussian", 
         "func": filterGaussian },
        {"name": "median", 
         "func": filterMedian },
    ]
    fouriers = [
        None, 
        {"name": "butterworth", 
         "func": filterButterworth},
    ]
    thresholds = [
        None,
        {"name": "isodata", 
        "func": filterIsodata},
        {"name": "li", 
        "func": filterLi},
        {"name": "mean", 
        "func": filterMean},
        {"name": "minimum", 
        "func": filterMinimum},
        {"name": "otsu", 
        "func": filterOtsu},
        {"name": "triangle", 
        "func": filterTriangle},
        {"name": "yen", 
        "func": filterYen},
    ]
    means = [
        None, 
        {"name": "percentile",
         "func": filterPercentileSmooth
         },
        {"name": "bilateral",
         "func": filterBilateralSmooth
         },
        {"name": "normal",
         "func": filterNormalSmooth
         }
    ]
    for smoothing in smoothings:
        for fourier in fouriers:
            for threshold in thresholds:
                for mean in means:
                    if smoothing is None and fourier is None and threshold is None and mean is None:
                        continue
                    combo = []
                    if smoothing is not None:
                        combo.append(smoothing)
                    if fourier is not None:
                        combo.append(fourier)
                    if threshold is not None:
                        combo.append(threshold)
                    if mean is not None:
                        combo.append(mean)
                    combos.append(combo)
    # for smoothing in smoothings:
    #         for threshold in thresholds:
    #             combo = [smoothing, threshold]
    #             combos.append(combo)
    return combos

def display(image):
    io.imshow(image)
    plt.show()

if __name__ == '__main__':
    max_disparity = 30
    window_length = 9
    window_size = (window_length, window_length)
    # image_names = ["Adirondack", "ArtL", "Jadeplant", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"]
    image_names = ["Adirondack", "Jadeplant", "Piano", "Playroom"]
    
    start = time.time()
    for filterCombo in generate_filters():
        for image_name in image_names:
            image1_raw = io.imread(f"data/input/{image_name}/im0.png", as_gray=True)
            image2_raw = io.imread(f"data/input/{image_name}/im1.png", as_gray=True)
            
            # image1_raw = util.img_as_float(image1_raw)
            # image2_raw = util.img_as_float(image2_raw)
            image1_raw = util.img_as_uint(image1_raw)
            image2_raw = util.img_as_uint(image2_raw)
            
            for filterInfo in filterCombo:
                print(filterInfo["name"])
                image1, image2 = filterInfo["func"](image1_raw, image2_raw)
            
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
            for disparity in range(2, max_disparity + 1):
                temp_cache = cache[:,:,:disparity]
                disparity_values = np.argmin(cache, axis=2) 
                disparity_values = disparity_values / (disparity - 1)
                disparity_float32 = np.float32(disparity_values)   
                filterComboName = ""
                for filterInfo in filterCombo:
                    filterComboName += filterInfo["name"]
                directory = f"./results/{filterComboName}/pfm"
                Path(directory).mkdir(parents=True, exist_ok=True) 
                pfm_path = f"{directory}/{image_name}-{disparity}.pfm"
                justpfm.write_pfm(file_name=pfm_path, data=disparity_float32)
            print_time(start)