from skimage import filters, restoration, exposure
import numpy as np
from skimage.morphology import disk
from skimage.filters import rank

def filterGaussian(image1_raw, image2_raw):
    image1 = filters.gaussian(image1_raw)
    image2 = filters.gaussian(image2_raw)
    return image1, image2

def filterMedian(image1_raw, image2_raw):
    footprint = disk(20)
    image1 = filters.median(image1_raw, footprint)
    image2 = filters.median(image2_raw, footprint)
    return image1, image2

def filterButterworth(image1_raw, image2_raw):
    image1 = filters.butterworth(image1_raw)
    image2 = filters.butterworth(image2_raw)
    return image1, image2

def filterGreater(image, filterFunc):
    thresh = filterFunc(image)
    image[image <= thresh] = 0
    return image

def filterLessThan(image, filterFunc):
    thresh = filterFunc(image)
    image[image > thresh] = 0
    return image

def filterIsodata(image1_raw, image2_raw):
    image1 = filterGreater(image1_raw, filters.threshold_isodata)
    image2 = filterGreater(image2_raw, filters.threshold_isodata)
    return image1, image2

def filterLi(image1_raw, image2_raw):
    image1 = filterGreater(image1_raw, filters.threshold_li)
    image2 = filterGreater(image2_raw, filters.threshold_li)
    return image1, image2

def filterMean(image1_raw, image2_raw):
    image1 = filterGreater(image1_raw, filters.threshold_mean)
    image2 = filterGreater(image2_raw, filters.threshold_mean)
    return image1, image2

def filterMinimum(image1_raw, image2_raw):
    image1 = filterGreater(image1_raw, filters.threshold_minimum)
    image2 = filterGreater(image2_raw, filters.threshold_minimum)
    return image1, image2

def filterOtsu(image1_raw, image2_raw):
    image1 = filterLessThan(image1_raw, filters.threshold_otsu)
    image2 = filterLessThan(image2_raw, filters.threshold_otsu)
    return image1, image2

def filterTriangle(image1_raw, image2_raw):
    image1 = filterGreater(image1_raw, filters.threshold_triangle)
    image2 = filterGreater(image2_raw, filters.threshold_triangle)
    return image1, image2

def filterYen(image1_raw, image2_raw):
    image1 = filterLessThan(image1_raw, filters.threshold_yen)
    image2 = filterLessThan(image2_raw, filters.threshold_yen)
    return image1, image2

def filterPercentileSmooth(image1_raw, image2_raw):
    footprint = disk(20)
    image1 = rank.mean_percentile(image1_raw, footprint=footprint, p0=0.1, p1=0.9)
    image2 = rank.mean_percentile(image2_raw, footprint=footprint, p0=0.1, p1=0.9)
    return image1, image2

def filterBilateralSmooth(image1_raw, image2_raw):
    footprint = disk(20)
    image1 = rank.mean_bilateral(image1_raw, footprint=footprint, s0=500, s1=500)
    image2 = rank.mean_bilateral(image2_raw, footprint=footprint, s0=500, s1=500)
    return image1, image2

def filterNormalSmooth(image1_raw, image2_raw):
    footprint = disk(20)
    image1 = rank.mean(image1_raw, footprint=footprint)
    image2 = rank.mean(image2_raw, footprint=footprint)
    return image1, image2

def wiener(image1_raw, image2_raw, psf):
    deconvolved1, _ = restoration.unsupervised_wiener(image1_raw, psf)
    deconvolved2, _ = restoration.unsupervised_wiener(image2_raw, psf)
    return deconvolved1, deconvolved2

def richardsonLucy(image1_raw, image2_raw, psf):
    deconvolved1 = restoration.richardson_lucy(image1_raw, psf, num_iter=30)
    deconvolved2 = restoration.richardson_lucy(image2_raw, psf, num_iter=30)
    return deconvolved1, deconvolved2

def constrastStretching(image1_raw, image2_raw):
    p2, p98 = np.percentile(image1_raw, (2, 98))
    img_rescale1 = exposure.rescale_intensity(image1_raw, in_range=(p2, p98))
    p2, p98 = np.percentile(image2_raw, (2, 98))
    img_rescale2 = exposure.rescale_intensity(image2_raw, in_range=(p2, p98))
    return img_rescale1, img_rescale2

def equalization(image1_raw, image2_raw):
    hist1 = exposure.equalize_hist(image1_raw)
    hist2 = exposure.equalize_hist(image2_raw)
    return hist1, hist2

def adaptiveEqualization(image1_raw, image2_raw):
    img_adapteq1 = exposure.equalize_adapthist(image1_raw, clip_limit=0.03)
    img_adapteq2 = exposure.equalize_adapthist(image2_raw, clip_limit=0.03)
    return img_adapteq1, img_adapteq2