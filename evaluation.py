import numpy as np

def delete_edge(ground_truth, start_idx, end_idx, axis):
    ground_truth_reduced = np.delete(ground_truth, [i for i in range(start_idx, end_idx)], axis = axis)
    return ground_truth_reduced

def reduce_ground_truth(ground_truth):
    height,width, _ = ground_truth.shape
    shave = 4
    ground_truth_reduced = delete_edge(ground_truth, height - shave, height, 0)
    ground_truth_reduced = delete_edge(ground_truth_reduced, 0, shave, 0)
    ground_truth_reduced = delete_edge(ground_truth_reduced, width - shave, width, 1)
    ground_truth_reduced = delete_edge(ground_truth_reduced, 0, shave, 1)
    return ground_truth_reduced
    
def filter_ground_truth(ground_truth):
    ground_truth[(ground_truth == -np.inf) | (ground_truth == np.inf)] = 1
    
def adjusted_computed(computed, max_disparity):
    computed = computed * max_disparity
    return computed

def adjusted_ground_truth(ground_truth, max_disparity):
    ground_truth = ground_truth * max_disparity
    ground_truth_reduced = reduce_ground_truth(ground_truth)
    filter_ground_truth(ground_truth_reduced)
    return ground_truth_reduced

def rms_error(computed, ground_truth, max_disparity):
    computed = adjusted_computed(computed, max_disparity)
    ground_truth = adjusted_ground_truth(ground_truth, max_disparity)
    diff = computed - ground_truth
    squared = np.square(diff)
    mean = np.mean(squared)
    square_root = np.sqrt(mean)
    return square_root

def bad_match(computed, ground_truth, threshold, max_disparity):
    computed = adjusted_computed(computed, max_disparity)
    ground_truth = adjusted_ground_truth(ground_truth, max_disparity)
    diff = computed - ground_truth
    absolute = np.abs(diff)
    above = [absolute > threshold]
    unique, counts = np.unique(above, return_counts=True)
    false_above_count = counts[0]
    true_above_count = counts[1]
    total = false_above_count + true_above_count
    percentage = true_above_count / total
    return percentage

def get_quantiles(computed, ground_truth, max_disparity):
    computed = adjusted_computed(computed, max_disparity)
    ground_truth = adjusted_ground_truth(ground_truth, max_disparity)
    diff = computed - ground_truth
    absolute = np.abs(diff)
    probs = [0.5, 0.9, 0.95, 0.99]
    res = np.quantile(absolute, probs)
    return res

def evaluation_suite(computed, ground_truth, max_disparity):
    rms_error_val = rms_error(computed, ground_truth, max_disparity)
    print(f'rms_error_val: {rms_error_val}')
    bad_pixels = [0.5, 1, 2, 4]
    for bad_pixel in bad_pixels:
        bad_match_val = bad_match(computed, ground_truth, bad_pixel, max_disparity)
        print(f'bad_match_val {bad_pixel}: {bad_match_val}')
    quantiles = get_quantiles(computed, ground_truth, max_disparity)
    print(f'quantiles: {quantiles}')
    