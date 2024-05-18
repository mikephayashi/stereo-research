import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from justpfm import justpfm

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

def output(f, prefix, content, results_map, results_key):
    combined_string = ""
    if prefix is not None:
        combined_string = prefix + ": " + str(content)
    else:
        combined_string = str(content)
    f.write(combined_string + "\n")
    if results_map is not None:
        results_map[results_key].append(content)
    print(content)

rms_error_val_key = "rms_error_val"
bad_pixels_half_key = "bad_pixels_half"
bad_pixels_one_key = "bad_pixels_one"
bad_pixels_two_key = "bad_pixels_two"
bad_pixels_four_key = "bad_pixels_four"
quantiles_key = "quantiles"

keys_list = [rms_error_val_key, bad_pixels_half_key, bad_pixels_one_key, bad_pixels_two_key, bad_pixels_four_key, quantiles_key]

def evaluation_suite(f, computed, ground_truth, max_disparity, results_map):
    rms_error_val = rms_error(computed, ground_truth, max_disparity)
    output(f, "rms_error_val", rms_error_val, results_map, rms_error_val_key)
    # bad_pixels = [0.5, 1, 2, 4]
    bad_pixels = [(0.5, bad_pixels_half_key), (1, bad_pixels_one_key), (2, bad_pixels_two_key), (4, bad_pixels_four_key)]
    for bad_pixel_pair in bad_pixels:
        bad_pixel = bad_pixel_pair[0]
        bad_pixel_key = bad_pixel_pair[1]
        bad_match_val = bad_match(computed, ground_truth, bad_pixel, max_disparity)
        output(f, "bad_match_val", bad_match_val, results_map, bad_pixel_key)
    quantiles = get_quantiles(computed, ground_truth, max_disparity)
    output(f, "quantiles", quantiles, results_map, quantiles_key)
    
    
if __name__ == '__main__':
    max_disparity = 50
    # image_names = ["Adirondack", "ArtL", "Jadeplant", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"]
    image_names = ["Adirondack"]
    base_path = "./results/slidingWindow-dotProduct/"
    for image_name in image_names:
        results_map = {
            rms_error_val_key: [],
            bad_pixels_half_key: [],
            bad_pixels_one_key: [],
            bad_pixels_two_key: [],
            bad_pixels_four_key: [],
            quantiles_key: []
        }
        base_results_path = base_path + "/results/"
        f = open(base_results_path + f"{image_name}.txt", "w")
        for disparity in range(2, max_disparity + 1):
            combined_name = f"{image_name}-{disparity}"
            pfm_path = base_path + f"/pfm/{combined_name}.pfm"
            gt_pfm = justpfm.read_pfm(file_name=f"./data/gt/{image_name}/disp0GT.pfm")
            disparity_values = justpfm.read_pfm(file_name=pfm_path)
            output(f, None, f"{image_name}-{disparity}", None, None)
            evaluation_suite(f, disparity_values, gt_pfm, disparity, results_map)
            output(f, None, "", None, None)
            # io.imshow(disparity_values)
            # plt.show()
        f.close()
        for current_key in keys_list:
            x = [i for i in range(2, max_disparity + 1)]
            y = results_map[current_key]
            plt.plot(x, y)
            plt.title(current_key)  # add title
            # plt.show()
            plt.savefig(base_results_path + current_key)
            plt.close()
            