import numpy as np
import matplotlib.pyplot as plt
from justpfm import justpfm
from pathlib import Path
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

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
    
def averageAcrossImages(f, results_map, averages_data):
    quantile_05 = []
    quantile_90 = []
    quantile_95 = []
    quantile_99 = []
    for sample in results_map[quantiles_key]:
        first, second, third, fourth = sample
        quantile_05.append(first)
        quantile_90.append(second)
        quantile_95.append(third)
        quantile_99.append(fourth)
    rms_error_average = np.mean(results_map[rms_error_val_key])
    bad_pixels_half_average = np.mean(results_map[bad_pixels_half_key])
    bad_pixels_one_average = np.mean(results_map[bad_pixels_one_key])
    bad_pixels_two_average = np.mean(results_map[bad_pixels_two_key])
    bad_pixels_four_average = np.mean(results_map[bad_pixels_four_key])
    quantile_05_average = np.mean(quantile_05)
    quantile_90_average = np.mean(quantile_90)
    quantile_95_average = np.mean(quantile_95)
    quantile_99_average = np.mean(quantile_99)
    averages_data.append([
        rms_error_average, 
        bad_pixels_half_average,
        bad_pixels_one_average,
        bad_pixels_two_average,
        bad_pixels_four_average,
        quantile_05_average,
        quantile_90_average,
        quantile_95_average,
        quantile_99_average
    ])
    f.write("RMS Average: " + str(rms_error_average) + '\n')
    f.write("Bad Half Average: " + str(bad_pixels_half_average) + '\n')
    f.write("Bad One Average: " + str(bad_pixels_one_average) + '\n')
    f.write("Bad Two Average: " + str(bad_pixels_two_average) + '\n')
    f.write("Bad Four Average: " + str(bad_pixels_four_average) + '\n')
    f.write("Quantile 0.5 Average: " + str(quantile_05_average) + '\n')
    f.write("Quantile 0.90 Average: " + str(quantile_90_average) + '\n')
    f.write("Quantile 0.95 Average: " + str(quantile_95_average) + '\n')
    f.write("Quantile 0.99 Average: " + str(quantile_99_average) + '\n')
    f.close()
    # return rms_error_average, bad_pixels_half_average, bad_pixels_one_average, bad_pixels_two_average, bad_pixels_four_average, quantile_05_average, quantile_90_average, quantile_95_average, quantile_99_average
    
def generateDirNames():
    # dirNames = []
    # smoothings = ["gaussian", "median", "percentile", "bilateral", "normal"]
    # deconvolutions = ["wiener", "richardsonLucy"]
    # histograms = ["constrastStretching", "equalization", "adaptiveEqualization"]
    # for smoothing in smoothings:
    #     for deconvolution in deconvolutions:
    #         for histogram in histograms:
    #             comboName = smoothing + deconvolution + histogram
    #             dirNames.append(comboName)
    # return dirNames
    return ["baseline"]
    
    
if __name__ == '__main__':
    max_disparity = 30
    # image_names = ["Adirondack", "ArtL", "Jadeplant", "MotorcycleE", "Piano", "PianoL", "Pipes", "Playroom", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"]
    image_names = ["Adirondack", "Jadeplant", "Piano", "Playroom"]
    # override = ["gaussianwienerconstrastStretching"]
    labels_list = []
    averages_data = []
    for dirName in generateDirNames():
        print("------")
        print(dirName)
        print("------")
        print()
    # for dirName in override:
        base_path = f"./results/{dirName}/"
        results_map = {
            rms_error_val_key: [],
            bad_pixels_half_key: [],
            bad_pixels_one_key: [],
            bad_pixels_two_key: [],
            bad_pixels_four_key: [],
            quantiles_key: []
        }
        base_results_path = base_path + "/results"
        for image_name in image_names:
            Path(base_results_path).mkdir(parents=True, exist_ok=True) 
            f = open(base_results_path + f"/{image_name}.txt", "w")
            combined_name = f"{image_name}-{max_disparity}"
            pfm_path = base_path + f"/pfm/{combined_name}.pfm"
            gt_pfm = justpfm.read_pfm(file_name=f"./data/gt/{image_name}/disp0GT.pfm")
            gt_pfm_max = np.max(np.ma.masked_invalid(gt_pfm))
            if gt_pfm_max > 1:
                gt_pfm = (gt_pfm)/(gt_pfm_max)
            # import pdb; pdb.set_trace()
            # with open(pfm_path) as fp:
            #     scale = float(fp.readline().decode().rstrip())
            # import pdb; pdb.set_trace()
            disparity_values = justpfm.read_pfm(file_name=pfm_path)
            output(f, None, f"{image_name}-{max_disparity}", None, None)
            evaluation_suite(f, disparity_values, gt_pfm, max_disparity, results_map)
            output(f, None, "", None, None)
            # io.imshow(disparity_values)
            # plt.show()
            f.close()
        f = open(base_results_path + f"/average.txt", "w")
        labels_list.append(dirName)
        averageAcrossImages(f, results_map, averages_data)

            