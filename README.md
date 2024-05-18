python3 setup.py build_ext --inplace
cython -a x.py

Normal Loop:
16.49659490585327

nditer:
14.9828462600708

numba: 
4.33

/Users/michaelhayashi/development/cs231a/stereo/michael-stereo/speedup.py:80: NumbaPerformanceWarning: np.dot() is faster on contiguous arrays, called on (Array(float64, 2, 'A', True, aligned=True), Array(float64, 2, 'A', True, aligned=True))

# Results

## Sliding Window Dot Product
rms_error_val: 4.441102027893066
bad_match_val 0.5: 0.9655494071313867
bad_match_val 1: 0.9274988761789519
bad_match_val 2: 0.7975033502589736
bad_match_val 4: 0.4702358822631353
quantiles: [3.81564057 6.93271942 7.2667141  7.88772534]

## Sliding Window Absolute Difference
rms_error_val: 4.353975772857666
bad_match_val 0.5: 0.9613329577839097
bad_match_val 1: 0.9179405661230859
bad_match_val 2: 0.7790619062832199
bad_match_val 4: 0.45571915358604936
quantiles: [3.7281816  6.86637115 7.18272562 7.62365422]

## Sliding Window Squared Difference
rms_error_val: 4.353400230407715
bad_match_val 0.5: 0.9614008111302106
bad_match_val 1: 0.9180292702372606
bad_match_val 2: 0.7791403617148802
bad_match_val 4: 0.45563486700744127
quantiles: [3.72811306 6.8658843  7.18269987 7.62457155]