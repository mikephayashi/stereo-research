python3 setup.py build_ext --inplace
cython -a x.py

Normal Loop:
16.49659490585327

nditer:
14.9828462600708

numba: 
4.33

/Users/michaelhayashi/development/cs231a/stereo/michael-stereo/speedup.py:80: NumbaPerformanceWarning: np.dot() is faster on contiguous arrays, called on (Array(float64, 2, 'A', True, aligned=True), Array(float64, 2, 'A', True, aligned=True))