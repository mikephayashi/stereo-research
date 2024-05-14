import numpy as np
a = np.arange(6).reshape(2,3)
# for x in np.nditer(a, flags=['external_loop']):
#     print(x)

for x in np.nditer(a, flags=['external_loop'], order='F'):
    print(x, end=' ')