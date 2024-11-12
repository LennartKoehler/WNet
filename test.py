import numpy as np
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
array = np.array([1,2,3])

np.save("models/test.npy", array)