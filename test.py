import matplotlib.pyplot as plt
from data import ReadDataset
import numpy as np
from models.swin_transformer_v2 import *





x = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])

WA = WindowAttention(5, (3,5), 2)

print(WA._buffers)