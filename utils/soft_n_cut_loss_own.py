import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _pair, _quadruple
import matplotlib.pyplot as plt

def calc_weights(raw_image, window_size):
    image_size = raw_image.size()[-1]
    expand_matrix = raw_image.expand(-1,-1,image_size)
    value_difference_matrix = expand_matrix - torch.transpose(expand_matrix, 2, 3)

    range_tensor = torch.arange(0, image_size)
    distance_matrix = torch.mul(range_tensor, torch.transpose(range_tensor))
    distance_matrix = distance_matrix.fil

