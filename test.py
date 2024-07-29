import matplotlib.pyplot as plt
import torch

radius = 5
kh = radius*2 + 1
ox = 4

distance_weights = torch.abs(torch.arange(1, kh + 1) - radius - 1)
print(distance_weights)
distance_weights = torch.exp(torch.div(-1*(distance_weights), ox**2)) # exp(-||X(i)-X(j)||^2_2  /  sigma^2_X)
print(distance_weights)

