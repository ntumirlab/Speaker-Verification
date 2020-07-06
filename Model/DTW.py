import torch
import torch.nn as nn
from torch.autograd import Function
from dtaidistance import dtw


class DTWDistance(nn.Module):
    def __init__(self):
        super(DTWDistance, self).__init__()

    def forward(self, x1, x2):
        # assert len(x1) == len(x2)
        a = torch.rand(10)
        b = torch.rand(9)
        distance = dtw.distance(a, b)
        return distance
