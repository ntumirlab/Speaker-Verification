import torch
import torch.nn as nn
import math
from torch.autograd import Function


class TripletCosMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletCosMarginLoss, self).__init__()
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, anchor, positive, negative):
        d_p = self.cos(anchor, positive)
        d_n = self.cos(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_n - d_p, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


if __name__ == "__main__":
    a = torch.randn(4)
    print(a)
    print(torch.diagonal(a, dim))
    print(torch.acos(a))