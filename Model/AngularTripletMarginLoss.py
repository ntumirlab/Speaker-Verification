import torch
import torch.nn as nn
import math
from torch.autograd import Function


class AngularTripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin=0.5, eps=1e-7):
        super(AngularTripletMarginLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def bdot(self, a, b):
        B = a.shape[0]
        S = a.shape[1]
        return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

    def forward(self, anchor, positive, negative):
        # print(self.bdot(anchor, positive))
        # print(torch.clamp(self.bdot(anchor, positive)))
        d_p = torch.acos(torch.clamp(self.bdot(anchor, positive), -1.+self.eps, 1-self.eps))
        d_n = torch.acos(torch.clamp(self.bdot(anchor, negative), -1.+self.eps, 1-self.eps))
        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.sum(dist_hinge)
        return loss



if __name__ == "__main__":
    criterion = AngularTripletMarginLoss(margin=0.4)
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    c = torch.randn(3, 4)

    # print(a)
    # print(b)
    # print(torch.dot(a[0], b[0]))
    # print(bdot(a, b))
    print(criterion.forward(a, b, c))
    # print(torch.acos(a))
