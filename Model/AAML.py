import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 1.0 if not s else s
            self.m = 0.2 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 5.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        # self.in_features = in_features
        # self.out_features = out_features
        # self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0

        # print("x.transpose(0, 1)", x.transpose(0, 1))
        # print("x.transpose(0, 1).shape", x.transpose(0, 1).shape)
        # print("labels", labels)
        # print("x.transpose(0, 1)[labels]", x.transpose(0, 1)[labels])
        # print("diagonal", torch.diagonal(x.transpose(0, 1)[labels]))
        # print("clamp", torch.clamp(torch.diagonal(x.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps))
        # print("acos", torch.acos(torch.clamp(torch.diagonal(x.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))
        # print("cos", torch.cos(torch.acos(torch.clamp(torch.diagonal(x.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m))
        # exit()




        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(x.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(x.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(x.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))
        # print(numerator)
        # for i, y in enumerate(labels):
        #     temp = torch.cat((x[i, :y], x[i, y+1:])).unsqueeze(0)
        #     print(i, y)
        #     print(temp.shape)
        #     input()
        excl = torch.cat([torch.cat((x[i, :y], x[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)

        # print(excl)

        log_sum = torch.sum(torch.exp(self.s * excl), dim=1)
        # print(log_sum)
        
        denominator = torch.exp(numerator) + log_sum
        # print(denominator)
        # print(torch.log(denominator))

        L = numerator - torch.log(denominator)
        # print(L)
        # exit()

        return -torch.mean(L)

if __name__ == "__main__":
    print(torch.exp(torch.FloatTensor([-2.])))
    # a = torch.randint(1, 10, (1,10))
    # print(a)
    # print(torch.prod(a))
    
    # b = torch.max(a)
    # print(b)
    # print(a-b)
    # c = torch.exp(a-b)
    # print(torch.log(c))