import torch
import torch.nn as nn
import math

from torch.autograd import Function
# from AAML import AngularPenaltySMLoss

class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)
class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs):

        # batch_size = encoder_outputs.size(0)
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        # print("energy:", energy.shape)
        weights = self.softmax(energy.squeeze(-1))
        # print("weights:", weights.unsqueeze(-1).shape)

        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        # print("encoder_outputs * weights.unsqueeze(-1)", encoder_outputs * weights.unsqueeze(-1).shape)
        # print("encoder_outputs:", encoder_outputs.shape)
        # print("outputs:", outputs.shape)

        return outputs, weights


class ReLU(nn.Hardtanh):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0, 20, inplace)

    def __repr__(self):
        inplace_str = 'inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class AttentiveStatsPool(nn.Module):

    def __init__(self, floor=1e-10, bessel=False):
        super(AttentiveStatsPool, self).__init__()
        self.floor = floor
        self.bessel = bessel
        self.attention = SelfAttention(2048)

    def forward(self, x):
        x = x.transpose(2, 1)
        means, w = self.attention(x)
        _, t, _ = x.shape
        if self.bessel:
            t = t - 1
        # print("x", x.shape)
        # print("w", w.shape)
        # print("residuals", (x * w.unsqueeze(-1)).shape)
        # print("means", means.shape)
        residuals = (x * w.unsqueeze(-1)) - means.unsqueeze(1)
        # print("residuals", residuals.shape)
        numerator = torch.sum(residuals**2, dim=1)
        # print("numerator", numerator.shape)

        stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        x = torch.cat([means, stds], dim=-1)
        return x
        # means = torch.mean(x, dim=-1)
        # _, t, _ = x.shape
        # if self.bessel:
        #     t = t - 1
        # residuals = x - means.unsqueeze(-1)
        # numerator = torch.sum(residuals**2, dim=-1)
        # stds = torch.sqrt(torch.clamp(numerator, min=self.floor)/t)
        # x = torch.cat([means, stds], dim=-1)
        # return x


def conv1x3(in_channels, out_channels, stride=1, padding=0, bias=False):
    "3x3 convolution with padding"
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, bias=bias)

def conv1x1(in_channels, out_channels, stride=1, padding=0, bias=False):
    "3x3 convolution with padding"
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride,
                     padding=padding, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = ReLU(inplace=True)
        self.conv2 = conv1x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.relu = ReLU(inplace=True)

        self.conv1 = conv1x1(in_channels=in_channels, out_channels=out_channels, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv1x3(in_channels=out_channels, out_channels=out_channels, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = conv1x1(in_channels=out_channels, out_channels=out_channels*self.expansion, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels*self.expansion)

        if self.downsampling:
            self.downsample = nn.Sequential(
                conv1x1(in_channels=in_channels, out_channels=out_channels*self.expansion, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels*self.expansion)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print(out.shape)

        out = self.conv3(out)
        out = self.bn3(out)
        # print(out.shape)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class ResNet50(nn.Module):

    def __init__(self, layers=[3,4,6,3], expansion=4, num_classes=1000, embedding_size=64):

        super(ResNet50, self).__init__()

        self.alpha = 10
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf

        self.relu = ReLU(inplace=True)
        self.expansion = expansion
        self.embedding_size = embedding_size
        self.conv1 = nn.Conv1d(39, 64, kernel_size=3, bias=False)

        self.layer1 = self.make_layer(in_channels=64, out_channels=64, layer=layers[0], stride=1)
        self.layer2 = self.make_layer(in_channels=256, out_channels=128, layer=layers[1], stride=2)
        self.layer3 = self.make_layer(in_channels=512, out_channels=256, layer=layers[2], stride=2)
        self.layer4 = self.make_layer(in_channels=1024, out_channels=512, layer=layers[3], stride=2)

        self.pooling = AttentiveStatsPool()
        self.fc1 = nn.Linear(1024*self.expansion, 512)
        self.fc2 = nn.Linear(512, self.embedding_size)

        self.classifier = nn.Linear(self.embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def make_layer(self, in_channels, out_channels, layer, stride):

        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride, downsampling=True))
        for i in range(1, layer):
            layers.append(Bottleneck(out_channels*self.expansion, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.pooling(x)
        # print(x.shape)
        # exit()
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        x = self.relu(x)
        x = self.fc2(x)
        # print(x.shape)


        x = torch.nn.functional.normalize(x, p=2, dim=1)



        # x = self.l2_norm(x)

        # x = x * self.alpha
        # print(x.shape)
        return x

    def forward_classifier(self, x):
        x = self.forward(x)
        # x = torch.nn.functional.normalize(x, p=2, dim=1)

        # x = self.l2_norm(x)   
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf

        # x = x * self.alpha
        # x = self.relu(x)

        x = self.classifier(x)
        return x


if __name__ == "__main__":

    x = torch.randn(32, 64, 100)

    model = ResNet50(num_classes=340, embedding_size=64)
    x = model(x)
    # print(x.shape)

    # b1 = Bottleneck(64, 64)
    # b2 = Bottleneck(256, 128, stride=2)
    # x = b1(x)
    # print(x.shape)
    # x = b2(x)
    # print(x.shape)