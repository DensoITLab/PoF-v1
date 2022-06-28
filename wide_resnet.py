import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import OrderedDict

class wide_basic(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class Wide_ResNet_fet(nn.Module):
    def __init__(self, args):
        super(Wide_ResNet_fet, self).__init__()
        self.in_channels = 16

        assert ((args.depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((args.depth-4)/6)
        k = args.factor

        print('| Wide-Resnet %dx%d' %(args.depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.numFeatureDim = nStages[3]
        self.conv1 = nn.Conv2d(args.input_channels, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, stride=2)
        self.bn_out = nn.BatchNorm2d(nStages[3], momentum=0.1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, mode='fan_out')
            if isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight.data, 1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_out(out))
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        return out

    def _wide_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(OrderedDict([(f"num_{i}", val) for i,val in enumerate(layers)]))