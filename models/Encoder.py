import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.relu = nn.ReLU(inplace=True)
        self.leakyRelu = nn.LeakyReLu(negative_slope=0.2)

        self.layer1 = self._make_layer(BasicBlock, 64, 2,stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2,stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2,stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2,stride=2)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.leakyRelu2 = nn.LeakyReLu(negative_slope=0.2)
        self.fc = nn.Sigmoid()

    def _make_layer(self, block, num_blocks, out_channels, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        #x = self.relu(x)
        x = self.leakyRelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv5(x)
        x = self.leakyRelu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
