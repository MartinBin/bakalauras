import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

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
    def __init__(self, latent_dim=1024):
        super(Encoder, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2)

        self.layer1 = self._make_layer(BasicBlock, 64, 2,stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2,stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2,stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2,stride=2)

        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.leakyRelu2 = nn.LeakyReLU(negative_slope=0.2)

        self.fc = nn.Linear(512 * 64 * 64, latent_dim)
        self.fc_activation = nn.ReLU()

    def _make_layer(self, block, out_channels,num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyRelu(x)

        x = cp.checkpoint(self.layer1,x, use_reentrant=False)
        x = cp.checkpoint(self.layer2,x, use_reentrant=False)
        x = cp.checkpoint(self.layer3,x, use_reentrant=False)
        x = cp.checkpoint(self.layer4,x, use_reentrant=False)

        x = self.conv5(x)
        x = self.leakyRelu2(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.fc_activation(x)
        return x
