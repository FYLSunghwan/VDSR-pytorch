import torch
import torch.nn as nn

class Conv_Block(nn.Module):
    def __init__(self):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_uniform_(self.input.weight, mode='fan_in', nonlinearity='relu')
        self.conv_layer = self.make_layer(Conv_Block, 18)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        nn.init.kaiming_uniform_(self.output.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        res = x
        out = self.input(x)
        out = self.conv_layer(out)
        out = self.output(out)
        out = out + res
        return out