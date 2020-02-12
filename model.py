import torch
import torch.nn as nn

class Conv_Block(nn.Module):
    def __init__(self):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('relu'))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.input.weight, gain=nn.init.calculate_gain('relu'))
        self.conv_layer = self.make_layer(Conv_Block, 18)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('relu'))
        self.relu = nn.ReLU(inplace=True)


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        res = x
        out = self.relu(self.input(x))
        out = self.conv_layer(out)
        out = self.output(out)
        out = out + res
        return out

    def withoutres(self, x):
        res = x
        out = self.relu(self.input(x))
        out = self.conv_layer(out)
        out = self.output(out)
        out = out
        return out


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=9 // 2)
        self.conv = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=5 // 2)
        self.output = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

        nn.init.xavier_uniform_(self.input.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        out = self.relu(self.input(x))
        out = self.relu(self.conv(out))
        out = self.output(out)
        return out