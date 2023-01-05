from torch import nn


class ResidualBlock(nn.Module):
    '''
    Residual block has two convolutional layers and adds in 'skip connections' after two convs
    where input is added to layer outputs in forward pass.
    Uses ReLU act function
    '''

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        # Force stride 1 convolution after normal stride conv1
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        # XXX changed to leaky
        self.lrelu = nn.LeakyReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x) # Note conv1 has relu in it
        out = self.conv2(out)
        # In order to concatenate the residual with the output we needed to save the downsample and apply to input
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.lrelu(out)
        return out
