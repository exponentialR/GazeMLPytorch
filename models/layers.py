from torch import nn

Pool = nn.MaxPool2d


def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)


class Conv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, batch_norm=False, relu=True):
        super(Conv, self).__init__()
        self.input_dimension = input_dim
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        assert x.size()[1] == self.input_dimension, "{} {}".format(x.size()[1], self.input_dimension)
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Residual, self).__init__()
        self.relu = nn.ReLU
        self.batch_norm_1 = nn.BatchNorm2d(input_dimension)
        self.conv1 = Conv(input_dimension, int(output_dimension / 2), 1, relu=False)
        self.batch_norm_2 = nn.BatchNorm2d(int(output_dimension / 2))
        self.conv2 = Conv(int(output_dimension / 2), int(output_dimension / 2), 3, relu=False)
        self.batch_norm_3 = nn.BatchNorm2d(int(output_dimension / 2))
        self.conv3 = Conv(int(output_dimension / 2), output_dimension, 1, relu=False)
        self.skip_layer = Conv(input_dimension, output_dimension, 1, relu=False)
        if input_dimension == output_dimension:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        output = self.batch_norm_1(x)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.batch_norm_2(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.batch_norm_3(output)
        output = self.relu(output)
        output = self.conv3(output)
        output += residual
        return output


class Hourglass(nn.Module):
    def __int__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, f)

        # lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf)
        self.n = n

        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, bn=bn)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, f)

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = nn.functional.interpolate(low3, x.shape[2:], mode='bilinear')
        return up1 + up2
