import torch
import torch.nn as nn

class UNet2D(nn.Module):
    def __init__(self, in_channel=1, n_classes=3):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet2D, self).__init__()
        bias_flag = False
        bn_flag = True
        self.ec0 = self.encoder(self.in_channel, 32, bias=bias_flag, batchnorm=bn_flag)
        self.ec1 = self.encoder(32, 64, bias=bias_flag, batchnorm=bn_flag)
        self.ec2 = self.encoder(64, 64, bias=bias_flag, batchnorm=bn_flag)
        self.ec3 = self.encoder(64, 128, bias=bias_flag, batchnorm=bn_flag)
        self.ec4 = self.encoder(128, 128, bias=bias_flag, batchnorm=bn_flag)
        self.ec5 = self.encoder(128, 256, bias=bias_flag, batchnorm=bn_flag)
        self.ec6 = self.encoder(256, 256, bias=bias_flag, batchnorm=bn_flag)
        self.ec7 = self.encoder(256, 512, bias=bias_flag, batchnorm=bn_flag)

        self.pool0 = nn.MaxPool2d(2)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=bias_flag)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=bias_flag)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=bias_flag)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.dc0 = self.decoder(64, n_classes, kernel_size=1, stride=1, bias=bias_flag)

        self.softmax = nn.Softmax(dim=1)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return self.softmax(d0)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    image_size = 128
    x = torch.Tensor(1, 1, image_size, image_size)
    x.to(device)
    print("x size: {}".format(x.size()))

    model = UNet2D(1, 3)

    out = model(x)
    print("out size: {}".format(out.size()))

    from torchsummary import summary
    print(model)
    summary(model.to(device), (1, 256, 256), device=device)