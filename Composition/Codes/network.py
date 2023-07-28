import torch
import torch.nn as nn
import torch.nn.functional as F



def build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor):

    out  = net(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

    learned_mask1 = (mask1_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*out
    learned_mask2 = (mask2_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*(1-out)
    stitched_image = (warp1_tensor+1.) * learned_mask1 + (warp2_tensor+1.)*learned_mask2 - 1.

    out_dict = {}
    out_dict.update(learned_mask1=learned_mask1, learned_mask2=learned_mask2, stitched_image = stitched_image)


    return out_dict


class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation = dilation))
        blk.append(nn.ReLU(inplace=True))
        blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation = dilation))
        blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layer(x)

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation):
        super(UpBlock, self).__init__()
        #self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.halfChanelConv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )

        self.conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):

        x1 = F.interpolate(x1, size = (x2.size()[2], x2.size()[3]), mode='nearest')
        x1 = self.halfChanelConv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# predict the composition mask of img1
class Network(nn.Module):
    def __init__(self, nclasses=1):
        super(Network, self).__init__()


        self.down1 = DownBlock(3, 32, 1, pool=False)
        self.down2 = DownBlock(32, 64, 2)
        self.down3 = DownBlock(64, 128,3)
        self.down4 = DownBlock(128, 256, 4)
        self.down5 = DownBlock(256, 512, 5)
        self.up1 = UpBlock(512, 256, 4)
        self.up2 = UpBlock(256, 128, 3)
        self.up3 = UpBlock(128, 64, 2)
        self.up4 = UpBlock(64, 32, 1)


        self.out = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, y, m1, m2):


        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        y1 = self.down1(y)
        y2 = self.down2(y1)
        y3 = self.down3(y2)
        y4 = self.down4(y3)
        y5 = self.down5(y4)

        res = self.up1(x5-y5, x4-y4)
        res = self.up2(res, x3-y3)
        res = self.up3(res, x2-y2)
        res = self.up4(res, x1-y1)
        res = self.out(res)

        return res


