import torch
import torch.nn as nn
import torch.nn.functional as F

###1.增加特征融合模块FPN
class FPN(nn.Module):
    def __init__(self, channels):
        super(FPN, self).__init__()
        
        self.upconv4 = nn.Conv2d(channels[3], channels[2], 1, 1, 0)
        self.upconv3 = nn.Conv2d(channels[2], channels[1], 1, 1, 0)
        self.upconv2 = nn.Conv2d(channels[1], channels[0], 1, 1, 0)
        
    def forward(self, x4, x3, x2, x1):
        up4 = F.upsample(self.upconv4(x4), size=x3.size()[2:])
        up3 = F.upsample(self.upconv3(x3 + up4), size=x2.size()[2:])
        up2 = F.upsample(self.upconv2(x2 + up3), size=x1.size()[2:])
        
        return up2

#---------U_Net-------------------------------------------------------#
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),#BN
            # nn.Dropout2d(0.3),#dropout0.5
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            # nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            
            )
            
    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class FPN_UNet(nn.Module):

    def __init__(self, in_ch=4, out_ch=1):#in_ch:通道（波段）数；out_ch:分类个数（包含0）
        super(FPN_UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(192, filters[0])

        self.fpn = FPN([64, 128, 256, 512])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # FPN
        up2 = self.fpn(e4, e3, e2, e1)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2,up2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)
        return out
    

###########################################################
class MultiScaleConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MultiScaleConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=3, dilation=3, bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.bn3 = nn.BatchNorm2d(ch_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.relu1(self.bn1(self.conv1(x)))
        out2 = self.relu2(self.bn2(self.conv2(x)))
        out3 = self.relu3(self.bn3(self.conv3(x)))

        x = out1 + out2 + out3
        return x

class FPN_MultiScaleConv_UNet(nn.Module):

    def __init__(self, in_ch=4, out_ch=1):#in_ch:通道（波段）数；out_ch:分类个数（包含0）
        super(FPN_MultiScaleConv_UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = MultiScaleConvBlock(in_ch, filters[0])
        self.Conv2 = MultiScaleConvBlock(filters[0], filters[1])
        self.Conv3 = MultiScaleConvBlock(filters[1], filters[2])
        self.Conv4 = MultiScaleConvBlock(filters[2], filters[3])
        self.Conv5 = MultiScaleConvBlock(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = MultiScaleConvBlock(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = MultiScaleConvBlock(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = MultiScaleConvBlock(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = MultiScaleConvBlock(192, filters[0])

        self.fpn = FPN([64, 128, 256, 512])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # FPN
        up2 = self.fpn(e4, e3, e2, e1)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2,up2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)
        return out
    
#############################################
class FPN_MultiScaleConv_UNet2(nn.Module):

    def __init__(self, in_ch=4, out_ch=1):#in_ch:通道（波段）数；out_ch:分类个数（包含0）
        super(FPN_MultiScaleConv_UNet2, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = MultiScaleConvBlock(in_ch, filters[0])
        self.Conv2 = MultiScaleConvBlock(filters[0], filters[1])
        self.Conv3 = MultiScaleConvBlock(filters[1], filters[2])
        self.Conv4 = MultiScaleConvBlock(filters[2], filters[3])
        self.Conv5 = MultiScaleConvBlock(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(192, filters[0])

        self.fpn = FPN([64, 128, 256, 512])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # FPN
        up2 = self.fpn(e4, e3, e2, e1)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2,up2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #d1 = self.active(out)
        return out