import torch.nn as nn
import torch
import torch.nn.functional as F

#----------------- CBAM-----------------------
## channel attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x)) #self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out =self.shared_MLP(self.max_pool(x)) #self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

## spatial attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

## CBAM
class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x  #  执行通道注意力机制，并为通道赋予权重
        x = self.sa(x) * x  #  执行空间注意力机制，并为通道赋予权重

        return x

#----------------------------CNN_CBAM Block---------------------------------------------------------------------------------------------------------------

class CNN_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, pad):
        super(CNN_CBAM, self).__init__()

        self.conv_cbam = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.cbam = CBAM(out_channels)
        self.act_cbam = nn.ReLU()

    def forward(self, x):
        x = self.conv_cbam(x)
        x = self.cbam(x)
        x = self.act_cbam(x)

        return x

#-------------------------ResNet_CBAM Block--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride, pad):
        super(Resblock, self).__init__()

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.Attention = CBAM(out_channels)

        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv_bn_relu(x)
        x = self.Attention(x)
        identity = self.shortcut(identity)
        x = x + identity
        x = self.activation(x)

        return x

#----------------------------Center Block------------------------------------------------------------------------------------------------------------------------------------------------------

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)


    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x))
        dilate2_out = F.relu(self.dilate2(dilate1_out))
        dilate3_out = F.relu(self.dilate3(dilate2_out))
        dilate4_out = F.relu(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out

#-------------------------------FEN--------------------------------------------------------------------------------------------

class FEN(nn.Module):
    def __init__(self):
        super(FEN, self).__init__()
        # 输入部分
        self.input = CNN_CBAM(in_channels=100, out_channels=100, k_size=3, stride=1, pad=1) #要做concat
        self.pool_input = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器部分
        self.encoder1_1 = Resblock(in_channels=100, out_channels=200, k_size=3, stride=1, pad=1)
        self.encoder1_2 = Resblock(in_channels=200, out_channels=200, k_size=3, stride=1, pad=1) #要做concat
        self.pool_encoder1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2_1 = Resblock(in_channels=200, out_channels=400, k_size=3, stride=1, pad=1)
        self.encoder2_2 = Resblock(in_channels=400, out_channels=400, k_size=3, stride=1, pad=1) #要做concat
        self.pool_encoder2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3_1 = Resblock(in_channels=400, out_channels=800, k_size=3, stride=1, pad=1)
        self.encoder3_2 = Resblock(in_channels=800, out_channels=800, k_size=3, stride=1, pad=1)

        #中间部分
        self.dblock = Dblock(800)

        #解码器部分
        self.upconv_1 = nn.ConvTranspose2d(in_channels=800, out_channels=400, kernel_size=2, stride=2) #要做concat
        self.encoder4_1 = Resblock(in_channels=800, out_channels=400, k_size=3, stride=1, pad=1)
        self.encoder4_2 = Resblock(in_channels=400, out_channels=400, k_size=3, stride=1, pad=1)

        self.upconv_2 = nn.ConvTranspose2d(in_channels=400, out_channels=200, kernel_size=2, stride=2) #要做concat
        self.encoder5_1 = Resblock(in_channels=400, out_channels=200, k_size=3, stride=1, pad=1)
        self.encoder5_2 = Resblock(in_channels=200, out_channels=200, k_size=3, stride=1, pad=1)

        self.upconv_3 = nn.ConvTranspose2d(in_channels=200, out_channels=100, kernel_size=2, stride=2) #要做concat
        self.encoder6_1 = Resblock(in_channels=200, out_channels=100, k_size=3, stride=1, pad=1)
        self.encoder6_2 = Resblock(in_channels=100, out_channels=100, k_size=3, stride=1, pad=1)

        #输出部分
        self.output_1 = CNN_CBAM(in_channels=100, out_channels=50, k_size=3, stride=1, pad=1)
        self.output_2 = CNN_CBAM(in_channels=50, out_channels=25, k_size=3, stride=1, pad=1)

        self.final_output = nn.Conv2d(in_channels=25, out_channels=1, kernel_size=1)

    def forward(self, x):
        #编码器
        out_input = self.input(x)     # 要做concat
        out_pool_input = self.pool_input(out_input)

        out_encoder1_1 = self.encoder1_1(out_pool_input)
        out_encoder1_2 = self.encoder1_2(out_encoder1_1) #要做concat
        out_pool_1 = self.pool_encoder1(out_encoder1_2)

        out_encoder2_1 = self.encoder2_1(out_pool_1)
        out_encoder2_2 = self.encoder2_2(out_encoder2_1) #要做concat
        out_pool_2 =  self.pool_encoder2(out_encoder2_2)

        out_encoder3_1 = self.encoder3_1(out_pool_2)
        out_encoder3_2 = self.encoder3_2(out_encoder3_1)

        #中间
        out_Center = self.dblock(out_encoder3_2)

        #解码器
        out_upconv_1 = self.upconv_1(out_Center)
        concat_1 = torch.cat((out_encoder2_2, out_upconv_1), dim=1)
        out_encoder4_1 = self.encoder4_1(concat_1)
        out_encoder4_2 = self.encoder4_2(out_encoder4_1)

        out_upconv_2 = self.upconv_2(out_encoder4_2)
        concat_2 = torch.cat((out_encoder1_2, out_upconv_2), dim=1)
        out_encoder5_1 = self.encoder5_1(concat_2)
        out_encoder5_2 = self.encoder5_2(out_encoder5_1)

        out_upconv_3 = self.upconv_3(out_encoder5_2)
        concat_3 = torch.cat((out_input, out_upconv_3), dim=1)
        out_encoder6_1 = self.encoder6_1(concat_3)
        out_encoder6_2 = self.encoder6_2(out_encoder6_1)

        #输出
        out_output_1 = self.output_1(out_encoder6_2)
        out_output_2 = self.output_2(out_output_1)
        output = self.final_output(out_output_2)

        return torch.sigmoid(output)
























