import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)

        return x


class DFN(nn.Module):
    def __init__(self):
        super(DFN, self).__init__()

        #Path-TOP:
        self.conv1_Top = ConvBlock(1, 5)
        self.conv2_Top = ConvBlock(10, 5)
        self.conv3_Top = ConvBlock(20, 5)
        self.conv4_Top = ConvBlock(30, 5)
        self.conv5_Top = ConvBlock(40, 5)

        #Path-Bottom:
        self.conv1_Bottom = ConvBlock(1, 5)
        self.conv2_Bottom = ConvBlock(10, 5)
        self.conv3_Bottom = ConvBlock(20, 5)
        self.conv4_Bottom = ConvBlock(30, 5)
        self.conv5_Bottom = ConvBlock(40, 5)

    def forward(self, SST, ADT):
        #----------First Layer-----------
        y1t = self.conv1_Top(SST)
        y1b = self.conv1_Bottom(ADT)

        #---------Second Layer-----------
        # concatenate
        y2t_i = torch.cat((y1t, y1b), dim=1)
        y2b_i = torch.cat((y1b, y1t), dim=1)

        y2t_o = self.conv2_Top(y2t_i)
        y2b_o = self.conv2_Bottom(y2b_i)

        #---------Third Layer------------
        # concatenate
        y3t_i = torch.cat((y2t_i, y2t_o, y2b_o), dim=1)
        y3b_i = torch.cat((y2b_i, y2b_o, y2t_o), dim=1)

        y3t_o = self.conv3_Top(y3t_i)
        y3b_o = self.conv3_Bottom(y3b_i)

        #----------Fourth Layer-------------
        # concatenate
        y4t_i = torch.cat((y3t_i, y3t_o, y3b_o), dim=1)
        y4b_i = torch.cat((y3b_i, y3b_o, y3t_o), dim=1)

        y4t_o = self.conv4_Top(y4t_i)
        y4b_o = self.conv4_Bottom(y4b_i)

        #---------Five Layer---------------
        # concatenate
        y5t_i = torch.cat((y4t_i, y4t_o, y4b_o), dim=1)
        y5b_i = torch.cat((y4b_i, y4b_o, y4t_o), dim=1)

        y5t_o = self.conv5_Top(y5t_i)
        y5b_o = self.conv5_Bottom(y5b_i)

        #----------Fusion Layer-------------
        # concatenate
        y_fusion_t_i = torch.cat((y5t_i, y5t_o, y5b_o), dim=1)
        y_fusion_b_i = torch.cat((y5b_i, y5b_o, y5t_o), dim=1)
        #data fusion concatenate
        fusion = torch.cat((y_fusion_t_i, y_fusion_b_i), dim=1)

        return fusion


































