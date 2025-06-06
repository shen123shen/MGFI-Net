import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.InceptionNext import inceptionnext_tiny

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


from pytorch_wavelets import DWTForward



class MGFI_Net(nn.Module):
    def __init__(self, out_planes=1, encoder='inceptionnext_tiny'):
        super(MGFI_Net, self).__init__()
        self.encoder = encoder
        if self.encoder == 'inceptionnext_tiny':
            mutil_channel = [96, 192, 384, 768]
            self.backbone = inceptionnext_tiny()

        self.dropout = torch.nn.Dropout(0.3)  # 添加 Dropout
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.LFIU1 = LFIU(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)
        self.LFIU2 = LFIU(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)
        self.LFIU3 = LFIU(mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3], lenn=1)


        self.decoder4 = BasicConv2d(mutil_channel[3], mutil_channel[2], 3, padding=1)
        self.decoder3 = BasicConv2d(mutil_channel[2], mutil_channel[1], 3, padding=1)
        self.decoder2 = BasicConv2d(mutil_channel[1], mutil_channel[0], 3, padding=1)
        self.decoder1 = nn.Sequential(nn.Conv2d(mutil_channel[0], 64, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(64, out_planes, kernel_size=1, stride=1))

        self.fu1 = DSEM(96, 192,  96)
        self.fu2 = DSEM(192, 384, 192)
        self.fu3 = DSEM(384, 768,  384)
    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        x1, x2, x3, x4 = self.LFIU1(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.LFIU2(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.LFIU3(x1, x2, x3, x4)

        x_f_3 = self.fu3(x3, x4)
        x_f_2 = self.fu2(x2, x_f_3)
        x_f_1 = self.fu1(x1, x_f_2)

        d1 = self.decoder1(x_f_1)
        d1 = self.dropout(d1)
        d1 = F.interpolate(d1, scale_factor=4, mode='bilinear')
        return d1


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class WECA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(WECA, self).__init__()

        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*3, in_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_ch),
                                    nn.ReLU(inplace=True),
                                    )
        self.outconv_bn_relu_L = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_H = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_glb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.outconv_bn_relu_local = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.weight_conv_L = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1)
        self.weight_conv_H = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.cam = CAM_Module(out_ch)
    def forward(self, x):
        _, _, h, w = x.shape

        yL, yH = self.wt(x)

        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]

        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yH = self.conv_bn_relu(yH)

        yH_L = yH + yL
        yH_L_up = F.interpolate(yH_L, size=(h, w), mode='bilinear', align_corners=True)
        output = self.cam(yH_L_up)


        return output

class DSEM(nn.Module):
    def __init__(self, l_dim, g_dim, out_dim):
        super(DSEM,self).__init__()
        self.extra_l = MFE(l_dim)
        self.bn = nn.BatchNorm2d(out_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3x3 = BasicConv2d(g_dim, out_dim, 3, padding=1)
        self.selection = nn.Conv2d(out_dim, 1, 1)
        self.conv3x3_1 = BasicConv2d(out_dim, 2, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = BasicConv2d(out_dim*2, out_dim, 1, padding=0)
        self.conv1x1 = BasicConv2d(out_dim*2, out_dim, 1, padding=0)
        self.weca = WECA(out_dim, out_dim)
    def forward(self,l,g):
        l = self.extra_l(l)
        g = self.conv3x3(self.upsample(g))
        fuse = self.proj(torch.cat([l, g], dim=1))
        fuse = self.weca(fuse)
        att = self.conv3x3_1(fuse)
        att = F.softmax(att, dim=1)

        att_1 = att[:, 0, :, :].unsqueeze(1)
        att_2 = att[:, 1, :, :].unsqueeze(1)
        output = att_1 * l + att_2 * g
        output = self.conv1x1(torch.cat([output, g], dim=1))
        return output

class MFE(nn.Module):

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3,],
                 channel_split=[1, 3, 4,],
                ):
        super(MFE, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x




def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
            self,
            num_in_filters,
            num_out_filters,
            kernel_size,
            stride=(1, 1),
            activation="LeakyReLU",
    ):

        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            groups=8

        )
        self.num_in_filters = num_in_filters
        self.num_out_filters = num_out_filters
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = channel_shuffle(x, gcd(self.num_in_filters, self.num_out_filters))
        x = self.conv1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.sqe(x)


class Conv2d_channel(torch.nn.Module):


    def __init__(self, num_in_filters, num_out_filters):

        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))


class ChannelSELayer(torch.nn.Module):

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()

        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)

    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.fc2(out)

        out = self.sigmoid(out)

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out

class CARAFE(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        N, C, H, W = x.size()

        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        # print("up shape:",out_tensor.shape)
        return out_tensor



class GSG(nn.Module):
    def __init__(self, in_channels, width=128, up_kwargs=None):
        super(GSG, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], in_channels[-1], 3, padding=1, groups=in_channels[-1]),
            Conv2d_batchnorm(in_channels[-1], width, 1))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2],in_channels[-2], 3, padding=1, groups=in_channels[-2], bias=False),
            Conv2d_batchnorm(in_channels[-2], width, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], in_channels[-3], 3, padding=1, groups=in_channels[-3], bias=False),
            Conv2d_batchnorm(in_channels[-3], width, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], in_channels[-4], 3, padding=1, groups=in_channels[-4], bias=False),
            Conv2d_batchnorm(in_channels[-4], width, 1))

        self.conv_out = nn.Sequential(
            Conv2d_batchnorm(4 * width, width, (1, 1)))

        self.up = CARAFE(width, width)
    def forward(self, x1, x2, x3, x4):
        feats = [
            self.conv5(x4),  # 768 -> 64
            self.conv4(x3),  # 384 -> 64
            self.conv3(x2),  # 192 -> 64
            self.conv2(x1)  # 96 -> 64
        ]

        _, _, h, w = feats[-1].size()
        feats[-2] = self.up(feats[-2])
        feats[-3] = self.up(self.up(feats[-3]))  # x3
        feats[-4] = self.up(self.up(self.up(feats[-4]))) # x4

        feat_1 = feats[-1]
        feat_2 = feats[-2]
        feat_3 = feats[-3]
        feat_4 = feats[-4]
        feat = torch.cat([feats[0], feats[1], feats[2], feats[3]],1)
        feat = self.conv_out(feat)
        return feat,feat_1,feat_2,feat_3,feat_4


class LFIU(torch.nn.Module):

    def __init__(self, in_filters1, in_filters2, in_filters3, in_filters4, width=96, lenn=1):


        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.in_filters4 = in_filters4
        self.in_filters = (
                in_filters1 + in_filters2 + in_filters3 + in_filters4
        )  # total number of channels
        self.in_filters3_4 = (
                in_filters3 + in_filters4
        )
        self.in_filters2_3_4 = (
                in_filters2 + in_filters3 + in_filters4
        )

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling

        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.cnv_blks1 = torch.nn.ModuleList([])
        self.cnv_blks2 = torch.nn.ModuleList([])
        self.cnv_blks3 = torch.nn.ModuleList([])
        self.cnv_blks4 = torch.nn.ModuleList([])

        self.cnv_mrg1 = torch.nn.ModuleList([])
        self.cnv_mrg2 = torch.nn.ModuleList([])
        self.cnv_mrg3 = torch.nn.ModuleList([])
        self.cnv_mrg4 = torch.nn.ModuleList([])

        self.bns1 = torch.nn.ModuleList([])
        self.bns2 = torch.nn.ModuleList([])
        self.bns3 = torch.nn.ModuleList([])
        self.bns4 = torch.nn.ModuleList([])

        self.bns_mrg1 = torch.nn.ModuleList([])
        self.bns_mrg2 = torch.nn.ModuleList([])
        self.bns_mrg3 = torch.nn.ModuleList([])
        self.bns_mrg4 = torch.nn.ModuleList([])

        for i in range(lenn):

            self.cnv_blks1.append(
                Conv2d_batchnorm(width*4, in_filters1, (1, 1))
            )

            self.cnv_mrg1.append(Conv2d_batchnorm(in_filters1, in_filters1, (1, 1)))

            self.bns1.append(torch.nn.BatchNorm2d(in_filters1))

            self.bns_mrg1.append(torch.nn.BatchNorm2d(in_filters1))
            self.cnv_blks2.append(
                Conv2d_batchnorm(width*3, in_filters2, (1, 1))
            )
            self.cnv_mrg2.append(Conv2d_batchnorm(2 * in_filters2, in_filters2, (1, 1)))
            self.bns2.append(torch.nn.BatchNorm2d(in_filters2))
            self.bns_mrg2.append(torch.nn.BatchNorm2d(in_filters2))

            self.cnv_blks3.append(
                Conv2d_batchnorm(width*2, in_filters3, (1, 1))
            )
            self.cnv_mrg3.append(Conv2d_batchnorm(2 * in_filters3, in_filters3, (1, 1)))
            self.bns3.append(torch.nn.BatchNorm2d(in_filters3))
            self.bns_mrg3.append(torch.nn.BatchNorm2d(in_filters3))

            self.cnv_blks4.append(
                Conv2d_batchnorm(width, in_filters4, (1, 1))
            )

            self.bns4.append(torch.nn.BatchNorm2d(in_filters4))
            self.bns_mrg4.append(torch.nn.BatchNorm2d(in_filters4))
        self.act = torch.nn.LeakyReLU()

        self.sqe1 = ChannelSELayer(in_filters1)
        self.sqe2 = ChannelSELayer(in_filters2)
        self.sqe3 = ChannelSELayer(in_filters3)
        self.sqe4 = ChannelSELayer(in_filters4)

        self.dropout = nn.Dropout(0.3)
        self.fuse = GSG([in_filters1, in_filters2, in_filters3, in_filters4], width=width, up_kwargs=up_kwargs)

    def forward(self, x1, x2, x3, x4):

        batch_size, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        _, _, h3, w3 = x3.shape
        _, _, h4, w4 = x4.shape

        fuse, x1_1, x2_1, x3_1, x4_1 = self.fuse(x1, x2, x3, x4)
        fuse = self.dropout(fuse)

        for i in range(len(self.cnv_blks1)):
            x_c1 = torch.mul(x1_1, fuse)
            x_c2 = torch.mul(x2_1, fuse)
            x_c3 = torch.mul(x3_1, fuse)
            x_c4 = torch.mul(x4_1, fuse)

            x_c1 = self.act(
                self.bns_mrg1[i](
                    self.cnv_blks1[i](
                       torch.cat([x_c1, x_c2, x_c3, x_c4], dim=1)
                    )
                )
            )
            x_c2 = self.act(
                self.bns_mrg2[i](
                    self.cnv_blks2[i](
                        torch.cat([x_c2, x_c3, x_c4], dim=1)
                    )
                )
            )
            x_c3 = self.act(
                self.bns_mrg3[i](
                    self.cnv_blks3[i](
                        torch.cat([x_c3, x_c4], dim=1)
                    )
                )
            )
            x_c4 = self.act(
                self.bns_mrg4[i](
                    self.cnv_blks4[i](
                        x_c4
                    )
                )
            )
        x_c1 = self.sqe1(x_c1) + x1
        x_c2 = self.sqe2(x_c2)
        x_c3 = self.sqe3(x_c3)
        x_c4 = self.sqe4(x_c4)

        x1 = x_c1
        x2 = self.no_param_down(x_c2) + x2
        x3 = self.no_param_down(self.no_param_down(x_c3)) + x3
        x4 = self.no_param_down(self.no_param_down(self.no_param_down(x_c4))) + x4
        return x1, x2, x3, x4


import thop
if __name__ == '__main__':
    x=torch.randn(1, 3, 224, 224)
    model = MGFI_Net(out_planes=1, encoder='inceptionnext_tiny')
    MACs,Params = thop.profile(model,inputs = (x,),verbose=False)
    FLOPs = MACs * 2
    MACs, FLOPs, Params = thop.clever_format([MACs,FLOPs,Params],"%.3f")

    print(f"MACs:{MACs}")
    print(f"FLOPs:{FLOPs}")
    print(f"Params:{Params}")
    output = model(x)
    print(output.shape)

