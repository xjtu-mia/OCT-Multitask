import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from timm.models.layers import DropPath
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull-requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                             stride=stride, padding=small_kernel//2, groups=groups, dilation=1)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            #   add to the central part
            eq_k += nn.functional.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 4)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = get_conv2d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class Att(nn.Module):
    def __init__(self, channels, shape, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1
class FReLU(nn.Module):
    def __init__(self, c1, k=3):  # ch_in, kernel
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))
class AconC(nn.Module):
    r""" ACON activation (activate or not).
    # AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    # according to "Activate or Not: Learning Customized Activation" <https://arxiv.org/pdf/2009.04759.pdf>.
    """

    def __init__(self, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1, requires_grad=True),requires_grad=True)
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1, requires_grad=True),requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1, requires_grad=True),requires_grad=True)

    def forward(self, x):
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(self.beta * (self.p1 * x - self.p2 * x)) + self.p2 * x
class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x
    """

    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(width, max(4, width // r), kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(max(4, width // r), width, kernel_size=1, stride=1, bias=True)
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1,requires_grad=True), requires_grad=True)
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1,requires_grad=True), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        beta = torch.sigmoid(
            self.fc2(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x
class Acon_FReLU(nn.Module):
    r""" ACON activation (activate or not) based on FReLU:
    # eta_a(x) = x, eta_b(x) = dw_conv(x), according to
    # "Funnel Activation for Visual Recognition" <https://arxiv.org/pdf/2007.11824.pdf>.
    """
    def __init__(self, width, stride=1):
        super().__init__()
        self.stride = stride

        # eta_b(x)
        self.conv_frelu = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=width, bias=True)
        self.bn1 = nn.BatchNorm2d(width)

        # eta_a(x)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        if self.stride == 2:
            x1 = self.maxpool(x)
        else:
            x1 = x

        x2 = self.bn1(self.conv_frelu(x))

        return self.bn2( (x1 - x2) * self.sigmoid(x1 - x2) + x2 )
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(
            dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(
            dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding='same')

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.channel_pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(self.channel_pool(x))
        return out * self.sigmod(out)


class TripletAttention(nn.Module):
    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate()
        self.width_gate = SpatialGate()
        if self.spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return (1/3) * (x_out1 + x_out2 + x_out3)
        else:
            return (1/2) * (x_out1 + x_out2)
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.acon1 = MetaAconC(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.acon2 = MetaAconC(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.acon1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.acon2(self.bn2(out)))
        out += shortcut
        return out
class co_PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(co_PreActBottleneck, self).__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.acon1 = MetaAconC(in_planes)
        #self.acon1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.acon2 = MetaAconC(planes)
        #self.acon2 = nn.ReLU(inplace=True)
        self.CA = CoordAtt(planes, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(planes)
        self.acon3 = MetaAconC(planes)
        #self.acon3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            if stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=1, bias=False),
                    nn.MaxPool2d(2)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=1, bias=False)
                )
    def forward(self, x):
        out = self.bn1(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.bn2(out))
        if self.stride!=1:
            out = self.pool(out)
        out = self.conv3(self.CA(self.acon3(self.bn3(out))))
        out += shortcut
        return out
    '''def forward(self, x):
        out = self.acon1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.acon2(self.bn2(out)))
        if self.stride!=1:
            out = self.pool(out)
        out = self.conv3(self.acon3(self.bn3(out)))

        out += shortcut
        return out'''

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, stage=1, drop_path=0.1, save_mem=True):
        super(PreActBottleneck, self).__init__()
        self.stride = stride
        self.stage = stage
        self.save_mem = save_mem
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.acon1 = MetaAconC(in_planes)
        #self.acon1 = nn.GELU()


        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.acon2 = MetaAconC(planes)
        self.acon2 = nn.GELU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.pool = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(planes)
        self.acon3 = MetaAconC(planes)
        #self.acon3 = nn.GELU()
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            if stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=1, bias=False),
                    nn.MaxPool2d(2)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=1, bias=False)
                )
    def forward(self, x):
        if self.save_mem:
            out = self.bn1(x)
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = checkpoint.checkpoint(self.conv1, out)
            out = self.bn2(out)
            out = checkpoint.checkpoint(self.conv2, out)
            if self.stride!=1:
                out = self.pool(out)
            out = checkpoint.checkpoint(self.acon3, self.bn3(out))
            out = checkpoint.checkpoint(self.conv3, out)
            out = shortcut + out
            return out
        else:
            out = self.bn1(x)
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.conv2(out)
            if self.stride != 1:
                out = self.pool(out)
            out = self.conv3((self.acon3(self.bn3(out))))
            out += shortcut
            return out
    '''def forward(self, x):
        out = self.acon1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.acon2(self.bn2(out)))
        if self.stride!=1:
            out = self.pool(out)
        out = self.conv3(self.acon3(self.bn3(out)))

        out += shortcut
        return out'''
class MSCA_PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, stage=1, drop_path=0.1, save_mem=True):
        super(MSCA_PreActBottleneck, self).__init__()
        self.stride = stride
        self.stage = stage
        self.save_mem = save_mem
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.acon1 = MetaAconC(in_planes)
        #self.acon1 = nn.GELU()


        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.acon2 = MetaAconC(planes)
        self.conv2 = AttentionModule(planes)

        self.pool = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(planes)
        self.acon3 = MetaAconC(planes)
        #self.acon3 = nn.GELU()
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1)

        if stride != 1 or in_planes != self.expansion*planes:
            if stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=1),
                    nn.MaxPool2d(2)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=1)
                )
    def forward(self, x):
        if self.save_mem:
            out = self.bn1(x)
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = checkpoint.checkpoint(self.conv1, out)
            out = self.bn2(out)
            out = checkpoint.checkpoint(self.conv2, out)
            if self.stride != 1:
                out = self.pool(out)
            out = checkpoint.checkpoint(self.acon3, self.bn3(out))
            out = checkpoint.checkpoint(self.conv3, out)
            out = shortcut + out
            return out
        else:
            out = self.bn1(x)
            shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.conv2(out)
            if self.stride != 1:
                out = self.pool(out)
            out = self.conv3((self.acon3(self.bn3(out))))
            out += shortcut
            return out
    '''def forward(self, x):
        out = self.acon1(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.acon2(self.bn2(out)))
        if self.stride!=1:
            out = self.pool(out)
        out = self.conv3(self.acon3(self.bn3(out)))

        out += shortcut
        return out'''

from torch.utils import checkpoint
class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks,in_ch=1, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, stage=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, stage=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, stage=3)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, stage=4)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, stage):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, stage))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def PreActResNet18():
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50(in_ch=1):
    return PreActResNet(PreActBottleneck, [3,3,9,3], in_ch=in_ch)
def MSCA_PreActResNet50():
    return PreActResNet(MSCA_PreActBottleneck, [3,3,9,3])
def co_PreActResNet50():
    return PreActResNet(co_PreActBottleneck, [3,3,9,3])
def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])
