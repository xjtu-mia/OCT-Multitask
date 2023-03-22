
import cv2
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch
from padding_same_conv2d import Conv2d
import torch.nn.functional as F
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from backbone.shuffle_transformer import ShuffleTransformer
import math
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=0, b = 1, gamma = 2):
        super(eca_layer, self).__init__()
        if k_size == 0:
            k_size = int(math.log2(channel)/gamma + b/gamma)
            if k_size%2 == 0:
                k_size += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0., stride=False):
        super().__init__()
        self.stride = stride
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size=1, shuffle=False, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., relative_pos_embedding=False):
        super().__init__()
        self.num_heads = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.shuffle = shuffle

        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)

        if self.shuffle:
            q, k, v = rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                                qkv=3, ws1=self.ws, ws2=self.ws)
        else:
            q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                                qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        if self.shuffle:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (ws1 hh) (ws2 ww)', h=self.num_heads, b=b,
                            hh=h // self.ws, ws1=self.ws, ws2=self.ws)
        else:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b,
                            hh=h // self.ws, ws1=self.ws, ws2=self.ws)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class Block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, window_size=1, shuffle=False, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, stride=False,
                 relative_pos_embedding=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, relative_pos_embedding=relative_pos_embedding)
        self.local = Conv2d(dim, dim, window_size, 1, window_size // 2, groups=dim, bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer,
                       drop=drop, stride=stride)
        self.norm3 = norm_layer(dim)
        print("input dim={}, output dim={}, stride={}, expand={}, num_heads={}".format(dim, out_dim, stride, shuffle,
                                                                                       num_heads))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.local(self.norm2(x))  # local connection
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = nn.BatchNorm2d(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input dim={self.dim}, out dim={self.out_dim}"


class StageModule(nn.Module):
    def __init__(self, layers, dim, out_dim, num_heads, window_size=1, shuffle=True, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, relative_pos_embedding=False):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        if dim != out_dim:
            self.patch_partition1 = PatchMerging(dim, out_dim)
            self.patch_partition2 = PatchMerging(out_dim, dim)
        else:
            self.patch_partition = None

        num = layers // 2
        self.layers = nn.ModuleList([])
        for idx in range(num):
            the_last = (idx == num - 1)
            self.layers.append(nn.ModuleList([
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=False,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                      relative_pos_embedding=relative_pos_embedding),
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                      relative_pos_embedding=relative_pos_embedding)
            ]))

    def forward(self, x):
        if self.patch_partition1:
            x = self.patch_partition1(x)

        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        if self.patch_partition2:
            x = self.patch_partition2(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return x
# Conv2d默认padding='same'
'''class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x
    """

    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Conv2d(width, max(width // 16, 4), kernel_size=1, stride=1, bias=False)
        self.fc2 = nn.Conv2d(max(width // 16, 4), width, kernel_size=1, stride=1, bias=False)

        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1)
                               )
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, **kwargs):
        beta = self.sigmoid(self.fc2(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))
        return (self.p1 * x - self.p2 * x) * self.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

class singleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, act=False):
        super(singleConv, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(in_ch, out_ch, kernel, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.activation = act
        self.act = MetaAconC(out_ch)
        #self.act = nn.ReLU(inplace=True)
    def forward(self, input):
        out = checkpoint.checkpoint(self.conv, input)
        if self.activation:
            out = self.act(out)
        return out
class adept_singleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, act=False):
        super(adept_singleConv, self).__init__()
        self.bn_act = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            MetaAconC(in_ch))
            #nn.ReLU(inplace=True))
        self.conv = nn.Sequential(
            Conv2d(in_ch, out_ch, kernel, bias=False),
            nn.BatchNorm2d(out_ch))

        self.activation = act
        self.act = MetaAconC(out_ch)
        #self.act = nn.ReLU(inplace=True)
    def forward(self, input):
        out = checkpoint.checkpoint(self.conv, self.bn_act(input))
        if self.activation:
            out = self.act(out)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(in_ch, out_ch, kernel, bias=False),
            nn.BatchNorm2d(out_ch))
            #nn.ReLU(inplace=True),
        self.act1 = MetaAconC(out_ch)
        self.conv2 = nn.Sequential(
            Conv2d(out_ch, out_ch, kernel, bias=False),
            nn.BatchNorm2d(out_ch))
        self.act2 = MetaAconC(out_ch)  #remove
            #nn.ReLU(inplace=True)


    def forward(self, input):
        out = self.act1(checkpoint.checkpoint(self.conv1, input))
        out = self.act2(checkpoint.checkpoint(self.conv2, out))
        return out

class upConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            Conv2d(in_ch, out_ch, 2, bias=False),
            nn.BatchNorm2d(out_ch),

            #nn.ReLU(inplace=True),
        )

        self.act = MetaAconC(out_ch)
    def forward(self, input):
        return self.act(checkpoint.checkpoint(self.conv, input))

class AM(nn.Module):
    def __init__(self, in_ch, out_ch, h=None, w=None):
        super(AM, self).__init__()
        self.conv1 = singleConv(in_ch, in_ch, 1, act=False)
        self.act1 = MetaAconC(in_ch)
        #self.act1 = nn.ReLU(inplace=True)
        self.conv2 = singleConv(in_ch, in_ch, 1, act=False)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = singleConv(in_ch, out_ch, 3, act=False)
        self.conv_r = singleConv(in_ch, out_ch, 1, act=True)
        self.act2 = MetaAconC(out_ch)
        #self.act2 = nn.ReLU(inplace=True)

    def forward(self, input_x, input_y):
        # x = torch.cat([self.ECA1(input_x), self.ECA2(input_y)], dim=1)
        x = torch.cat([input_x, input_y], dim=1)
        y = checkpoint.checkpoint(self.conv1, x)
        y = self.act1(y)
        y = checkpoint.checkpoint(self.conv2, y)
        y = self.sigmoid(y)
        y = torch.mul(x, y)
        # y = self.ECA1(y)
        y = checkpoint.checkpoint(self.conv3, y)

        y = self.act2(y)
        return y'''
class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x
    """

    def __init__(self, width, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(width, max(4, width // r), kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(max(4, width // r), width, kernel_size=1, stride=1, bias=True)
        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1,requires_grad=True),requires_grad=True)
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1,requires_grad=True),requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Hardtanh()
    def forward(self, x):
        beta = torch.sigmoid(
            self.fc2(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))
        return (self.p1 * x - self.p2 * x) * torch.sigmoid(beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

import math
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=0, b = 1, gamma = 2):
        super(eca_layer, self).__init__()
        if k_size == 0:
            k_size = int(math.log2(channel)/gamma + b/gamma)
            if k_size%2 == 0:
                k_size += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


class singleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, act=False, save_mem=True, function='gelu', norm='bn'):
        self.save_mem = save_mem
        self.activation = act
        super(singleConv, self).__init__()
        if norm == 'bn':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=(kernel - 1) // 2, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        elif norm == None:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=(kernel - 1) // 2, bias=True)
        if function == 'acon':
            self.act = MetaAconC(out_ch)
        else:
            self.act = nn.GELU()
    def forward(self, input):
        if self.save_mem:
            out = checkpoint.checkpoint(self.conv, input)
        else:
            out = self.conv(input)
        if self.activation:
            if self.save_mem:
                out = checkpoint.checkpoint(self.act, out)
                #out = self.act(out)
            else:
                out = self.act(out)
        return out
class adept_singleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, act=False, save_mem=True):
        self.save_mem = save_mem
        super(adept_singleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            MetaAconC(in_ch),
            #nn.GELU(),
            nn.Conv2d(in_ch, out_ch, kernel, padding='same', bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.activation = act
        self.act = MetaAconC(out_ch)
        #self.act = nn.GELU()
    def forward(self, input):
        if self.save_mem:
            out = checkpoint.checkpoint(self.conv, input)
        else:
            out = self.conv(input)
        if self.activation:
            if self.save_mem:
                out = checkpoint.checkpoint(self.act, out)
                #out = self.act(out)
            else:
                out = self.act(out)
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, save_mem=True):
        self.save_mem = save_mem
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=(kernel - 1) // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            #nn.GELU(),
            MetaAconC(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel, padding=(kernel - 1) // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            #MetaAconC(out_ch)  #remove
            nn.GELU()
        )

    def forward(self, input):
        if self.save_mem:
            return checkpoint.checkpoint(self.conv, input)
        else:
            return self.conv(input)

class upConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=2, act=True,save_mem=True,transconv=False, scale_factor=2):
        super(upConv, self).__init__()
        self.save_mem = save_mem
        self.activation = act
        if transconv:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, k, stride=2, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, k, bias=False, padding='same'),
                nn.BatchNorm2d(out_ch)
            )
        #self.act = MetaAconC(out_ch)
        self.act = nn.GELU()

    def forward(self, input):

        if self.save_mem:
            x = checkpoint.checkpoint(self.conv, input)
            if self.activation:
                x = checkpoint.checkpoint(self.act, x)
        else:
            x = self.conv(input)
            if self.activation:
                x = self.act(x)
        return x


class AM(nn.Module):
    def __init__(self, in_ch, out_ch, h=None, w=None, save_mem=True):
        self.save_mem = save_mem
        super(AM, self).__init__()
        self.conv1 = singleConv(in_ch, in_ch, 1, act=False)
        #self.act1 = MetaAconC(in_ch)
        self.act1 = nn.GELU()
        self.conv2 = singleConv(in_ch, in_ch, 1, act=False)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = singleConv(in_ch, out_ch, 3, act=False)
        self.act2 = MetaAconC(out_ch)
        #self.act2 = nn.GELU()

    def forward(self, input_x, input_y):
        if self.save_mem:
            x = torch.cat([input_x, input_y], dim=1)
            y = checkpoint.checkpoint(self.conv1, x)
            #y = self.act1(y)
            y = checkpoint.checkpoint(self.act1, y)
            y = checkpoint.checkpoint(self.conv2, y)
            y = self.sigmoid(y)
            y = torch.mul(x, y)
            # y = self.ECA1(y)
            y = checkpoint.checkpoint(self.conv3, y)
            #y = self.act2(y)
            y = checkpoint.checkpoint(self.act2, y)
        else:
            x = torch.cat([input_x, input_y], dim=1)
            y = self.conv1(x)
            y = self.act1(y)
            y = self.conv2(y)
            y = self.sigmoid(y)
            y = torch.mul(x, y)
            y = self.conv3(y)
            y = self.act2(y)
        return y


class masks_from_regression:
    def __init__(self, input, eta=1, dim=2, device='cuda', add=False, convexhull=True):
        self.input = input
        self.eta = eta
        self.dim = dim
        self.device = device
        self.add = add
        self.convexhull = convexhull

    def posission_constraint(self, input):
        c = input.shape[1]
        x = input
        for i in range(c - 1):
            x[:, i + 1, :] = x[:, i, :] + nn.functional.relu(x[:, i + 1, :] - x[:, i, :])
            x# [:, i + 1, :] = x[:, i, :] + nn.functional.softplus(x[:, i + 1, :] - x[:, i, :])
        return x

    def ConvexHull(self, img):
        img = img.astype(np.uint8)
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        h, w = img.shape
        # 图片轮廓
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        # 寻找凸包并绘制凸包（轮廓）
        hull = cv2.convexHull(cnt)
        length = len(hull)
        im = np.zeros_like(img)
        for m in range(len(hull)):
            cv2.line(im, tuple(hull[m][0]), tuple(hull[(m + 1) % length][0]), 255, 1, lineType=cv2.LINE_AA)
        y_index = np.zeros(im.shape[1])
        for m in range(im.shape[1]):
            y_index[m] = np.argmax(im[1:-1, m])
        return y_index

    def run(self):
        device = torch.device(self.device)
        b, c, h, w = self.input.shape
        x = nn.functional.softmax(self.eta * self.input, dim=self.dim).cpu()
        indices = torch.unsqueeze(torch.linspace(0, h - 1, h), dim=0)
        y = torch.zeros((b, c, w))
        if self.add:
            mask = torch.ones((b, c - 1, h, w)) * -1
        else:
            mask = torch.zeros((b, c - 1, h, w))
        BM_img = np.zeros((b, h, w))
        act = True
        avg_th = 0

        for i in range(b):
            for k in range(c):
                y[i, k, :] = torch.mm(indices, x[i, k, :, :])
        y = self.posission_constraint(y)  # 保证层之间不会出现拓扑错误
        # print(torch.mean(y, dim=(0, 2)), c)
        for i in range(b):
            for k in range(c - 1):
                top_index = torch.clamp(torch.round(y[i, k, :]), 0, h-1)
                bottom_index = torch.clamp(torch.round(y[i, k + 1, :]), 0, h-1)
                '''if torch.isnan(bottom_index).any() or torch.isnan(top_index).any():
                    continue'''
                if self.convexhull:
                    for j in range(w):
                        # print( BM_img[i, 0: int(bottom_index[j]), int(j)])
                        BM_img[i, 0: int(bottom_index[j]), int(j)] = 255
                    if k == 5:
                        bottom_index[:] = torch.tensor(self.ConvexHull(BM_img[i, :, :]))

                for j in range(w):
                    pixel_top = max(int(top_index[j].item()), 0)
                    pixel_bottom = int(bottom_index[j].item())
                    avg_th = avg_th + (pixel_bottom - pixel_top)
                    if pixel_bottom - pixel_top > 1:
                        mask[i, k, pixel_top:pixel_bottom, j] = torch.ones((1, pixel_bottom - pixel_top))
                        # print(mask[i, 0, pixel_top:pixel_bottom, j])

        avg_th = avg_th / b / w
        if avg_th > 40:
            act = True
        else:
            act = True
        print(avg_th, act)
        return mask.to(device), act


def softargmax2d_col(input, eta=1, dim=2, device='cuda', add=False):
    def ConvexHull(img):
        img = img.astype(np.uint8)
        if len(img.shape) > 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        h, w = img.shape
        # 图片轮廓
        _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        # 寻找凸包并绘制凸包（轮廓）
        hull = cv2.convexHull(cnt)
        length = len(hull)
        im = np.zeros_like(img)
        for m in range(len(hull)):
            cv2.line(im, tuple(hull[m][0]), tuple(hull[(m + 1) % length][0]), 255, 1, lineType=cv2.LINE_AA)
        y_index = np.zeros(im.shape[1])
        for m in range(im.shape[1]):
            y_index[m] = np.argmax(im[1:-1, m])
        return y_index

    device = torch.device("cuda")
    b, c, h, w = input.shape
    x = nn.functional.softmax(eta * input, dim=dim).cpu()
    indices = torch.unsqueeze(torch.linspace(0, h - 1, h), dim=0)
    y = torch.zeros((b, c, w))
    if add:
        mask = torch.ones((b, 1, h, w)) * -1
    else:
        mask = torch.zeros((b, 1, h, w))
    BM_img = np.zeros((b, h, w))
    for i in range(b):
        for j in range(c):
            y[i, j, :, ] = torch.mm(indices, x[i, j, :, :])
    top_index = torch.round(y[:, 0, :])
    bottom_index = torch.round(y[:, -1, :])

    for i in range(b):
        for j in range(w):
            BM_img[i, 0: int(bottom_index[i, j]), int(j)] = 255
        bottom_index[i, :] = torch.tensor(ConvexHull(BM_img[i, :, :]))
    act = True
    avg_th = 0
    for i in range(b):
        for j in range(w):
            pixel_top = int(top_index[i, j].item())
            pixel_bottom = int(bottom_index[i, j].item())
            avg_th = avg_th + (pixel_bottom - pixel_top)
            if pixel_bottom - pixel_top > 0:
                mask[i, 0, pixel_top:pixel_bottom, j] = torch.ones((1, pixel_bottom - pixel_top))
                # print(mask[i, 0, pixel_top:pixel_bottom, j])
    avg_th = avg_th / b / w

    if avg_th > 20:
        act = True
    else:
        act = False

    print(avg_th, act)
    return mask.to(device), act

class tinyquadra_multi_resUnet_ly_masks_s3(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2, out_ch3, multitask=False, device='cuda', resize=(336,560), branch_num=4):
        super(tinyquadra_multi_resUnet_ly_masks_s3, self).__init__()
        self.multitask = multitask
        self.device = device
        self.resize = resize
        self.out_ch1 = out_ch1
        self.device = device
        self.branch_num = branch_num
        '''self.conv1 = DoubleConv(in_ch, 32, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(128, 256, 3)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(256, 512, 3)'''
        #encoder = PreActResNet50()
        #encoder = co_PreActResNet50()
        encoder = ShuffleTransformer(img_size=resize, window_size=7,in_chans=in_ch)
        pretext_model = torch.load('/home/wanghualin/python_project/OCT/OCT_fluid_layer_seg/pytorch_ver/backbone/shuffle_transformer_T.pth')
        model2_dict = encoder.state_dict()
        state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
        model2_dict.update(state_dict)
        encoder.load_state_dict(model2_dict)
        '''self.trans1 = StageModule(2, 64, 32, 2, window_size=16,drop_path=0.1)
        self.trans2 = StageModule(2, 256, 64, 4, window_size=16,drop_path=0.1)
        self.trans3 = StageModule(2, 512, 128, 8, window_size=16,drop_path=0.1)
        self.trans4 = StageModule(2, 1024, 256, 16, window_size=16, drop_path=0.1)
        self.trans5 = StageModule(2, 2048, 512, 32, window_size=8, drop_path=0.1)'''
        self.conv1 = encoder.to_token1
        self.adp_conv1 = singleConv(96, 32, 1, act=False)
        self.pool1 = nn.MaxPool2d(2)
        #self.pool1_ = encoder.to_token1
        #self.pool1_ = singleConv(32, 96, 1, stride=2, act=True)
        #self.conv2 = encoder.layer1

        self.conv2 = encoder.stage1
        self.adp_conv2 = singleConv(96, 64, 1, act=True)
        #self.conv3 = encoder.layer2

        self.conv3 = encoder.stage2
        self.adp_conv3 = singleConv(192, 128, 1, act=True)
        #self.conv4 = encoder.layer3
        self.conv4 = encoder.stage3
        self.adp_conv4 = singleConv(384, 256, 1, act=True)
        #self.conv5 = encoder.layer4
        self.conv5 = encoder.stage4
        self.adp_conv5 = singleConv(768, 512, 1, act=True)

        self.drop = nn.Dropout(0.1)

        self.up1 = upConv(512, 256)
        self.conv6 = DoubleConv(512, 256, 3)
        self.up2 = upConv(256, 128)
        self.conv7 = DoubleConv(256, 128, 3)
        self.up3 = upConv(128, 64)
        self.conv8 = DoubleConv(128, 64, 3)
        self.up4 = upConv(64, 32)
        self.conv9 = DoubleConv(64, 32, 3)
        self.conv9_eca = eca_layer(32)
        self.conv9_m = singleConv(32, 8, 3, act=True, function='gelu')
        self.conv9_mm = singleConv(32 + 8 * (out_ch1 - 1), 32, 1, act=True, function='acon')
        self.conv8_eca = eca_layer(64)
        self.conv8_m = singleConv(64, 16, 3, act=True, function='gelu')
        self.conv8_mm = singleConv(64 + 16 * (out_ch1 - 1), 64, 1, act=True, function='acon')
        self.conv7_eca = eca_layer(128)
        self.conv7_m = singleConv(128, 32, 3, act=True, function='gelu')
        self.conv7_mm = singleConv(128 + 32 * (out_ch1 - 1), 128, 1, act=True, function='acon')
        self.conv6_eca = eca_layer(256)
        self.conv6_m = singleConv(256, 64, 3, act=True, function='gelu')
        self.conv6_mm = singleConv(256 + 64 * (out_ch1 - 1), 256, 1, act=True, function='acon')
        self.conv5_eca = eca_layer(512)
        self.conv5_m = singleConv(512, 128, 3, act=True, function='gelu')
        self.conv5_mm = singleConv(512 + 128 * (out_ch1 - 1), 512, 1, act=True, function='acon')
        self.pool5 = nn.MaxPool2d(8)
        self.up5 = upConv(512, 256)
        self.AM1 = AM(512, 256)
        self.up6 = upConv(256, 128)
        self.AM2 = AM(256, 128)
        self.up7 = upConv(128, 64)
        self.AM3 = AM(128, 64)
        '''self.irf_gate = eca_layer(64)
        self.up8_1 = upConv(64, 32)'''
        self.AM4 = AM(64, 64)
        self.up8 = upConv(64, 32)
        self.up8_1 = upConv(64, 32)
        self.AM4_1 = nn.Sequential(
            singleConv(32, 32, 3, act=False),
            nn.GELU(),
            singleConv(32, 32, 3, act=False),
            nn.GELU()
        )
        self.up9 = upConv(512, 256)
        self.AM5 = AM(512, 256)
        self.up10 = upConv(256, 128)
        self.AM6 = AM(256, 128)
        self.up11 = upConv(128, 64)
        self.AM7 = AM(128, 64)
        self.up12 = upConv(64, 32)
        self.AM8 = AM(96, 64)
        self.conv10 = nn.Conv2d(32, out_ch1, 1, padding='same')
        self.conv10_1 = nn.Conv2d(64, out_ch1, 1, padding='same')
        self.conv11 = nn.Conv2d(64, out_ch2, 1, padding='same')
        if not multitask:
            self.conv11_1 = nn.Conv2d(32, out_ch2, 1, padding='same')
        self.conv12 = nn.Conv2d(32, out_ch3, 1, padding='same')

    def forward(self, x, act=True):
        '''c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)'''

        if self.resize:
            h1, w1 = x.shape[2], x.shape[3]
            x = F.interpolate(x, self.resize, mode='nearest')
        c1 = checkpoint.checkpoint(self.conv1, x)
        adp_conv1 = self.adp_conv1(c1)
        c1 = self.pool1(c1)
        # c1 = checkpoint.checkpoint(self.self.trans1, c1)
        # print(c1.shape)
        c2 = checkpoint.checkpoint(self.conv2, c1)
        adp_conv2 = self.adp_conv2(c2)
        # c2 = self.trans2(c2)
        c3 = checkpoint.checkpoint(self.conv3, c2)
        adp_conv3 = self.adp_conv3(c3)
        # c3 = self.trans3(c3)
        c4 = self.conv4(c3)
        adp_conv4 = self.adp_conv4(c4)

        # c4 = self.trans4(c4)
        c5 = self.conv5(c4)
        # c5 = self.trans5(c5)
        c5 = self.drop(c5)
        adp_conv5 = self.adp_conv5(c5)
        # adp_conv5 = self.ASPP(adp_conv5)

        up_1 = self.up1(adp_conv5)
        merge1 = torch.cat([up_1, adp_conv4], dim=1)
        c6 = self.conv6(merge1)
        up_2 = self.up2(c6)
        merge2 = torch.cat([up_2, adp_conv3], dim=1)
        c7 = self.conv7(merge2)
        up_3 = self.up3(c7)
        # print(up_3.shape)
        merge3 = torch.cat([up_3, adp_conv2], dim=1)
        c8 = self.conv8(merge3)
        up_4 = self.up4(c8)
        # print(up_4.shape, adp_conv1.shape)
        merge4 = torch.cat([up_4, adp_conv1], dim=1)
        c9 = self.conv9(merge4)
        c10 = self.conv10(c9)
        '''c1_1 = self.conv1_1(x).relu()
        c1 = torch.cat([c1, c1_1], dim=1)'''

        if self.multitask:

            mask_irf = 0
            masks, act = masks_from_regression(c10, eta=1, dim=2, device=self.device, add=False).run()
            for i in range(self.out_ch1 - 1):
                mask_irf += masks[:, i, :, :]
            # self.gussconv.weight = nn.Parameter(self.w1, requires_grad=True)
            # print(self.gussconv.weight)
            mask_irf = torch.unsqueeze(mask_irf, dim=1)

            c9_ = c9
            c8_ = c8
            c7_ = c7
            c6_ = c6
            c5_ = adp_conv5

            for i in range(c10.shape[1] - 1):
                mask_1 = torch.unsqueeze(masks[:, i, :, :], dim=1)
                tmp = self.conv9_eca(torch.mul(c9, mask_1) if act else c9)

                c9_m = self.conv9_m(tmp)
                c9_ = torch.cat([c9_, c9_m], dim=1)
                # print(c9_.shape)
                mask_2 = self.pool1(mask_1)
                c8_m = self.conv8_m(self.conv8_eca(torch.mul(c8, mask_2) if act else c8))
                c8_ = torch.cat([c8_, c8_m], dim=1)
                mask_3 = self.pool1(mask_2)

                c7_m = self.conv7_m(self.conv7_eca(torch.mul(c7, mask_3) if act else c7))
                c7_ = torch.cat([c7_, c7_m], dim=1)
                mask_4 = self.pool1(mask_3)
                # self.featuremap1 = torch.mul(c7, mask_3)
                c6_m = self.conv6_m(self.conv6_eca(torch.mul(c6, mask_4) if act else c6))
                c6_ = torch.cat([c6_, c6_m], dim=1)
                mask_5 = self.pool1(mask_4)
                c5_m = self.conv5_m(self.conv5_eca(torch.mul(adp_conv5, mask_5) if act else adp_conv5))
                c5_ = torch.cat([c5_, c5_m], dim=1)
                # print(c5_.shape)

            c9 = self.conv9_mm(self.drop(c9_))
            c8 = self.conv8_mm(self.drop(c8_))
            c7 = self.conv7_mm(self.drop(c7_))
            c6 = self.conv6_mm(self.drop(c6_))
            c5 = self.conv5_mm(self.drop(c5_))
            # c5 = adp_conv5
            up_5 = self.up5(c5)
            a1 = self.AM1(up_5, c6)
            up_6 = self.up6(a1)
            a2 = self.AM2(up_6, c7)
            up_7 = self.up7(a2)
            a3 = self.AM3(up_7, c8)
            up_8 = self.up8(a3)
            up_8_1 = self.up8_1(a3)
            a4 = self.AM4(up_8, c9)
            a4_1 = self.AM4_1(torch.mul(up_8_1, mask_irf) if act else up_8_1)
            # a4_1 = self.AM4_1(up_8_1)
            # a4_1 = self.AM4_1(torch.mul(torch.cat([up_8_1, adp_conv1], dim=1), mask_irf) if act else torch.cat(up_8, adp_conv1))
            if self.branch_num == 4:
                up_9 = self.up9(c5)
                a5 = self.AM5(up_9, a1)
                up_10 = self.up10(a5)
                a6 = self.AM6(up_10, a2)
                up_11 = self.up11(a6)
                a7 = self.AM7(up_11, a3)
                up_12 = self.up12(a7)
                a8 = self.AM8(up_12, a4)
                # self.featuremap1 = a8
                # print(self.featuremap1.shape)
                c10_1 = self.conv10_1(a8)

                self.featuremap1 = torch.cat([c10_1, x * 0.5 + 0.5], dim=1)
            c12 = self.conv11(a4)
            c13 = self.conv12(a4_1)
            # c13 = torch.cat([torch.unsqueeze(c12[:, 0, :, :], dim=1), torch.unsqueeze(c12[:, -1, :, :], dim=1)], dim=1)
            '''
            c11 = self.conv10(a9)
            c12_bk = torch.unsqueeze(c12[:,0 ,:,:], dim=1)
            c12_fl = torch.unsqueeze(c12[:,-1 ,:,:], dim=1)'''
            if self.resize:
                c10_1 = F.interpolate(c10_1, (h1, w1), mode='nearest')
                c10 = F.interpolate(c10, (h1, w1), mode='nearest')
                c12 = F.interpolate(c12, (h1, w1), mode='nearest')
                c13 = F.interpolate(c13, (h1, w1), mode='nearest')
            if self.branch_num == 4:
                # return c10_1, c10, c12
                return c10_1, c10, c12, c13
            elif self.branch_num == 3:
                return c10, c12, c13
        else:
            # out = nn.Sigmoid()(c10) self.output(c10)
            c12 = self.conv11_1(c9)
            c13 = self.conv12(c9)
            return c10, c12, c13

if __name__ == '__main__':
    from torchsummary import summary
    model = tinyquadra_multi_resUnet_ly_masks(1,7,7,2, device='cpu')
    summary(model, (1,224,224), device='cpu')