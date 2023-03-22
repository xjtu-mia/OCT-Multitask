#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WMSELoss(nn.Module):
    def __init__(self, weight):
        super(WMSELoss, self).__init__()
        self.weight = weight
    def forward(self, inputs, targets):
        b, c, w = inputs.shape
        e = 1e-6
        norm_weight = self.weight / torch.sum(self.weight)
        var_x = torch.var(inputs, dim=(0, 2))
        var_y = torch.var(targets, dim=(0, 2))
        tmp = torch.abs(var_x - var_y) * norm_weight
        loss = torch.mean(tmp)
        return loss
class WMSE_SmoothL1Loss(nn.Module):
    def __init__(self, beta, weight, ratio=0.1, group = 16):
        super(WMSE_SmoothL1Loss, self).__init__()
        self.weight = weight
        self.beta = beta
        self.ratio = ratio
        self.group = group
    def forward(self, inputs, targets):
        b, c, w = inputs.shape
        e = 1e-6
        norm_weight = self.weight / torch.sum(self.weight)
        inputs = inputs.reshape(b, c, self.group, -1)
        targets = targets.reshape(b, c, self.group, -1)
        var_x = torch.mean(torch.var(inputs, dim=3), dim=(0, 2))
        var_y = torch.mean(torch.var(targets, dim=3), dim=(0,2))
        tmp = var_x * norm_weight
        loss = torch.mean(tmp)
        print(loss)
        return loss * self.ratio + F.smooth_l1_loss(inputs, targets, beta=self.beta)
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        #predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]

class tversky_coef_fun(nn.Module):
    def __init__(self, alpha, beta):
        super(tversky_coef_fun, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, 'predict & target shape do not match'
        epsilon = 1e-6
        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true
        # 求得每个sample的每个类的dice

        fn = torch.sum(p1 * g0, dim=(2, 3))
        fp = torch.sum(p0 * g1, dim=(2, 3))
        tp = torch.sum(p0 * g0, dim=(2, 3))
        den = torch.zeros_like(tp)
        for c in range(len(self.alpha)):

            den[:,c] = tp[:,c] + self.alpha[c] * fp[:,c] + self.beta[c] * fn[:,c]  # fn、fp可能写反了
        T = tp / (den + epsilon)  # [batchsize,class_num]
        T = torch.mean(T, dim=(0,1))
        return 1-T
class tversky_CEv2_loss_fun(nn.Module):
    def __init__(self, alpha, beta, w, gamma,ignored_class=None, mean_class=True):
        super(tversky_CEv2_loss_fun, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.gamma = gamma
        self.ignored_class = ignored_class
        self.mean_class = mean_class
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, 'predict & target shape do not match'
        epsilon = 1e-6
        #tversky loss

        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true

        # 求得每个sample的每个类的dice

        fn = torch.sum(p1 * g0, dim=(2, 3))
        fp = torch.sum(p0 * g1, dim=(2, 3))
        tp = torch.sum(p0 * g0, dim=(2, 3))
        den = torch.zeros_like(tp)
        for c in range(len(self.alpha)):
            den[:,c] = tp[:,c] + self.alpha[c] * fp[:,c] + self.beta[c] * fn[:,c]  # fn、fp可能写反了
        TL = tp / (den + epsilon)  # [batchsize,class_num]
        if self.ignored_class:
            TL = torch.mean(TL, dim=(0))
            for i in self.ignored_class:
                TL[i] = 1.
            TL = 1 - torch.mean(TL)
        else:
            if self.mean_class:
                TL = 1-torch.mean(TL, dim=(0,1))
            else:
                TL = 1 - torch.mean(TL, dim=0)
        #categorical_crossentropy loss
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)  # [bs,classes,x,y]
        p_loss = torch.zeros_like(y_pred)
        p_loss = (y_true * torch.log(y_pred))
        p_loss = torch.mean(p_loss, dim=(2, 3))
        for c in range(len(self.w)):
            p_loss[:, c] = p_loss[:, c] * self.w[c]
        if self.mean_class:
            CL = -torch.mean(p_loss, dim=1)
            CL = torch.mean(CL)
        else:
            CL = -torch.mean(p_loss, dim=0)
        #print(CL)
        #return TL + CL
        return self.gamma*TL + CL
'''
tversky_CEv2_loss_fun
def tversky_coef_fun(alpha, beta):
    def tversky_coef(y_true, y_pred):
        epsilon = 1e-6
        p0 = y_pred  # proba that voxels are class i
        p1 = 1 - y_pred  # proba that voxels are not class i
        g0 = y_true
        g1 = 1 - y_true
        # 求得每个sample的每个类的dice
        fn = K.sum(p1 * g0, axis=(1, 2))
        fp = K.sum(p0 * g1, axis=(1, 2))
        tp = K.sum(p0 * g0, axis=(1, 2))
       
        # 求得每个sample的每个类的dice
        # num = K.sum(p0 * g0, axis=( 1, 2))
        den = tp + alpha * fp + beta * fn + epsilon  # fn、fp可能写反了
        T = tp / den  # [batchsize,class_num]
        T = K.mean(T, axis=0)
        # 求得每个类的dice

        return K.mean(T, axis=-1)

    return tversky_coef
'''
class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, mean_class=True):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.mean_class = mean_class
    def forward(self, y_pred, y_true):
        diff = torch.abs(y_pred - y_true)
        cond = diff < self.beta
        loss = torch.where(cond, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        if self.mean_class:
            return torch.mean(loss)
        return torch.mean(loss, dim=(0,-1))

class categorical_crossentropy_v2(nn.Module):
    def __init__(self, w, mean_class=True):
        super(categorical_crossentropy_v2,self).__init__()
        self.w = w
        self.mean_class = mean_class
    def forward(self, y_pred, y_true):
        epsilon = 1e-6
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)  # [bs,classes,x,y]
        p_loss = torch.zeros_like(y_pred)
        if isinstance(self.w, list):
            for c in range(len(self.w)):
                p_loss[:,c,:,:] = (y_true * torch.log(y_pred))[:,c,:,:] * self.w[c]
        else:
            p_loss = y_true * torch.log(y_pred)
        p_loss = -torch.sum(p_loss,dim=2)
        if self.mean_class:
            loss = torch.mean(p_loss)
        else:
            loss = torch.mean(p_loss, dim=(0, -1))
        return loss

class pcloss(nn.Module):
    def __init__(self, w=1):
        super(pcloss,self).__init__()
        self.w = w
    def forward(self, y_pred, y_true):
        epsilon = 1e-6
        c = y_pred.shape[1]
        loss = 0
        for i in range(1,c,1):
            l = torch.square(torch.mean(nn.functional.relu(-(y_pred[:,i,:] - y_pred[:,i-1,:]))))
            loss=loss+l
        print(loss)
        return loss / (c-1) + nn.functional.smooth_l1_loss(y_pred, y_true)
'''
def categorical_crossentropy_v2(w=[0.1, 0.9, 0.9, 0.9, 0.9]):
    # weights = K.variable(w1,w2)
    # weights = (K.sum(y_true,axis=(0,1,2))+epsilon)
    epsilon =1e-6

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)  # [bs,x,y,classes]
        p_loss = -K.sum(y_true * K.log(y_pred) * w, axis=(-1))
        # n_loss = -K.sum((1-y_true) * K.log(1-y_pred) , axis=(-1))
        # loss = K.mean((w1*p_loss+w2*n_loss),axis=0)
        loss = K.mean(p_loss, axis=0)
        return loss

    return loss
'''
class index_diceloss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super(index_diceloss,self).__init__()
        self.reduction = reduction
    def forward(self, y_pred, y_true):
        eps = 1e-6
        b, ch, n = y_true.shape
        ious = torch.zeros((b, ch-1), dtype=torch.float)
        for c in range(ch-1):
            y1 = torch.where(y_pred[:,c] >= y_true[:,c], y_pred[:,c], y_true[:,c])
            y2 = torch.where(y_pred[:,c+1] <= y_true[:,c+1], y_pred[:,c+1], y_true[:,c+1])
            hi = (y2-y1).clamp(0.)
            hp = (y_pred[:,c+1] - y_pred[:,c]).clamp(0.)
            hg = (y_true[:,c+1] - y_true[:,c]).clamp(0.)
            #h = (y2 - y1 + 1.0).clamp(0.)
            inters = torch.sum(hi, dim=-1)
            #print("inters:\n", inters)
            uni = torch.sum(hp, dim=-1) + torch.sum(hg, dim=-1) + inters
            #print("uni:\n", uni)
            ious[:, c] = (2*inters / (uni+eps))
        loss = ious
        if self.reduction == 'mean':
            loss = -torch.mean(loss).log()
        elif self.reduction == 'sum':
            loss = -torch.sum(loss).log()
        else:
            raise NotImplementedError
        return loss

class PJcurvature_v2(nn.Module):
    def __init__(self, device=None, k_size=35):
        super(PJcurvature_v2,self).__init__()
        self.device = device
        self.kernel1 = torch.zeros((1, 1, k_size // 2 + 1), dtype=torch.float32).to(self.device)
        self.kernel2 = torch.zeros((1, 1, k_size), dtype=torch.float32).to(self.device)
        self.kernel1[0, 0, 0], self.kernel1[0, 0, -1] = 1., -1.
        self.kernel2[0, 0, 0], self.kernel2[0, 0, -1] = 1., 1.
        self.kernel2[0, 0, k_size // 2] = -2.

    def forward(self, index, target):
        epsilon = 1e-6
        loss = 0
        index = index[:, 4:, :]
        b, c, w = index.shape
        kappa = 0
        kernel2 = torch.repeat_interleave(self.kernel2, c, dim=0)
        kernel1 = torch.repeat_interleave(self.kernel1, c, dim=0)
        y1 = F.conv1d(index, kernel1, padding='valid', groups=c)
        y2 = F.conv1d(index, kernel2, padding='valid', groups=c)
        t1 = F.conv1d(target, kernel1, padding='valid', groups=c)
        t2 = F.conv1d(target, kernel2, padding='valid', groups=c)
        loss_diff1 = nn.L1Loss()(y1, t1)
        loss_diff2 = nn.L1Loss()(y2, t2)


        return (torch.mean(torch.abs(y2)))/5

class PJcurvature(nn.Module):
    def __init__(self, device=None, k_size=51):
        super(PJcurvature,self).__init__()
        self.device = device
        self.kernel1 = torch.zeros((1,k_size//2+1), dtype=torch.float32).to(self.device)
        self.kernel2 = torch.zeros((1,k_size), dtype=torch.float32).to(self.device)
        self.kernel1[0,0], self.kernel1[0,-1] = 1., -1.
        self.kernel2[0,0], self.kernel2[0,-1] = 1., 1.
        self.kernel2[0,k_size//2] = -2.
    def forward(self, index, target=None):
        epsilon = 1e-6
        loss = 0
        b, w = index.shape
        kappa = 0
        kernel2 = torch.unsqueeze(self.kernel2, dim=0)
        kernel1 = torch.unsqueeze(self.kernel1, dim=0)
        index = torch.unsqueeze(index, dim=1)
        target = torch.unsqueeze(target, dim=1)
        y1 = F.conv1d(index, kernel1, padding='valid')
        y2 = F.conv1d(index, kernel2, padding='valid')
        t1 = F.conv1d(target, kernel1, padding='valid')
        t2 = F.conv1d(target, kernel2, padding='valid')
        #loss_diff1 = nn.L1Loss()(y1, t1)
        #loss_diff2 = nn.L1Loss()(y2, t2)

        loss_diff2 = torch.mean(torch.abs(y2))
        #print(loss_diff1, loss_diff2)
        return (loss_diff2)/50*10

class DSCLoss(torch.nn.Module):

    def __init__(self, alpha: float = 1.0, smooth: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.gather(probs, dim=1, index=targets.unsqueeze(1))

        probs_with_factor = ((1 - probs) ** self.alpha) * probs
        loss = 1 - (2 * probs_with_factor + self.smooth) / (probs_with_factor + 1 + self.smooth)

        if self.reduction == "mean":
            return loss.mean()
class DSC_CEv2_loss_fun(nn.Module):
    def __init__(self, beta, w, gamma,ignored_class=None, alpha=0.5):
        super(DSC_CEv2_loss_fun, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.w = w
        self.gamma = gamma
        self.ignored_class = ignored_class
    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, 'predict & target shape do not match'
        epsilon = 1e-6
        smooth =1.
        #tversky loss
        b, c = y_pred.size(0), y_pred.size(1)

        p0 = y_pred  # proba that voxels are class i
        p1 = (1 - y_pred) # proba that voxels are not class i
        g0 = y_true
        g1 = (1 - y_true)
        # 求得每个sample的每个类的dice

        '''fn = torch.sum(p1 * g0, dim=(2, 3))
        fp = torch.sum(p0 * g1, dim=(2, 3))
        tp = torch.sum(p0 * g0, dim=(2, 3))'''
        #den = torch.zeros_like(tp)
        probs_with_factor = p0 * (torch.pow(p1, 1))
        den = torch.sum(probs_with_factor, dim=(2,3)) + torch.sum(g0, dim=(2,3))
        num = torch.sum(2 * probs_with_factor * g0, dim=(2,3))
        #print(torch.mean(num), torch.mean(den))
        '''for c in range(len(self.alpha)):
            den[:,c] = tp[:,c] + self.alpha[c] * fp[:,c] + self.beta[c] * fn[:,c]  # fn、fp可能写反了'''
        dsc = (num + smooth) / (den + smooth)  # [batchsize,class_num]

        if self.ignored_class:
            dsc = torch.mean(dsc, dim=(0))
            for i in self.ignored_class:
                dsc[i] = 1.
            dsc = 1 - torch.mean(dsc)
        else:
            dsc = 1 - torch.mean(dsc)
        #categorical_crossentropy loss
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)  # [bs,classes,x,y]
        p_loss = torch.zeros_like(y_pred)
        for c in range(len(self.w)):
            p_loss[:, c, :, :] = (y_true * torch.log(y_pred))[:, c, :, :] * self.w[c]
        CL = -torch.sum(p_loss, dim=1)
        CL = torch.mean(CL)
        return self.gamma*dsc + CL