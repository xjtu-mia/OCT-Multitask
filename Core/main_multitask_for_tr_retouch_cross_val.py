import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchsummary import summary
import sys
sys.path.append(sys.path[0]+'/multitask_models')
from loss import DiceLoss, tversky_CEv2_loss_fun, categorical_crossentropy_v2, pcloss, PJcurvature
from dataset_multitask import train_transform_data, val_transform_data, saveResult, drawmask, drawmask_truth
from evaluate import get_y_true, get_y_pred, AUPR, ROC_auto, compute_pr_f1, MAD, RMSE
import cv2
import os
import xlwt
import json
from matplotlib import pyplot as plt
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torchvision
import random
# 是否使用cuda
from scipy.signal import savgol_filter
from cul_AVD import main as cul_AVD
"""
# 把多个步骤整合到一起, channel=（channel-mean）/std, 因为是分别对三个通道处理
x_transforms = transforms.Compose([       #transforms.Compose()串联多个transform操作
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()
# 参数解析器,用来解析从终端读取的命令
"""
parse = argparse.ArgumentParser()
device = torch.device("cuda")


# pytorch tensor to tensorflow tensor
def torch2tensor(torch):
    #print(torch.shape)
    shape = torch.shape
    if len(shape) == 4:
        tensor = np.empty((shape[0], shape[2], shape[3], shape[1]), dtype=np.float16)
        for n in range(shape[0]):
            for c in range(shape[1]):
                tensor[n, :, :, c] = torch[n, c, :, :]
    elif len(shape) == 3:
        tensor = np.empty((shape[0], shape[2], shape[1]), dtype=np.float16)
        for n in range(shape[0]):
            for c in range(shape[1]):
                tensor[n, :, c] = torch[n, c, :]
    else:
        tensor = torch
    #print(tensor.shape)
    return tensor


def softargmax2d_col(input, eta=1, dim=2, dynamic_beta=False, epoch=None, max_epoch=None):
    if dynamic_beta:
        if epoch < 1:
            eta = 10 * eta
        else:
            eta = eta

    b, c, h, w = input.shape
    '''for i, e in enumerate(eta):
        input[:, i, :, :] = input[:, i, :, :] * e'''
    x = nn.functional.softmax(input * eta, dim=dim)
    indices = torch.unsqueeze(torch.linspace(0, h - 1, h), dim=0).to(device)
    y = torch.zeros((b, c, w)).to(device)
    for i in range(b):
        for j in range(c):
            y[i, j, :, ] = torch.mm(indices, x[i, j, :, :])
    return y
    


def posission_constraint(input, KF=False, device='cuda'):
    device = torch.device(device)
    b, c, w = input.shape[0], input.shape[1], input.shape[2]
    x = input
    for i in range(c - 1): #BM-->INL_OPL
        #print(i)
        # x[:,i+1,:] = x[:,i,:] + nn.functional.relu(x[:,i+1,:] - x[:,i,:])
        x[:, i + 1, :] = x[:, i, :] + nn.functional.softplus(x[:, i+1, :] - x[:, i, :])

    '''for i in range(2, c - 1):  #ILM-->INL_OPL
        #print(i)
        #x[:,i+1,:] = x[:,i,:] + nn.functional.relu(x[:,i+1,:] - x[:,i,:])
        x[:, i + 1, :] = x[:, i, :] + nn.functional.softplus(x[:, i + 1, :] - x[:, i, :])'''

    return x
class masks_from_regression1:
    def __init__(self, input, eta=1, dim=2, device='cuda', add=False, KF=False, convexhull=True):
        self.input = input
        self.eta = eta
        self.dim = dim
        self.device = device
        self.add = add
        self.KF = KF
        self.convexhull = convexhull


    def posission_constraint(self, input, KF=True):
        b, c, w = input.shape[0], input.shape[1], input.shape[2]
        x = input
        for i in range(c - 1):  # BM-->INL_OPL
            # print(i)
            # x[:,i+1,:] = x[:,i,:] + nn.functional.relu(x[:,i+1,:] - x[:,i,:])
            x[:, i + 1, :] = x[:, i, :] + nn.functional.softplus(x[:, i + 1, :] - x[:, i, :])

        '''for i in range(2, c - 1):  # ILM-->INL_OPL
            #print(i)
            # x[:,i+1,:] = x[:,i,:] + nn.functional.relu(x[:,i+1,:] - x[:,i,:])
            x[:, i + 1, :] = x[:, i, :] + nn.functional.softplus(x[:, i + 1, :] - x[:, i, :])'''
        return x

    def ConvexHull(self, img):
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
        y = self.posission_constraint(y, KF=self.KF)  #保证层之间不会出现拓扑错误
        for i in range(b):
            for k in range(c - 1):
                top_index = torch.round(y[i, k, :])
                bottom_index = torch.round(y[i, k + 1, :])
                if self.convexhull:
                    for j in range(w):
                        # print( BM_img[i, 0: int(bottom_index[j]), int(j)])
                        BM_img[i, 0: int(bottom_index[j]), int(j)] = 255
                    if k == 5:
                        bottom_index[:] = torch.tensor(self.ConvexHull(BM_img[i, :, :]))

                for j in range(w):
                    pixel_top = int(top_index[j].item())
                    pixel_bottom = int(bottom_index[j].item())
                    avg_th = avg_th + (pixel_bottom - pixel_top)
                    if k == c-2:
                        thresh = 2 #BM
                    elif k == c-4:
                        thresh = 2 #SRF
                    else:
                        thresh = 1
                    if pixel_bottom - pixel_top > thresh:
                        mask[i, k, pixel_top:pixel_bottom, j] = torch.ones((1, pixel_bottom - pixel_top))
                        # print(mask[i, 0, pixel_top:pixel_bottom, j])
                    else:
                        if k > 0:
                            mask[i, k-1, pixel_top:pixel_bottom, j] = torch.ones((1, pixel_bottom - pixel_top))
        avg_th = avg_th / b / c / w

        if avg_th > 10:
            act = True
        else:
            act = True
        print(avg_th, act)
        return mask.to(device), act

def index2img(input, target_size):
    shape = input.shape
    x = np.zeros((shape[0], shape[1], target_size[0], shape[2]))
    for b in range(shape[0]):
        for c in range(shape[1]):
            index = input[b, c, :]
            for w in range(shape[2]):
                y_co = np.round(index[w])
                x[b, c, int(y_co), w] = 1
    return x


def save_to_exel(x_list, y_list, write_path):
    xls = xlwt.Workbook()
    sht1 = xls.add_sheet("Sheet1")
    sht1.write(0, 0, 'recall')
    sht1.write(0, 1, 'precision')
    for i, x in enumerate(x_list):
        sht1.write(i + 1, 0, x)
        sht1.write(i + 1, 1, y_list[i])
    xls.save(write_path)

def save_to_exel2(x_list, write_path):
    xls = xlwt.Workbook()
    sht1 = xls.add_sheet("Sheet1")
    start_row = 0
    for k, v in x_list.items():
        sht1.write(start_row, 0, k)
        n = len(v[1])
        for i in range(n):
            sht1.write(start_row+1, i, v[1][i])
        for i, x in enumerate(v[0]):
            sht1.write(start_row+2, i, x)
        start_row += 3
    xls.save(write_path)

def evl(results,
        dataset='retouch',
        method='PR',
        threshold_num=33,
        classes=5,
        target_size=(420, 420),
        task=None,
        flag_multi_class=True,
        groundtruth_path=None,
        save_path=None,
        regression=False):
    if flag_multi_class:
        classes = classes
        if classes == 1:
            print('classes must > 1')
    else:
        classes = 1
    if classes == 8 and task == None:
        classes_list = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6', 'layer7']
    if classes == 7 and task == None:
        classes_list = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6']
    if classes == 2 and task == None:
        classes_list = ['fluid']
    y_true = get_y_true(groundtruth_path, dataset=dataset, ly_classes=classes, target_size=target_size, regression=regression,
                        task=task)  # transforming every image into array and saving in a list
    y_pred = get_y_pred(results, classes, target_size=target_size)
    au_dict = {}

    """
    evaluation methods

    """

    # 多分类，softmax
    sum_AUPR = 0
    sum_AUC = 0
    if method == 'f1':  # AUC, AUPR, F1-score, Dice-score
        au_dict = compute_pr_f1(y_pred, y_true, class_num=classes, task=task)  # compute pr,recall,f1
        print('f1:', au_dict)
        if task == 'index2area':
            filename = 'f1_final'
        else:
            filename = 'f1'
        with open(os.path.join(save_path, filename+'.json'), 'a') as outfile:
            for k, v in au_dict.items():
                json.dump({k: v}, outfile, ensure_ascii=False, indent=4)
                outfile.write('\n')

        f = open(os.path.join(save_path, filename+'.txt'), 'w', encoding='utf-8')  # 以'w'方式打开文件
        for k, v in au_dict.items():  # 遍历字典中的键值
            s2 = str(v)  # 把字典的值转换成字符型
            f.write(k + '\n')  # 键和值分行放，键在单数行，值在双数行
            f.write(s2 + '\n')
        f.close()  # 关闭文件
    else:
        print("error")
    return au_dict


def train_model(model,
                criterion,
                optimizer,
                dataload,
                ly_classes=8,
                pixel_ly_classes=9,
                fl_classes=2,
                epoch=0,
                max_epoch=100,
                eta=1,
                regression=False,
                softargmax=False,
                multitask=False):
    dt_size = len(dataload.dataset)
    epoch_loss = 0
    step = 0
    mean_loss_list = np.zeros(9)

    if multitask:
        for x, y1, y2, y3, y4 in tqdm(dataload, ncols=80):
            # for x, y in dataload:
            step += 1

            inputs = x.to(device)
            labels_1 = y1.to(device)
            labels_2 = y2.to(device)
            labels_3 = y3.to(device)
            labels_4 = y4.to(device)
            label_diff = torch.tensor(savgol_filter(np.array(y1[:, -1, :]), 51, 2), dtype=torch.float).to(device)
            # print(labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs_1, outputs_1_0, outputs_3, outputs_4 = model(inputs)
            # print(outputs)if classes == 1:
            outputs_2 = outputs_1
            outputs_2_0 = outputs_1_0
            # labels = labels.squeeze(1)  #使用内置的交叉熵函数时需要压缩维度，且不需要softmax
            h = labels_1.shape[2]
            outputs_1 = softargmax2d_col(outputs_1, dim=2, eta=eta, dynamic_beta=False, epoch=epoch,
                                         max_epoch=max_epoch)
            outputs_1 = posission_constraint(outputs_1)
            pcurv_loss_1 = PJcurvature(device=device)(outputs_1[:, -1, :], label_diff)
            outputs_1_0 = softargmax2d_col(outputs_1_0, dim=2, eta=eta, dynamic_beta=False, epoch=epoch,
                                         max_epoch=max_epoch)
            outputs_1_0 = posission_constraint(outputs_1_0)
            pcurv_loss_1_0 = PJcurvature(device=device)(outputs_1_0[:, -1, :], label_diff)
            '''
            不计算没有label点的loss，即output与label在该点始终具有相同值
            '''
            outputs_1 = torch.where(labels_1 == 0, labels_1, outputs_1)
            outputs_1_0 = torch.where(labels_1 == 0, labels_1, outputs_1_0)
            outputs_2 = nn.Softmax(dim=2)(outputs_2)
            outputs_2_0 = nn.Softmax(dim=2)(outputs_2_0)
            outputs_3 = nn.Softmax(dim=1)(outputs_3)
            outputs_4 = nn.Softmax(dim=1)(outputs_4)
            loss_1 = criterion[0](outputs_1, labels_1)
            loss_2 = criterion[1](outputs_2, labels_2)
            loss_1_0 = criterion[0](outputs_1_0, labels_1)
            loss_2_0 = criterion[1](outputs_2_0, labels_2)
            # print(outputs_3.shape, labels_3.shape)
            loss_3 = criterion[2](outputs_3, labels_3)
            loss_4 = criterion[3](outputs_4, labels_4)
            loss = 1 * loss_1 + 1 * loss_1_0 + 1 * loss_2 + 1 * loss_2_0 + 10 * loss_3 + 10 * loss_4+0*pcurv_loss_1+0*pcurv_loss_1_0
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            epoch_loss += loss.item()
            loss_list = np.array([loss_1.item(), loss_1_0.item(), loss_2.item(), loss_2_0.item(),
                                  loss_3.item(), loss_4.item(), pcurv_loss_1.item(), pcurv_loss_1_0.item(),
                                  loss.item()])
            mean_loss_list = mean_loss_list + loss_list
            print("%d/%d,train_loss:" % (step, (dt_size - 1) // dataload.batch_size + 1), loss_list)
            '''feature_output1 = model.featuremap1.cpu()
            print(feature_output1.shape)
            out = torchvision.tools.make_grid(feature_output1[0])
            feature_imshow(out)'''
    else:
        for x, y in tqdm(dataload, ncols=80):
            # for x, y in dataload:
            step += 1

            inputs = x.to(device)
            labels = y.to(device)
            # print(labels.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            # print(outputs)
            if ly_classes == 1:
                outputs = nn.Sigmoid()(outputs)
                loss = criterion(outputs, labels)
            else:
                # labels = labels.squeeze(1)  #使用内置的交叉熵函数时需要压缩维度，且不需要softmax
                if not regression:
                    outputs = nn.Softmax(dim=1)(outputs)
                else:
                    if softargmax:
                        h = labels.shape[2]
                        outputs = softargmax2d_col(outputs, dim=2, eta=eta, dynamic_beta=False, epoch=epoch,
                                                   max_epoch=max_epoch)
                        '''
                        不计算没有label点的loss，即output与label在该点始终具有相同值
                        '''
                        outputs_ = torch.where(labels == 0, labels, outputs)
                        # outputs = posission_constraint(outputs)
                    else:
                        outputs = nn.Softmax(dim=2)(outputs)
                loss = criterion(outputs_, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
    # scheduler.step()
    print("第%d个epoch的lr, weight_decay：%f, %f" % (
        epoch, optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['weight_decay']))
    mean_epoch_loss = epoch_loss / ((dt_size - 1) // dataload.batch_size + 1)
    mean_loss_list = mean_loss_list / ((dt_size - 1) // dataload.batch_size + 1)
    print("epoch %d mean_train_loss:" % (epoch), mean_loss_list)
    return mean_loss_list, model


def val_model(model,
              criterion,
              dataload,
              dataset='duke',
              ly_classes=8,
              pixel_ly_classes=9,
              fl_classes=2,
              epoch=0,
              max_epoch=100,
              eta=1,
              regression=False,
              softargmax=False,
              multitask=False):
    model.eval()
    with torch.no_grad():  # 不进行梯度计算和反向传播
        dt_size = len(dataload.dataset)
        target_size = (dataload.dataset[0][0].shape[1], dataload.dataset[0][0].shape[2])
        print(target_size)
        epoch_loss = 0
        step = 0
        img_arr_ly = np.empty((len(dataload.dataset), target_size[0], target_size[1], ly_classes))
        img_arr_fl = np.empty((len(dataload.dataset), target_size[0], target_size[1], fl_classes))
        n = 0
        mean_loss_list = np.zeros(9)
        if dataset == 'duke':
            name_list = ['ILM', 'RNFL-GCL', 'IPL-INL', 'INL-OPL', 'OPL-ONL', 'IS-OS', 'OS-RPE', 'BM', 'mean']
        if dataset == 'yifuyuan' or dataset == 'retouch':
            name_list = ['ILM', 'IPL-INL', 'INL_OPL', 'ONL_NE', 'IRPE', 'ORPE', 'BM', 'mean']

        tmp_mad = np.zeros((len(name_list) - 1))
        tmp_rmse = np.zeros((len(name_list) - 1))
        mean_absolute_distance = {}
        root_mean_square_error = {}
        for name in name_list:
            mean_absolute_distance[name] = 0
        if multitask:
            for x, y1, y2, y3, y4 in tqdm(dataload, ncols=60):
                # for x, y in dataload:
                step += 1

                inputs = x.to(device)
                labels_1 = y1.to(device)
                labels_2 = y2.to(device)
                labels_3 = y3.to(device)
                labels_4 = y4.to(device)
                label_diff = torch.tensor(savgol_filter(np.array(y1[:, -1, :]), 51, 2), dtype=torch.float).to(device)
                # print(labels.shape)
                # zero the parameter gradients
                # forward
                outputs_1, outputs_1_0, outputs_3, outputs_4 = model(inputs)
                # print(outputs)if classes == 1:

                # labels = labels.squeeze(1)  #使用内置的交叉熵函数时需要压缩维度，且不需要softmax
                h = labels_3.shape[2]
                outputs_2 = outputs_1
                outputs_2_0 = outputs_1_0
                outputs_1 = softargmax2d_col(outputs_1, dim=2, eta=eta, dynamic_beta=False, epoch=epoch,
                                             max_epoch=max_epoch)
                outputs_1 = posission_constraint(outputs_1)
                pcurv_loss_1 = PJcurvature(device=device)(outputs_1[:, -1, :], label_diff)
                outputs_1_0 = softargmax2d_col(outputs_1_0, dim=2, eta=eta, dynamic_beta=False, epoch=epoch,
                                             max_epoch=max_epoch)
                outputs_1_0 = posission_constraint(outputs_1_0)
                pcurv_loss_1_0 = PJcurvature(device=device)(outputs_1_0[:, -1, :], label_diff)
                '''
                不计算没有label点的loss，即output与label在该点始终具有相同值
                '''
                outputs_1 = torch.where(labels_1 == 0, labels_1, outputs_1)
                outputs_1_0 = torch.where(labels_1 == 0, labels_1, outputs_1_0)
                outputs_2 = nn.Softmax(dim=2)(outputs_2)
                outputs_2_0 = nn.Softmax(dim=2)(outputs_2_0)
                outputs_3 = nn.Softmax(dim=1)(outputs_3)
                outputs_4 = nn.Softmax(dim=1)(outputs_4)
                loss_1 = criterion[0](outputs_1, labels_1)
                loss_1_0 = criterion[0](outputs_1_0, labels_1)
                loss_2 = criterion[1](outputs_2, labels_2)
                loss_2_0 = criterion[1](outputs_2_0, labels_2)
                loss_3 = criterion[2](outputs_3, labels_3)
                loss_4 = criterion[3](outputs_4, labels_4)
                loss = 1 * loss_1 + 1 * loss_1_0 + 1 * loss_2 + 1 * loss_2_0 + 10 * loss_3 + 10 * loss_4+0*pcurv_loss_1+0*pcurv_loss_1_0
                loss_list = np.array([loss_1.item(), loss_1_0.item(), loss_2.item(), loss_2_0.item(),
                                      loss_3.item(), loss_4.item(), pcurv_loss_1.item(), pcurv_loss_1_0.item(),
                                      loss.item()])
                mean_loss_list = mean_loss_list + loss_list
                epoch_loss += loss.item()

                print("%d/%d,val_loss:" % (step, (dt_size - 1) // dataload.batch_size + 1), loss_list)
                img_y_ly = outputs_1.cpu().numpy()  # cuda输出的tensor无法直接进行numpy操作，因此需要转换成cpu tensor
                img_true_ly = labels_1.cpu().numpy()
                tmp_mad = tmp_mad + MAD(img_y_ly, img_true_ly, softargmax=softargmax)
                tmp_rmse = tmp_rmse + RMSE(img_y_ly, img_true_ly, softargmax=softargmax)

                img_y_ly = torch2tensor(img_y_ly)
                img_arr_ly[n] = img_y_ly[0]

                img_y_fl = outputs_4.cpu().numpy()  # cuda输出的tensor无法直接进行numpy操作，因此需要转换成cpu tensor
                img_true_fl = labels_4.cpu().numpy()
                img_y_fl = torch2tensor(img_y_fl)
                img_arr_fl[n] = img_y_fl[0]
                n += 1
        else:
            for x, y in tqdm(dataload, ncols=60):
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                # forward
                outputs = model(inputs)
                # print(outputs)
                if ly_classes == 1:
                    outputs = nn.Sigmoid()(outputs)
                    loss = criterion(outputs, labels)
                else:
                    # labels = labels.squeeze(1)  #使用内置的交叉熵函数时需要压缩维度，且不需要softmax
                    if not regression:
                        outputs = nn.Softmax(dim=1)(outputs)
                    else:
                        if softargmax:
                            h = labels.shape[2]
                            outputs = softargmax2d_col(outputs, dim=2, eta=eta, dynamic_beta=False, epoch=epoch,
                                                       max_epoch=max_epoch)
                            '''
                            不计算没有label点的loss，即output与label在该点始终具有相同值
                            '''
                            outputs_ = torch.where(labels == 0, labels, outputs)
                            # outputs = posission_constraint(outputs)
                        else:
                            outputs = nn.Softmax(dim=2)(outputs)
                    loss = criterion(outputs_, labels)

                epoch_loss += loss.item()
                print("%d/%d,val_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
                img_y = outputs.cpu().numpy()  # cuda输出的tensor无法直接进行numpy操作，因此需要转换成cpu tensor
                img_y = torch2tensor(img_y)
                img_arr_ly[n] = img_y[0]
                n += 1
        mean_epoch_loss = epoch_loss / ((dt_size - 1) // dataload.batch_size + 1)
        mean_loss_list = mean_loss_list / ((dt_size - 1) // dataload.batch_size + 1)
        tmp_mad = tmp_mad / len(dataload.dataset)
        tmp_rmse = tmp_rmse / len(dataload.dataset)
        print(len(dataload.dataset))
        for i, name in enumerate(name_list):
            if name == 'mean':
                mmad = 0
                mrmse = 0
                for value in mean_absolute_distance.values():
                    mmad += value
                mmad = mmad / (len(name_list) - 1)
                mean_absolute_distance[name] = mmad
                for value in root_mean_square_error.values():
                    mrmse += value
                mrmse = mrmse / (len(name_list) - 1)
                root_mean_square_error[name] = mrmse
            else:
                mean_absolute_distance[name] = tmp_mad[i]
                root_mean_square_error[name] = tmp_rmse[i]
        print("epoch %d mean_val_loss:" % (epoch), mean_loss_list)

    return mean_loss_list, img_arr_ly, img_arr_fl, mean_absolute_distance, root_mean_square_error


# 训练模型
def train(root=None,
          dataset='duke',
          pretraining=None,
          backbone='unet',
          save_path=None,
          tr_image_fold=None,
          tr_layer_label_fold=None,
          tr_fluid_label_fold=None,
          tr_layer_fluid_label_fold=None,
          val_image_fold=None,
          val_layer_label_fold=None,
          val_fluid_label_fold=None,
          val_layer_fluid_label_fold=None,
          batch_size=1,
          learning_rate=0.0001,
          ly_classes=8,
          pixel_ly_classes=9,
          fl_classes=2,
          one_hot='n',
          regression=False,
          softargmax=False,
          multitask=False,
          fluid_label=False,
          eta=1,
          max_epoch=20,
          target_size=(512, 512),
          num_workers=0,
          writer=None,
          image_color_mode='gray'):
    def lr_decay(optimizer, epoch, method='step_decay', warm_epoch=0, max_epoch=40, initial_lr=0.0001, power=0.9):
        ite = epoch + 1
        if ite <= warm_epoch:
            l = max(ite * initial_lr / warm_epoch, 1e-5)
        else:
            l = optimizer.param_groups[0]['lr']
        if method == 'step_decay':  # step_decay
            if ite == round(max_epoch * 0.9):
                l = l * 0.3
            elif ite == round(max_epoch * 0.95):
                l = l * 0.1
            elif ite == round(max_epoch * 0.95) + 1:
                l = l * 0.1
            elif ite == round(max_epoch * 0.95) + 2:
                l = l * 0.1
            elif ite == round(max_epoch * 0.95) + 3:
                l = l * 0.1

        elif method == 'poly':

            if ite >= int(max_epoch * 0.8):
                l = l * ((1 - ((ite - int(max_epoch * 0.8)) / (max_epoch - int(max_epoch * 0.8)))) ** power)
            if l <= 1e-8:
                l = 1e-8
        for param_group in optimizer.param_groups:
            param_group['lr'] = l
            param_group['weight_decay'] = l

    global criterion, gamma, beta, alpha, w
    if image_color_mode == 'gray':
        channel = 1
    else:
        channel = 3
    '''
    selecting model
    '''
    print(backbone)
    if backbone == 'resnetv2':
        from multitask_models.multitask_resnestv2 import tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_resUnet_ly_masks
        model = tinyquadra_multi_resUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                  multitask=multitask).to(device)
    elif backbone == 'vgg':
        from multitask_models.multitask_VGG import tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_vggUnet_ly_masks
        model = tinyquadra_multi_vggUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                  multitask=multitask).to(device)
    elif backbone == 'resnet':
        from multitask_models.multitask_resnestv2 import R50_Unet as tinyquadra_multi_oresUnet_ly_masks
        model = tinyquadra_multi_oresUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                   multitask=multitask).to(device)
    elif backbone == 'convnext':
        from multitask_models.multitask_convnext import tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_convnextUnet_ly_masks
        model = tinyquadra_multi_convnextUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                       multitask=multitask).to(device)
    elif backbone == 'swintrans':
        from multitask_models.multitask_swintrans import tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_swinUnet_ly_masks
        model = tinyquadra_multi_swinUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                   multitask=multitask).to(device)
    elif backbone == 'shuffletrans':
        from multitask_models.multitask_shuffletrans import tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_shufflUnet_ly_masks
        model = tinyquadra_multi_shufflUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                     multitask=multitask).to(device)
    elif backbone == 'mpvit':
        from multitask_models.multitask_mpvit import tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_mpvitUnet_ly_masks
        model = tinyquadra_multi_mpvitUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                    multitask=multitask).to(device)
    else:
        raise Exception("need a backbone model name")
    #summary(model, (channel,) + target_size)
    '''from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
    inputs = torch.randn(1,1,416,512).to(device)
    flops = FlopCountAnalysis(model, inputs)
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acts = ActivationCountAnalysis(model, inputs)

    print(f"total flops : {flops.total()}")
    print(f"total activations: {acts.total()}")
    print(f"number of parameter: {param}")'''
    '''
    loading pretrained model
    '''
    if pretraining:
        '''print(pretraining)
        pretext_model = torch.load(pretraining)
        model2_dict = model.state_dict()
        state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
        model2_dict.update(state_dict)
        model.load_state_dict(model2_dict)'''


        model.load_state_dict(torch.load(pretraining))
        #print(model.state_dict()['conv11.0.weight'])
    '''
    loss function
    '''
    if multitask:

        criterion_1 = nn.SmoothL1Loss(beta=1)
        criterion_2 = categorical_crossentropy_v2(w=1)
        if dataset == 'duke':
            alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            beta = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            w = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
            gamma = 0.4
        elif dataset == 'yifuyuan' or dataset == 'retouch':
            if pixel_ly_classes == ly_classes:
                alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                beta = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                w = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                gamma = 0.4
            else:
                alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                beta = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                w = [1, 1, 1, 1, 1, 1, 1, 1]
                gamma = 0.4
        criterion_3 = tversky_CEv2_loss_fun(alpha=alpha,
                                            beta=beta,
                                            w=w,
                                            gamma=gamma,
                                            )
        criterion_4 = tversky_CEv2_loss_fun(alpha=[0.5, 0.5],
                                            beta=[0.5, 0.5],
                                            w=[1, 1],
                                            gamma=0.4)
        criterion = (criterion_1, criterion_2, criterion_3, criterion_4)


    else:
        if ly_classes == 1:
            criterion = nn.BCELoss()
        elif ly_classes > 1 and not regression:
            # criterion = nn.CrossEntropyLoss()  #pytorch CEloss 内置softmax函数，因此network不需要添加softmax！
            print('tversky_CEv2_loss')
            criterion = tversky_CEv2_loss_fun(alpha=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                              beta=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                              w=[0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                                              gamma=0.4)
        elif ly_classes > 1 and regression:
            if softargmax:
                criterion = nn.SmoothL1Loss(beta=1)
                # criterion = pcloss()
            else:
                criterion = categorical_crossentropy_v2(w=1)
    '''
    selecting optimizer
    '''
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=learning_rate)
    # scheduler = MultiStepLR(optimizer,
    #                        milestones=[int(max_epoch*0.5), int(max_epoch*0.7), int(max_epoch*0.9)],
    #                        gamma=0.3)
    '''
    initializing dataloaders 
    '''
    dataset_train = train_transform_data(root=os.path.join(root, 'train'),
                                         dataset=dataset,
                                         image_fold=tr_image_fold,
                                         layer_label_fold=tr_layer_label_fold,
                                         fluid_label_fold=tr_fluid_label_fold,
                                         layer_fluid_label_fold=tr_layer_fluid_label_fold,
                                         ly_classes=ly_classes,
                                         pixel_ly_classes=pixel_ly_classes,
                                         fl_classes=fl_classes,
                                         one_hot=one_hot,
                                         regression=regression,
                                         softargmax=softargmax,
                                         multitask=multitask,
                                         fluid_label=fluid_label,
                                         target_size=target_size)
    dataloaders_train = DataLoader(dataset_train,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers)
    dataset_val = val_transform_data(root=os.path.join(root, 'val'),
                                     dataset=dataset,
                                     image_fold=val_image_fold,
                                     layer_label_fold=val_layer_label_fold,
                                     fluid_label_fold=val_fluid_label_fold,
                                     layer_fluid_label_fold=val_layer_fluid_label_fold,
                                     ly_classes=ly_classes,
                                     pixel_ly_classes=pixel_ly_classes,
                                     fl_classes=fl_classes,
                                     one_hot=one_hot,
                                     regression=regression,
                                     softargmax=softargmax,
                                     multitask=multitask,
                                     fluid_label=fluid_label,
                                     target_size=target_size)
    dataloaders_val = DataLoader(dataset_val,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=num_workers
                                 )

    '''
    training
    '''
    # train_names = ['train_loss']
    # val_names = ['val_loss',
    #             'EX_AUPR', 'HE_AUPR', 'MA_AUPR', 'SE_AUPR', 'mAUPR']
    PR_area = []
    val_epoch_loss_tmp = []
    val_epoch_mad_tmp = []
    print('saving model...')
    torch.save(model.state_dict(), os.path.join(save_path, 'weights_initial.pth'))
    for epoch in range(max_epoch):
        print('Epoch {}/{}'.format(epoch, max_epoch - 1))
        print('-' * 10)
        train_epoch_loss, model = train_model(model, criterion, optimizer,
                                              dataloaders_train,
                                              ly_classes=ly_classes,
                                              pixel_ly_classes=pixel_ly_classes,
                                              fl_classes=fl_classes,
                                              epoch=epoch,
                                              max_epoch=max_epoch,
                                              eta=eta,
                                              regression=regression,
                                              softargmax=softargmax,
                                              multitask=multitask)
        writer.add_scalar('train_loss_1', train_epoch_loss[0], global_step=epoch)
        writer.add_scalar('train_loss_1_0', train_epoch_loss[1], global_step=epoch)
        writer.add_scalar('train_loss_2', train_epoch_loss[2], global_step=epoch)
        writer.add_scalar('train_loss_2_0', train_epoch_loss[3], global_step=epoch)
        writer.add_scalar('train_loss_3', train_epoch_loss[4], global_step=epoch)
        writer.add_scalar('train_loss_4', train_epoch_loss[5], global_step=epoch)
        writer.add_scalar('train_curvature_1', train_epoch_loss[6], global_step=epoch)
        writer.add_scalar('train_curvature_1_0', train_epoch_loss[7], global_step=epoch)
        writer.add_scalar('train_loss', train_epoch_loss[8], global_step=epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
        lr_decay(optimizer, epoch, method='poly', warm_epoch=0, max_epoch=max_epoch, initial_lr=learning_rate,
                 power=1.5)
        val_epoch_loss, img_arr_ly, img_arr_fl, mean_absolute_distance, root_mean_square_error = val_model(model,
                                                                                                           criterion,
                                                                                                           dataloaders_val,
                                                                                                           dataset=dataset,
                                                                                                           ly_classes=ly_classes,
                                                                                                           pixel_ly_classes=pixel_ly_classes,
                                                                                                           fl_classes=fl_classes,
                                                                                                           epoch=epoch,
                                                                                                           max_epoch=max_epoch,
                                                                                                           eta=eta,
                                                                                                           regression=regression,
                                                                                                           softargmax=softargmax,
                                                                                                           multitask=multitask)

       

        # print(au_dict)
        # PR_area.append(au_dict['mAUPR'])
        writer.add_scalar('val_loss_1', val_epoch_loss[0], global_step=epoch)
        writer.add_scalar('val_loss_1_0', val_epoch_loss[1], global_step=epoch)
        writer.add_scalar('val_loss_2', val_epoch_loss[2], global_step=epoch)
        writer.add_scalar('val_loss_2_0', val_epoch_loss[3], global_step=epoch)
        writer.add_scalar('val_loss_3', val_epoch_loss[4], global_step=epoch)
        writer.add_scalar('val_loss_4', val_epoch_loss[5], global_step=epoch)
        writer.add_scalar('val_curvature_1', val_epoch_loss[6], global_step=epoch)
        writer.add_scalar('val_curvature_1_0', val_epoch_loss[7], global_step=epoch)
        writer.add_scalar('val_loss', val_epoch_loss[8], global_step=epoch)
        for key in mean_absolute_distance.keys():
            writer.add_scalar(key, mean_absolute_distance[key], global_step=epoch)
        print(mean_absolute_distance, '\n', root_mean_square_error)
       
        #saving checkpoints
        
        if epoch == 0:
            val_epoch_loss_tmp.append(val_epoch_loss[-1])
            val_epoch_mad_tmp.append(mean_absolute_distance['mean'])
            # torch.save(model.state_dict(), os.path.join(save_path, 'weights_0.pth'))
        else:

            min_epoch_loss = min(val_epoch_loss_tmp)
            min_epoch_mad = min(val_epoch_mad_tmp)
            if val_epoch_loss[-1] < min_epoch_loss:
                print('mean_val_loss = {} < min_mean_val_loss = {}, saving model...'.format(val_epoch_loss[-1],
                                                                                            min_epoch_loss))
                torch.save(model.state_dict(), os.path.join(save_path, 'weights_loss.pth'))
                # if epoch > 10:
                #    torch.save(model.state_dict(), os.path.join(save_path, 'weights_'+ str(epoch)+'.pth'))
                # if epoch > 15:
                #    saveResult(save_path, test_path, target_size, img_arr, flag_multi_class=True, classes=classes,regression=regression)
            if mean_absolute_distance['mean'] < min_epoch_mad:
                print('mean_val_mad = {} < min_epoch_mad = {}, saving model...'.format(mean_absolute_distance['mean'],
                                                                                       min_epoch_mad))
                torch.save(model.state_dict(), os.path.join(save_path, 'weights_mad.pth'))
            val_epoch_loss_tmp.append(val_epoch_loss[-1])
            val_epoch_mad_tmp.append(mean_absolute_distance['mean'])
    
    writer.close()
    torch.save(model.state_dict(), os.path.join(save_path, 'weights_final.pth'))

'''
testing and visualizing results
'''


def feature_imshow(inp, title=None):
    """Imshow for Tensor."""

    inp = inp.detach().numpy().transpose((1, 2, 0))

    mean = np.array([0.5, 0.5, 0.5])

    std = np.array([0.5, 0.5, 0.5])

    # inp = std * inp + mean

    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated


def test(root=None,
         dataset='duke',
         backbone='unet',
         save_path=None,
         image_fold=None,
         layer_label_fold=None,
         fluid_label_fold=None,
         layer_fluid_label_fold=None,
         layer_area_label_fold=None,
         ckp=None,
         ly_classes=8,
         pixel_ly_classes=9,
         fl_classes=2,
         one_hot='n',
         regression=False,
         softargmax=False,
         multitask=False,
         fluid_label=False,
         eta=1,
         target_size=(512, 512),
         image_color_mode='gray',
         device = torch.device("cuda")):
    print('testing...')
    if image_color_mode == 'gray':
        channel = 1
    else:
        channel = 3
    #device = torch.device("cuda")

    print(backbone)
    if backbone == 'resnetv2':
        from multitask_models.multitask_resnestv2 import \
            tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_resUnet_ly_masks
        model = tinyquadra_multi_resUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                  multitask=multitask).to(device)
    elif backbone == 'vgg':
        from multitask_models.multitask_VGG import \
            tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_vggUnet_ly_masks
        model = tinyquadra_multi_vggUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                  multitask=multitask).to(device)
    elif backbone == 'resnet':
        from multitask_models.multitask_resnestv2 import R50_Unet as tinyquadra_multi_oresUnet_ly_masks
        model = tinyquadra_multi_oresUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                   multitask=multitask).to(device)
    elif backbone == 'convnext':
        from multitask_models.multitask_convnext import \
            tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_convnextUnet_ly_masks
        model = tinyquadra_multi_convnextUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                       multitask=multitask).to(device)
    elif backbone == 'swintrans':
        from multitask_models.multitask_swintrans import \
            tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_swinUnet_ly_masks
        model = tinyquadra_multi_swinUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                   multitask=multitask).to(device)
    elif backbone == 'shuffletrans':
        from multitask_models.multitask_shuffletrans import \
            tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_shufflUnet_ly_masks
        model = tinyquadra_multi_shufflUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                     multitask=multitask).to(device)
    elif backbone == 'mpvit':
        from multitask_models.multitask_mpvit import \
            tinyquadra_multi_resUnet_ly_masks_s3 as tinyquadra_multi_mpvitUnet_ly_masks
        model = tinyquadra_multi_mpvitUnet_ly_masks(channel, ly_classes, pixel_ly_classes, fl_classes,
                                                    multitask=multitask).to(device)
    else:
        raise Exception("need a backbone model name")
    print('loading model...', ckp)
    model.load_state_dict(torch.load(ckp, map_location='cuda'))


    model.eval()
    #summary(model, (channel,) + target_size)
    dataset_test = val_transform_data(root=os.path.join(root, 'val'),
                                      dataset=dataset,
                                      image_fold=image_fold,
                                      layer_label_fold=layer_label_fold,
                                      fluid_label_fold=fluid_label_fold,
                                      layer_fluid_label_fold=layer_fluid_label_fold,
                                      ly_classes=ly_classes,
                                      pixel_ly_classes=pixel_ly_classes,
                                      fl_classes=fl_classes,
                                      one_hot=one_hot,
                                      regression=regression,
                                      softargmax=softargmax,
                                      multitask=multitask,
                                      fluid_label=fluid_label,
                                      target_size=target_size)
    dataload = DataLoader(dataset_test,
                          batch_size=1,
                          shuffle=False,
                          num_workers=0
                          )

    # import matplotlib.pyplot as plt
    # plt.ion()
    i = 0
    n = 0
    if dataset == 'duke':
        name_list = ['ILM', 'RNFL-GCL', 'IPL-INL', 'INL-OPL', 'OPL-ONL', 'IS-OS', 'OS-RPE', 'BM', 'mean']
    if dataset == 'yifuyuan' or dataset== 'retouch':
        name_list = ['ILM', 'IPL-INL', 'INL_OPL', 'ONL_NE', 'IRPE', 'ORPE', 'BM', 'mean']
    img_arr_ly = np.empty((len(dataload.dataset), target_size[0], target_size[1], ly_classes), dtype=np.float16)
    img_arr_fl = np.empty((len(dataload.dataset), target_size[0], target_size[1], fl_classes), dtype=np.float16)
    img_arr_lyfl = np.empty((len(dataload.dataset), target_size[0], target_size[1], pixel_ly_classes), dtype=np.float16)
    img_arr_ly_area = np.empty((len(dataload.dataset), target_size[0], target_size[1], ly_classes), dtype=np.float16)
    mean_absolute_distance = {}
    root_mean_square_error = {}
    for name in name_list:
        mean_absolute_distance[name] = 0
    with torch.no_grad():  # 不进行梯度计算和反向传播
        dt_size = len(dataload.dataset)
        step = 0
        mean_loss = 0
        mean_loss_list = np.zeros(7)
        tmp_mad = np.zeros((len(name_list) - 1))
        tmp_rmse = np.zeros((len(name_list) - 1))
        if multitask:
            for x, y1, y2, y3, y4 in tqdm(dataload, ncols=60):
                step += 1
                inputs = x.to(device)
                labels_1 = y1.to(device)
                labels_2 = y2.to(device)
                labels_3 = y3.to(device)
                labels_4 = y4.to(device)
                outputs_1, outputs_1_0, outputs_3, outputs_4 = model(inputs)
                '''feature_output1 = model.featuremap1.transpose(1, 0).cpu()

                out = torchvision.tools.make_grid(feature_output1)
                feature_imshow(out)'''

                h = outputs_1.shape[2]
                outputs_2 = outputs_1
                outputs_2_0 = outputs_1_0
                img_y_ly_area, _ = masks_from_regression1(outputs_1,
                                                         eta=1, dim=2, device='cuda',
                                                         add=False, convexhull=False).run()
                outputs_1 = softargmax2d_col(outputs_1, dim=2, eta=eta)
                outputs_1 = posission_constraint(outputs_1)
                outputs_1_0 = softargmax2d_col(outputs_1_0, dim=2, eta=eta)
                outputs_1_0 = posission_constraint(outputs_1_0)
                outputs_2 = nn.Softmax(dim=2)(outputs_2)
                outputs_2_0= nn.Softmax(dim=2)(outputs_2_0)
                outputs_3 = nn.Softmax(dim=1)(outputs_3)
                outputs_4 = nn.Softmax(dim=1)(outputs_4)
                outputs_1_ = outputs_1
                outputs_1 = torch.where(labels_1 == 0, labels_1, outputs_1)
                outputs_1_0 = torch.where(labels_1 == 0, labels_1, outputs_1_0)
                criterion_1 = nn.SmoothL1Loss(beta=1)
                criterion_2 = categorical_crossentropy_v2(w=1)
                if dataset == 'duke':
                    alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                    beta = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                    w = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                    gamma = 0.4
                elif dataset == 'yifuyuan' or dataset=='retouch':
                    if pixel_ly_classes == ly_classes:
                        alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                        beta = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                        w = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                        gamma = 0.4
                    else:
                        alpha = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                        beta = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
                        w = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2]
                        gamma = 0.4
                criterion_3 = tversky_CEv2_loss_fun(alpha=alpha,
                                                    beta=beta,
                                                    w=w,
                                                    gamma=gamma
                                                    )
                criterion_4 = tversky_CEv2_loss_fun(alpha=[0.5, 0.5],
                                                    beta=[0.5, 0.5],
                                                    w=[0.5, 0.5],
                                                    gamma=0.4)
                criterion = (criterion_1, criterion_2, criterion_3, criterion_4)
                loss_1 = criterion[0](outputs_1, labels_1).cpu().numpy()
                loss_2 = criterion[1](outputs_2, labels_2).cpu().numpy()
                loss_1_0 = criterion[0](outputs_1_0, labels_1).cpu().numpy()
                loss_2_0 = criterion[1](outputs_2_0, labels_2).cpu().numpy()
                loss_3 = criterion[2](outputs_3, labels_3).cpu().numpy()
                loss_4 = criterion[3](outputs_4, labels_4).cpu().numpy()
                loss = 1 * loss_1 + 1 * loss_1_0 + 1 * loss_2 + 1 * loss_2_0 + 10 * loss_3 + 10 * loss_4
                loss_list = np.array([loss_1.item(), loss_1_0.item(), loss_2.item(), loss_2_0.item(), loss_3.item(), loss_4.item(), loss.item()])
                print("%d/%d,test_loss:" % (step, (dt_size - 1) // dataload.batch_size + 1), loss_list)
                img_y_ly = outputs_1.cpu().numpy() # cuda输出的tensor无法直接进行numpy操作，因此需要转换成cpu tensor
                img_y_ly_ = outputs_1_.cpu().numpy()
                img_true = labels_1.cpu().numpy()
                tmp_mad = tmp_mad + MAD(img_y_ly, img_true, softargmax=softargmax)
                tmp_rmse = tmp_rmse + RMSE(img_y_ly, img_true, softargmax=softargmax)

                img_y_ly = index2img(img_y_ly_, target_size=target_size)
                img_y_ly = torch2tensor(img_y_ly)
                img_arr_ly[n] = img_y_ly[0]
                img_y_fl = outputs_4.cpu().numpy() # cuda输出的tensor无法直接进行numpy操作，因此需要转换成cpu tensor
                img_y_fl = torch2tensor(img_y_fl)
                img_arr_fl[n] = img_y_fl[0]
                img_y_lyfl = outputs_3.cpu().numpy()  # cuda输出的tensor无法直接进行numpy操作，因此需要转换成cpu tensor
                img_y_lyfl = torch2tensor(img_y_lyfl)
                img_arr_lyfl[n] = img_y_lyfl[0]
                img_y_ly_area = img_y_ly_area.cpu().numpy()  # cuda输出的tensor无法直接进行numpy操作，因此需要转换成cpu tensor
                img_y_ly_area = torch2tensor(img_y_ly_area)
                img_arr_ly_area[n, :, :, 1:] = img_y_ly_area[0]
                n += 1
                mean_loss = mean_loss + loss
                mean_loss_list = mean_loss_list + loss_list
            mean_loss = mean_loss / len(dataload.dataset)
            mean_loss_list = mean_loss_list / len(dataload.dataset)
            print(mean_loss_list)

            test_path = os.path.join(root, 'val', image_fold)
            #groundtruth_path = os.path.join(root, 'val', fluid_label_fold.split('/')[0], fluid_label_fold.split('/')[1]) #only for mix
            groundtruth_path = os.path.join(root, 'val', fluid_label_fold.split('/')[0])
            tmp_mad = tmp_mad / len(dataload.dataset)
            tmp_rmse = tmp_rmse / len(dataload.dataset)
            print(len(dataload.dataset))
            for i, name in enumerate(name_list):
                if name == 'mean':
                    mmad = 0
                    mrmse = 0
                    for value in mean_absolute_distance.values():
                        mmad += value
                    mmad = mmad / (len(name_list) - 1)
                    mean_absolute_distance[name] = mmad
                    for value in root_mean_square_error.values():
                        mrmse += value
                    mrmse = mrmse / (len(name_list) - 1)
                    root_mean_square_error[name] = mrmse
                else:
                    mean_absolute_distance[name] = tmp_mad[i]
                    root_mean_square_error[name] = tmp_rmse[i]
            print(img_arr_ly_area.shape)
            results_item = {'MAD': [list(mean_absolute_distance.values()), list(mean_absolute_distance.keys())],
                            'RMSE': [list(root_mean_square_error.values()), list(root_mean_square_error.keys())]}
            saveResult(save_path,
                       test_path,
                       target_size,
                       dataset=dataset,
                       npyfile_ly=img_arr_ly,
                       npyfile_fl=img_arr_fl,
                       npyfile_ly_fl=img_arr_lyfl,
                       npyfile_ly_area=img_arr_ly_area,
                       flag_multi_class=True,
                       ly_classes=ly_classes,
                       pixel_ly_classes=pixel_ly_classes,
                       fl_classes=fl_classes,
                       regression=regression)
            drawmask(test_path,
                     save_path,
                     dataset=dataset,
                     ly_classes=ly_classes,
                     pixel_ly_classes=pixel_ly_classes,
                     fl_classes=fl_classes,
                     target_size=target_size,
                     regression=regression,
                     split_IRF=True,
                     task='index2area')  # visualization
            print(groundtruth_path,test_path, save_path)
            drawmask_truth(test_path,
                     groundtruth_path,
                     save_path,
                     dataset=dataset,
                     ly_classes=ly_classes,
                     pixel_ly_classes=pixel_ly_classes,
                     fl_classes=fl_classes,
                     target_size=target_size,
                     regression=regression)
            au_dict_fl = evl(img_arr_fl,
                             method='f1',
                             threshold_num=33,
                             classes=fl_classes,
                             target_size=target_size,
                             flag_multi_class=True,
                             groundtruth_path=os.path.join(root, 'val', fluid_label_fold),
                             save_path=save_path,
                             dataset=dataset)
            # print(au_dict)
            au_dict_lyfl = evl(img_arr_lyfl,
                               method='f1',
                               threshold_num=33,
                               classes=pixel_ly_classes,
                               target_size=target_size,
                               flag_multi_class=True,
                               groundtruth_path=os.path.join(root, 'val', layer_fluid_label_fold),
                               save_path=save_path,
                               dataset=dataset)
            # print(au_dict_lyfl)
            au_dict_lyfl['class_name'].append('IRF branch')
            au_dict_lyfl['F1-score'] += au_dict_fl['F1-score']
            au_dict_lyfl['class_name'].append('mean')
            au_dict_lyfl['F1-score'].append(
                sum(au_dict_lyfl['F1-score'][:-2] + au_dict_fl['F1-score']) / (pixel_ly_classes - 1))
            results_item['F1-score'] = [au_dict_lyfl['F1-score'], au_dict_lyfl['class_name']]
            '''save_to_exel2(au_dict_lyfl['F1-score'], au_dict_lyfl['class_name'], 
                          os.path.join(save_path, 'results.xlsx'), start_row=6, tag='F1-score')'''
            au_dict_ly = evl(img_arr_ly_area,
                             method='f1',
                             task='index2area',
                             threshold_num=33,
                             classes=ly_classes,
                             target_size=target_size,
                             flag_multi_class=True,
                             groundtruth_path=os.path.join(root, 'val', layer_area_label_fold),
                             save_path=save_path,
                             dataset=dataset)
            au_dict_ly['class_name'].append('mean')
            au_dict_ly['F1-score'] += [au_dict_ly['mean_F1']]
            au_dict_ly['class_name'].append('mean_irf')
            au_dict_ly['F1-score'].append(
                sum(au_dict_ly['F1-score'][:-1] + au_dict_fl['F1-score']) / (pixel_ly_classes - 1))
            results_item['Reg F1-score'] = [au_dict_ly['F1-score'], au_dict_ly['class_name']]
            save_to_exel2(results_item, os.path.join(save_path, 'results.xls'))

        with open(os.path.join(save_path, 'test_results.json'), 'a') as outfile:
            outfile.seek(0)
            outfile.truncate()  # 清空文件
            json.dump({'loss': mean_loss_list.tolist()}, outfile, ensure_ascii=False, indent=4)
            outfile.write('\n')
            if regression:
                json.dump({'MAD': mean_absolute_distance}, outfile, ensure_ascii=False, indent=4)
                outfile.write('\n')
                json.dump({'RMSE': root_mean_square_error}, outfile, ensure_ascii=False, indent=4)
                outfile.write('\n')




'''
parse = argparse.ArgumentParser()
parse.add_argument("--action", type=str, help="train or test",default='test')
parse.add_argument("--batch_size", type=int, default=2)
parse.add_argument("--one_hot", type=str, default='n')
parse.add_argument("--epoch", type=int, default=1)
parse.add_argument("--target_size", type=tuple, default=(512,512))
parse.add_argument("--ckp", type=str, help="the path of model weight file", default="C:/Users/whl/python_project/torch-DR_seg-master/weights_0.pth")
parse.add_argument("--classes", type=int, default=1)
args = parse.parse_args()
if __name__ == '__main__':
    root = "C:/Users/whl/python_project/torch-DR_seg-master/data/val"
    if args.action == 'test':
        test(args.batch_size,args.ckp,classes=args.classes,one_hot=args.one_hot,target_size=args.target_size)
    else:
        train(args.batch_size, classes=args.classes,one_hot=args.one_hot,max_epoch=args.epoch,target_size=args.target_size)
'''


def main(dataset='duke',
         action='test',
         pretraining=None,
         tr_image_fold='image',
         tr_layer_label_fold='label/layers',
         tr_fluid_label_fold='label/fluid',
         tr_layer_fluid_label_fold='label/covered',
         tr_layer_area_label_fold='label/covered_wo_irf',
         val_image_fold='image',
         val_layer_label_fold='label/layers',
         val_fluid_label_fold='label/fluid',
         val_layer_fluid_label_fold='label/covered',
         val_layer_area_label_fold='label/covered_wo_irf',
         image_color_mode='gray',
         regression=True,
         softargmax=False,
         multitask=False,
         fluid_label=True,
         eta=1,
         backbone='swin_multiUnet',
         batch_size=2,
         one_hot='y',
         epoch=40,
         target_size=(512, 512),
         ly_classes=8,
         pixel_ly_classes=9,
         fl_classes=2,
         learning_rate=0.0001,
         ckp="C:/Users/whl/python_project/torch-DR_seg-master/weights_loss.pth",
         root="C:/Users/whl/python_project/torch-DR_seg-master/data/val",
         save_path=None,
         log_path=None
         ):
    writer = SummaryWriter(log_path)
    print(action)
    if action == 'test':
        test(root=root,
             dataset=dataset,
             backbone=backbone,
             save_path=save_path,
             image_fold=val_image_fold,
             layer_label_fold=val_layer_label_fold,
             fluid_label_fold=val_fluid_label_fold,
             layer_fluid_label_fold=val_layer_fluid_label_fold,
             layer_area_label_fold=val_layer_area_label_fold,
             ckp=ckp,
             ly_classes=ly_classes,
             pixel_ly_classes=pixel_ly_classes,
             fl_classes=fl_classes,
             one_hot=one_hot,
             regression=regression,
             eta=eta,
             softargmax=softargmax,
             multitask=multitask,
             fluid_label=fluid_label,
             target_size=target_size,
             image_color_mode=image_color_mode)
    elif action == 'train':
        train(root=root,
              dataset=dataset,
              pretraining=pretraining,
              backbone=backbone,
              save_path=save_path,
              tr_image_fold=tr_image_fold,
              tr_layer_label_fold=tr_layer_label_fold,
              tr_fluid_label_fold=tr_fluid_label_fold,
              tr_layer_fluid_label_fold=tr_layer_fluid_label_fold,
              val_image_fold=val_image_fold,
              val_layer_label_fold=val_layer_label_fold,
              val_fluid_label_fold=val_fluid_label_fold,
              val_layer_fluid_label_fold=val_layer_fluid_label_fold,
              batch_size=batch_size,
              learning_rate=learning_rate,
              ly_classes=ly_classes,
              pixel_ly_classes=pixel_ly_classes,
              fl_classes=fl_classes,
              one_hot=one_hot,
              regression=regression,
              softargmax=softargmax,
              multitask=multitask,
              fluid_label=fluid_label,
              eta=eta,
              max_epoch=epoch,
              target_size=target_size,
              writer=writer,
              image_color_mode=image_color_mode)


def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)



def run(
        device_id= '0',
        id = 0,
        action = 'train',
        epoch = 25,
        seed = 8830,
        dataset='retouch',
        oct_device = 'Topcon',
        root='./datasets',
        backbone='resnetv2',  # resnetv2,vgg,resnet,convnext,swintrans,shuffletrans,mpvit
        log_name=None,
        pretraining_path='./datasets/yifuyuan/result/yifuyuan_resnetv2_seed_8830/weights_final.pth'
        ):
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    #seedlist = np.array([8830, 3655, 4313, 1090, 7622])
    print(seed)
    root = os.path.join(root, dataset)
    seed_torch(seed=seed)
    root_ = os.path.join(root, oct_device+'_yifuyuan', 'mix/cross_valid/'+str(id))
    if not log_name:
        log_name = 'cross_valid'+str(id)+'_'+dataset + '_' + backbone + '_seed'
        log_name_ = log_name + '_' + str(seed)
    save_path = os.path.join(root_, "result", log_name_)
    test_path = os.path.join(root_, "result", log_name_)
    ckp = os.path.join(save_path, 'weights_loss.pth')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = os.path.join(root_, "log", log_name_)

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    #ckp = '/data/whl/OCT_layer_seg/RETOUCH/Cirrus_yifuyuan/mix/cross_valid/'+str(j)+'/result/crossval'+str(j)+'_retouch_bs_2_esares50unetv2_bn_adamw_acon_7layers_beta1_wIRF_tinyquadraatt_multitask_R_C_mul_ly_masks_fl_IN_test'+str(i)+'/weights_final.pth'
    main(dataset='retouch',
         action=action,
         pretraining=pretraining_path,
         tr_image_fold='flatted_IN_img_512',
         tr_layer_label_fold='pseudo_label'+'/layers',
         tr_fluid_label_fold='pseudo_label'+'/fluid3',
         tr_layer_fluid_label_fold='pseudo_label'+'/covered',
         tr_layer_area_label_fold='pseudo_label'+'/covered_wo_irf',
         val_image_fold='flatted_IN_img_512',
         val_layer_label_fold='pseudo_label'+'/layers',
         val_fluid_label_fold='pseudo_label'+'/fluid3',
         val_layer_fluid_label_fold='pseudo_label'+'/covered',
         val_layer_area_label_fold='pseudo_label'+'/covered_wo_irf',
         regression=True,
         softargmax=True,
         multitask=True,
         fluid_label=True,
         # eta=[0.1,0.3,0.3,0.5,0.3,0.3, 0.01],
         eta=1,
         backbone=backbone,
         batch_size=2,
         learning_rate=0.0001,
         one_hot='y',
         epoch=epoch,
         target_size=(352, 512),
         ly_classes=7,
         pixel_ly_classes=8,
         fl_classes=2,
         ckp=ckp,
         root=root_,
         save_path=save_path,
         log_path=log_path)
    #culculate AVD metric
    if action == 'test':
        cul_AVD(
            save_path=save_path,
            val_path=os.path.join(root_, 'val', 'pseudo_label'),
            oct_device=oct_device
        )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='命令行参数示例')
    parser.add_argument('--k', type=int, default=6,
                        help="k-fold validation")
    parser.add_argument('--device_id', type=str, default='0', help='设备ID')
    parser.add_argument('--dataset', type=str, default='retouch', help='dataset name')
    parser.add_argument('--oct_device', type=str, default='Topcon', help='oct type')
    parser.add_argument('--action', type=str, default='train', help='操作类型')
    parser.add_argument('--epoch', type=int, default=25, help='max epoch for training')
    parser.add_argument('--seedlist', type=int, nargs='+', default=[8830], help='种子列表')
    parser.add_argument('--root', type=str, default='../datasets', help='根目录路径')
    parser.add_argument('--log_name', type=str, default=None, help='日志名称')
    parser.add_argument('--backbone', type=str, default='resnetv2', help='骨干网络')
    parser.add_argument('--pretrain_path', type=str, default='../datasets/yifuyuan/result/yifuyuan_resnetv2_seed_8830/weights_final.pth', help='pre-trained weights path')
    args = parser.parse_args()
    for seed in args.seedlist:
        for i in range(args.k):
            run(device_id=args.device_id,
                id = i,
                action=args.action,
                epoch=args.epoch,
                dataset=args.dataset,
                oct_device=args.oct_device,
                seed=seed,
                root=args.root,
                pretraining_path=args.pretrain_path,
                log_name=args.log_name,
                backbone=args.backbone)





