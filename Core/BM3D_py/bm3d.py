import sys
sys.path.append(sys.path[0]+'/tools/BM3D_py')
from bm3d_1st_step import bm3d_1st_step
from bm3d_2nd_step import bm3d_2nd_step
from psnr import compute_psnr
import os
import cv2
import numpy as np
def symetrize(img, N):
    img_pad = np.pad(img, ((N, N), (N, N)), 'symmetric')
    return img_pad
def run_bm3d(noisy_im, sigma,
             n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
             n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W):
    k_H = 8 if (tau_2D_H == 'BIOR' or sigma < 40.) else 12
    k_W = 8 if (tau_2D_W == 'BIOR' or sigma < 40.) else 12

    noisy_im_p = symetrize(noisy_im, n_H)
    img_basic = bm3d_1st_step(sigma, noisy_im_p, n_H, k_H, N_H, p_H, lambda3D_H, tauMatch_H, useSD_H, tau_2D_H)
    img_basic = img_basic[n_H: -n_H, n_H: -n_H]

    assert not np.any(np.isnan(img_basic))
    img_basic_p = symetrize(img_basic, n_W)
    noisy_im_p = symetrize(noisy_im, n_W)
    img_denoised = bm3d_2nd_step(sigma, noisy_im_p, img_basic_p, n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)
    img_denoised = img_denoised[n_W: -n_W, n_W: -n_W]

    return img_basic, img_denoised
def roi(img):
    h, w = img.shape
    max = np.max(img)
    #cv2.imshow('i', img)
    ret, thr = cv2.threshold(img, max-1, max,cv2.THRESH_BINARY)
    #cv2.imshow('thr', thr)
    NpKernel = np.uint8(np.ones((7, 7)))
    thr = cv2.erode(thr, NpKernel)
    # 显示腐蚀后的图
    # 膨胀图像
    thr = cv2.dilate(thr, NpKernel)
    #cv2.imshow('j', thr)
    #cv2.waitKey(1000)
    up = []
    down = []
    for i in range(w):
        y1 = 0
        y2 = 0
        y = thr[:, i]
        for j in range(h-1):
            if y[j] == 255 and y[j+1]==0:
                y1 = j
                break
            else:
                y1 = 0
        for j in range(h-1, 0, -1):
            if y[j] == 255 and y[j-1]==0:
                y2 = j
                break
            else:
                y2 = h-1
        up.append(y1)
        down.append(y2)

    img = img[np.max(up): np.min(down), 0:w]
    return img

def fill_boundary(img, filled_value = 0, value = 'same'):
    h, w = img.shape
    boundary = np.zeros((w))
    for j in range(w):
        flag = 0
        for i in range(h // 3):
            if img[i, j] != filled_value:
                flag += 1
                if flag == 3:
                    for k in range(i):
                        if value == 'same':
                            img[k, j] = img[i, j]
                        else:
                            img[k, j] = value
                        boundary[j] = i + 10  # 视野边界坐标
                    break
        flag = 0
        for i in range(h - 1, h // 3 * 2, -1):
            if img[i, j] != filled_value:
                flag += 1
                if flag == 3:
                    for k in range(h - 1, i, -1):
                        if value == 'same':
                            img[k, j] = img[i, j]
                        else:
                            img[k, j] = value
                    break
    return img, boundary
def main(image, sigma,refactor=4):

    # <hyper parameter> -------------------------------------------------------------------------------
    n_H = 16
    k_H = 8
    N_H = 16
    p_H = 3
    lambda3D_H = 2.7  # ! Threshold for Hard Thresholding
    useSD_H = False
    tau_2D_H = 'BIOR'
    n_W = 16
    k_W = 8
    N_W = 32
    p_W = 3
    useSD_W = True
    tau_2D_W = 'DCT'
    # for im_name in os.listdir(im_dir):

    sigma_list = [sigma]
    for sigma in sigma_list:
        tauMatch_H = 2500 if sigma < 35 else 5000  # ! threshold determinates similarity between patches
        tauMatch_W = 400 if sigma < 35 else 3500  # ! threshold determinates similarity between patches
        noisy_dir = 'test_data/sigma' + str(sigma)
        im = image
        h, w = im.shape
        #im, _ = fill_boundary(im)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # im = clahe.apply(im)
        # im[np.where(im==255)]=0
        im = cv2.resize(im, (w // refactor, h // refactor))
        # noisy_im = cv2.imread(noisy_im_path, cv2.IMREAD_GRAYSCALE)
        noisy_im = im
        im1, im2 = run_bm3d(noisy_im, sigma,
                            n_H, k_H, N_H, p_H, tauMatch_H, useSD_H, tau_2D_H, lambda3D_H,
                            n_W, k_W, N_W, p_W, tauMatch_W, useSD_W, tau_2D_W)

        psnr_1st = compute_psnr(im, im1)
        psnr_2nd = compute_psnr(im, im2)

        im1 = (np.clip(im1, 0, 255)).astype(np.uint8)
        im2 = (np.clip(im2, 0, 255)).astype(np.uint8)
        im2 = cv2.resize(im2, (w, h))
        # save_name = im_name[:-4] + '_s' + str(sigma) + '_py_1st_P' + '%.4f' % psnr_1st + '.png'
        # cv2.imwrite(os.path.join(save_dir1, save_name), im1)
        return im2


