import os
import cv2
import numpy as np
from os.path import join as oj
import math
def resize_label(label, size):
    resized_label = cv2.resize(label, size)

    h, w = resized_label.shape
    y = np.argmax(resized_label, axis=0)
    '''#fix error in cv2.resize
    if math.fabs(y[0]-y[1])>10*math.fabs(y[1]-y[2]):
        y[0] = y[1]
    if math.fabs(y[size[0]-2]-y[-1])>10*math.fabs(y[size[0]-3]-y[size[0]-2]):
        y[-1] = y[size[0]-2]'''
    resized_label = np.zeros_like(resized_label)
    for i in range(w):
        resized_label[y[i], i] = 255
    #cv2.imshow('l', resized_label)
    #cv2.waitKey(10000)

    return resized_label
def fix_boundary(img):
    if len(img.shape) == 3:
        h, w = img.shape[:2]
        img[:,:, 0] = img[:,:, 1]
        img[:,:, w - 1] = img[:,:, w - 2]
    elif len(img.shape) == 2:
        h, w = img.shape
        img[:,0] = img[:,1]
        img[:, w-1] = img[:, w-2]
    else:
        w = img.shape[0]
        img[0] = img[1]
        img[w - 1] = img[w - 2]
    return img
if __name__ == '__main__':
    root = r'H:\python project\OCT_layer_seg\2ColoredBScan_145\preprocessing_data'
    ly_path = oj(root, 'flatted_label', 'layers')
    fl1_path = oj(root,'flatted_label', 'fluid1')
    fl2_path = oj(root, 'flatted_label', 'fluid2')
    fl3_path = oj(root, 'flatted_label', 'fluid3')
    cw_path = oj(root, 'flatted_label', 'covered')
    img_path = oj(root, 'flatted_IN_img')
    save_ly_path = oj(root, 'flatted_label_512', 'layers')
    save_fl1_path = oj(root, 'flatted_label_512', 'fluid1')
    save_fl2_path = oj(root, 'flatted_label_512', 'fluid2')
    save_fl3_path = oj(root, 'flatted_label_512', 'fluid3')
    save_cw_path = oj(root, 'flatted_label_512', 'covered')
    save_img_path = oj(root, 'flatted_IN_img_512')
    os.makedirs(save_ly_path, exist_ok=True)
    os.makedirs(save_fl1_path, exist_ok=True)
    os.makedirs(save_fl2_path, exist_ok=True)
    os.makedirs(save_fl3_path, exist_ok=True)
    os.makedirs(save_cw_path, exist_ok=True)
    os.makedirs(save_img_path, exist_ok=True)
    imglist = os.listdir(cw_path)
    for name in imglist:
        cw_label = cv2.imread(oj(cw_path, name),-1)
        fl1_label = cv2.imread(oj(fl1_path, name), -1)
        fl2_label = cv2.imread(oj(fl2_path, name), -1)
        fl3_label = cv2.imread(oj(fl3_path, name), -1)

        img = cv2.imread(oj(img_path, name), -1)
        h, w = cw_label.shape
        cw_label = fix_boundary(cv2.resize(cw_label, (w//2, h//2), interpolation=cv2.INTER_NEAREST))
        fl1_label = fix_boundary(cv2.resize(fl1_label, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST))
        fl2_label = fix_boundary(cv2.resize(fl2_label, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST))
        fl3_label = fix_boundary(cv2.resize(fl3_label, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST))
        img = fix_boundary(cv2.resize(img, (w // 2, h // 2)))
        cv2.imwrite(oj(save_fl1_path, name), fl1_label)
        cv2.imwrite(oj(save_fl2_path, name), fl2_label)
        cv2.imwrite(oj(save_fl3_path, name), fl3_label)
        cv2.imwrite(oj(save_cw_path, name), cw_label)
        cv2.imwrite(oj(save_img_path, name), img)
        #cv2.imshow('cw_label', cw_label*30)
        for i in range(7):
            os.makedirs(oj(save_ly_path,'layer'+str(i)), exist_ok=True)
            ly_label = cv2.imread(oj(ly_path,'layer'+str(i), name), -1)
            #cv2.imshow('ly_label1', ly_label)
            ly_label = fix_boundary(resize_label(ly_label, (w // 2, h // 2)))
            #cv2.imshow('ly_label', ly_label)
            #cv2.waitKey(1000)
            cv2.imwrite(oj(save_ly_path,'layer'+str(i), name), ly_label)

