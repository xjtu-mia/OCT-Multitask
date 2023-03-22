import cv2
import os
from os.path import join as oj
import numpy as np

def main(root,
         dataset,
         oct_device):
    root = oj(root, dataset, oct_device, 'preprocessing_data')
    img_dir = oj(root, 'flatted_IN_img_512')
    img_list = os.listdir(img_dir)
    ly_dirs = oj(root, 'flatted_label_512', 'layers')
    original_label_dir = oj(root, 'flatted_label_512', 'covered')
    cw_dir = oj(root, 'flatted_label_512', 'covered')
    cw_woirf_dir = oj(root, 'flatted_label_512', 'covered_wo_irf')
    fl1_dir = oj(root, 'flatted_label_512', 'fluid1')
    fl2_dir = oj(root, 'flatted_label_512', 'fluid2')
    fl3_dir = oj(root, 'flatted_label_512', 'fluid3')
    os.makedirs(cw_dir, exist_ok=True)
    os.makedirs(cw_woirf_dir, exist_ok=True)
    os.makedirs(fl1_dir, exist_ok=True)
    os.makedirs(fl2_dir, exist_ok=True)
    os.makedirs(fl3_dir, exist_ok=True)
    classes = 7
    for name in img_list:
        cw_label = cv2.imread(oj(original_label_dir, name), -1)
        cw_label[np.where(cw_label == 1)] = 7
        cw_label[np.where(cw_label == 3)] = 6
        cw_label[np.where(cw_label == 2)] = 4
        cv2.imwrite(oj(cw_dir, name), cw_label)

        h, w = cw_label.shape
        ep_ly_label = np.zeros_like(cw_label)
        ep_cw_woirf_label = np.zeros_like(cw_label)
        for i in range(classes):
            ly_dir = oj(ly_dirs, 'layer' + str(i))
            os.makedirs(ly_dir, exist_ok=True)
            cv2.imwrite(oj(ly_dir, name), ep_ly_label)

        cw_woirf_label = cw_label.copy()

        cw_woirf_label[np.where(cw_woirf_label==7) ] = 0
        #print(np.max(cw_label), np.max(cw_woirf_label))
        cv2.imwrite(oj(cw_woirf_dir, name), cw_woirf_label)

        fl1_label = np.zeros_like(cw_label)
        fl1_label[np.where(cw_label == 4)] = 1
        cv2.imwrite(oj(fl1_dir, name), fl1_label)
        fl2_label = np.zeros_like(cw_label)
        fl2_label[np.where(cw_label == 6)] = 1
        cv2.imwrite(oj(fl2_dir, name), fl2_label)
        fl3_label = np.zeros_like(cw_label)
        fl3_label[np.where(cw_label == 7)] = 1
        cv2.imwrite(oj(fl3_dir, name), fl3_label)
if __name__ == '__main__':
    main(root='./datasets',
         dataset = 'retouch',
         oct_device='Topcon')
