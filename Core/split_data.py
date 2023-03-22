import cv2
import numpy as np
import os
from os.path import join as oj
import random
from PIL import Image
import json

def save_img(img, path=None, extension=None):
    if extension == '.tiff' and extension == '.tif':
        img = Image.fromarray(img)
        img.save(path + extension)
    else:
        cv2.imwrite(path + extension, img)
def read_img(path=None, extension=None):
    if extension == '.tiff' and extension == '.tif':
        img = Image.open(path + extension)
    else:
        img = cv2.imread(path + extension, -1)
    return img

def split_data_yifuyuan(root,random_select=False):
    imgfolder = 'flatted_IN_img_512'
    labelfolder = 'flatted_label_512'
    original_imglist =  os.listdir(oj(root,'preprocessing_data', imgfolder))
    #trainlist=os.listdir('/data/whl/OCT_layer_seg/2ColoredBScan_145/preprocessing_data/train/flatted_IN_img_512')
    #vallist=[x for x in original_imglist if x not in trainlist]

    with open(oj(root, 'train.json'), 'r') as f1:
        train_imglist = json.load(f1)
    with open(oj(root, 'test.json'), 'r') as f2:
        val_imglist = json.load(f2)
    label_path = oj(root,  'preprocessing_data', labelfolder)
    print(train_imglist)
    for n, name in enumerate(train_imglist):
        realname, extension = os.path.splitext(name)

        img = cv2.imread(oj(root, 'preprocessing_data', imgfolder, name), -1)

        train_img_path = oj(root, 'train', imgfolder)
        os.makedirs(train_img_path, exist_ok=True)
        cv2.imwrite(oj(train_img_path, realname+'---'+str(n)+'---'+extension), img)
        print(realname+'---'+str(n)+'---'+extension)
        fly_label_path = oj(label_path, 'covered')
        fly2_label_path = oj(label_path, 'covered_wo_irf')
        fly_label = cv2.imread(oj(fly_label_path, name), 0)
        fly2_label = cv2.imread(oj(fly2_label_path, name), 0)
        train_fly_label_path = oj(root, 'train', labelfolder, 'covered')
        train_fly2_label_path = oj(root, 'train', labelfolder, 'covered_wo_irf')
        train_fl_label_path = oj(root, 'train', labelfolder, 'fluid3')
        train_fl1_label_path = oj(root, 'train', labelfolder, 'fluid1')
        train_fl2_label_path = oj(root, 'train', labelfolder, 'fluid2')
        os.makedirs(train_fly_label_path, exist_ok=True)
        os.makedirs(train_fl_label_path, exist_ok=True)
        os.makedirs(train_fl1_label_path, exist_ok=True)
        os.makedirs(train_fl2_label_path, exist_ok=True)
        os.makedirs(train_fly2_label_path, exist_ok=True)
        for i in range(7):
            ly_label_path = oj(label_path, 'layers', 'layer'+str(i))
            ly_label = cv2.imread(oj(ly_label_path, name), 0)
            #train_ly_label_path1 = oj(root, 'train', labelfolder, 'layer' + str(i))
            train_ly_label_path2 = oj(root, 'train', labelfolder, 'layers', 'layer' + str(i))
            #os.makedirs(train_ly_label_path1, exist_ok=True)
            os.makedirs(train_ly_label_path2, exist_ok=True)
            #cv2.imwrite(oj(train_ly_label_path1, realname+'---'+str(n)+'---'+extension), ly_label)
            cv2.imwrite(oj(train_ly_label_path2, realname + '---' + str(n) + '---' + extension), ly_label)
        cv2.imwrite(oj(train_fly_label_path, realname+'---'+str(n)+'---'+extension), fly_label)
        cv2.imwrite(oj(train_fly2_label_path, realname + '---' + str(n) + '---' + extension), fly2_label)
        fl_label = cv2.imread(oj(root, 'preprocessing_data', labelfolder, 'fluid3', name), 0)
        fl1_label = cv2.imread(oj(root, 'preprocessing_data', labelfolder, 'fluid1', name), 0)
        fl2_label = cv2.imread(oj(root, 'preprocessing_data', labelfolder, 'fluid2', name), 0)
        cv2.imwrite(oj(train_fl_label_path, realname + '---' + str(n) + '---' + extension),
                    fl_label)
        cv2.imwrite(oj(train_fl1_label_path, realname + '---' + str(n) + '---' + extension),
                    fl1_label)
        cv2.imwrite(oj(train_fl2_label_path, realname + '---' + str(n) + '---' + extension),
                    fl2_label)
    for n, name in enumerate(val_imglist):
        realname, extension = os.path.splitext(name)
        img = cv2.imread(oj(root, 'preprocessing_data', imgfolder, name), -1)
        val_img_path = oj(root,'val', imgfolder)
        os.makedirs(val_img_path, exist_ok=True)
        cv2.imwrite(oj(val_img_path, realname+'---'+str(len(train_imglist)+n)+'---'+extension), img)
        for i in range(7):
            ly_label_path = oj(label_path, 'layers', 'layer'+str(i))
            ly_label = cv2.imread(oj(ly_label_path, name), 0)
            val_ly_label_path1 = oj(root, 'val', labelfolder, 'layer' + str(i))
            val_ly_label_path2 = oj(root, 'val', labelfolder, 'layers', 'layer'+str(i))
            #os.makedirs(val_ly_label_path1, exist_ok=True)
            os.makedirs(val_ly_label_path2, exist_ok=True)
            #cv2.imwrite(oj(val_ly_label_path1, realname+'---'+ str(len(train_imglist)+n)+'---'+extension), ly_label)
            cv2.imwrite(oj(val_ly_label_path2, realname + '---' + str(len(train_imglist) + n) + '---' + extension),
                        ly_label)
        fly_label_path = oj(label_path, 'covered')
        fly2_label_path = oj(label_path, 'covered_wo_irf')
        fly_label = cv2.imread(oj(fly_label_path, name), 0)
        fly2_label = cv2.imread(oj(fly2_label_path, name), 0)
        val_fly_label_path = oj(root, 'val', labelfolder, 'covered')
        val_fly2_label_path = oj(root, 'val', labelfolder, 'covered_wo_irf')
        val_fl_label_path = oj(root, 'val', labelfolder, 'fluid3')
        val_fl1_label_path = oj(root, 'val', labelfolder, 'fluid1')
        val_fl2_label_path = oj(root, 'val', labelfolder, 'fluid2')

        os.makedirs(val_fly_label_path, exist_ok=True)
        os.makedirs(val_fl_label_path, exist_ok=True)
        os.makedirs(val_fl1_label_path, exist_ok=True)
        os.makedirs(val_fl2_label_path, exist_ok=True)
        os.makedirs(val_fly2_label_path, exist_ok=True)
        cv2.imwrite(oj(val_fly_label_path, realname+'---'+str(len(train_imglist)+n)+'---'+extension), fly_label)
        cv2.imwrite(oj(val_fly2_label_path, realname + '---' + str(len(train_imglist) + n) + '---' + extension),
                    fly2_label)
        fl_label = cv2.imread(oj(root, 'preprocessing_data', labelfolder, 'fluid3', name), 0)
        fl1_label = cv2.imread(oj(root, 'preprocessing_data', labelfolder, 'fluid1', name), 0)
        fl2_label = cv2.imread(oj(root, 'preprocessing_data', labelfolder, 'fluid2', name), 0)
        cv2.imwrite(oj(val_fl_label_path, realname + '---' + str(len(train_imglist) + n) + '---' + extension),
                    fl_label)
        cv2.imwrite(oj(val_fl1_label_path, realname + '---' + str(len(train_imglist) + n) + '---' + extension),
                    fl1_label)
        cv2.imwrite(oj(val_fl2_label_path, realname + '---' + str(len(train_imglist) + n) + '---' + extension),
                    fl2_label)


def split_data_duke(root,random_select=False):
    imgfolder = 'flatted_IN_img_512'
    labelfolder = 'flatted_label_512'
    original_img_path = oj(root, 'preprocessing_data', imgfolder)
    original_label_path = oj(root, 'preprocessing_data', labelfolder)
    img_list = os.listdir(original_img_path)
    print(img_list)
    with open(oj(root, 'train.json'), 'r') as f1:
        train_list=json.load(f1)
    with open(oj(root, 'test.json'), 'r') as f2:
        val_list=json.load(f2)
    #saving splitted data
    train_img_path = oj(root, 'train', imgfolder)
    val_img_path = oj(root, 'val', imgfolder)
    test_img_path = oj(root, 'test', imgfolder)


    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)
    if not os.path.exists(val_img_path):
        os.makedirs(val_img_path)
    if not os.path.exists(test_img_path):
        os.makedirs(test_img_path)

    for train_name in train_list:
        realname, extension = os.path.splitext(train_name)
        img = cv2.imread(oj(original_img_path, train_name), 0)
        cv2.imwrite(oj(train_img_path, train_name), img)

        label = read_img(oj(original_label_path, 'covered', realname), extension='.png')
        label_wo_fl= read_img(oj(original_label_path, 'covered_wo_irf', realname), extension='.png')
        label_fl = read_img(oj(original_label_path, 'fluid', realname), extension='.png')

        if not os.path.exists(oj(root, 'train', labelfolder, 'covered')):
            os.makedirs(oj(root, 'train', labelfolder, 'covered'))
        if not os.path.exists(oj(root, 'train', labelfolder, 'covered_wo_irf')):
            os.makedirs(oj(root, 'train', labelfolder, 'covered_wo_irf'))
        if not os.path.exists(oj(root, labelfolder, 'label', 'fluid')):
            os.makedirs(oj(root, 'train', labelfolder, 'fluid'), exist_ok=True)
        if not os.path.exists(oj(root, 'train', labelfolder, 'layers')):
            os.makedirs(oj(root, 'train', labelfolder, 'layers'))
        save_img(label, path=oj(oj(root, 'train', labelfolder, 'covered'), realname), extension='.png')
        save_img(label_wo_fl, path=oj(oj(root, 'train', labelfolder, 'covered_wo_irf'), realname),
                 extension='.png')
        save_img(label_fl, path=oj(oj(root, 'train', labelfolder, 'fluid'), realname), extension='.png')
        #save_img(label_ly, path=oj(oj(root, 'train', labelfolder, 'layers'), realname), extension='.png')
        for i in range(8):
            train_label_path = oj(root, 'train', labelfolder)
            #os.makedirs(oj(train_label_path, 'layer'+str(i)),exist_ok=True)
            os.makedirs(oj(train_label_path, 'layers', 'layer'+str(i)),exist_ok=True)
            label = read_img(oj(original_label_path, 'layers', 'layer' + str(i), realname), extension='.png')
            #save_img(label, path=oj(train_label_path, 'layer'+str(i), realname), extension='.png')
            save_img(label, path=oj(train_label_path, 'layers', 'layer'+str(i),realname), extension='.png')

    for val_name in val_list:
        realname, extension = os.path.splitext(val_name)
        img = cv2.imread(oj(original_img_path, val_name), 0)
        cv2.imwrite(oj(val_img_path, val_name), img)
        label = read_img(oj(original_label_path, 'covered', realname), extension='.png')
        label_wo_fl = read_img(oj(original_label_path, 'covered_wo_irf', realname), extension='.png')
        label_fl = read_img(oj(original_label_path, 'fluid', realname), extension='.png')
        if not os.path.exists(oj(root, 'val', labelfolder, 'covered')):
            os.makedirs(oj(root, 'val', labelfolder, 'covered'))
        if not os.path.exists(oj(root, 'val', labelfolder, 'covered_wo_irf')):
            os.makedirs(oj(root, 'val', labelfolder, 'covered_wo_irf'))
        if not os.path.exists(oj(root, 'val', labelfolder, 'fluid')):
            os.makedirs(oj(root, 'val', labelfolder, 'fluid'))
        if not os.path.exists(oj(root, 'val', labelfolder, 'layers')):
            os.makedirs(oj(root, 'val', labelfolder, 'layers'))
        save_img(label, path=oj(oj(root, 'val', labelfolder, 'covered'), realname), extension='.png')
        save_img(label_wo_fl, path=oj(oj(root, 'val', labelfolder, 'covered_wo_irf'), realname),
                 extension='.png')
        save_img(label_fl, path=oj(oj(root, 'val', labelfolder, 'fluid'), realname), extension='.png')

        for i in range(8):
            val_label_path = oj(root, 'val', labelfolder)
            #os.makedirs(oj(val_label_path, 'layer' + str(i)),exist_ok=True)
            os.makedirs(oj(val_label_path, 'layers', 'layer' + str(i)),exist_ok=True)
            label = read_img(oj(original_label_path, 'layers','layer' + str(i), realname), extension='.png')
            #save_img(label, path=oj(val_label_path, 'layer' + str(i), realname), extension='.png')
            save_img(label, path=oj(val_label_path, 'layers', 'layer' + str(i), realname),extension='.png')
def main(root='./datasets', dataset='yifuyuan'):
    if dataset == 'yifuyuan':
        split_data_yifuyuan(os.path.join(root, dataset))
    elif dataset == 'duke':
        split_data_duke(os.path.join(root, dataset))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./datasets', help='数据集的根目录')
    parser.add_argument('--dataset', default='yifuyuan', help='数据集名称')
    args = parser.parse_args()
    main(root=args.root, dataset=args.dataset)