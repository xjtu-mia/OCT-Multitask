import cv2
import numpy as np
import os
import random
from PIL import Image

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

def splitting(indir,root,rescale_fl_label=False,oct_device='Cirrus', seed=1, id=0):
    original_img_path = os.path.join(indir, 'preprocessing_data/flatted_IN_img_512')
    original_label_path = os.path.join(indir, 'preprocessing_data/pseudo_label')
    if not os.path.exists(original_label_path):
        raise ValueError('please run --generate_pseudo first!')
    img_list = os.listdir(original_img_path)
    print(len(img_list))
    selected_val_dict = {
        'Cirrus':
        [[1,2,3,6],
        [4,5,10,11],
        [7,14,18,15],
        [8,20,21,23],
        [9,13,16,24],
        [12,17,19,22]],
        'Spectralis':
        [[1, 6, 10, 11],
        [2, 3, 7, 13],
        [4, 9, 12, 19],
        [5, 14, 15, 21],
        [8, 18, 20, 23],
        [16, 17, 22, 24]],
        'Topcon':
        [[1, 3, 13, 19],
        [2, 7, 12, 16],
        [4, 8, 9, 18],
        [5, 6, 11, 15],
        [10, 14, 17],
        [20, 21, 22]]
    }
    selected_val_list = selected_val_dict[oct_device]
    m=id
    img_list_tmp = img_list.copy()
    train_num = int(len(img_list) * 0.5)
    val_num = int(len(img_list) * 0.5)
    test_num = int(len(img_list)) - train_num - val_num
    val_list = []
    print(m, selected_val_list)
    val_list_tmp = list(map(str, selected_val_list[m]))


    for name in img_list:
        #print(name.split('_'))
        if name.split('_')[1] in val_list_tmp:
            val_list.append(name)
    for t in val_list:
        img_list_tmp.remove(t)

    train_list = img_list_tmp

    #saving splitted data
    train_img_path = os.path.join(root, str(m), 'train', 'flatted_IN_img_512')
    val_img_path = os.path.join(root, str(m), 'val', 'flatted_IN_img_512')
    #test_img_path = os.path.join(root, 'test', 'flatted_IN_img')


    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)
    if not os.path.exists(val_img_path):
        os.makedirs(val_img_path)

    print(len(train_list))
    print(len(val_list))
    for train_name in train_list:
        realname, extension = os.path.splitext(train_name)
        img = cv2.imread(os.path.join(original_img_path, train_name), 0)
        cv2.imwrite(os.path.join(train_img_path, train_name), img)
        label = read_img(os.path.join(original_label_path, 'covered', realname), extension='.png')
        if rescale_fl_label:
            label[np.where(label == 4)] = 1
            label[np.where(label == 6)] = 2
            label[np.where(label == 7)] = 3
        label_wo_irf = read_img(os.path.join(original_label_path, 'covered_wo_irf', realname), extension='.png')
        label_ly = read_img(os.path.join(original_label_path, 'layers', realname), extension='.png')
        label_fl1 = read_img(os.path.join(original_label_path, 'fluid1', realname), extension='.png')
        label_fl2 = read_img(os.path.join(original_label_path, 'fluid2', realname), extension='.png')
        label_fl3 = read_img(os.path.join(original_label_path, 'fluid3', realname), extension='.png')
        label_save_path = os.path.join(root, str(m), 'train', 'pseudo_label')
        if not os.path.exists(os.path.join(label_save_path, 'covered')):
            os.makedirs(os.path.join(label_save_path, 'covered'))
        if not os.path.exists(os.path.join(label_save_path, 'covered_wo_irf')):
            os.makedirs(os.path.join(label_save_path, 'covered_wo_irf'))
        if not os.path.exists(os.path.join(label_save_path, 'fluid1')):
            os.makedirs(os.path.join(label_save_path, 'fluid1'))
        if not os.path.exists(os.path.join(label_save_path, 'fluid2')):
            os.makedirs(os.path.join(label_save_path, 'fluid2'))
        if not os.path.exists(os.path.join(label_save_path, 'fluid3')):
            os.makedirs(os.path.join(label_save_path, 'fluid3'))
        if not os.path.exists(os.path.join(label_save_path, 'layers')):
            os.makedirs(os.path.join(label_save_path, 'layers'))
        save_img(label, path=os.path.join(os.path.join(label_save_path, 'covered'), realname), extension='.png')
        save_img(label_wo_irf, path=os.path.join(os.path.join(label_save_path, 'covered_wo_irf'), realname),
                 extension='.png')

        save_img(label_fl1, path=os.path.join(os.path.join(label_save_path, 'fluid1'), realname), extension='.png')
        save_img(label_fl2, path=os.path.join(os.path.join(label_save_path, 'fluid2'), realname),
                 extension='.png')
        save_img(label_fl3, path=os.path.join(os.path.join(label_save_path, 'fluid3'), realname),
                 extension='.png')

        for i in range(7):
            train_label_path = os.path.join(label_save_path, 'layers', 'layer' + str(i))
            if not os.path.exists(train_label_path):
                os.makedirs(train_label_path)
            label = read_img(os.path.join(original_label_path, 'layers', 'layer' + str(i), realname), extension='.png')
            save_img(label, path=os.path.join(train_label_path, realname), extension='.png')

    for val_name in val_list:
        realname, extension = os.path.splitext(val_name)
        img = cv2.imread(os.path.join(original_img_path, val_name), 0)
        cv2.imwrite(os.path.join(val_img_path, val_name), img)
        label = read_img(os.path.join(original_label_path, 'covered', realname), extension='.png')
        if rescale_fl_label:
            label[np.where(label == 4)] = 1
            label[np.where(label == 6)] = 2
            label[np.where(label == 7)] = 3
        label_wo_irf = read_img(os.path.join(original_label_path, 'covered_wo_irf', realname), extension='.png')
        label_ly = read_img(os.path.join(original_label_path, 'layers', realname), extension='.png')
        label_fl1 = read_img(os.path.join(original_label_path, 'fluid1', realname), extension='.png')
        label_fl2 = read_img(os.path.join(original_label_path, 'fluid2', realname), extension='.png')
        label_fl3 = read_img(os.path.join(original_label_path, 'fluid3', realname), extension='.png')
        label_save_path = os.path.join(root, str(m), 'val', 'pseudo_label')
        if not os.path.exists(os.path.join(label_save_path, 'covered')):
            os.makedirs(os.path.join(label_save_path, 'covered'))
        if not os.path.exists(os.path.join(label_save_path, 'covered_wo_irf')):
            os.makedirs(os.path.join(label_save_path, 'covered_wo_irf'))
        if not os.path.exists(os.path.join(label_save_path, 'fluid1')):
            os.makedirs(os.path.join(label_save_path, 'fluid1'))
        if not os.path.exists(os.path.join(label_save_path, 'fluid2')):
            os.makedirs(os.path.join(label_save_path, 'fluid2'))
        if not os.path.exists(os.path.join(label_save_path, 'fluid3')):
            os.makedirs(os.path.join(label_save_path, 'fluid3'))
        if not os.path.exists(os.path.join(label_save_path, 'layers')):
            os.makedirs(os.path.join(label_save_path, 'layers'))

        save_img(label, path=os.path.join(os.path.join(label_save_path, 'covered'), realname), extension='.png')
        save_img(label_wo_irf,
                 path=os.path.join(os.path.join(label_save_path, 'covered_wo_irf'), realname),
                 extension='.png')

        save_img(label_fl1, path=os.path.join(os.path.join(label_save_path, 'fluid1'), realname), extension='.png')
        save_img(label_fl2, path=os.path.join(os.path.join(label_save_path, 'fluid2'), realname), extension='.png')
        save_img(label_fl3, path=os.path.join(os.path.join(label_save_path, 'fluid3'), realname), extension='.png')

        for i in range(7):
            val_label_path = os.path.join(label_save_path, 'layers', 'layer' + str(i))
            if not os.path.exists(val_label_path):
                os.makedirs(val_label_path)

            label = read_img(os.path.join(original_label_path, 'layers', 'layer' + str(i), realname), extension='.png')
            save_img(label, path=os.path.join(val_label_path, realname), extension='.png')




if __name__ == '__main__':
    indir = './datasets/retouch/Topcon'
    save_dir = './datasets/retouch/Topcon_yifuyuan/mix/cross_valid'
    rescale_fl_label = False
    from rename_yifuyuan2retouch import main as rename
    for id in range(6):
        splitting(indir, save_dir, rescale_fl_label, oct_device='Topcon', seed=8830, id=id)
        rename(out_root=os.path.join(save_dir, str(id), 'train'), label_folder='pseudo_label')