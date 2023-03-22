import matplotlib.pyplot as plt
import torch.utils.data as data
import PIL.Image as Image
import os
from skimage import io
import torchvision.transforms as transforms
import torch
import numpy as np
import skimage.transform as trans
import cv2
import random
"""
def make_dataset(root):
    imgs = []
    image_dir = os.path.join(root,'image')
    mask_dir = os.path.join(root,'label')
    n = len(os.listdir(image_dir)) 
    img_list = os.listdir(image_dir)
    for name in img_list:
        img = os.path.join(image_dir, name)
        mask = os.path.join(mask_dir, name.split('.')[0]+'_mask.png')
        #img = os.path.join(image_dir, "%03d.png" % i)
        #mask = os.path.join(mask_dir, "%03d_mask.png" % i)
        
    
        imgs.append((img, mask))
    return imgs
"""
#保证listdir()读取的图片名顺序固定
def get_filelist(root, dataset='yifuyuan'):
    filelist = os.listdir(root)
    if dataset=='duke':
        filelist.sort(key=lambda x: int(x.split(".")[0].split("_")[0]) * 100 + int(x.split(".")[0].split("_")[3]))
        # filelist.sort(key=lambda x: int(x.split(".")[0].split("_")[1]) * 10 + int(x.split(".")[0].split("_")[2]))
    elif dataset=='yifuyuan':
        filelist.sort(key=lambda x: int(x.split("---")[1]))
    elif dataset=='retouch':
        filelist.sort(key=lambda x: int(x.split(".")[0].split("_")[1]) * 1000 + int(x.split(".")[0].split("_")[2]))
    return filelist

def make_dataset(root,image_fold,label_fold_ly,label_fold_fl, label_fold_lyfl,
                 dataset='yifuyuan', ly_classes=8,multitask=True,oversampling=False):
    imgs = []
    image_dir = os.path.join(root,image_fold)
    mask_dir_ly = os.path.join(root,label_fold_ly)
    mask_dir_fl = os.path.join(root, label_fold_fl)
    mask_dir_lyfl = os.path.join(root, label_fold_lyfl)
    '''image_v_dir = os.path.join(root, 'm1/flatted_label/coordinate_map_v')
    image_h_dir = os.path.join(root, 'm1/flatted_label/coordinate_map_h')'''
    n = len(os.listdir(image_dir)) 
    img_list = get_filelist(image_dir, dataset=dataset)
    #oversampling the subjects which have IRF
    if oversampling:
        img_list += ['manual-chengpengxiao-QYZ_39---9---.png', 'manual-chengpengxiao-QYZ_59---10---.png',
                     'manual-dudeyin-QYZ_76---22---.png' ,'manual-duhaining-QYZ_56---24---.png',
                     'manual-duhaining-QYZ_72---25---.png', 'manual-zhangchangchun-QYZ_46---37---.png',
                     'manual-zhanghongen-QYZ_86---40---.png', 'manual-zhangsuiyou-QYZ_47---42---.png',
                     'manual-zhangyuxiang-QYZ_52---46---.png', 'manual-zhangyuxiang-QYZ_67---47---.png',
                     'manual-zhangyuxiang-QYZ_74---48---.png', 'manual-zhaofangxia-QYZ_68---49---.png',
                     'manual-zhaofangxia-QYZ_79---50---.png', 'manual-zhaoyuanbiao-QYZ_78---54---.png'
                     ]
    print(img_list)
    for name in img_list:
        if multitask:
            #print(int(name.split(".")[0].split("_")[1])*10+int(name.split(".")[0].split("_")[2]))
            realname, extension = os.path.splitext(name)
            img = os.path.join(image_dir, name)
            #mask_ly = os.path.join(mask_dir_ly, realname + extension)
            mask_fl = os.path.join(mask_dir_fl, realname + extension)
            mask_lyfl = os.path.join(mask_dir_lyfl, realname + extension)
            '''image_v = os.path.join(image_v_dir, realname + '.tif')
            image_h = os.path.join(image_h_dir, realname + '.tif')'''
            layer_list = []
            for i in range(ly_classes):
                mask_ly_i = os.path.join(mask_dir_ly, 'layer'+str(i), realname + extension)
                layer_list.append(mask_ly_i)
            #img = os.path.join(image_dir, "%03d.png" % i)
            #mask = os.path.join(mask_dir, "%03d_mask.png" % i)
            #print(layer_list)
            imgs.append((img, layer_list, mask_lyfl, mask_fl))
        else:
            realname, extension = os.path.splitext(name)
            img = os.path.join(image_dir, name)
            # mask_ly = os.path.join(mask_dir_ly, realname + extension)
            mask_fl = os.path.join(mask_dir_fl, realname + extension)

            # img = os.path.join(image_dir, "%03d.png" % i)
            # mask = os.path.join(mask_dir, "%03d_mask.png" % i)
            # print(layer_list)
            imgs.append((img, mask_fl))
    return imgs

class image_aug:
    def __init__(self, img, scale, zoom=True, shift=True, fill_mode=0, resample=0):
        self.img = img
        self.scale = scale
        self.zoom = zoom
        self.shift = shift
        self.fill_mode = fill_mode
        self.resample = resample
    def aug(self):

        def zoom_img(img, scale, resample=0, mode=0):
            resample_dict = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]
            mode_dict = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE]
            np_img = np.array(img)

            if len(np_img.shape) > 2:
                original_size = np_img.shape[:2]
                height, width = int(np_img.shape[1] * scale[0]), int(np_img.shape[2] * scale[1])
            else:
                original_size = np_img.shape
                height, width = int(np_img.shape[0] * scale[0]), int(np_img.shape[1] * scale[1])
            np_img = cv2.resize(np_img, (width, height), interpolation=resample_dict[resample])

            h_dis = (original_size[0] - height) // 2
            w_dis = (original_size[1] - width) // 2
            if h_dis < 0:
                np_img = np_img[-original_size[0] + height:height, 0:original_size[1]]
                h_dis = 0
            np_img = cv2.copyMakeBorder(np_img, h_dis, h_dis, w_dis, w_dis, mode_dict[mode])
            return Image.fromarray(np_img)
        resample_dict = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]
        mode_dict = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE]
        if self.zoom:
            img = zoom_img(self.img,
                           self.scale,
                           resample=resample_dict[self.resample],
                           mode=mode_dict[self.fill_mode])
        else:
            img = self.img
        return img

def zoom_img(img, scale, resample=0, mode=0):
        resample_dict = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]
        mode_dict = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE]
        np_img = np.array(img)
        if len(np_img.shape) > 2:
            original_size = np_img.shape[:2]
            height, width = int(np_img.shape[1] * scale[0]), int(np_img.shape[2] * scale[1])
        else:
            original_size = np_img.shape
            height, width = int(np_img.shape[0] * scale[0]), int(np_img.shape[1] * scale[1])
        np_img = cv2.resize(np_img, (width, height), interpolation=resample_dict[resample])

        h_dis = (original_size[0] - height) // 2
        w_dis = (original_size[1] - width) // 2
        if h_dis < 0:
            np_img = np_img[-original_size[0] + height:height, 0:original_size[1]]
            h_dis = 0
        np_img = cv2.copyMakeBorder(np_img, h_dis, h_dis, w_dis, w_dis, mode_dict[mode])
        np_img = cv2.resize(np_img, (original_size[1],original_size[0]), interpolation=resample_dict[resample])
        return Image.fromarray(np_img)
def shift_img(img, scale,  mode=0):
    mode_dict = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE,cv2.BORDER_REFLECT]
    np_img = np.array(img)
    if len(np_img.shape) > 2:
        original_size = np_img.shape[:2]
        height, width = int(np_img.shape[1]*scale[0]), int(np_img.shape[2]*scale[1])
    else:
        original_size = np_img.shape
        height, width = int(np_img.shape[0] * scale[0]), int(np_img.shape[1] * scale[1])
    if scale[0] < 0 and scale[1] < 0:
        np_img = np_img[0:original_size[0]+height, 0:original_size[1]+width]
        np_img = cv2.copyMakeBorder(np_img, -height, 0, 0, 0, mode_dict[mode])
        np_img = cv2.copyMakeBorder(np_img, 0, 0, -width, 0, mode_dict[2])
    elif scale[0] > 0 and scale[1] < 0:
        np_img = np_img[height:original_size[0], 0:original_size[1]+width]
        np_img = cv2.copyMakeBorder(np_img, 0, height, 0, 0, mode_dict[mode])
        np_img = cv2.copyMakeBorder(np_img, 0, 0, -width, 0, mode_dict[2])
    elif scale[0] > 0 and scale[1] > 0:
        np_img = np_img[height:original_size[0], width:original_size[1]]
        np_img = cv2.copyMakeBorder(np_img, 0, height, 0, 0, mode_dict[mode])
        np_img = cv2.copyMakeBorder(np_img, 0, 0, width, 0, mode_dict[2])
    elif scale[0] < 0 and scale[1] > 0:
        np_img = np_img[0:original_size[0] + height, width:original_size[1]]
        np_img = cv2.copyMakeBorder(np_img, -height, 0, 0, 0, mode_dict[mode])
        np_img = cv2.copyMakeBorder(np_img, 0, 0, 0, width, mode_dict[2])
    return Image.fromarray(np_img)
def resize(img, target_size, resample=0):
    resample_dict = [cv2.INTER_NEAREST, cv2.INTER_LINEAR]
    np_img = np.array(img)
    np_img = cv2.resize(np_img, (target_size[1], target_size[0]), interpolation=resample_dict[resample])
    return Image.fromarray(np_img)
class imageDataset(data.Dataset):           #pytorch 读取数据的标准格式
    def __init__(self,
                 root,
                 stage = 'train',
                 dataset = 'yifuyuan',
                 image_fold=None,
                 layer_label_fold=None,
                 fluid_label_fold=None,
                 target_size=None,
                 ly_classes=8,
                 pixel_ly_classes=9,
                 fl_classes=2,
                 one_hot='n',
                 regression = False,
                 softargmax = False,
                 multitask = False,
                 fluid_label = False,
                 layer_fluid_label_fold = False,
                 oversampling=False):
        imgs = make_dataset(root,image_fold,layer_label_fold,fluid_label_fold,
                            layer_fluid_label_fold, oversampling=oversampling, dataset=dataset,
                            ly_classes=ly_classes, multitask=multitask)

        self.stage = stage
        self.imgs = imgs
        self.target_size = target_size
        self.one_hot = one_hot
        self.ly_classes = ly_classes
        self.pixel_ly_classes = pixel_ly_classes
        self.fl_classes = fl_classes
        self.regression = regression
        self.softargmax = softargmax
        self.multitask = multitask
        self.fluid_label = fluid_label
       # self.target_size = target_size
    def __getitem__(self, index):
        if self.stage == 'train':
            p1 = np.random.choice([0, 1])
            p2 = np.random.choice([0, 1])  # 在0，1二者中随机取一个# ，
            zoom_scale_h = random.uniform(1, 1)
            shift_scale = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
            #print(shift_scale)
            x_transform = transforms.Compose([  # transforms.Compose()串联多个transform操作
                #transforms.Lambda(lambda img: zoom_img(img, (zoom_scale_h, 1), resample=1, mode=1)),
                transforms.Lambda(lambda img: shift_img(img, shift_scale, mode=1)),
                transforms.Lambda(lambda img: resize(img, self.target_size, resample=1)),
                #transforms.Resize(self.target_size, 2),
                transforms.RandomHorizontalFlip(p=p1),
                transforms.RandomVerticalFlip(p=0),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),  # -> [0,1]
                transforms.Normalize([0.5], [0.5]),  # ->[-1,1]
            ])
            xvh_transform = transforms.Compose([  # transforms.Compose()串联多个transform操作
                #transforms.Lambda(lambda img: zoom_img(img, (zoom_scale_h, 1), resample=1, mode=1)),
                #transforms.Lambda(lambda img: shift_img(img, shift_scale, mode=1)),
                transforms.Lambda(lambda img: resize(img, self.target_size, resample=1)),
                #transforms.Resize(self.target_size, 2),
                #transforms.RandomHorizontalFlip(p=p1),
                #transforms.RandomVerticalFlip(p=0),
                transforms.ToTensor(),  # -> [0,1]
                #transforms.Normalize([0.5], [0.5])  # ->[-1,1]
            ])
            y_transform = transforms.Compose([  # transforms.Compose()串联多个transform操作
                #transforms.Lambda(lambda img: zoom_img(img, (zoom_scale_h, 1), resample=0, mode=1)),
                transforms.Lambda(lambda img: shift_img(img, shift_scale, mode=1)),
                transforms.Lambda(lambda img: resize(img, self.target_size, resample=1)),
                #transforms.Resize(self.target_size, 3),
                transforms.RandomHorizontalFlip(p=p1),
                transforms.RandomVerticalFlip(p=0),
                transforms.ToTensor(),  # -> [0,1]
            ])
        else:
            x_transform = transforms.Compose([  # transforms.Compose()串联多个transform操作
                #transforms.Resize(self.target_size, 2),
                transforms.Lambda(lambda img: resize(img, self.target_size, resample=1)),
                transforms.ToTensor(),  # -> [0,1]
                transforms.Normalize([0.5], [0.5])  # ->[-1,1]
            ])
            xvh_transform = transforms.Compose([  # transforms.Compose()串联多个transform操作
                #transforms.Resize(self.target_size, 2),
                transforms.Lambda(lambda img: resize(img, self.target_size, resample=1)),
                transforms.ToTensor(),  # -> [0,1]
                #transforms.Normalize([0.5], [0.5])  # ->[-1,1]
            ])
            y_transform = transforms.Compose([  # transforms.Compose()串联多个transform操作
                #transforms.Resize(self.target_size, 3),
                transforms.Lambda(lambda img: resize(img, self.target_size, resample=0)),
                transforms.ToTensor(),  # -> [0,1]
            ])
        if self.multitask:
            x_path, y_path_1, y_path_2, y_path_3 = self.imgs[index]
            img_x = Image.open(x_path)

            img_y_ly_ilst = []
            for i, path in enumerate(y_path_1):
                img_y_ly_ilst.append(Image.open(path))
            #img_y_ly = np.array(img_y_ly_ilst)
            img_y_lyfl =Image.open(y_path_2)
            '''img_y_lyfl = np.array(Image.open(y_path_2))
            max_value = np.max(img_y_lyfl)

            if max_value >= self.pixel_ly_classes-1:
                img_y_lyfl[np.where(img_y_lyfl==max_value)] = 0'''
            if self.fluid_label:
                img_y_fl = Image.open(y_path_3)
            #cv2.imshow('1', np.array(img_y_fl))
            #cv2.waitKey(1000)
            # img_x = img_x.resize(self.target_size, Image.BICUBIC)
            # img_y = img_y.resize(self.target_size, Image.NEAREST)
            '''
            if self.common_transform is not None:
                img_x = self.common_transform(img_x)
                # img_y = self.target_transform(img_y)
                img_y_ly = self.common_transform(img_y_ly)
                if self.fluid_label:
                    img_y_fl = self.common_transform(img_y_fl)
            '''


            if x_transform is not None:

                img_x = x_transform(img_x)
            '''if xvh_transform is not None:
                img_x_v = np.array(Image.open(x_v_path))/255
                #print(img_x_v)
                img_x_h = np.array(Image.open(x_h_path))/255
                img_x_v = xvh_transform(img_x_v)
                img_x_h = xvh_transform(img_x_h)
                #print(img_x_h)
                img_x = torch.cat([img_x, img_x_v, img_x_h], dim=0)  #img+coordinate_map_v++coordinate_map_h'''
            if y_transform is not None:
                img_y_ly = torch.zeros((self.ly_classes, img_x.shape[1], img_x.shape[2]))
                for i, img_y_ly_i in enumerate(img_y_ly_ilst):
                    #img_y_ly_i = Image.fromarray(cv2.GaussianBlur(np.array(img_y_ly_i), (1,3), sigmaX=0,sigmaY=1))
                    '''plt.figure()
                    plt.imshow(img_y_ly_i)
                    plt.show()'''
                    img_y_ly_i = y_transform(img_y_ly_i)
                    img_y_ly[i] = img_y_ly_i
                img_y_lyfl = y_transform(img_y_lyfl) * 255    #check the max value in tensor
                if self.fluid_label:
                    img_y_fl = y_transform(img_y_fl)
                    if torch.max(img_y_fl) < 1:
                        img_y_fl = img_y_fl * 255
                    #print(torch.max(img_y_fl))
                    #img_y_fl = img_y_fl
                    #print(torch.max(img_y_fl))

            img_y_new_1 = torch.zeros((self.pixel_ly_classes, img_y_ly.shape[1], img_y_ly.shape[2]))
            #img_y_new_2 = torch.zeros((self.ly_classes, img_y_ly.shape[1], img_y_ly.shape[2]))

            # pixel wise classification

            img_y3 = img_y_new_1.scatter_(0, img_y_lyfl.long(), 1)  # (1,h,w)-->(classes,h,w

            #for i in range(self.ly_classes):
            #    img_y_new_2[i][torch.where(img_y_ly[0] == i + 1)] = 1
            # column wise classification
            img_y_new_2 = img_y_ly
            img_y2 = img_y_new_2

            '''
            pytorch ver<=1.6.0
            torch.argmax值相同时返回最大索引，np.argmax则返回最小索引0
            pytorch ver>=1.7.0
            torch.argmax与np.argmax一致，返回最小索引0
            '''
            img_y_new_2 = torch.argmax(img_y_new_2, dim=1)
            img_y_new_2 = img_y_new_2.to(torch.float32)
            # y index regression
            img_y1 = img_y_new_2
            # print(torch.max(img_y_new))
            if self.fluid_label:
                img_y_new_3 = torch.zeros((self.fl_classes, img_y_fl.shape[1], img_y_fl.shape[2]))
                for i in range(self.fl_classes):
                    img_y_new_3 = img_y_new_3.scatter_(0, img_y_fl.long(), 1)  # (1,h,w)-->(classes,h,w
                    #img_y_new_3[i][torch.where(img_y_fl[0] == i)] = 1
                img_y4 = img_y_new_3  # (1,h,w)-->(classes,h,w


                return img_x, img_y1, img_y2, img_y3, img_y4
            else:
                return img_x, img_y1, img_y2, img_y3
        else:
            x_path, y_path = self.imgs[index]
            img_x = Image.open(x_path)
            img_y = Image.open(y_path)
            #img_x = img_x.resize(self.target_size, Image.BICUBIC)
            #img_y = img_y.resize(self.target_size, Image.NEAREST)

            if x_transform is not None:
                img_x = x_transform(img_x)
            if y_transform is not None:
                img_y = y_transform(img_y)
            if torch.max(img_y) < 1:
                img_y = img_y * 255
            if self.regression:
                self.one_hot = 'n'
                img_y_new = torch.zeros((self.ly_classes, img_y.shape[1], img_y.shape[2]))
                for i in range(self.ly_classes):
                    img_y_new[i][torch.where(img_y[0] == i + 1)] = 1
                if self.softargmax:
                    '''
                    pytorch ver<=1.6.0
                    torch.argmax值相同时返回最大索引，np.argmax则返回最小索引0
                    pytorch ver>=1.7.0
                    torch.argmax与np.argmax一致，返回最小索引0
                    '''
                    img_y_new = torch.argmax(img_y_new, dim=1)
                    img_y_new = img_y_new.to(torch.float32)
                img_y = img_y_new
                    #print(torch.max(img_y_new))
            if self.one_hot=='y':
                img_y_onehot = torch.zeros((self.fl_classes, img_y.shape[1], img_y.shape[2]))
                img_y_onehot.scatter_(0, img_y.long(), 1)  #(1,h,w)-->(classes,h,w
                img_y = img_y_onehot

            return img_x, img_y

    def __len__(self):
        return len(self.imgs)
    
def train_transform_data(root=None,
                         dataset='yifuyuan',
                         image_fold=None,
                         layer_label_fold=None,
                         fluid_label_fold=None,
                         layer_fluid_label_fold=None,
                         fl_classes=2,
                         ly_classes=8,
                         pixel_ly_classes = 9,
                         one_hot="n",
                         target_size=(512,512),
                         regression = False,
                         softargmax = False,
                         multitask = False,
                         fluid_label = False,
                         oversampling=False):


    # 参数解析器,用来解析从终端读取的命令
    dataset = imageDataset(root,
                           stage='train',
                           dataset=dataset,
                           image_fold=image_fold,
                           layer_label_fold=layer_label_fold,
                           fluid_label_fold=fluid_label_fold,
                           layer_fluid_label_fold=layer_fluid_label_fold,
                           fl_classes = fl_classes,
                           ly_classes = ly_classes,
                           pixel_ly_classes=pixel_ly_classes,
                           target_size=target_size,
                           one_hot=one_hot,
                           regression = regression,
                           softargmax = softargmax,
                           multitask=multitask,
                           fluid_label = fluid_label,
                           oversampling=oversampling)
    return dataset


def val_transform_data(root=None,
                      dataset='yifuyuan',
                       image_fold=None,
                       layer_label_fold=None,
                       fluid_label_fold=None,
                        layer_fluid_label_fold=None,
                       fl_classes=2,
                       ly_classes=8,
                        pixel_ly_classes = 9,
                       one_hot="n",
                       target_size=(512, 512),
                       regression = False,
                       softargmax = False,
                       multitask = False,
                       fluid_label = False):

    # 参数解析器,用来解析从终端读取的命令
    dataset = imageDataset(root,
                           stage='test',
                           dataset=dataset,
                           image_fold=image_fold,
                           layer_label_fold=layer_label_fold,
                           fluid_label_fold=fluid_label_fold,
                           layer_fluid_label_fold=layer_fluid_label_fold,
                           fl_classes=fl_classes,
                           ly_classes=ly_classes,
                           pixel_ly_classes = pixel_ly_classes,
                           target_size=target_size,
                           one_hot=one_hot,
                           regression=regression,
                           softargmax=softargmax,
                           multitask=multitask,
                           fluid_label=fluid_label)
    return dataset


COLOR_DICT_1 = np.asarray([[255,255,255],
              [255,0,255],
              [0,255,255],
              [255,255,0],
              [0,0,255],
              [0,255,0],
              [255,0,0],
              [0,127,255]])
COLOR_DICT_2 = np.asarray([[255,255,0],
                            [0,255,0],
                            [255,0,0],
                           ])
COLOR_DICT_3 = np.asarray([
              [255,255,255],
              [255,0,255],
              [0,255,255],
              [255,255,0],
              [0,0,255],
              [0,255,0],
              [255,0,0],
              [255,127,127],
              [127,255,255]
            ])
def saveResult(save_path,
               test_path,
               target_size,
               dataset = 'yifuyuan',
               npyfile_ly=np.zeros((1)),
               npyfile_fl=np.zeros((1)),
               npyfile_ly_fl=np.zeros((1)),
               npyfile_ly_area=np.zeros((1)),
               flag_multi_class=True,
               ly_classes=8,
               pixel_ly_classes=9,
               fl_classes=2,
               regression=False,
               task=None):
    def soft_argmax(data, axis=0, beta=10):
        dim = data.shape
        if len(dim) == 2:
            index = np.array(list(range(0, dim[axis])))
            for i in range(dim[1 - axis]):
                data[:, i] = np.exp(data[:, i] * beta) / np.sum(np.exp(data[:, i] * beta)) * index
        return np.sum(data, axis=axis)
    filelist_test = get_filelist(test_path, dataset=dataset)
    name = []
    for filename in filelist_test:
        (realname, extension) = os.path.splitext(filename)
        name.append(realname)

    if npyfile_ly.any():
        for i, item in enumerate(npyfile_ly):
            cd1 = []
            cd2 = []
            img_test = io.imread(os.path.join(test_path, '%s.png' % name[i]))
            img_test = trans.resize(img_test, target_size)
            io.imsave(os.path.join(save_path, '%s.png' % name[i]), img_test)
            if flag_multi_class:
                if regression:
                    img = item
                    img_out_list = []
                    h, w = img[:, :, 0].shape
                    for c in range(ly_classes):
                        img_out = np.zeros(img[:, :, 0].shape + (3,))
                        y_index = np.argmax(img[:,:,c], axis=0)

                        if c <= ly_classes-2:
                            '''cv2.imshow('0', img[:,:,c])
                            cv2.imshow('1', img[:,:,c+1])
                            cv2.waitKey(3000)'''
                            img_mask = np.zeros(img[:, :, 0].shape + (3,))

                            y_index_top = np.argmax(img[:,:,c], axis=0)
                            y_index_bottom = np.argmax(img[:,:,c+1], axis=0)
                            #print(img.shape)
                            for j in range(w):
                                if y_index_bottom[j] - y_index_top[j]>1:
                                    for b in range(y_index_top[j], y_index_bottom[j]+1, 1):
                                        img_mask[b, j] = [255,255,255]
                            #img_mask = trans.resize(img_mask, target_size)
                            cd2.append(img_mask)
                        for j in range(w):
                            img_out[y_index[j], j] = [255,255,255]

                        #img_out = trans.resize(img_out, target_size)
                        cd1.append(img_out)
                    img = img_out.astype(np.uint8)
                    img = trans.resize(img, target_size)
                    #io.imsave(os.path.join(save_path, '%s_predict.png' % name[i]), img)
                    lp_list = []
                    ly_list = []
                    for l in range(ly_classes):
                        ly_list.append('layer'+str(l))
                    for l in range(ly_classes-1):
                        lp_list.append('layer'+str(l)+'_pixelwise')
                    for j, s in enumerate(ly_list):
                        path = os.path.join(save_path, s + '_predict')
                        if not os.path.exists(path):
                            os.makedirs(path)
                        #io.imsave(os.path.join(path, '%s_predict_argmax_map.png' % (s + '_' + name[i])), cd1[j])
                        io.imsave(os.path.join(path, '%s_predict_map.png' % (s + '_' + name[i])), item[:, :, j] * 255)
                    '''for j, s in enumerate(lp_list):
                        path = os.path.join(save_path, s + '_predict')
                        if not os.path.exists(path):
                            os.makedirs(path)


                        cv2.imwrite(os.path.join(path, '%s_predict.png' % (s + '_' + name[i])), cd2[j])'''


    if npyfile_fl.any():
        for i, item in enumerate(npyfile_fl):
            cd = []
            img = item
            h, w = img[:, :, 0].shape
            index_of_class = np.argmax(img, axis=-1)
            for c in range(1, fl_classes, 1):
                img_out = np.zeros(img[:, :, 0].shape + (3,))

                img_out[np.where(index_of_class == c)] = COLOR_DICT_2[c - 1][::-1]
                img_out = trans.resize(img_out, target_size)
                cd.append(img_out)
            img = img_out.astype(np.uint8)
            img = trans.resize(img, target_size)
            if fl_classes == 2:
                label_list = ['fluid']
            elif fl_classes == 4:
                label_list = ['SRF', 'PED', 'IRF']
            for j, s in enumerate(label_list):
                path = os.path.join(save_path, s + '_predict')
                if not os.path.exists(path):
                    os.makedirs(path)
                io.imsave(os.path.join(path, '%s_predict.png' % (s + '_' + name[i])), cd[j])
                io.imsave(os.path.join(path, '%s_predict_map.png' % (s + '_' + name[i])), item[:, :, j + 1] * 255)
    if npyfile_ly_fl.any():
        for i, item in enumerate(npyfile_ly_fl):
            cd = []
            img = item
            h, w = img[:, :, 0].shape
            index_of_class = np.argmax(img, axis=-1)
            for c in range(1, pixel_ly_classes, 1):
                img_out = np.zeros(img[:, :, 0].shape + (3,))
                img_out[np.where(index_of_class == c)] = COLOR_DICT_3[c - 1][::-1]
                img_out = trans.resize(img_out, target_size)
                cd.append(img_out)
            img = img_out.astype(np.uint8)
            img = trans.resize(img, target_size)
            lf_list = []
            for l in range(pixel_ly_classes-1):
                lf_list.append('layer'+str(l)+'&fluid')
            for j, s in enumerate(lf_list):
                path = os.path.join(save_path, s + '_predict')
                if not os.path.exists(path):
                    os.makedirs(path)
                io.imsave(os.path.join(path, '%s_predict.png' % (s + '_' + name[i])), cd[j])
                io.imsave(os.path.join(path, '%s_predict_map.png' % (s + '_' + name[i])), item[:, :, j + 1] * 255)
    if npyfile_ly_area.any():

        for i, item in enumerate(npyfile_ly_area):
            cd = []
            img = item
            h, w = img[:, :, 0].shape
            index_of_class = np.argmax(img, axis=-1)
            for c in range(1, ly_classes, 1):
                img_out = np.zeros(img[:, :, 0].shape + (3,))
                img_out[np.where(index_of_class == c)] = [255,255,255]
                img_out = trans.resize(img_out, target_size)
                cd.append(img_out)
            img = img_out.astype(np.uint8)
            img = trans.resize(img, target_size)
            lf_list = []
            for l in range(ly_classes-1):
                lf_list.append('layer'+str(l)+'_pixelwise')
            for j, s in enumerate(lf_list):
                path = os.path.join(save_path, s + '_predict')
                if not os.path.exists(path):
                    os.makedirs(path)
                io.imsave(os.path.join(path, '%s_predict.png' % (s + '_' + name[i])), cd[j])
                io.imsave(os.path.join(path, '%s_predict_map.png' % (s + '_' + name[i])), item[:, :, j+1] * 255)
def draw(img_origin, img_mask, color, dilate=True):
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))  # 定义结构元素的形状和大小
        img_mask = cv2.dilate(img_mask, kernel)  # 膨胀操作
    img_origin[np.where(img_mask != 0)] = color
    return img_origin

def drawmask(origin_path,
             mask_path,
             dataset = 'yifuyuan',
             ly_classes=8,
             pixel_ly_classes=9,
             fl_classes=2,
             task=None,
             target_size=(1024, 1024),
             regression=False,
             mix=True,
             split_IRF=False
             ):

    if not regression:
        classes_ly = ly_classes - 1
    fl_mask_path_list = []
    ly_mask_path_list = []
    lyfl_mask_path_list = []
    ly_area_path_list = []
    label_list = []
    for l in range(ly_classes):
        label_list.append('layer'+str(l))
    for i, layer_name in enumerate(label_list):
        ly_mask_path_list.append(os.path.join(mask_path, label_list[i] + '_predict'))
    fileList1 = get_filelist(origin_path, dataset=dataset)
    if os.path.exists(ly_mask_path_list[0]):
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName)  # 分离文件名和后缀
            img_origin = cv2.imread(os.path.join(origin_path, fileName))
            img_origin = cv2.resize(img_origin, (target_size[1],target_size[0]))
            #img_origin = np.zeros_like(img_origin)
            for i, layer_name in enumerate(label_list):
                if not regression:
                    img_mask = cv2.imread(
                        os.path.join(ly_mask_path_list[i], '%s_predict_map.png' % (label_list[i] + '_' + realname)), 0)
                else:
                    img_mask = cv2.imread(os.path.join(ly_mask_path_list[i], '%s_predict_map.png' % (label_list[i] +'_'+ realname)), 0)
                img_mask = cv2.resize(img_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                img_origin = draw(img_origin, img_mask, COLOR_DICT_1[i])
            cv2.imwrite(os.path.join(mask_path, '%s_layer_predict.png' % realname), img_origin)
    fl_label_list = []
    if fl_classes == 2:
        fl_label_list = ['fluid']
        for i, fluid_name in enumerate(fl_label_list):
            fl_mask_path_list.append(os.path.join(mask_path, fl_label_list[i] + '_predict'))
        fileList1 = get_filelist(origin_path, dataset=dataset)
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName)  # 分离文件名和后缀
            img_origin = cv2.imread(os.path.join(origin_path, fileName))
            img_origin = cv2.resize(img_origin, (target_size[1], target_size[0]))
            for i, fluid_name in enumerate(fl_label_list):
                img_mask = cv2.imread(
                    os.path.join(fl_mask_path_list[i], '%s_predict.png' % (fl_label_list[i] + '_' + realname)),
                    0)
                img_mask = cv2.resize(img_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                img_origin = draw(img_origin, img_mask, COLOR_DICT_2[-1])
            cv2.imwrite(os.path.join(mask_path, '%s_fluid_predict.png' % realname), img_origin)
    if fl_classes == 4:
        fl_label_list = ['SRF','PED','IRF']
        for i, fluid_name in enumerate(fl_label_list):
            fl_mask_path_list.append(os.path.join(mask_path, fl_label_list[i] + '_predict'))
        fileList1 = get_filelist(origin_path, dataset=dataset)
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName)  # 分离文件名和后缀
            img_origin = cv2.imread(os.path.join(origin_path, fileName))
            img_origin = cv2.resize(img_origin, (target_size[1], target_size[0]))
            for i, fluid_name in enumerate(fl_label_list):
                img_mask = cv2.imread(
                    os.path.join(fl_mask_path_list[i], '%s_predict.png' % (fl_label_list[i] + '_' + realname)),
                    0)
                img_mask = cv2.resize(img_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                img_origin = draw(img_origin, img_mask, COLOR_DICT_2[i])
            cv2.imwrite(os.path.join(mask_path, '%s_fluid_predict.png' % realname), img_origin)
    if mix:
        fileList = get_filelist(origin_path, dataset=dataset)
        for fileName in fileList:
            (realname, extension) = os.path.splitext(fileName)
            layer_predict = cv2.imread(os.path.join(mask_path,  '%s_layer_predict.png' % realname))
            layer_predict = cv2.resize(layer_predict, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            if len(fl_label_list)>0:
                for i, fluid_name in enumerate(fl_label_list):
                    fluid_mask = cv2.imread(
                        os.path.join(fl_mask_path_list[i], '%s_predict.png' % (fl_label_list[i] + '_' + realname)),0)
                    fluid_mask = cv2.resize(fluid_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                    mix_predict = draw(layer_predict, fluid_mask, COLOR_DICT_2[i])
                cv2.imwrite(os.path.join(mask_path, '%s_predict_1.png' % realname), mix_predict)
        lyfl_label_list = []
        ly_area_label_list = []
        for l in range(pixel_ly_classes-1):
            lyfl_label_list.append('layer'+str(l)+'&fluid')

        for i, layer_fluid_name in enumerate(lyfl_label_list):
            lyfl_mask_path_list.append(os.path.join(mask_path, lyfl_label_list[i] + '_predict'))
        for l in range(ly_classes-1):  #regression branch
            ly_area_label_list.append('layer'+str(l)+'_pixelwise')

        for i, layer_fluid_name in enumerate(ly_area_label_list):
            ly_area_path_list.append(os.path.join(mask_path, ly_area_label_list[i] + '_predict'))
        fileList1 = os.listdir(origin_path)
        for i, layer_name in enumerate(label_list):
            ly_mask_path_list.append(os.path.join(mask_path, label_list[i] + '_predict'))
        fileList1 = get_filelist(origin_path, dataset=dataset)
        for fileName in fileList1:
            (realname, extension) = os.path.splitext(fileName)  # 分离文件名和后缀
            img_origin = cv2.imread(os.path.join(origin_path, fileName))
            img_origin = cv2.resize(img_origin, (target_size[1], target_size[0]))
            img_origin_ = img_origin.copy()
            if len(fl_label_list)>0:
                if split_IRF:
                    if len(lyfl_label_list) == 7:
                        del lyfl_label_list[-1]
                    for i, fluid_name in enumerate(fl_label_list):
                        fluid_mask = cv2.imread(
                            os.path.join(fl_mask_path_list[i], '%s_predict.png' % (fl_label_list[i] + '_' + realname)), 0)
                else:
                    fluid_mask = cv2.imread(
                        os.path.join(lyfl_mask_path_list[-1], '%s_predict.png' % (lyfl_label_list[-1] + '_' + realname)),
                        0)
                fluid_mask = cv2.resize(fluid_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                #img_origin = np.zeros_like(img_origin)
            for i, layer_fluid_name in enumerate(lyfl_label_list):
                img_mask = cv2.imread(
                    os.path.join(lyfl_mask_path_list[i], '%s_predict.png' % (lyfl_label_list[i] + '_' + realname)),
                    0)
                img_mask = cv2.resize(img_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                img_origin = draw(img_origin, img_mask, COLOR_DICT_3[i])

            if split_IRF:
                if dataset=='duke':
                    img_origin = draw(img_origin, fluid_mask, COLOR_DICT_3[len(lyfl_label_list)-1])
                else:
                    img_origin = draw(img_origin, fluid_mask, COLOR_DICT_3[len(lyfl_label_list)])
            else:
                fluid_mask = cv2.imread(
                    os.path.join(lyfl_mask_path_list[-1], '%s_predict.png' % (lyfl_label_list[-1] + '_' + realname)), 0)
                fluid_mask = cv2.resize(fluid_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(mask_path, '%s_predict_2.png' % realname), img_origin)

            if task == 'index2area':
                for i, layer_fluid_name in enumerate(ly_area_label_list):
                    img_mask = cv2.imread(
                        os.path.join(ly_area_path_list[i], '%s_predict.png' % (ly_area_label_list[i] + '_' + realname)),
                        0)
                    img_mask = cv2.resize(img_mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
                    img_origin_ = draw(img_origin_, img_mask, COLOR_DICT_3[i])
                cv2.imwrite(os.path.join(mask_path, '%s_predict_3.png' % realname), img_origin_)
                img_origin_ = draw(img_origin_, fluid_mask, COLOR_DICT_3[pixel_ly_classes-2])
                cv2.imwrite(os.path.join(mask_path, '%s_predict_3_fluid.png' % realname), img_origin_)

def drawmask_truth(origin_path,
             mask_path,
             save_path,
            dataset = 'yifuyuan',
             ly_classes=8,
              pixel_ly_classes=9,
             fl_classes=2,
             task=None,
             target_size=(1024, 1024),
             regression=False,
             mix=True):
    global img_origin_2
    fl_mask_path_list = []
    ly_mask_path_list = []
    label_list = []
    for l in range(ly_classes):
        label_list.append('layer' + str(l))
    for i, layer_name in enumerate(label_list):
        ly_mask_path_list.append(os.path.join(mask_path, 'layers', label_list[i]))
    fileList1 = get_filelist(origin_path, dataset = dataset)
    for fileName in fileList1:
        (realname, extension) = os.path.splitext(fileName)  # 分离文件名和后缀
        img_origin = cv2.imread(os.path.join(origin_path, fileName))
        #img_origin=np.zeros_like(img_origin)
        img_origin = cv2.resize(img_origin, (target_size[1], target_size[0]))
        img_origin_1 = img_origin.copy()
        if fl_classes == 4:
            img_origin_2 = img_origin.copy()
            fl_label = cv2.imread(os.path.join(mask_path, 'covered_fluid', fileName), 0)
            fl_label = cv2.resize(fl_label, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            for i in range(fl_classes - 1):
                img_mask3 = np.zeros_like(fl_label)
                img_mask3[np.where(fl_label == i + 1)] = 255
                img_origin_2 = draw(img_origin_2, img_mask3, COLOR_DICT_2[i])
            cv2.imwrite(os.path.join(save_path, '%s_groundtruth_4.png' % realname), img_origin_2)
        for i, layer_name in enumerate(label_list):
            img_mask1 = cv2.imread(os.path.join(ly_mask_path_list[i], fileName), 0)
            img_mask1 = cv2.resize(img_mask1, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            img_origin_1 = draw(img_origin_1, img_mask1, COLOR_DICT_1[i])
        cv2.imwrite(os.path.join(save_path, '%s_groundtruth_1.png' % realname), img_origin_1)

        lyfl_label = cv2.imread(os.path.join(mask_path, 'covered', fileName), 0)
        lyfl_label = cv2.resize(lyfl_label, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        for i in range(pixel_ly_classes-1):
            img_mask2 = np.zeros_like(lyfl_label)
            img_mask2[np.where(lyfl_label == i+1)] = 255
            img_origin = draw(img_origin, img_mask2, COLOR_DICT_3[i])
        cv2.imwrite(os.path.join(save_path, '%s_groundtruth_2.png' % realname), img_origin)


