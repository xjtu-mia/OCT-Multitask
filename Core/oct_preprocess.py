import cv2
import numpy as np
import os
import math
import sys
sys.path.append(sys.path[0]+'/tools')
from resize_and_fix_img_label import resize_label, fix_boundary
from BM3D_py.bm3d import main as bm3d
from preprocess_utils import intp, flat_oct, fill_boundary, intensity_normalization, ConvexHull

def process_yifuyuan(dataset='yifuyuan',
    oct_device='Cirrus',
    root='./datasets',
    bottom_dis = 80,
    img_height = 352,
    label_exist=True,
    bm3d_img_exist=False,
    wo_cvx=False):

    from scipy import signal
    bm3d_img_root = os.path.join(root, dataset, 'preprocessing_data/bm3d_img')
    label_root = os.path.join(root, dataset, 'original_label')
    original_img_root = os.path.join(root, dataset, 'original_image')
    savepath_BM = os.path.join(root, dataset, 'preprocessing_data/BM')
    savepath_ILM = os.path.join(root, dataset, 'preprocessing_data/ILM')
    savepath_bound = os.path.join(root, dataset, 'preprocessing_data/boundary')
    savepath_fimg = os.path.join(root, dataset, 'preprocessing_data/flatted_img_512')
    savepath_infimg = os.path.join(root, dataset, 'preprocessing_data/flatted_IN_img_512')
    savepath_flabel = os.path.join(root, dataset, 'preprocessing_data/flatted_label_512')
    sigma = {'yifuyuan': 10, 'retouch': 20, 'duke': 20}
    bm_dis, height = bottom_dis, img_height
    os.makedirs(savepath_BM, exist_ok=True)
    os.makedirs(savepath_ILM, exist_ok=True)
    os.makedirs(savepath_fimg, exist_ok=True)
    os.makedirs(savepath_infimg, exist_ok=True)
    os.makedirs(savepath_bound, exist_ok=True)
    os.makedirs(savepath_flabel, exist_ok=True)
    imglist = os.listdir(os.path.join(label_root, 'covered'))
    imgset = set()
    for i in range(len(imglist)):
        tm = imglist[i].split('_')
        tmm = tm[0].split('-')
        if len(tmm) > 1:
            imgset.add(tmm[1])
        else:
            imgset.add(tmm[0])
    print('num_image:{}, num_patient:{}'.format(str(len(imglist)), str(len(imgset))))
    exist_imglist = os.listdir(savepath_fimg)
    # imglist = [i for i in imglist if i not in exist_imglist]
    for name in imglist:
        print(name)
        original_img = cv2.imread(os.path.join(original_img_root, name), -1)
        h, w = original_img.shape
        if h * w >= 1024 * 1024:
            h, w = h // 2, w // 2
        original_img = cv2.resize(original_img, (w, h))
        if not bm3d_img_exist:
            os.makedirs(bm3d_img_root, exist_ok=True)
            img = bm3d(original_img, sigma[dataset], refactor=4)
            cv2.imwrite(os.path.join(bm3d_img_root, name), img)
        else:
            img = cv2.imread(os.path.join(bm3d_img_root, name),-1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        '''
        最小化视野边界对梯度的影响
        '''
        img, boundary = fill_boundary(img, value='same')
        bound_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        flat_img = original_img
        print(original_img.shape)

        layers = []
        for i in range(7):
            label = cv2.imread(os.path.join(label_root, 'layers', 'layer' + str(i), name), -1)
            layers.append(label)
        cw_label = cv2.imread(os.path.join(label_root, 'covered', name), -1)
        cw_wo_irf_label = cv2.imread(os.path.join(label_root, 'covered_wo_irf', name), -1)
        fl_label1 = cv2.imread(os.path.join(label_root, 'fluid1', name), -1)
        fl_label2 = cv2.imread(os.path.join(label_root, 'fluid2', name), -1)
        fl_label3 = cv2.imread(os.path.join(label_root, 'fluid3', name), -1)
        h, w = img.shape

        '''
            水平方向中值滤波，连接小的血管伪影
        '''
        for i in range(h):
            img[i, :] = signal.medfilt(img[i, :], 15)
        for i in range(w):
            img[:, i] = signal.medfilt(img[:, i], 15)
        # img[np.where(img == 255)] = 0
        '''
        sobel滤波得到垂直梯度
        '''
        img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        #cv2.imwrite(os.path.join(savepath_grad, name), img.astype(np.uint8))
        t_l = np.mean(img[np.where(img < 0)])
        # img[np.where(img <= -300)] = 0
        y_index_bm = np.zeros((w))
        y_index_ilm = np.zeros((w))
        BM_img = np.zeros_like(img, dtype=np.uint8)
        ILM_img = np.zeros_like(img, dtype=np.uint8)
        BM_img_ = BM_img.copy()
    
        zero_points = []
        head = 0
        '''
        找到上界膜
    
        '''
        for i in range(w):
            col = img[:, i]
            # print(i, np.mean(col[np.where(col > 0)]), np.median(col[np.where(col > 0)]))
            for j in range(int(boundary[i] + 0), h, 1):
                if col[j] > np.mean(col[np.where(col > 0)])*1.5:
                    y_index_ilm[i] = int(j)
                    ILM_img[j, i] = 255
                    break
    
        for i in range(w):
            '''#ILM下方50pixels以下区域的最小梯度值作为初始阈值'''
            thr = int(np.max(img[int(y_index_ilm[i] + 40):h - 1, i]))
            min_thr = int(15)
            for t in range(20):
                #print(thr, t, i)
                if thr < min_thr:
                    thr = int(np.max(img[int(y_index_ilm[i] + 40 - t):h - 1, i]))
                    if t == 19:
                        thr = min_thr
    
                else:
    
                    break
            # print(thr, i)
    
            thr_ = thr
            col = img[:, i]
            '''
            自适应阈值寻找BM, 从下向上遍历每一列(ILM以下)，如果找到的边界点与上一个不为0坐标的边界点垂直距离小于5像素，
            则认为该点为BM上的点，否则增大阈值再次遍历。
            '''
            wrong_points = set()
            # print(zero_points)
            '''
            标记第一段0值
            '''
            if sum(y_index_bm[0:i - 2]) == 0 and y_index_bm[i - 1] != 0 and i > 1:
                head = 1
            # print(y_index_bm[0:i-1], y_index_bm[i-1],head)
            for t in range(1, thr - min_thr, 1):
                for j in range(h - 1, int(y_index_ilm[i] + 40), -1):
                    # m = np.min(col[j-20: j+1])
                    if col[j] >= thr_:
                        # print(j, col[j], thr_ )
                        if i > 0:
                            if head == 1 and math.fabs(j - y_index_bm[i - 1]) <= 2 or \
                                    math.fabs(j - y_index_bm[i - len(zero_points) - 1]) <= (
                                    1.5 + 1 * 1 / (len(zero_points) + 1)) * (len(zero_points) + 1):
                                y_index_bm[i] = int(j)
                                break
                            elif head == 0 and y_index_bm[i - 1] == 0:
                                y_index_bm[i] = int(j)
                                break
                        elif i == 0:
                            y_index_bm[i] = int(j)
                            break
    
                if y_index_bm[i - 1] > 0 and i > 0:
                    if y_index_bm[i] - y_index_bm[i - 1] < -2:
                        thr_ = thr - t
                        wrong_points.add(y_index_bm[i])
                    elif y_index_bm[i] - y_index_bm[i - 1] > 2:
                        thr_ = thr + t
                        wrong_points.add(y_index_bm[i])
                    # print(wrong_points)
                    # print(y_index_bm[i - 1])
    
                elif y_index_bm[i - 1] == 0 and head == 1 and i > 0:
                    if y_index_bm[i] - y_index_bm[i - len(zero_points) - 1] < -(1.5 + 1 * 1 / (len(zero_points) + 1)) * (
                            len(zero_points) + 1):
                        thr_ = thr - t
                        wrong_points.add(y_index_bm[i])
                    elif y_index_bm[i] - y_index_bm[i - len(zero_points) - 1] > (1.5 + 1 * 1 / (len(zero_points) + 1)) * (
                            len(zero_points) + 1):
                        thr_ = thr + t
                        wrong_points.add(y_index_bm[i])
    
                elif thr_ <= min_thr:
                    if y_index_bm[i - 1] != 0 and y_index_bm[i - 1] not in wrong_points:
                        y_index_bm[i] = y_index_bm[i - 1]
                    else:
                        y_index_bm[i] = 0
                    break
                else:
                    break
            y_index_bm[i] = 0 if y_index_bm[i] > h - 20 else y_index_bm[i]
            if y_index_bm[i] == 0:
                zero_points.append(i)
            else:
                zero_points = []

    
        for i in range(w):
            BM_img_[int(y_index_bm[i]), int(i)] = 255
        y_index_bm = intp(y_index_bm)
        print(y_index_bm.shape)
        for i in range(w):
            BM_img[int(y_index_bm[i]), int(i)] = 255

        '''
            平滑曲线
        '''
        # using ConvexHull
        if not wo_cvx:
            hull_img = np.zeros((h, w))
            for i in range(w):
                hull_img[0: int(y_index_bm[i]), int(i)] = 255
            y_index_bm = ConvexHull(hull_img)
        y_index_bm = signal.savgol_filter(y_index_bm, 51, 3)
        BM_img = np.zeros_like(BM_img)
        BM_img_ = np.zeros_like(BM_img)
        for i in range(w):
            BM_img[int(y_index_bm[i]), int(i)] = 255
            BM_img_[int(y_index_bm[i])+15, int(i)] = 255
        # BM_img = cv2.resize(BM_img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
        y_index_bm = fix_boundary(np.argmax(BM_img, axis=0))
    
        '''
            拉平图像和标签
        '''
        H, W = flat_img.shape

        flat_img, baseline = flat_oct(flat_img, y_index_bm, dis=bm_dis, height=height)
        flat_img.astype(np.uint8)
        flat_img_in = intensity_normalization(flat_img).astype(np.uint8)
        h, w = flat_img.shape
        # print(bm3d_img.shape, BM_img.shape, ILM_img.shape)
        bound_img[np.where(cv2.dilate(BM_img, kernel=np.ones((2, 2))) == 255)] = [0, 255, 255]
        bound_img[np.where(cv2.dilate(ILM_img, kernel=np.ones((2, 2))) == 255)] = [0, 255, 255]
        bound_img[np.where(cv2.dilate(BM_img_, kernel=np.ones((2, 2))) == 255)] = [0, 255, 0]
        # BM_img = cv2.bitwise_or(cv2.bitwise_or(bm3d_img, BM_img), ILM_img)
        cv2.imwrite(os.path.join(savepath_bound, name), bound_img)
        cv2.imwrite(os.path.join(savepath_BM, name), BM_img)
        cv2.imwrite(os.path.join(savepath_ILM, name), ILM_img)
        cv2.imwrite(os.path.join(savepath_fimg, name), flat_img)
        cv2.imwrite(os.path.join(savepath_infimg, name), flat_img_in)


        for i in range(7):
            label = fix_boundary(flat_oct(resize_label(layers[i], (W, H)), y_index_bm, dis=bm_dis, height=height)[0])
            os.makedirs(os.path.join(savepath_flabel, 'layers', 'layer' + str(i)), exist_ok=True)
            cv2.imwrite(os.path.join(savepath_flabel, 'layers', 'layer' + str(i), name), label)
        cw_label = fix_boundary(flat_oct(cv2.resize(cw_label, (W, H), interpolation=cv2.INTER_NEAREST),
                                         y_index_bm, dis=bm_dis, height=height)[0])
        cw_wo_irf_label = fix_boundary(
            flat_oct(cv2.resize(cw_wo_irf_label, (W, H), interpolation=cv2.INTER_NEAREST),
                     y_index_bm, dis=bm_dis, height=height)[0])
        fl_label1 = fix_boundary(flat_oct(cv2.resize(fl_label1, (W, H), interpolation=cv2.INTER_NEAREST),
                                          y_index_bm, dis=bm_dis, height=height)[0])
        fl_label2 = fix_boundary(flat_oct(cv2.resize(fl_label2, (W, H), interpolation=cv2.INTER_NEAREST),
                                          y_index_bm, dis=bm_dis, height=height)[0])
        fl_label3 = fix_boundary(flat_oct(cv2.resize(fl_label3, (W, H), interpolation=cv2.INTER_NEAREST),
                                          y_index_bm, dis=bm_dis, height=height)[0])
        os.makedirs(os.path.join(savepath_flabel, 'covered'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'covered_wo_irf'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'fluid1'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'fluid2'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'fluid3'), exist_ok=True)
        cv2.imwrite(os.path.join(savepath_flabel, 'covered', name), cw_label)
        cv2.imwrite(os.path.join(savepath_flabel, 'covered_wo_irf', name), cw_wo_irf_label)
        cv2.imwrite(os.path.join(savepath_flabel, 'fluid1', name), fl_label1)
        cv2.imwrite(os.path.join(savepath_flabel, 'fluid2', name), fl_label2)
        cv2.imwrite(os.path.join(savepath_flabel, 'fluid3', name), fl_label3)
        cv2.imwrite(os.path.join(savepath_flabel, 'covered_wo_irf', name), cw_wo_irf_label)


def process_duke(
    dataset='duke',
    oct_device='Spectralis',
    root='./datasets',
    bottom_dis = 15,
    img_height = 224,
    bm3d_img_exist=False,
    wo_cvx=False):

    from scipy import signal
    dataset = 'duke'
    bm3d_img_root = os.path.join(root, dataset, 'preprocessing_data/bm3d_img')
    label_root = os.path.join(root, dataset, 'original_label')
    original_img_root = os.path.join(root, dataset, 'original_image')
    savepath_BM = os.path.join(root, dataset, 'preprocessing_data/BM')
    savepath_ILM = os.path.join(root, dataset, 'preprocessing_data/ILM')
    savepath_bound = os.path.join(root, dataset, 'preprocessing_data/boundary')
    savepath_fimg = os.path.join(root, dataset, 'preprocessing_data/flatted_img_512')
    savepath_infimg = os.path.join(root, dataset, 'preprocessing_data/flatted_IN_img_512')
    savepath_flabel = os.path.join(root, dataset, 'preprocessing_data/flatted_label_512')
    sigma = {'yifuyuan': 10, 'retouch': 20, 'duke': 20}
    bm_dis, height = bottom_dis, img_height
    os.makedirs(savepath_BM, exist_ok=True)
    os.makedirs(savepath_ILM, exist_ok=True)
    os.makedirs(savepath_fimg, exist_ok=True)
    os.makedirs(savepath_infimg, exist_ok=True)
    os.makedirs(savepath_bound, exist_ok=True)
    os.makedirs(savepath_flabel, exist_ok=True)
    imglist = os.listdir(os.path.join(label_root, 'covered'))
    print('num_image:{}'.format(str(len(imglist))))
    exist_imglist = os.listdir(savepath_fimg)
    # imglist = [i for i in imglist if i not in exist_imglist]
    for name in imglist:
        print(name)
        original_img = cv2.imread(os.path.join(original_img_root, name), -1)
        h, w = original_img.shape
        if h * w >= 1024 * 1024:
            h, w = h // 2, w // 2
        original_img = cv2.resize(original_img, (w, h))
        if not bm3d_img_exist:
            os.makedirs(bm3d_img_root, exist_ok=True)
            img = bm3d(original_img, sigma[dataset], refactor=4)
            cv2.imwrite(os.path.join(bm3d_img_root, name), img)
        else:
            img = cv2.imread(os.path.join(bm3d_img_root, name), -1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        '''
        最小化视野边界对梯度的影响
        '''
        img, boundary = fill_boundary(img, value='same')
        bound_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        flat_img = original_img
        print(original_img.shape)

        layers = []
        for i in range(8):
            label = cv2.imread(os.path.join(label_root,'layers', 'layer' + str(i), name), -1)
            layers.append(label)
        cw_label = cv2.imread(os.path.join(label_root, 'covered', name), -1)
        cw_wo_irf_label = cv2.imread(os.path.join(label_root, 'covered_wo_irf', name), -1)
        fl_label1 = cv2.imread(os.path.join(label_root, 'fluid', name), -1)
        h, w = img.shape

        '''
            水平方向中值滤波，连接小的血管伪影
        '''
        for i in range(h):
            img[i, :] = signal.medfilt(img[i, :], 15)
        '''for i in range(w):
            img[:, i] = signal.medfilt(img[:, i], 15)'''
        # img[np.where(img == 255)] = 0
        '''
        sobel滤波得到垂直梯度
        '''
        img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        y_index_bm = np.zeros((w))
        y_index_ilm = np.zeros((w))
        BM_img = np.zeros_like(img, dtype=np.uint8)
        ILM_img = np.zeros_like(img, dtype=np.uint8)
        BM_img_ = BM_img.copy()

        zero_points = []
        head = 0
        '''
        找到上界膜

        '''
        for i in range(w):
            col = img[:, i]
            for j in range(int(boundary[i] + 1), h, 1):
                if col[j] > np.mean(col[np.where(col > 5)]) * 1.5:
                    y_index_ilm[i] = int(j)
                    ILM_img[j, i] = 255
                    break

        for i in range(w):
            '''#ILM下方50pixels以下区域的最小梯度值作为初始阈值'''
            thr = int(np.min(img[int(y_index_ilm[i] + 50):h - 1, i]))
            for t in range(20):
                if thr > -30:
                    thr = int(np.min(img[int(y_index_ilm[i] + 50 - t):h - 1, i]))
                    if t == 19:
                        thr = -30
                else:

                    break
            thr_ = thr
            col = img[:, i]
            '''
            自适应阈值寻找BM, 从下向上遍历每一列(ILM以下)，如果找到的边界点与上一个不为0坐标的边界点垂直距离小于5像素，
            则认为该点为BM上的点，否则增大阈值再次遍历。
            '''
            wrong_points = []
            #print(zero_points)
            '''
            标记第一段0值
            '''
            if sum(y_index_bm[0:i-2]) == 0 and y_index_bm[i-1]!=0 and i > 1:
                head = 1
            for t in range(1,-thr-30,1):
                for j in range(h-80, int(y_index_ilm[i]+50), -1):
                    if col[j] <= thr_:
                        if i > 0:
                            if head == 1 and math.fabs(j - y_index_bm[i - 1])<=2 or \
                                math.fabs(j - y_index_bm[i - len(zero_points)-1])<=(0.5+1.5*1/(len(zero_points)+1))*(len(zero_points)+1):
                                y_index_bm[i] = int(j)
                                break
                            elif head == 0 and y_index_bm[i - 1] == 0:
                                y_index_bm[i] = int(j)
                                break
                        elif i == 0:
                            y_index_bm[i] = int(j)
                            break


                if y_index_bm[i - 1] > 0 and i > 0:
                    if y_index_bm[i] - y_index_bm[i - 1] < -2:
                        thr_ = thr + t
                        wrong_points.append(y_index_bm[i])
                    elif y_index_bm[i] - y_index_bm[i - 1] > 2:
                        thr_ = thr - t
                        wrong_points.append(y_index_bm[i])
                    #print(wrong_points)
                    #print(y_index_bm[i - 1])

                elif y_index_bm[i-1] == 0 and head == 1  and i > 0:
                    if y_index_bm[i] - y_index_bm[i - len(zero_points)-1] < -(0.5+1.5*1/(len(zero_points)+1))*(len(zero_points)+1):
                        thr_ = thr + t
                        wrong_points.append(y_index_bm[i])
                    elif y_index_bm[i] - y_index_bm[i - len(zero_points)-1] > (0.5+1.5*1/(len(zero_points)+1))*(len(zero_points)+1):
                        thr_ = thr - t
                        wrong_points.append(y_index_bm[i])

                elif thr_ > -30:
                    if y_index_bm[i - 1] != 0 and y_index_bm[i - 1] not in wrong_points:
                        y_index_bm[i] = y_index_bm[i - 1]
                    else:
                        y_index_bm[i] = 0
                    break
                else:
                    break
            y_index_bm[i] = 0 if y_index_bm[i] > h - 50 else y_index_bm[i]
            if y_index_bm[i] == 0:
                zero_points.append(i)
            else:
                zero_points = []

        for i in range(w):
            BM_img_[int(y_index_bm[i]), int(i)] = 255
        y_index_bm = intp(y_index_bm)
        print(y_index_bm.shape)
        for i in range(w):
            BM_img[int(y_index_bm[i]), int(i)] = 255

        '''
            平滑曲线
        '''
        # using ConvexHull
        if not wo_cvx:
            hull_img = np.zeros((h, w))
            for i in range(w):
                hull_img[0: int(y_index_bm[i]), int(i)] = 255
            y_index_bm = ConvexHull(hull_img)
        y_index_bm = signal.savgol_filter(y_index_bm, 127, 2)
        BM_img = np.zeros_like(BM_img)
        BM_img_ = np.zeros_like(BM_img)
        for i in range(w):
            BM_img[int(y_index_bm[i]), int(i)] = 255
            BM_img_[int(y_index_bm[i]) + 15, int(i)] = 255
        # BM_img = cv2.resize(BM_img, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
        y_index_bm = fix_boundary(np.argmax(BM_img, axis=0))

        '''
            拉平图像和标签
        '''
        H, W = flat_img.shape

        flat_img, baseline = flat_oct(flat_img, y_index_bm, dis=bm_dis, height=height)
        flat_img.astype(np.uint8)
        flat_img_in = intensity_normalization(flat_img).astype(np.uint8)
        h, w = flat_img.shape
        # print(bm3d_img.shape, BM_img.shape, ILM_img.shape)
        bound_img[np.where(cv2.dilate(BM_img, kernel=np.ones((2, 2))) == 255)] = [0, 255, 255]
        bound_img[np.where(cv2.dilate(ILM_img, kernel=np.ones((2, 2))) == 255)] = [0, 255, 255]
        bound_img[np.where(cv2.dilate(BM_img_, kernel=np.ones((2, 2))) == 255)] = [0, 255, 0]
        # BM_img = cv2.bitwise_or(cv2.bitwise_or(bm3d_img, BM_img), ILM_img)
        cv2.imwrite(os.path.join(savepath_bound, name), bound_img)
        cv2.imwrite(os.path.join(savepath_BM, name), BM_img)
        cv2.imwrite(os.path.join(savepath_ILM, name), ILM_img)
        cv2.imwrite(os.path.join(savepath_fimg, name), flat_img)
        cv2.imwrite(os.path.join(savepath_infimg, name), flat_img_in)
        for i in range(8):
            label = fix_boundary(
                flat_oct(resize_label(layers[i], (W, H)), y_index_bm, dis=bm_dis, height=height)[0])
            os.makedirs(os.path.join(savepath_flabel, 'layers', 'layer' + str(i)), exist_ok=True)
            cv2.imwrite(os.path.join(savepath_flabel, 'layers', 'layer' + str(i), name), label)
        cw_label = fix_boundary(flat_oct(cv2.resize(cw_label, (W, H), interpolation=cv2.INTER_NEAREST),
                                         y_index_bm, dis=bm_dis, height=height)[0])
        cw_wo_irf_label = fix_boundary(
            flat_oct(cv2.resize(cw_wo_irf_label, (W, H), interpolation=cv2.INTER_NEAREST),
                     y_index_bm, dis=bm_dis, height=height)[0])
        fl_label1 = fix_boundary(flat_oct(cv2.resize(fl_label1, (W, H), interpolation=cv2.INTER_NEAREST),
                                          y_index_bm, dis=bm_dis, height=height)[0])

        os.makedirs(os.path.join(savepath_flabel, 'covered'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'covered_wo_irf'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'fluid'), exist_ok=True)
        cv2.imwrite(os.path.join(savepath_flabel, 'covered', name), cw_label)
        cv2.imwrite(os.path.join(savepath_flabel, 'covered_wo_irf', name), cw_wo_irf_label)
        cv2.imwrite(os.path.join(savepath_flabel, 'fluid', name), fl_label1)
        cv2.imwrite(os.path.join(savepath_flabel, 'covered_wo_irf', name), cw_wo_irf_label)

def process_retouch(
    dataset='retouch',
    oct_device='Topcon',
    root='./datasets',
    bottom_dis=80,
    img_height=352,
    bm3d_img_exist=False,
    wo_cvx=False):

    from scipy import signal
    bm3d_img_root = os.path.join(root, dataset, oct_device, 'preprocessing_data/bm3d_img')
    label_root = os.path.join(root, dataset, oct_device, 'original_label')
    original_img_root = os.path.join(root, dataset, oct_device, 'original_image')
    savepath_BM = os.path.join(root, dataset, oct_device, 'preprocessing_data/BM')
    savepath_ILM = os.path.join(root, dataset, oct_device, 'preprocessing_data/ILM')
    savepath_bound = os.path.join(root, dataset, oct_device, 'preprocessing_data/boundary')
    savepath_fimg = os.path.join(root, dataset, oct_device, 'preprocessing_data/flatted_img_512')
    savepath_infimg = os.path.join(root, dataset, oct_device, 'preprocessing_data/flatted_IN_img_512')
    savepath_flabel = os.path.join(root, dataset, oct_device, 'preprocessing_data/flatted_label_512')

    sigma = {'yifuyuan': 20, 'retouch': 20, 'duke': 20}
    min_thr_dict = {'Cirrus': 15, 'Spectralis': 30, 'Topcon': 15}
    bm_dis, height = bottom_dis, img_height
    os.makedirs(savepath_BM, exist_ok=True)
    os.makedirs(savepath_ILM, exist_ok=True)
    os.makedirs(savepath_fimg, exist_ok=True)
    os.makedirs(savepath_infimg, exist_ok=True)
    os.makedirs(savepath_bound, exist_ok=True)
    os.makedirs(savepath_flabel, exist_ok=True)

    imglist = sorted(os.listdir(os.path.join(original_img_root)))
    # imglist = [i for i in imglist if i not in exist_imglist]
    for name in imglist:
        # name = 'index_5_046.png'
        min_thr = min_thr_dict[oct_device]  # 15 for topcon, 30 for cirrus and Spectralis
        print(os.path.join(original_img_root, name))
        original_img = cv2.imread(os.path.join(original_img_root, name), -1)
        h, w = original_img.shape
        if h * w >= 1024 * 1024:
            h, w = h // 2, w // 2
        original_img = cv2.resize(original_img, (w, h))
        if not bm3d_img_exist:
            os.makedirs(bm3d_img_root, exist_ok=True)
            img = bm3d(original_img, sigma[dataset], refactor=4)
            cv2.imwrite(os.path.join(bm3d_img_root, name), img)
        else:
            img = cv2.imread(os.path.join(bm3d_img_root, name))

        '''
            最小化视野边界对梯度的影响
        '''
        img, boundary = fill_boundary(img, value='same')
        '''
            自适应直方均衡化
        '''
        h, w = img.shape
        flat_img = original_img
        print(flat_img.shape)

        cw_label = cv2.imread(os.path.join(label_root, name), -1)
        cw_label = cv2.resize(cw_label, (w, h), interpolation=cv2.INTER_NEAREST)

        h, w = img.shape

        '''
            水平方向中值滤波，连接小的血管伪影
        '''
        for i in range(h):
            img[i, :] = signal.medfilt(img[i, :], 15)
        '''for i in range(w):
            img[:, i] = signal.medfilt(img[:, i],15)'''

        # img[np.where(img == 255)] = 0
        '''
        sobel滤波得到垂直梯度
        '''
        img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        t_l = np.mean(img[np.where(img < 0)])
        # img[np.where(img <= -300)] = 0
        y_index_bm = np.zeros((w))
        y_index_ilm = np.zeros((w))
        BM_img = np.zeros_like(img)
        ILM_img = np.zeros_like(img)
        BM_img_ = np.zeros_like(BM_img)
        for i in range(w):
            BM_img[int(y_index_bm[i]), int(i)] = 255
            BM_img_[int(y_index_bm[i]) + 15, int(i)] = 255
        zero_points = []
        head = 0
        '''
        找到上界膜

        '''
        for i in range(w):
            col = img[:, i]
            # print(i, np.mean(col[np.where(col > 0)]), np.median(col[np.where(col > 0)]))

            for j in range(int(boundary[i] + 0), h, 1):
                if col[j] > np.mean(col[np.where(col > 0)]) * 1.5:
                    y_index_ilm[i] = int(j)
                    ILM_img[j, i] = 255
                    break

        for i in range(w):
            # print(np.min(img[:,i]),np.where(img[:,i]==np.min(img[:,i])))
            thr = int(np.max(img[int(y_index_ilm[i] + 40):h - 1, i]))
            for t in range(20):
                if thr < min_thr:
                    thr = int(np.max(img[int(y_index_ilm[i] + 40 - t):h - 1, i]))
                    if t == 19:
                        thr = min_thr
                    # print(thr, t, i)
                else:

                    break
            # print(thr, i)

            thr_ = thr
            col = img[:, i]
            '''
            自适应阈值寻找BM, 从下向上遍历每一列(ILM以下)，如果找到的边界点与上一个不为0坐标的边界点垂直距离小于5像素，
            则认为该点为BM上的点，否则增大阈值再次遍历。
            '''
            wrong_points = []
            # print(zero_points)
            '''
            标记第一段0值
            '''
            if sum(y_index_bm[0:i - 2]) == 0 and y_index_bm[i - 1] != 0 and i > 1:
                head = 1
            # print(y_index_bm[0:i-1], y_index_bm[i-1],head)
            for t in range(1, thr - min_thr, 1):
                for j in range(h - 1, int(y_index_ilm[i] + 40), -1):
                    # m = np.min(col[j-20: j+1])
                    if col[j] >= thr_:
                        # print(j, col[j], thr_ )
                        if i > 0:
                            if head == 1 and math.fabs(j - y_index_bm[i - 1]) <= 2 or \
                                    math.fabs(j - y_index_bm[i - len(zero_points) - 1]) <= (
                                    1.5 + 1 * 1 / (len(zero_points) + 1)) * (len(zero_points) + 1):
                                y_index_bm[i] = int(j)
                                break
                            elif head == 0 and y_index_bm[i - 1] == 0:
                                y_index_bm[i] = int(j)
                                break
                        elif i == 0:
                            y_index_bm[i] = int(j)
                            break

                if y_index_bm[i - 1] > 0 and i > 0:
                    if y_index_bm[i] - y_index_bm[i - 1] < -2:
                        thr_ = thr - t
                        wrong_points.append(y_index_bm[i])
                    elif y_index_bm[i] - y_index_bm[i - 1] > 2:
                        thr_ = thr + t
                        wrong_points.append(y_index_bm[i])
                    # print(wrong_points)
                    # print(y_index_bm[i - 1])

                elif y_index_bm[i - 1] == 0 and head == 1 and i > 0:
                    if y_index_bm[i] - y_index_bm[i - len(zero_points) - 1] < -(
                            1.5 + 1 * 1 / (len(zero_points) + 1)) * (len(zero_points) + 1):
                        thr_ = thr - t
                        wrong_points.append(y_index_bm[i])
                    elif y_index_bm[i] - y_index_bm[i - len(zero_points) - 1] > (
                            1.5 + 1 * 1 / (len(zero_points) + 1)) * (len(zero_points) + 1):
                        thr_ = thr + t
                        wrong_points.append(y_index_bm[i])

                elif thr_ <= min_thr:
                    if y_index_bm[i - 1] != 0 and y_index_bm[i - 1] not in wrong_points:
                        y_index_bm[i] = y_index_bm[i - 1]
                    else:
                        y_index_bm[i] = 0
                    break
                else:
                    break
            y_index_bm[i] = 0 if y_index_bm[i] > h - 20 else y_index_bm[i]
            if y_index_bm[i] == 0:
                zero_points.append(i)
            else:
                zero_points = []

        for i in range(w):
            BM_img_[int(y_index_bm[i]), int(i)] = 255
        print(y_index_bm.shape)
        y_index_bm = intp(y_index_bm)
        print(y_index_bm.shape)
        for i in range(w):
            BM_img[int(y_index_bm[i]), int(i)] = 255

        from scipy import signal

        '''
            平滑曲线
        '''
        #using ConvexHull
        if not wo_cvx:
            hull_img = np.zeros((h, w))
            for i in range(w):
                hull_img[0: int(y_index_bm[i]), int(i)] = 255
            y_index_bm = ConvexHull(hull_img)
        y_index_bm = signal.savgol_filter(y_index_bm, 51, 3)
        BM_img = np.zeros_like(BM_img)
        for i in range(w):
            BM_img[int(y_index_bm[i]), int(i)] = 255
        BM_img = cv2.resize(BM_img, (w * 1, h * 1), interpolation=cv2.INTER_NEAREST)
        print(BM_img.shape)
        y_index_bm = np.argmax(BM_img, axis=0)


        '''
            拉平图像和标签
        '''
        H, W = flat_img.shape
        # BM_img = cv2.resize(BM_img, (W, H), interpolation=cv2.INTER_NEAREST)
        # y_index_bm = np.argmax(BM_img, axis=)

        flat_img, baseline = flat_oct(flat_img, y_index_bm, dis=bm_dis, height=height)
        cv2.imwrite(os.path.join(savepath_fimg, name), flat_img)
        flat_img = intensity_normalization(flat_img)
        cv2.imwrite(os.path.join(savepath_BM, name), BM_img)
        cv2.imwrite(os.path.join(savepath_ILM, name), ILM_img)
        cv2.imwrite(os.path.join(savepath_infimg, name), flat_img)
        bound_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        bound_img[np.where(cv2.dilate(BM_img, kernel=np.ones((2, 2))) == 255)] = [0, 255, 255]
        bound_img[np.where(cv2.dilate(ILM_img, kernel=np.ones((2, 2))) == 255)] = [0, 255, 255]
        bound_img[np.where(cv2.dilate(BM_img_, kernel=np.ones((2, 2))) == 255)] = [0, 255, 0]
        cv2.imwrite(os.path.join(savepath_bound, name), bound_img)

        cw_label, baseline = flat_oct(cw_label, y_index_bm, dis=bm_dis, height=height)
        cw_label[np.where(cw_label == 1)] = 7
        cw_label[np.where(cw_label == 2)] = 4
        cw_label[np.where(cw_label == 3)] = 6
        cw_label_woirf = cw_label.copy()
        cw_label_woirf[np.where(cw_label == 7)] = 0
        fl_label3 = np.zeros_like(cw_label)
        fl_label3[np.where(cw_label == 7)] = 255
        fl_label2 = np.zeros_like(cw_label)
        fl_label2[np.where(cw_label == 6)] = 255
        fl_label1 = np.zeros_like(cw_label)
        fl_label1[np.where(cw_label == 4)] = 255
        os.makedirs(os.path.join(savepath_flabel), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'covered'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'covered_wo_irf'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'fluid1'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'fluid2'), exist_ok=True)
        os.makedirs(os.path.join(savepath_flabel, 'fluid3'), exist_ok=True)
        cv2.imwrite(os.path.join(savepath_flabel, 'fluid1', name), fl_label1)
        cv2.imwrite(os.path.join(savepath_flabel, 'fluid2', name), fl_label2)
        cv2.imwrite(os.path.join(savepath_flabel, 'fluid3', name), fl_label3)
        cv2.imwrite(os.path.join(savepath_flabel, 'covered', name), cw_label)
        cv2.imwrite(os.path.join(savepath_flabel, 'covered_wo_irf', name), cw_label_woirf)
def main(
    dataset='retouch',
    oct_device='Topcon',
    root='./datasets/',
    bm3d_img_exist=False,
    wo_cvx=True):
    if dataset == 'yifuyuan':
        bottom_dis = 80
        img_height = 352
        process_yifuyuan(
            dataset=dataset,
            oct_device=oct_device,
            root=root,
            bottom_dis=bottom_dis,
            img_height=img_height,
            bm3d_img_exist=bm3d_img_exist,
            wo_cvx=wo_cvx)
    elif dataset == 'duke':
        bottom_dis = 70
        img_height = 224
        process_duke(
            dataset=dataset,
            oct_device=oct_device,
            root=root,
            bottom_dis=bottom_dis,
            img_height=img_height,
            bm3d_img_exist=bm3d_img_exist,
            wo_cvx=wo_cvx)
    elif dataset == 'retouch':
        bottom_dis = 80
        img_height = 352
        process_retouch(
            dataset=dataset,
            oct_device=oct_device,
            root=root,
            bottom_dis=bottom_dis,
            img_height=img_height,
            bm3d_img_exist=bm3d_img_exist,
            wo_cvx=wo_cvx)
    else:
        raise ValueError('dataset must be yifuyuan, duke or retouch.')
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='retouch', help='数据集名称')
    parser.add_argument('--oct_device', default='Spectralis', help='OCT设备名称（仅当dataset为retouch时有效）')
    parser.add_argument('--root', default='./datasets', help='数据集的根目录')
    parser.add_argument('--bm3d_img_exist', action='store_true', help='是否存在BM3D图像')
    parser.add_argument('--wo_cvx', action='store_true', help='是否使用CVX')
    args = parser.parse_args()
    # 将参数传递给主函数
    main(
        dataset=args.dataset,
        oct_device=args.oct_device,
        root=args.root,
        bm3d_img_exist=args.bm3d_img_exist,
        wo_cvx=args.wo_cvx
    )

