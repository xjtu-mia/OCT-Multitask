import cv2
import math
import os
import numpy as np


def crop_img(img, size):
    if len(img.shape) == 2:
        h, w = img.shape

        return img[:, (w - size) // 2:w - (w - size) // 2]
    else:
        h, w = img.shape[:2]
        return img[:, (w - size) // 2:w - (w - size) // 2, :]


def gene_classwise_label(ly_label_list, fl_label=np.array([None])):
    def intp(fp):
        fp_ = fp.copy().tolist()
        xp_ = np.arange(len(fp_))
        original_length = len(fp)
        index = []
        start = 0
        end = len(fp_) - 1
        flag = 0
        t = []
        for i in range(len(fp)):
            if fp[i] != 0 and flag == 1:

                if len(t) > 0:
                    if t[0] == 0:
                        start = t[-1]
                    if t[0] > 0:
                        index.append(t)
                t = []
                flag = 0

            elif fp[i] == 0:
                t.append(i)
                fp_.remove(0)
                flag = 1
        if len(t) > 0:
            end = t[0]
        s = fp_[0]
        for i in range(start + 1):
            # print('start', start)
            if start > 0:
                fp_.insert(i, s)  # 起始点未知则默认插入最近的一个有值点的值
        e = fp_[-1]
        for i in range(end, len(fp), 1):
            # print('end', end)
            if end <= len(fp) - 1:
                fp_.insert(i, e)  # 结束点未知则默认插入最近的一个有值点的值
        xp_ = np.arange(len(fp_))

        for group in index:
            v = np.linspace(group[0] - 1, group[0], group[-1] - group[0] + 3)
            x = np.array(v[1:-1])
            # print('x', x)
            # print('xp_', xp_)
            # print('fp_', fp_)
            int = np.interp(x, xp_, fp_)
            # print('group', group)
            # print('int', int)
            # print(int.shape)
            for i, y in enumerate(int):
                fp_.insert(group[0] + i, y)
            xp_ = np.arange(len(fp_))
        if len(fp_) < original_length:
            for l in range(original_length - len(fp_)):
                fp_.insert(len(fp_) + l, fp_[-1])
        if len(fp_) > original_length:
            for l in range(len(fp_) - original_length):
                del fp_[len(fp_) - 1 - l]
        return np.round(np.array(fp_))

    new_label = np.zeros_like(ly_label_list[0], dtype=np.uint8)

    h, w = new_label.shape
    for c in range(0, 6, 1):
        ly_label_top = ly_label_list[c]
        ly_label_top = np.argmax(ly_label_top, axis=0)
        ly_label_bottom = ly_label_list[c + 1]
        ly_label_bottom = np.argmax(ly_label_bottom, axis=0)

        s_flag = 0
        # interp(ly_label_top)
        if np.where(ly_label_top == 0)[0].shape[0] > 0:
            ly_label_top = intp(ly_label_top)
        if np.where(ly_label_bottom == 0)[0].shape[0] > 0:
            ly_label_bottom = intp(ly_label_bottom)
        for i in range(min(len(ly_label_top), len(ly_label_bottom))):
            y_top = ly_label_top[i]
            y_bottom = ly_label_bottom[i]
            if y_top != 0 and y_bottom != 0:
                for j in range(int(y_top), int(y_bottom), 1):
                    if y_bottom - y_top > 1:
                        new_label[j, i] = c + 1
                    else:
                        new_label[j, i] = c
        if fl_label.any():
            new_label[np.where(fl_label > 0)] = 7
        '''cv2.imshow('1', new_label)
        cv2.waitKey(100)'''

    return new_label


def gene_index_label(cw_label, reverse=False):
    new_label_up = np.zeros_like(cw_label, dtype=np.uint8)
    new_label_down = np.zeros_like(cw_label, dtype=np.uint8)
    new_label_down_ = new_label_down.copy()
    h, w = new_label_up.shape
    index_up = np.argmax(cw_label, axis=0)
    cw_label_ = np.rot90(cw_label, k=2)
    index_down_ = np.argmax(cw_label_, axis=0)
    for i in range(w):
        if index_down_[i] > 0:
            new_label_down_[index_down_[i], i] = 255
    new_label_down = np.rot90(new_label_down_, k=2)
    index_down = np.argmax(new_label_down, axis=0)

    for i in range(w):
        if index_up[i] > 0:
            new_label_up[index_up[i], i] = 255
        if index_down[i] > 0:
            new_label_down[index_down[i], i] = 255

    # index = np.argmax(new_label_up, axis=0)

    '''cv2.imshow('1', new_label_up+new_label_down)
    cv2.waitKey(100)'''

    return index_up, index_down


def main(root='./datasets', dataset='retouch', oct_device='Topcon', log_name=None, seed=8830):
    root = os.path.join(root, dataset, oct_device)
    in_root = os.path.join(root, 'result', log_name)
    out_root = os.path.join(root, 'preprocessing_data/pseudo_label')
    image_path = os.path.join(root, 'preprocessing_data/flatted_IN_img_512')
    label_path = in_root
    true_label_path = os.path.join(root, 'preprocessing_data/flatted_label_512')
    if not os.path.exists(os.path.join(out_root, 'covered')):
        os.makedirs(os.path.join(out_root, 'covered'))
    if not os.path.exists(os.path.join(out_root, 'covered_wo_irf')):
        os.makedirs(os.path.join(out_root, 'covered_wo_irf'))
    if not os.path.exists(os.path.join(out_root, 'fluid1')):
        os.makedirs(os.path.join(out_root, 'fluid1'))
    if not os.path.exists(os.path.join(out_root, 'fluid2')):
        os.makedirs(os.path.join(out_root, 'fluid2'))
    if not os.path.exists(os.path.join(out_root, 'fluid3')):
        os.makedirs(os.path.join(out_root, 'fluid3'))
    imglist = os.listdir(image_path)
    for name in imglist:
        print(name)
        # name = 'index_4_051.png'
        realname, extension = os.path.splitext(name)
        img = cv2.imread(os.path.join(image_path, name), 0)
        true_label = cv2.imread(os.path.join(true_label_path, 'covered', name), 0)
        true_irf_label = cv2.imread(os.path.join(true_label_path, 'fluid3', name), 0)
        true_label_srf = np.zeros_like(true_label)
        true_label_ped = true_label_srf.copy()
        # 找到srf上边界
        true_label_srf[np.where(true_label == 4)] = 255
        srf_index_up, srf_index_down = gene_index_label(true_label_srf)

        # 找到ped下边界(BM)
        true_label_ped[np.where(true_label == 6)] = 255
        ped_index_up, ped_index_down = gene_index_label(true_label_ped, reverse=True)
        h, w = img.shape
        label_list = []
        cv2.imwrite(os.path.join(out_root, 'fluid1', realname + extension), true_label_srf)
        cv2.imwrite(os.path.join(out_root, 'fluid2', realname + extension), true_label_ped)
        cv2.imwrite(os.path.join(out_root, 'fluid3', realname + extension), true_irf_label)
        for i in range(3):
            name = 'layer' + str(i) + '_' + realname + '_predict_map' + extension
            label = cv2.imread(os.path.join(label_path, 'layer' + str(i) + '_predict', name), 0)
            label_list.append(label)
            os.makedirs(os.path.join(out_root, 'layers', 'layer' + str(i)), exist_ok=True)
            cv2.imwrite(os.path.join(out_root, 'layers', 'layer' + str(i), realname + extension), label)
        for i in range(3, 7):
            os.makedirs(os.path.join(out_root, 'layers', 'layer' + str(i)), exist_ok=True)
        name = 'layer4' + '_' + realname + '_predict_map' + extension
        layer4_ = cv2.imread(os.path.join(label_path, 'layer4' + '_predict', name), 0)
        name = 'layer5' + '_' + realname + '_predict_map' + extension
        layer5_ = cv2.imread(os.path.join(label_path, 'layer5' + '_predict', name), 0)
        layer4__ = np.zeros_like(layer4_)
        layer3__ = np.zeros_like(layer4_)
        layer4 = np.zeros_like(layer4_)
        layer3 = np.zeros_like(layer4_)
        for j in range(w):

            if srf_index_up[j] == 0 and srf_index_down[j] == 0:
                layer3[np.argmax(layer4_[:, j]) - 1, j] = 255
                layer4[np.argmax(layer4_[:, j]), j] = 255
            else:
                layer3[srf_index_up[j], j] = 255
                layer4[srf_index_down[j], j] = 255

        layer5 = np.zeros_like(layer5_)
        layer6 = np.zeros_like(layer5_)
        for j in range(w):
            if ped_index_up[j] == 0 and ped_index_down[j] == 0:
                if np.argmax(layer5_[:, j]) <= np.argmax(layer4[:, j]):  # 修正拓扑错误
                    layer5[np.argmax(layer4[:, j]) + 1, j] = 255
                else:
                    layer5[np.argmax(layer5_[:, j]), j] = 255
                    layer6[np.argmax(layer5_[:, j]) + 1, j] = 255
            else:
                layer5[ped_index_up[j], j] = 255
                layer6[ped_index_down[j], j] = 255

        for j in range(w):
            if np.argmax(layer5[:, j]) - np.argmax(layer4[:, j]) >= 10:
                layer4__[np.argmax(layer5[:, j]) - 10, j] = 255
            elif np.argmax(layer5[:, j]) - np.argmax(layer4[:, j]) <= 0:
                layer4__[np.argmax(layer4__[:, j]), j] = 0
                layer4__[np.argmax(layer5[:, j]) - 1, j] = 255
            else:
                layer4__[np.argmax(layer4[:, j]), j] = 255
            if srf_index_up[j] == 0 and srf_index_down[j] == 0:
                layer3__[np.argmax(layer4__[:, j]) - 1, j] = 255
            else:
                layer3__[srf_index_up[j], j] = 255
                layer4__[np.argmax(layer4__[:, j]), j] = 0
                layer4__[srf_index_down[j], j] = 255

        label_list.append(layer3__)
        label_list.append(layer4__)
        label_list.append(layer5)
        label_list.append(layer6)
        cv2.imwrite(os.path.join(out_root, 'layers', 'layer3', realname + extension), layer3__)
        cv2.imwrite(os.path.join(out_root, 'layers', 'layer4', realname + extension), layer4__)
        cv2.imwrite(os.path.join(out_root, 'layers', 'layer5', realname + extension), layer5)
        cv2.imwrite(os.path.join(out_root, 'layers', 'layer6', realname + extension), layer6)

        cw_label_wo_irf = gene_classwise_label(label_list)
        cv2.imwrite(os.path.join(out_root, 'covered_wo_irf', realname + extension), cw_label_wo_irf)
        cw_label = cw_label_wo_irf.copy()
        cw_label[np.where(true_irf_label > 0)] = 7

        cv2.imwrite(os.path.join(out_root, 'covered', realname + extension), cw_label)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='程序说明')
    parser.add_argument('--root', type=str, default='./datasets',
                        help='root path')
    parser.add_argument('--oct_device', type=str, default='Topcon',
                        help='OCT设备类型')
    parser.add_argument('--log_name', type=str, default='test_retouch_resnetv2_seed_8830',
                        help='log name')
    parser.add_argument('--dataset', type=str, default='retouch', help='dataset name')
    args = parser.parse_args()
    main(root=args.root,
         dataset=args.dataset,
         oct_device=args.oct_device,
         log_name=args.log_name,
         )
