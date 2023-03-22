import cv2
import math
import os
import numpy as np
def main(in_root='./datasets/yifuyuan/train',
         out_root ='./datasets/yifuyuan_to_retouch/train',
         label_folder = 'pseudo_label',
         index_num=25):
    image_path = os.path.join(in_root, 'flatted_IN_img_512')
    label_path = os.path.join(in_root, 'flatted_label_512')
    out_image_path = os.path.join(out_root, 'flatted_IN_img_512')
    out_label_path = os.path.join(out_root, label_folder)
    label_fl3_path = os.path.join(label_path, 'fluid3')
    label_fl2_path = os.path.join(label_path, 'fluid2')
    label_fl1_path = os.path.join(label_path, 'fluid1')
    label_h_path = os.path.join(label_path, 'coordinate_map_h')
    label_v_path = os.path.join(label_path, 'coordinate_map_v')
    os.makedirs(os.path.join(out_label_path, 'fluid3'), exist_ok=True)
    os.makedirs(os.path.join(out_label_path, 'fluid2'), exist_ok=True)
    os.makedirs(os.path.join(out_label_path, 'fluid1'), exist_ok=True)
    if not os.path.exists(os.path.join(out_label_path, 'covered')):
        os.makedirs(os.path.join(out_label_path, 'covered'))
    if not os.path.exists(os.path.join(out_label_path, 'covered_wo_irf')):
        os.makedirs(os.path.join(out_label_path, 'covered_wo_irf'))

    imglist = os.listdir(image_path)
    imglist.sort(key=lambda x: int(x.split("---")[1]))
    print(imglist)
    for n, name in enumerate(imglist):

        if n < 10:
            new_name = 'index_'+str(index_num)+'_00' + str(n) + '.png'
        else:
            new_name = 'index_'+str(index_num)+'_0' + str(n) + '.png'
        print(new_name)
        realname, extension = os.path.splitext(name)
        img = cv2.imread(os.path.join(image_path, name), 0)
        os.makedirs(out_image_path, exist_ok=True)
        cv2.imwrite(os.path.join(out_image_path, new_name), img)
        h, w = img.shape
        label_list = []
        for i in range(7):
            label = cv2.imread(os.path.join(label_path, 'layers', 'layer' + str(i), name), 0)
            # label = crop_img(label, 500)
            print(os.path.join(label_path, 'layers', 'layer' + str(i), name))
            label_list.append(label)
            os.makedirs(os.path.join(out_label_path, 'layers', 'layer' + str(i)), exist_ok=True)
            cv2.imwrite(os.path.join(out_label_path, 'layers', 'layer' + str(i), new_name), label)
        cw_label = cv2.imread(os.path.join(label_path, 'covered', name), 0)
        cv2.imwrite(os.path.join(out_label_path, 'covered', new_name), cw_label)
        cw_label_wo_irf = cv2.imread(os.path.join(label_path, 'covered_wo_irf', name), 0)
        cv2.imwrite(os.path.join(out_label_path, 'covered_wo_irf', new_name), cw_label_wo_irf)

        label_fl1 = cv2.imread(os.path.join(label_fl1_path, name), 0)
        cv2.imwrite(os.path.join(out_label_path, 'fluid1', new_name), label_fl1)
        label_fl2 = cv2.imread(os.path.join(label_fl2_path, name), 0)
        cv2.imwrite(os.path.join(out_label_path, 'fluid2', new_name), label_fl2)
        label_fl3 = cv2.imread(os.path.join(label_fl3_path, name), 0)
        cv2.imwrite(os.path.join(out_label_path, 'fluid3', new_name), label_fl3)
        '''cw_label = gene_classwise_label(label_list, label_fl)
        cv2.imwrite(os.path.join(label_path, 'covered', name), cw_label)'''
if __name__ == '__main__':
    main()
