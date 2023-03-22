import collections
import os
import numpy as np
import cv2
import xlwt

def main(save_path=None,
         val_path=None,
         oct_device='Topcon'
):
    # 6x6 mm2
    def save_to_exel2(x_list, write_path, start_row=0, logname=''):
        xls = xlwt.Workbook()
        sht1 = xls.add_sheet("Sheet1")
        start_row = start_row
        i = 0
        for k, v in x_list.items():
            sht1.write(start_row, i, k)
            sht1.write(start_row + 1, i, v)
            i += 1
        xls.save(write_path)
    divice_config = {'Cirrus': [0.011742, 0.001955*2, 0.047244],
                     'Spectralis': [0.011254, 0.003872, 0.119678],
                     'Topcon1': [0.011720, 0.0035, 0.04688],
                     'Topcon2': [0.011720, 0.0026, 0.04688]}
    T1000 = [2,3,4,5,7,9,14,15,16,21]  #Topcon T1000

    pre_dir = save_path
    truth_dir = val_path
    print(truth_dir)
    name_list = os.listdir(os.path.join(truth_dir, 'covered'))
    name_list.sort(key=lambda x: int(x.split(".")[0].split("_")[1]) * 1000 + int(x.split(".")[0].split("_")[2]))
    vol_list = collections.defaultdict(list)
    for name in name_list:
        vol_list[int(name.split('_')[1])].append(name.split('.')[0].split('_')[2])
    print(len(vol_list.keys()))
    for task in range(2):  #eval on pixel classification and index regression
        mean_avd = {'SRF':0, 'PED':0, 'IRF':0}
        if task == 0:
            filename = 'AVD_final'
            start_row = 0
            fluid_list = {'SRF': ['layer3_pixelwise_', 'fluid1'], 'PED': ['layer5_pixelwise_', 'fluid2'], 'IRF': ['fluid_', 'fluid3']}
        else:
            filename = 'AVD'
            start_row = 0
            fluid_list = {'SRF': ['layer3&fluid_', 'fluid1'], 'PED': ['layer5&fluid_', 'fluid2'], 'IRF': ['layer6&fluid_', 'fluid3']}

        for vol in vol_list.keys():
            avd = {'SRF':0, 'PED':0, 'IRF':0}
            for fluid in ['SRF', 'PED', 'IRF']:
                truth_dir_ = os.path.join(truth_dir, fluid_list[fluid][1])
                pre_dir_ = os.path.join(pre_dir, fluid_list[fluid][0]+'predict')
                prefix = fluid_list[fluid][0]
                v_x, v_y = 0, 0
                for index in vol_list[vol]:
                    realname = 'index_'+str(vol)+'_'+index
                    name = realname+'.png'
                    if oct_device == 'Topcon':
                        if index in T1000:
                            c = 'Topcon1'
                        else:
                            oct_device = 'Topcon2'
                    pre = cv2.imread(os.path.join(pre_dir_, prefix+realname+'_predict.png'), 0)
                    gt = cv2.imread(os.path.join(truth_dir_, name), 0)

                    v_x += np.sum(pre/255)*(divice_config[oct_device][0]*divice_config[oct_device][1])
                    v_y += np.sum(gt/255)*(divice_config[oct_device][0]*divice_config[oct_device][1])
                    #print(fluid,np.sum(pre/255),np.sum(gt), v_x, v_y)
                mean_avd[fluid] += abs(v_y*divice_config[oct_device][2] - v_x*divice_config[oct_device][2])
        for k in mean_avd.keys():
            mean_avd[k] /= len(vol_list.keys())
        save_to_exel2(mean_avd, os.path.join(pre_dir, filename +'.xls'), start_row=start_row)
        print(mean_avd)
