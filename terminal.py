'''
oct_processing.py-->split_data-->main_multitask_for_tr_yifuyuan.py

'''
import argparse
import os
import numpy as np
import sys
import sys
sys.path.append(sys.path[0]+'/Core')
sys.path.append(sys.path[0]+'/Core/multitask_models')
sys.path.append(sys.path[0]+'/Core/BM3D_py')
parser = argparse.ArgumentParser()
'''
pre processing
'''
parser.add_argument('--preprocessing', action='store_true',
                    help="flatten and enhance of OCT image")
parser.add_argument('--dataset', default='yifuyuan', help="name of dataset, including yifuyuan, duke, retouch")
parser.add_argument('--oct_device', default='Cirrus', help='OCT设备名称（仅当dataset为retouch时有效）')
parser.add_argument('--root', default='./datasets', help='数据集的根目录')
parser.add_argument('--bm3d_img_exist', action='store_true', help='是否已存在BM3D图像')
parser.add_argument('--wo_cvx', action='store_true', help='是否使用CVX')


parser.add_argument('--split_dataset', action='store_true',
                    help="split dataset for training and testing")
parser.add_argument('--merge_dataset', action='store_true',
                    help="merging yifuyuan and retouch training set for training")
parser.add_argument('--split_cross_valid', action='store_true',
                    help="split retouch dataset as k-fold cross valid form")

'''
training and testing
'''
parser.add_argument("--device_id", type=str, default='0',
                    help="which GPU to use")
parser.add_argument('--k', type=int, default=6,
                    help="k-fold validation")
parser.add_argument('--training', action='store_true',
                    help="training the model")
parser.add_argument('--testing', action='store_true',
                    help="testing the model")
parser.add_argument('--cross_valid', action='store_true',
                    help="training the model with cross validation")
parser.add_argument('--epoch', type=int, default=65, help='max epoch for training, 65 for yifuyuan and duke, 25 for retouch.')
parser.add_argument('--generate_pseudo', action='store_true',
                    help="testing on retouch dataset for generating pseudo labels")
parser.add_argument('--backbone', type=str, default='resnetv2',
                    help="selecting a backbone model, "
                         "including resnetv2, vgg, resnet, convnext, swintrans, shuffletrans, mpvit")
parser.add_argument("--log_name", type=str,
                    default=None,
                    help="giving a log name. If is None, the log name will generate by the program, eg:dataset_backbone_seed_123.")
parser.add_argument('--pretrain_path', type=str,
                    default='./datasets/yifuyuan/result/yifuyuan_resnetv2_seed_8830/weights_final.pth',
                    help='pre-trained weights path')
parser.add_argument('--seedlist', type=int, nargs='*', default=[8830], help='input: 8830 1024 64')
parser.add_argument('--k_size', type=int, default=51, help='kernel size of laplacian conv ')

args = parser.parse_args()
if args.preprocessing:
    from Core import oct_preprocess
    oct_preprocess.main(
        dataset=args.dataset,
        oct_device=args.oct_device,
        root=args.root,
        bm3d_img_exist=args.bm3d_img_exist,
        wo_cvx=args.wo_cvx
    )
if args.split_dataset:
    from Core import split_data
    split_data.main(
        root=args.root,
        dataset=args.dataset
    )
if args.generate_pseudo:
    from Core import generate_empty_label
    from Core import generate_pseudo_label
    if args.training or args.dataset != 'retouch':
        raise ValueError('using testing operation for generating pseudo labels on retouch dataset!')
    for seed in args.seedlist:
        if not args.log_name:
            log_name = 'test_' + args.dataset + '_' + args.backbone + '_seed_' + str(seed)
        else:
            log_name = args.log_name
        generate_empty_label.main(root=args.root,
                                  dataset=args.dataset,
                                  oct_device=args.oct_device)
        if args.testing:
            from Core.main_multitask_for_te_retouch import run
            run(device_id=args.device_id,
                action='test',
                dataset=args.dataset,
                oct_device=args.oct_device,
                seed=seed,
                root=args.root,
                log_name=log_name,
                ckp_path=args.pretrain_path,
                backbone=args.backbone)
        generate_pseudo_label.main(
            root=args.root,
            dataset=args.dataset,
            oct_device=args.oct_device,
            log_name=log_name,
            seed=seed)
if args.merge_dataset:
    from Core import rename_yifuyuan2retouch
    from Core import copy_dataset
    save_dir = os.path.join(args.root, 'retouch', args.oct_device+'_yifuyuan', 'mix', 'all_train')
    for i, f in enumerate(['train', 'val']):
        rename_yifuyuan2retouch.main(in_root=os.path.join(args.root, 'yifuyuan', f),
                                    out_root=os.path.join(save_dir, f),
                                    label_folder='pseudo_label', index_num=25+i)
    for n in ['flatted_IN_img_512', 'pseudo_label']:
        copy_dataset.copytree(os.path.join(args.root, 'retouch', args.oct_device, 'preprocessing_data', n),
                      os.path.join(save_dir, 'train', n))
action = []
if args.training:
    action += ['train']
if args.testing:
    action += ['test']
for seed in args.seedlist:
    if args.dataset == 'yifuyuan':
        if action:
            from Core.main_multitask_for_tr_yifuyuan import run
            for a in action:
                print('%sing the model' % a)
                run(device_id=args.device_id,
                    action=a,
                    seed=seed,
                    root=args.root,
                    log_name=args.log_name,
                    backbone=args.backbone,
                    k_size=args.k_size)
    elif args.dataset == 'duke':
        if action:
            from Core.main_multitask_for_tr_duke import run
            for a in action:
                print('%sing the model' % a)
                run(device_id=args.device_id,
                    action=a,
                    seed=seed,
                    root=args.root,
                    log_name=args.log_name,
                    backbone=args.backbone)
    elif args.dataset == 'retouch':
        if args.cross_valid:
            if action:
                from Core.main_multitask_for_tr_retouch_cross_val import run
                for i in range(args.k):
                    for a in action:
                        print('%sing the model' % a)
                        run(device_id=args.device_id,
                            id=i,
                            action=a,
                            epoch=args.epoch,
                            dataset=args.dataset,
                            oct_device=args.oct_device,
                            seed=seed,
                            root=args.root,
                            pretraining_path=args.pretrain_path,
                            log_name=args.log_name,
                            backbone=args.backbone)
        else:
            if action:
                from Core.main_multitask_for_tr_retouch_yifuyuan import run
                for a in action:
                    print('%sing the model' % a)
                    run(device_id=args.device_id,
                        action=a,
                        epoch=args.epoch,
                        dataset=args.dataset,
                        oct_device=args.oct_device,
                        seed=seed,
                        root=args.root,
                        pretraining_path=args.pretrain_path,
                        log_name=args.log_name,
                        backbone=args.backbone)
    else:
        raise ValueError('wrong dataset name! please select a dataset from: yifuyuan, duke and retouch.')


if args.split_cross_valid:
    from Core import split_data_for_cross_val
    from Core import rename_yifuyuan2retouch
    indir = os.path.join(args.root, 'retouch', args.oct_device)
    save_dir = os.path.join(args.root, 'retouch', args.oct_device+'_yifuyuan', 'mix/cross_valid')
    for seed in args.seedlist:
        for i in range(args.k):
            split_data_for_cross_val.splitting(indir,
                                                   save_dir,
                                                   oct_device=args.oct_device,
                                                   seed=seed,
                                                   id = i)
            rename_yifuyuan2retouch.main(out_root=os.path.join(save_dir, str(i), 'train'),
                                         label_folder='pseudo_label',index_num=25)
