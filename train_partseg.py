'''
Author       : ZHP
Date         : 2021-05-28 23:05:16
LastEditors  : ZHP
LastEditTime : 2022-02-15 13:57:22
FilePath     : /train_partseg.py
Description  : 项目说明
Copyright    : ZHP
'''
import os
import sys
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traceback
import torch
from trainer import shapePart_train
import models
from utils import utils

SEED = 10

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='ShapePart Segmentation!')
    parser.add_argument('--task', type=str, default='PartSeg', help='Task')
    parser.add_argument('--model_name', type=str, default='Trans_Global_Part_SSG', help='model name')
    parser.add_argument('--ckpt_folder', type=str, default='Results/TrainLogs', help='Train Logs save dir')
    parser.add_argument('--model_save_dir', type=str, default='Results/models', help='model save dir')
    parser.add_argument('--enc_num', type=int, default=1, help='the transformer encoder num')
    parser.add_argument('--share', action='store_true', default=False, help='Resume training at a breakpoint')

    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--min_lr', type=float, default=0.001, help="weight decay")
    parser.add_argument('--lr_decay', action='store_false', default=True, help='decay rate for learning rate')
    parser.add_argument('--lr_step', type=int, default=40, help='decay step for learning rate')
    parser.add_argument('--decay_ratio', type=float, default=0.5, help='decay ratio for learning rate')
    parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--init', type=str, default='xavier', help='weight initial mode')
    parser.add_argument('--gpus', type=str, default='0,1', help='specify GPU devices')

    # data config
    parser.add_argument('--data_dir', type=str, default='/root/zhp/dataset/ShapeNet', help='data dir')
    parser.add_argument('--npoints', type=int, default=2048, help='point Number')
    parser.add_argument('--normalize', action='store_true', default=True, help='use normals')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training at a breakpoint')
    
    args = parser.parse_args()
    print('主要参数配置如下：\n')
    for key, value in args.__dict__.items():
        print(f'{key:^20} : {str(value):<}')
    return args


def main():
    torch.cuda.empty_cache()
    utils.fix_seed(SEED)
    args = get_args()    
    partseg = shapePart_train.PartSegTrainer(args.__dict__, models.get_loss, \
    {"enc_num" : args.enc_num, "part_num" : 50, "share" : args.share})
    try:
        partseg.run()
    except:
        partseg.trace_end()


if __name__ == "__main__":
    main()
    # python train_partseg.py --model_name=ProgressivePointCloudTransformer_Pre --nEpochs=300 --optimizer=SGD --learning_rate=0.06 --batch_size=8 --gpus=0 --lr_step=40 