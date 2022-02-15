'''
Author       : ZHP
Date         : 2021-12-09 19:38:41
LastEditors  : ZHP
LastEditTime : 2022-01-05 16:37:39
FilePath     : /TransformerPlus/utils/utils.py
Description  : 
Copyright 2021 ZHP, All Rights Reserved. 
2021-12-09 19:38:41
'''
import time
import functools
import numpy as np
import torch
import torch.nn.functional as F
import os
import math
import random


def cal_time(total_time):
    h = total_time // 3600
    minute = (total_time % 3600) // 60
    sec = int(total_time % 60)
    print(f'本次总时长为： {h} hours {minute} minutes {sec} seconds...')


def show_time_cost(func):
    '''自动显示耗时装饰器'''
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time_start = time.time()
        print('{} process start at {}'.format(func.__name__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        result = func(*args, **kw)
        time_end = time.time()
        print('{} process end at {}\n'.format(func.__name__, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        cal_time(time_end - time_start)
        return result
    return wrapper

def capture_time_cost(func):
    '''捕获耗时装饰器'''
    @functools.wraps(func)
    def wrapper(*args, **kw):
        time_start = time.time()
        result = func(*args, **kw)
        time_end = time.time()
        return result, round(time_end - time_start)
    return wrapper

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    # scales = np.random.uniform(scale_low, scale_high, B)
    scales = torch.zeros(B).uniform_(scale_low, scale_high)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    # shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    shifts = torch.zeros((B, 3)).uniform_(-shift_range, shift_range)
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def make_one_hot(y, num_classes):
    '''
    Author: ZHP
    func: 1-hot encodes a tensor
    param {torch.tensor} y : [*],每个数据表示类别，其值不大于num_classes
    param {scalar} num_classes ： 类别数目
    return {torch.tensor} [*, num_classes] : 对应的one-hot编码
    '''
    new_y = torch.eye(num_classes, device=y.device)[y.cpu().data.numpy(),]
    return new_y

def get_context_loss(pred, target, context, context_gt, num_classes=50):
        '''
        pred :[B, classes, N]
        target : [B, N]
        context : [B, N, N]
        context_gt : [B, N]  (target)
        '''
        # print("pred :{} || target : {} || context:{}||context_gt:{}".format(
        #     pred.shape, target.shape, context.shape, context_gt.shape
        # ))
        context_gt = make_one_hot(context_gt, num_classes)  # [B, N, num_classes]
        # get LL^T  --- Ideal Affinity Map\
        # print("context_gt.shape", context_gt.shape)
        context_gt = torch.bmm(context_gt,context_gt.permute(0, 2, 1)) # [B, N, N]
        
        
        loss = F.cross_entropy(pred+1e-8, target)
        # context prior loss
        loss_u=F.binary_cross_entropy_with_logits(context.float(),context_gt.float())  # 二分类交叉熵
        # precision
        # loss_g_p = torch.log((context.mul(context_gt)).sum(2)+1e-8) - torch.log(context.sum(2)+1e-8)
        loss_g_p = torch.log((context.mul(context_gt)).sum(2)+1e-8) - torch.log(context.sum(2)+1e-8)

        # recall
        loss_g_r = torch.log((context.mul(context_gt)).sum(2)+1e-8) - torch.log(context_gt.sum(2)+1e-8)
        
        # specificity  
        # loss_g_s = torch.log(((torch.ones_like(context) - context).mul((torch.ones_like(context_gt) - context_gt))).sum(2)+1e-8) - torch.log((1 - context_gt).sum(2)+1e-8)
        loss_g_s = torch.log(((torch.ones_like(context) - context).mul((torch.ones_like(context_gt) - context_gt))).sum(2)+1e-8) - torch.log((1 - context_gt).sum(2)+1e-8)
        loss_g = - torch.mean(loss_g_p + loss_g_r + loss_g_s)
       
        total_loss = loss + loss_u + loss_g   #+loss_fc
        return total_loss


def delete_useless_files(folder, use_file, suffix='pth'):
    '''删除文件夹下除use_file外的后缀为suffix的文件'''
    size = 0
    for file_name in os.listdir(folder):
        file_suffix = file_name.split(".")[-1]
        file_path = os.path.join(folder, file_name)
        if file_suffix == suffix and (file_name not in use_file):
            try:
                file_size = os.path.getsize(file_path) / 1024
                os.remove(file_path)
                size += file_size
            except:
                print(f"{file_path} delete fail")
    gb, mb, kb = display_size(size)
    print(f"共清理 {gb:^}GB {mb:^}MB {kb:^}KB 无用文件")


def display_size(size):
    '''size:以KB为单位'''
    RATE = 1024
    gb = size // (RATE * RATE)
    size %= (RATE * RATE)
    mb = size // RATE
    size %= RATE
    size = math.floor(size)
    return gb, mb, size

def save_model_structure(model, file_path):
    with open(file_path, 'a') as f:
        print(model, file=f)
        print("save model structure at {}".format(file_path))

def idx_pt(pts, idx):
    raw_size  = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(pts, 1, idx[..., None].expand(-1, -1, pts.size(-1)))
    return res.reshape(*raw_size,-1)

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
    
    