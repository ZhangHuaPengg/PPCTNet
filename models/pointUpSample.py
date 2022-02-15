'''
Author       : ZHP
Date         : 2022-01-12 20:32:57
LastEditors  : ZHP
LastEditTime : 2022-01-19 13:41:16
FilePath     : /models/pointUpSample.py
Description  : 
Copyright 2022 ZHP, All Rights Reserved. 
2022-01-12 20:32:57
'''
import torch
import torch.nn as nn
import copy
from models.pointnet.pointNet2_Ops import square_distance, index_points


def get_single_conv(input_size, in_channel, out_channel, kernel_size=1, bn=False, activate_fn=nn.ReLU()):
    layer = nn.ModuleList()
    if input_size == 3:
        layer.append(nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size))
        if bn:
            layer.append(nn.BatchNorm1d(out_channel))
    elif input_size == 4:
        layer.append(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size))
        if bn:
            layer.append(nn.BatchNorm2d(out_channel))
    else:
        print("Not implemented for {} input size yet".format(input_size))
    
    if activate_fn is not None:
        layer.append(activate_fn)
    
    return layer

class PUNetUpsample(nn.Module):
    def __init__(self, up_ratio, in_size, in_channel, out_list=[256, 128]):
        '''
        Author: ZHP
        func: 将数据复制up_ratio份，分别通过两个卷积，然后concat成N*up_ratio
        description: reference the paper[PU-Net: Point Cloud Upsampling Networ]: https://zhuanlan.zhihu.com/p/327983248
        param {scalar} up_ratio : 上采样比例
        param {scalar} in_size  ：输入数据shape长度
        param {scalar} in_channel : 输入channel
        param {list[int,int]} out_list : 两个卷积的输出channel
        '''    
        super().__init__()
        self.up_ratio, self.in_size = up_ratio, in_size
        conv1_layer = get_single_conv(in_size, in_channel, out_list[0], 1, False)
        self.conv1_layers = nn.ModuleList([copy.deepcopy(conv1_layer) for _ in range(up_ratio)])
        
        conv2_layer = get_single_conv(in_size, out_list[0], out_list[1], 1, False)
        self.conv2_layers = nn.ModuleList([copy.deepcopy(conv2_layer) for _ in range(up_ratio)])


    def forward(self, x):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} x: [B, C, *, N]
        return {torch.tensor} output : [B, C', * , up_ration*N]
        '''    
        in_size = len(list(x.shape))
        assert in_size == self.in_size, "input data must match the shape length of {},but got shape length {}!".format(self.in_size, in_size)

        new_points_list = []
        for i, conv1_layer in enumerate(self.conv1_layers):
            new_point = conv1_layer(x)
            new_point = self.conv2_layers[i](new_point)
            new_points_list.append(new_point)
        output = torch.cat(new_points_list, dim=-1)            # [B, C', *, N * up_ration]
        return output

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        '''
        Author: ZHP
        description: hierarchical Feature Propagation,用于分割任务中上采样阶段特征汇集,包括 插值 -> 拼接编码阶段特征 -> unit pointnet
        param {int} in_channel : 输入channel
        param {list} mlp : unit pointnet输出channel list
        '''    
        super().__init__()
        last_channel = in_channel
        self.layers = nn.ModuleList()
        for out_channel in mlp:
            self.layers.append(nn.Sequential(
                nn.Conv2d(last_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            ))
            last_channel = out_channel

    def forward(self, xyz_sa, xyz_now, points_sa, points_now, k=3):
        '''
        Author: ZHP
        description: 线性插值，距离越远的点权重越小。插值特征f = sum(w_i*f_i)/sum(w_i) = sum(weight_i * f_i)   w_i = 1/distance(x, x_i)^p     p=2, k=3
                     weight_i = w_i / sum(w_i)
                     N > N1  拿当前点云(基数N1)去SA时的点云(N)做KNN，得到当前点云中离高维点云每个点最近的K个点，然后用这K个点加权和作为插值，将当前点云的基数从N1升至N
        param {torch.tensor} xyz_sa : SA输出的点云坐标，[B, C, N]， 点云基数为N
        param {torch.tensor} xyz_now : 当前点云坐标，[B, C, S]  点云基数为S
        param {torch.tensor} points_sa: set abstraction层的特征,直接与插值后特征拼接  [B, D1, N] 
        param {torch.tensor} points_now: 分割阶段当前特征，shape [B, D2, S],S表示点个数，插值后为 [B, D2, N]
        param {scalar} k : knn中k

        return {torch.tensor} new_points : [B, D', N]  D'=mlp[-1]
        '''   
        xyz_sa = xyz_sa.permute(0, 2, 1)   # [B, N, C]
        xyz_now = xyz_now.permute(0, 2, 1) # [B, S, C]

        points_now = points_now.permute(0, 2, 1) # [B, S, D2]
        B, N, C = xyz_sa.shape
        _, S, _ = xyz_now.shape

        if S == 1:
            # 如果维度只有1，则直接扩展到N
            interpolated_points = points_now.repeat(1, N, 1)       # [B, N, D2]
        else:
            # KNN 插值
            dists = square_distance(xyz_sa, xyz_now)                    # [B, N, S]，距离的平方
            dists, idx = dists.sort(dim=-1)                             # [B, N, S]
            dists, idx = dists[:, :, :k], idx[:, :, :k]                 # [B, N, 3]  KNN取3
            # 以SA输出的点云(N个)为中心，计算当前点云距离其最近的K(3)个点,记录距离和索引
            # 注意，这里索引是在points2,即当前分割阶段点云数据的索引，即索引值都<S

            dist_recip = 1.0 / (dists + 1e-8)                   # 反向加权 w_i 防止为0 [B, N, 3]
            norm = torch.sum(dist_recip, dim=2, keepdim=True)   # 分母，[B, N, 1]
            weight = dist_recip / norm                          # weight_i = w_i / sum(w_i)

            # 在当前点云中取出[B, N, 3]个最近点数据[B, N, 3, D2]，乘以权重再求和得到新的插值点特征[B, N, D2], 作为新的插值点
            interpolated_points = torch.sum(index_points(points_now, idx) * weight.view(B, N, k, 1), dim=2)  # 插值特征:[B, N, D2]

        if points_sa is not None:
            # concatenate 两个level特征
            points1 = points_sa.permute(0, 2, 1)                              # [B, N, D1]
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B, N, D1+D2]
        else:
            new_points = interpolated_points                                # [B, N, D2]

        new_points = new_points.permute(0, 2, 1).unsqueeze(3)               # 经过插值和拼接后的新特征 [B, D1+D2/D2, N, 1]
        for i, layer in enumerate(self.layers):
            new_points = layer(new_points)                                  # [B, D', N, 1] D'=mlp[-1]
        new_points = new_points.squeeze(dim=-1)
        return new_points


