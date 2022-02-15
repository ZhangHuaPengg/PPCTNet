'''
Author       : ZHP
Date         : 2021-11-13 10:57:27
LastEditors  : ZHP
LastEditTime : 2022-01-14 21:28:40
FilePath     : /models/pointnet/pointNet2_Ops.py
Description  : PointNet++ 中的Ops
Copyright    : ZHP
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .pointNetUtils import TNetkd
from torchsummary import summary


def pc_normalize(pcs):  
    '''
    Author: ZHP
    func: 对输入点云进行归一化
    description: 减均值，除以方差
    param {ndarray} pcs 点云数据 [B, H, W]
    return {ndarray} pcs 归一化后的点云数据 [B, H, W]
    '''
    centroid = np.mean(pcs, axis=0)
    pcs = pcs - centroid       
    m = np.max(np.sqrt(np.sum(pcs**2, axis=1)))
    pcs = pcs / m
    return pcs


def square_distance(src, dst):
    '''
    Author: ZHP
    func: 计算两个点云每个点的欧氏距离的平方
    description: 点(x1, y1, z1)属于点云src, (x2, y2, z2)属于点云dst，则两点欧氏距离为
                dist = (x1 - x2)^2 + (y1 - y2)^2 + (z1- z2)^2 
                    = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst。
                因为有
                src^T * dst = x1 * x2 + y1 * y2 + z1 * z2；
                sum(src^2, dim=-1) = x1*x1 + y1*y1 + z1*z1;
                sum(dst^2, dim=-1) = x2*x2 + y2*y2 + z2*z2;
    param {torch.tensor} src: shape as [B, N, C]
    param {torch.tensor} dst: shape as [B, S, C]
    
    return {torch.tenosr} dist: shape as [B, N, S]
    '''
    B, N, _ = src.shape
    _, S, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, S)
    return dist


def index_points(points, idx):
    '''
    Author: ZHP
    func: 提取采样点数据 
    description: 提取通过FPS得到的点的索引对应的点云数据
    param {torch.tensor} points : 原始点云数据 [B, N, C]
    param {torch.tensor} idx : 采样点索引 [B, *, K]
    return {torch.tensor} new_points : 提取的对应的点云数据 [B, *, K, C]
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)   # 除batch维度外都置1 [B, 1, ..., 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)  # shape equal idx.shape [B, *, K]
    new_points = points[batch_indices, idx, :]   # [B, *, K, C]  
    return new_points  


def farthest_point_sample(xyz, npoint):
    '''
    Author: ZHP
    func: FPS(最远点采样)
    description: 
    param {torch.tensor} xyz：point cloud data to sample , shape as [B, N, 3]
    param {scalar} npoint, 采样的点个数
    
    return {torch.tensor} centroids ： 最远点采样的npoint个索引，sampled pointcloud index, [B, npoint]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 随机初始化最远点索引
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)    # 当前最远点坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)   # [B, N]，点云各点与当前最远点的欧氏距离
        mask = dist < distance
        distance[mask] = dist[mask]   # 更新distance
        farthest = torch.max(distance, -1)[1]  # 更新最远点索引
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    '''
    Author: ZHP
    func: 球形搜索
    description: 在点云中对FPS得到的S个点为球心做group搜索，每个球心搜索若干个点
    param {scalar} radius : 球形搜索半径
    param {scalar} nsample : 采样group内最大采样点的个数 
    param {torch.tensor} xyz : 原始点云数据 [B, N, C]
    param {torch.tensor} new_xyz : 从FPS获得的S个点云数据 [B, S, C] ，S个group中心(球心)
    
    return {torch.tensor} group_idx : 每个采样点球形区域内的点索引 [B, S, nsample]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])   # [B, S, N]
    sqrdists = square_distance(new_xyz, xyz)    # [B, S, N]
    group_idx[sqrdists > radius ** 2] = N       # initial，超出球形区域索引置位最大值N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]   # 取球形区域内按索引排列的前nsample个点的索引 [B, S, nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample]) 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]  # 所有group内，超出球形区域的索引默认用第一个点的球形搜索对应替换
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Author: ZHP
    func: 对点云进行sample和group
    description: 先进行FPS,然后得到每个点云的最远点索引,然后取出采样点数据new_xyz作为采样后点云数据，
                    球形搜索得到所有group的nsample采样点的数据，每个group减均值(球心)，然后视情况拼接旧的点云特征作为新的点云特征
    param {scalar} npoint : 球形区域个数,即采样中心点个数
    param {scalar} radius : 球形搜索区域半径
    param {scalar} nsample ： 每个球形区域采样点个数
    param {torch.tensor} xyz ：输入点云数据 [B, N, 3]
    param {torch.tensor} points ：点云旧特征 [B, N, D]，为None表示没有旧特征
    param {bool} returnfps : 是否返回最远点采样数据
    
    return {torch.tensor} new_xyz : 采样后新的点云数据, 即最远点(球心)的数据(坐标) [B, npoint, C]
    return {torch.tensor} new_points : 新的点云特征 [B, npoint, nsample, C + D](points有时) / [B, npoint, nsample, C]
    return {torch.tensor} grouped_xyz : 所有group内的nsample个采样点数据 [B, npoint, nsample, C], returnfps为True时返回
    return {torch.tensor} fps_idx : 最远点采样索引 [B, npoint], returnfps为True时返回
    """
    B, N, C = xyz.shape
    S = npoint

    # sampling
    fps_idx = farthest_point_sample(xyz, npoint)                # 每个点云的最远点(球心)索引 [B, npoint]
    new_xyz = index_points(xyz, fps_idx)                        # 最远点(球心)的数据(坐标) [B, npoint, C]
    
    # grouping
    idx = query_ball_point(radius, nsample, xyz, new_xyz)       # 每个球形区域内nsample个采样点的索引 [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)                        # 所有group内的nsample个采样点数据[B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)   # 新的点云特征，每个区域的点减去区域中心值(球心) [B, npoint, nsample, C]

    # 如果输入点云有特征，则与新特征concat返回，否则只返回新特征
    if points is not None:
        grouped_points = index_points(points, idx)                             # 所有group内nsample个采样点的旧特征 [B, npoint, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)      # 与新特征拼接，[B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points   # 返回新的点云数据 和 新的点云特征


def sample_and_group_all(xyz, points):
    '''
    Author: ZHP
    func: 全局sample 和 group
    description: 将所有点看成一个group，球心为原点，拼接旧的特征
    param {torch.tensor} xyz : 输入点云数据 [B, N, C]  C == 3
    param {torch.tensor} points : 点云旧特征 [B, N, D] 

    return {torch.tensor} new_xyz : 新的点云数据 [B, 1, C] 全0，表示原点，只有一个group
    return {torch.tensor} new_points ： 新的点云特征， [B, 1, N, C] / [B, 1, N, C + D]
    '''
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)  # 只有一个group， 球心为原点
    grouped_xyz = xyz.view(B, 1, N, C) # 新的点云特征，即为点云数据 [B, 1, N, C] 1代表1个group， N代表每个group有N个采样点
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # 拼接旧特征 [B, 1, N, C + D]
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

def get_mlp_layer(in_channel, out_channel, kernel_size, activate=True):
    """
    生成感知机,用于处理4D输入
    """
    model = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size),
        nn.BatchNorm2d(out_channel)
    )
    if activate:
        model.add_module("activateion_layer", nn.ReLU())
    return model


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        '''
        Author: ZHP
        description: hierarchical structure: sampling layer -> grouping layer -> pointnet
        param {int} npoint : 所有点云采样的group数
        param {scalar} radius ： group半径
        param {int} nsample : 每个group采样点数
        param {int} in_channel : 输入点云特征的channel
        param {list} mlp : 存储特征提取层(pointnet)的channels list
        param {bool} group_all : 是否进行group_all
        '''    
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channel = in_channel

        last_channel = in_channel + 3                      # in_channel = D
        self.layers = nn.ModuleList()
        for out_channel in mlp:
            self.layers.append(get_mlp_layer(last_channel, out_channel, 1))
            last_channel = out_channel
        self.max_pool = torch.max
        self.group_all = group_all

    def forward(self, xyz, points):
        '''
        Author: ZHP
        description: sampling -> grouping -> extract feature
        param {torch.tensor} xyz : 输入点云坐标 [B, C, N]  C == 3
        param {torch.tensor} points : 输入点云数据(特征) [B, D, N], 可能为None,初始时归一化坐标作为feature

        return {*} new_xyz : 采样的点云坐标(每个group中心) [B, C, npoint]/[B, C, 1](group all)
        return {*} new_points : 采样点云特征 [B, C', npoint]/[B, C', 1](group all)    C'=mlp[-1]
        '''    
        xyz = xyz.permute(0, 2, 1)                                                                          # [B, N, C]
        if points is not None:
            assert points.shape[1] == self.in_channel, "The second arguments <input points> should \
                have {0} channels, but shape as {1} with {2} channels".\
                    format(self.in_channel, points.shape, points.shape[1])
            points = points.permute(0, 2, 1)                                                                # [B, N, D]

        # sampling and grouping
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)                                         # [B, 1, C] / [B, 1, N, C+D]
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)     # [B, npoint, C] , [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)                                                         # [B, C+D, nsample,npoint]
        
        # extract feature
        for i, layer in enumerate(self.layers):
            new_points = layer(new_points)                                                                  # [B, C', nsample, npoint]
        new_points = self.max_pool(new_points, dim=2)[0]                                                    # new feature/points [B, C', npoint]

        
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        '''
        Author: ZHP
        description: Multi-scale grouping, sampling(1次) -> [grouping -> pointnet](多次) 
                    由于PointNetSetAbstraction每次
        param {int} npoint : 采样group数
        param {list} radius_list ： 半径列表
        param {list} nsample_list ： 采样点数列表，长度与radius_list相同，一一对应
        param {int} in_channel : 输入channel
        param {list[list]} mlp_list : 存储特征提取层(pointnet)的channels list
        '''    
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.in_channel = in_channel     # 输入点云特征的channels
        self.layers = nn.ModuleList()
        for i in range(len(mlp_list)):
            layer = nn.ModuleList()
            last_channel = in_channel + 3  # in_channel是点云特征channel D,加上输入点云坐标channel 3
            # 每个尺度的特征提取模块，输入channel保持一致
            for out_channel in mlp_list[i]:
                layer.append(get_mlp_layer(last_channel, out_channel, 1))
                last_channel = out_channel
            self.layers.append(layer)

    def forward(self, xyz, points):
        '''
        Author: ZHP
        description: 多种尺度(不同radius,nsample)的特征concat到一起
        param {torch.tensor} xyz : 输入点云 [B, C, N]
        param {torch.tensor} points : 点云旧特征 [B, D, N]
        
        return {torch.tensor} new_xyz : 采样点云数据[B, C, S]
        return {torch.tensor} new_points_concat : 新的特征 [B, D', S], D'是layer最后输出维度，即sum(mlp_list[-1])，S=npoint
        '''                                             
        xyz = xyz.permute(0, 2, 1)                                                  # [B, N, C]
        if points is not None:
            assert points.shape[1] == self.in_channel, "The second arguments <input points> should \
                have {0} channels, but shape as {1} with {2} channels".\
                    format(self.in_channel, points.shape, points.shape[1])
            points = points.permute(0, 2, 1)                                        # [B, N, D]

        B, N, C = xyz.shape
        S = self.npoint

        # sampling
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))                  # sample后的group形心数据[B, S, C]
        new_points_list = []

        for i, radius in enumerate(self.radius_list):
            # 每个(radius,nsample)做一次grouping + feature extract
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)                   # 每个group内采样点索引, [B, S, K]
            grouped_xyz = index_points(xyz, group_idx)                              # 所有采样点数据 [B, S, K, C]
            grouped_xyz -= new_xyz.view(B, S, 1, C)                                 # 减形心，成新特征 [B, S, K, C]
            if points is not None:
                # 如果有旧特征需要拼接
                grouped_points = index_points(points, group_idx)                    # [B, S, K, D]
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)   # [B, S, K, C+D]
            else:
                grouped_points = grouped_xyz                                        # [B, S, K, C]
            grouped_points = grouped_points.permute(0, 3, 2, 1)                     # [B, C, K, S]/[B, C+D, K, S]
            
            # extract feature
            for layer in self.layers[i]:
                grouped_points = layer(grouped_points)                              # [B, D_i, K, S]
            new_points = torch.max(grouped_points, 2)[0]                            # [B, D_i, S]  D_i=mlp_list[-1][-1]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)                                          # [B, C, S]
        new_points_concat = torch.cat(new_points_list, dim=1)                       # [B, sum(D_i), S], 将mulit scale的特征concat到一起
        return new_xyz, new_points_concat

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
            self.layers.append(get_mlp_layer(last_channel, out_channel, 1))     # unit pointnet
            last_channel = out_channel

 
    def forward(self, xyz_sa, xyz_now, points_sa, points_now):
        '''
        Author: ZHP
        description: 线性插值，距离越远的点权重越小。插值特征f = sum(w_i*f_i)/sum(w_i) = sum(weight_i * f_i)   w_i = 1/distance(x, x_i)^p     p=2, k=3
                     weight_i = w_i / sum(w_i)
                     N > N1  拿当前点云(基数N1)去SA时的点云(N)做KNN，得到当前点云中离高维点云每个点最近的K个点，然后用这K个点加权和作为插值，将当前点云的基数从N1升至N
        param {torch.tensor} xyz_sa : SA输出的点云坐标，[B, C, N]， 点云基数为N
        param {torch.tensor} xyz_now : 当前点云坐标，[B, C, S]  点云基数为S
        param {torch.tensor} points_sa: set abstraction层的特征,直接与插值后特征拼接  [B, D1, N] 
        param {torch.tensor} points_now: 分割阶段当前特征，shape [B, D2, S],S表示点个数，插值后为 [B, D2, N]
        
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
            dists, idx = dists[:, :, :3], idx[:, :, :3]                 # [B, N, 3]  KNN取3
            # 以SA输出的点云(N个)为中心，计算当前点云距离其最近的K(3)个点,记录距离和索引
            # 注意，这里索引是在points2,即当前分割阶段点云数据的索引，即索引值都<S

            dist_recip = 1.0 / (dists + 1e-8)                   # 反向加权 w_i 防止为0 [B, N, 3]
            norm = torch.sum(dist_recip, dim=2, keepdim=True)   # 分母，[B, N, 1]
            weight = dist_recip / norm                          # weight_i = w_i / sum(w_i)

            # 在当前点云中取出[B, N, 3]个最近点数据[B, N, 3, D2]，乘以权重再求和得到新的插值点特征[B, N, D2], 作为新的插值点
            interpolated_points = torch.sum(index_points(points_now, idx) * weight.view(B, N, 3, 1), dim=2)  # 插值特征:[B, N, D2]

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


if __name__ == "__main__":
    pass

