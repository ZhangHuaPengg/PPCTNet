'''
Author       : ZHP
Date         : 2022-01-14 15:00:20
LastEditors  : ZHP
LastEditTime : 2022-01-22 13:58:36
FilePath     : /models/PointFormer/pointformer_utils.py
Description  : 
Copyright 2022 ZHP, All Rights Reserved. 
2022-01-14 15:00:20
'''
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from models.pointnet.pointNet2_Ops import *
from models.PointFormer.trans_block import *


def sample_and_group_trans(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Author: ZHP
    func: 对点云进行sample和group,区别是不拼接采样坐标
    description: 先进行FPS,然后得到每个点云的最远点索引,然后取出采样点数据new_xyz作为采样后点云数据，
                    球形搜索得到所有group的nsample采样点的数据，每个group减均值(球心)，然后视情况拼接旧的点云特征作为新的点云特征
    param {scalar} npoint : 球形区域个数,即采样中心点个数
    param {scalar} radius : 球形搜索区域半径
    param {scalar} nsample ： 每个球形区域采样点个数
    param {torch.tensor} xyz ：输入点云数据 [B, N, C]  (c==3)
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
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)   # 新的点云特征，每个区域的点减去区域中心值(球心) [B, npoint, nsample, C] relative postion

    # 如果输入点云有特征，则与新特征concat返回，否则只返回新特征
    if points is not None:
        grouped_points = index_points(points, idx)                             # 所有group内nsample个采样点的旧特征 [B, npoint, nsample, D]
        new_points = grouped_points                                             # not concat with cooridinate
        # new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)      # 与新特征拼接，[B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points   # 返回新的点云数据 和 新的点云特征


def sample_and_group_all_trans(xyz, points):
    '''
    Author: ZHP
    func: 全局sample 和 group, without cooridinate
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
        # new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # 拼接旧特征 [B, 1, N, C + D]
        new_points = points.unsqueeze(1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def group_point(npoint, radius, nsample, xyz, points):
    """
    Author: ZHP
    func: 对点云进行sample和group
    description: 先进行FPS,然后得到每个点云的最远点索引,然后取出采样点数据new_xyz作为采样后点云数据，球形搜索得到所有group的nsample采样点的数据
    param {scalar} npoint : 球形区域个数,即采样中心点个数
    param {scalar} radius : 球形搜索区域半径
    param {scalar} nsample ： 每个球形区域采样点个数
    param {torch.tensor} xyz ：输入点云数据 [B, N, 3]
    param {torch.tensor} points ：点云旧特征 [B, N, D]，为None表示没有旧特征
    
    return {torch.tensor} grouped_xyz : 所有group内的nsample个采样点坐标 [B, npoint, nsample, C]
    return {torch.tensor} grouped_points : 所有group内的nsample个采样点数据 [B, npoint, nsample, D]
    """
    B, N, C = xyz.shape
    S = npoint

    # sampling
    fps_idx = farthest_point_sample(xyz, npoint)                # 每个点云的最远点(球心)索引 [B, npoint]
    new_xyz = index_points(xyz, fps_idx)                        # 最远点(球心)的数据(坐标) [B, npoint, C]
    
    # grouping
    idx = query_ball_point(radius, nsample, xyz, new_xyz)       # 每个球形区域内nsample个采样点的索引 [B, npoint, nsample]
    grouped_xyz = index_points(xyz, idx)                        # 所有group内的nsample个采样点数据[B, npoint, nsample, C]
    grouped_points = None
    if points is not None:
        grouped_points = index_points(points, idx)              # 所有group内nsample个采样点的旧特征 [B, npoint, nsample, D]
    return grouped_xyz, grouped_points


class TransformerSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, transformer, group_all):
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
        self.transform_block = transformer                      # extract feature, last output dimension is target_num
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
            new_xyz, new_points = sample_and_group_all(xyz, points) 
            grouped_xyz = xyz.unsqueeze(1)                                       # [B, 1, C] / [B, 1, N, C+D]
        else:
            new_xyz, new_points, grouped_xyz, _ = sample_and_group(self.npoint, self.radius,\
                self.nsample, xyz, points, returnfps=True)     # [B, npoint, C]/[B, npoint, nsample, D]/[B, npoint, nsample, C]
        
        # extract feature
        new_points = self.transform_block(new_points, grouped_xyz.transpose(1,3))                           # [B, npoint, nsample, target_num]
        new_points = self.max_pool(new_points, dim=2)[0].transpose(1,2)                                     # new feature/points [B, target_num, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        return new_xyz, new_points


class TransEncoderSA(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, linear_project, transformer, group_all):
        '''
        Author: ZHP
        description: hierarchical structure: sampling layer -> grouping layer -> extract local feature --> output
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

        self.linear_project = linear_project
        self.transformer_encoder = transformer
        self.max_pool = torch.max
        self.group_all = group_all

    def forward(self, xyz, points):
        '''
        Author: ZHP
        description: sampling -> grouping -> extract feature
                     [B, C, N] --> [B, C+D, nsample, npoint] --> [B, D', nsample, npoint] -->transformer-->[B, D', nsample, npoint] -->max->[B, D', npoint]
        param {torch.tensor} xyz : 输入点云坐标 [B, C, N]  C == 3
        param {torch.tensor} points : 输入点云数据(特征) [B, D, N], 可能为None,初始时归一化坐标作为feature

        return {*} new_xyz : 采样的点云坐标(每个group中心) [B, C, npoint]/[B, C, 1](group all)
        return {*} new_points : 采样点云特征 [B, C', npoint]/[B, C', 1](group all)    C'=mlp[-1]
        '''    
        xyz = xyz.permute(0, 2, 1)                                                                          # [B, N, C]
        if points is not None:
            assert points.shape[1] == self.in_channel, "The second arguments <input points> should have {0} channels, but shape as {1} with {2} channels".\
                    format(self.in_channel, points.shape, points.shape[1])
            points = points.permute(0, 2, 1)                                                                # [B, N, D]

        # sampling and grouping
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)                                         # [B, 1, C] / [B, 1, N, C+D]
            group_xyz = xyz.unsqueeze(1)
        else:
            # [B, npioint, C]  [B, npoint, nsample, C+D] [B, npoint, nsample, C]
            new_xyz, new_points, group_xyz, _ = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, returnfps=True)    
        
        # extract feature
        local_feature = self.linear_project(new_points.transpose(1, 3))                         # [B, D', nsample, npoint]
        local_feature = self.transformer_encoder(local_feature, group_xyz.transpose(1, 3))      # [B, npoint, nsample, D']
        new_points = self.max_pool(local_feature, dim=2)[0].transpose(1,2)                      # new feature/points [B, C', npoint]
    
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        return new_xyz, new_points


class Trans_Global_SSG(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, transformer):
        '''
        Author: ZHP
        description: hierarchical structure: sampling layer -> grouping layer -> pointnet (downsample)--> transformer
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
        self.transform_block = transformer
        self.max_pool = torch.max
        self.group_all = group_all

    def forward(self, xyz, points):
        '''
        Author: ZHP
        description: sampling -> grouping ->  mlp编码 ->max downsample -> transformer(dim=npoint)
        param {torch.tensor} xyz : 输入点云坐标 [B, C, N]  C == 3
        param {torch.tensor} points : 输入点云数据(特征) [B, D, N], 可能为None,初始时归一化坐标作为feature

        return {torch.tensor} new_xyz : 采样的点云坐标(每个group中心) [B, C, npoint]/[B, C, 1](group all)
        return {torch.tensor} new_points : 采样点云特征 [B, C', npoint]/[B, C', 1](group all)    C'=mlp[-1]
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
        for layer in self.layers:
            new_points = layer(new_points)                                                                  # [B, C', nsample, npoint]
        new_points = self.max_pool(new_points, dim=2)[0]                                                    # new feature/points [B, C', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        # tranformer encoding
        new_points = self.transform_block(new_points.transpose(1, 2), new_xyz)                              # [B, npoint, C']
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


class Trans_Global_SSG_v2(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, transformer):
        '''
        Author: ZHP
        description: hierarchical structure: sampling layer -> grouping layer -> pointnet--> transformer(dim=npoint) -> downsample
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
        self.transform_block = transformer
        self.max_pool = torch.max
        self.group_all = group_all

    def forward(self, xyz, points):
        '''
        Author: ZHP
        description: sampling -> grouping -> mlp编码->transformer (dim=npoint)->max downsample
        param {torch.tensor} xyz : 输入点云坐标 [B, C, N]  C == 3
        param {torch.tensor} points : 输入点云数据(特征) [B, D, N], 可能为None,初始时归一化坐标作为feature

        return {torch.tensor} new_xyz : 采样的点云坐标(每个group中心) [B, C, npoint]/[B, C, 1](group all)
        return {torch.tensor} new_points : 采样点云特征 [B, C', npoint]/[B, C', 1](group all)    C'=mlp[-1]
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
            group_xyz = xyz.unsqueeze(1)
        else:
            new_xyz, new_points, group_xyz, _ = sample_and_group(self.npoint, self.radius, self.nsample, 
            xyz, points, returnfps=True)         # [B, npoint, C] , new_points-[B, npoint, nsample, C+D], group_xyz-[B, npoint, nsample, C]
        new_points = new_points.permute(0, 3, 2, 1)                                                         # [B, C+D, nsample,npoint]
        
        # extract feature
        for layer in self.layers:
            new_points = layer(new_points)                                                                  # [B, C', nsample, npoint]
        new_points = new_points.permute(0, 2, 3, 1)                                                         # [B, nsample, npoint, C']
        new_points = self.transform_block(new_points, group_xyz.permute(0, 3, 1, 2))           
        new_points = self.max_pool(new_points, dim=1)[0]                                                    # new feature/points [B, npoint, C']
        
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        new_points = new_points.permute(0, 2, 1)                                                            # [B, C, npoint]
        return new_xyz, new_points


class Trans_Global_SSG_v3(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, transformer):
        '''
        Author: ZHP
        description: hierarchical structure: sampling layer -> grouping layer -> pointnet--> transformer(dim=nsample) -> downsample
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
        self.transform_block = transformer
        self.max_pool = torch.max
        self.group_all = group_all

    def forward(self, xyz, points):
        '''
        Author: ZHP
        description: sampling -> grouping -> mlp编码->transformer (dim=nsample)->max downsample
        param {torch.tensor} xyz : 输入点云坐标 [B, C, N]  C == 3
        param {torch.tensor} points : 输入点云数据(特征) [B, D, N], 可能为None,初始时归一化坐标作为feature

        return {torch.tensor} new_xyz : 采样的点云坐标(每个group中心) [B, C, npoint]/[B, C, 1](group all)
        return {torch.tensor} new_points : 采样点云特征 [B, C', npoint]/[B, C', 1](group all)    C'=mlp[-1]
        '''    
        xyz = xyz.permute(0, 2, 1)                                                                          # [B, N, C]
        if points is not None:
            assert points.shape[1] == self.in_channel, "The second arguments <input points> should \
                have {0} channels, but shape as {1} with {2} channels".\
                    format(self.in_channel, points.shape, points.shape[1])
            points = points.permute(0, 2, 1)                                                                 # [B, N, D]

        # sampling and grouping
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)                                         # [B, 1, C] / [B, 1, N, C+D]
            group_xyz = xyz.unsqueeze(1)
        else:
            new_xyz, new_points, group_xyz, _ = sample_and_group(self.npoint, self.radius, self.nsample, 
            xyz, points, returnfps=True)         # [B, npoint, C] , new_points-[B, npoint, nsample, C+D], group_xyz-[B, npoint, nsample, C]
        new_points = new_points.permute(0, 3, 2, 1)                                                         # [B, C+D, nsample,npoint]
        
        # extract feature
        for layer in self.layers:
            new_points = layer(new_points)                                                                  # [B, C', nsample, npoint]
        new_points = new_points.transpose(1, 3)                                                             # [B, npoint, nsample, C']
        new_points = self.transform_block(new_points, group_xyz.transpose(1,3))           
        new_points = self.max_pool(new_points, dim=2)[0]                                                    # new feature/points [B, npoint, C']
        
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        new_points = new_points.permute(0, 2, 1)                                                            # [B, C, npoint]
        return new_xyz, new_points


class Trans_Local_SSG(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, dim_model, group_all, transformer):
        '''
        Author: ZHP
        description: hierarchical structure: sampling layer -> grouping layer -> local transformer --> max downsample
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
        self.project = nn.Sequential(
            nn.Conv2d(last_channel, dim_model, 1),
            nn.BatchNorm2d(dim_model)
        )
        self.transform_block = transformer
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
            group_xyz = xyz.unsqueeze(1).transpose(1, 3)        # [B, C, N, 1]
        else:
            new_xyz, new_points, group_xyz, _= sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, returnfps=True)     # [B, npoint, C] , [B, npoint, nsample, C+D]
            group_xyz = group_xyz.transpose(1, 3)   # [B, C, nsample, npoint]
        new_points = new_points.permute(0, 3, 2, 1)                                                         # [B, C+D, nsample,npoint]
        
        # extract feature by tranformer 
        new_points = self.project(new_points)                                                               # [B, dim_model, nsample, npoint]
        new_points = self.transform_block(new_points.transpose(1, 3), group_xyz)
        new_points = self.max_pool(new_points, dim=2)[0].transpose(1, 2)                                    # new feature/points [B, C', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]

        return new_xyz, new_points

class Trans_token_SSG(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, dim_model, group_all, transformer, fc_layer):
        '''
        Author: ZHP
        description: 先做transformer --> sampling --> extract sample point feature(downsample)，从N*D的trans_feature提取npoint的
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

        last_channel = in_channel                     # in_channel = D
        self.project = nn.Sequential(
            nn.Conv2d(last_channel, dim_model, 1),
            nn.BatchNorm2d(dim_model)
        )
        self.transform_block = transformer
        self.fc = fc_layer
        self.group_all = group_all

    def forward(self, xyz, points):
        '''
        Author: ZHP
        description: transformer  --> sampling --> 提取sample点的特征作为新特征
        param {torch.tensor} xyz : 输入点云坐标 [B, C, N]  C == 3
        param {torch.tensor} points : 输入点云数据(特征) [B, D, N], 可能为None,初始时归一化坐标作为feature

        return {torch.tensor} new_xyz : 采样的点云坐标(每个group中心) [B, C, npoint]/[B, C, 1](group all)
        return {torch.tensor} new_points : 采样点云特征 [B, D', npoint]/[B, D', 1](group all)   
        '''    
        if points is not None:
            assert points.shape[1] == self.in_channel, "The second arguments <input points> should \
                have {0} channels, but shape as {1} with {2} channels".\
                    format(self.in_channel, points.shape, points.shape[1])

        features = self.project(points).transpose(1, 2)     # [B, N, D']                 
        features = self.transform_block(features, xyz)      # [B, N, D']
        xyz = xyz.permute(0, 2, 1)

        if self.group_all:
            device = xyz.device
            B, N, C = xyz.shape
            new_xyz = torch.zeros(B, 1, C).to(device)                   # 只有一个group， 球心为原点
            new_points = torch.max(features.unsqueeze(1), dim=2)[0]      # [B, 1, D']
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)            # 每个点云的最远点(球心)索引 [B, npoint]
            new_points = index_points(features, fps_idx)                # [B, npoint, D']
            new_xyz = index_points(xyz, fps_idx)                        # 最远点(球心)的数据(坐标) [B, npoint, C]

        new_points = torch.cat([new_points, new_xyz], dim=-1)             # [B, npoint, C+D']
        new_points = self.fc(new_points.transpose(1,-1))                  # [B, C', D'']            
        new_xyz = new_xyz.transpose(1, 2)                               # [B, C, npoint]
        return new_xyz, new_points


class PointTransBlock(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, transformer, group_all, project_layer):
        '''
        Author: ZHP
        description: hierarchical structure: sampling layer -> grouping layer -> transformer
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
        self.transform_block = transformer                      # extract feature, last output dimension is target_num
        self.project = project_layer
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
            new_xyz, new_points = sample_and_group_all(xyz, points) 
            grouped_xyz = xyz.unsqueeze(1)                                       # [B, 1, C] / [B, 1, N, C+D]
        else:
            new_xyz, new_points, grouped_xyz, _ = sample_and_group(self.npoint, self.radius,\
                self.nsample, xyz, points, returnfps=True)     # [B, npoint, C]/[B, npoint, nsample, D]/[B, npoint, nsample, C]
        
        # extract feature
        new_points = self.transform_block(new_points, grouped_xyz.transpose(1,3))                           # [B, npoint, nsample, target_num]
        new_points = self.max_pool(new_points, dim=2)[0].transpose(1,2)                                     # new feature/points [B, target_num, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        return new_xyz, new_points

class TransFeaturePropagation(nn.Module):
    def __init__(self, pre_block=None, transformer=None):
        '''
        Author: ZHP
        description: hierarchical Feature Propagation,用于分割任务中上采样阶段特征汇集,包括 插值 -> 拼接编码阶段特征 -> embeddding+transformer
        param {int} in_channel : 输入channel
        param {list} mlp : unit pointnet输出channel list
        '''    
        super().__init__()
        # TODO:将unit pointnet 改成transfomer结构
        self.pre_block = pre_block
        self.transformer = transformer

 
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

        if self.pre_block:
            new_points = self.pre_block(new_points.permute(0, 2, 1))
            new_points = new_points.permute(0, 2, 1)
        new_points = self.transformer(new_points, xyz_sa.permute(0, 2, 1))                           # [B, N, D1+D2/D2]
        new_points = new_points.transpose(1, -1)
        return new_points


class JointTransPointAbstraction(nn.Module):
    def __init__(self,
                npoint,
                radius, 
                nsample, 
                in_channel, 
                group_all, 
                transformer_1, 
                transformer_2, 
                pre_block=None,         # pre embedding
                back_block=None
                ):
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

        self.pre_block, self.back_block = pre_block, back_block
        self.trans_block_1 = transformer_1
        self.trans_block_2 = transformer_2

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
            group_xyz = xyz.unsqueeze(1)            # [B, 1, N, C]
        else:
            # new_points [B, npoint, nsample, C+D]   group_xyz [B, npoint, nsample, 3]
            new_xyz, new_points, group_xyz, _ = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, returnfps=True)     # [B, npoint, C] , [B, npoint, nsample, C+D]
        
        if self.pre_block:
            new_points = self.pre_block(new_points.transpose(1, -1))
            new_points = new_points.permute(0, 3, 2, 1)                                                     # [B, npoint, nsample, C+D]
        new_points = self.trans_block_1(new_points, group_xyz.transpose(1, -1))                             # [B, npoint, nsample, C+D]
        
        if self.back_block:
            new_points = self.back_block(new_points.transpose(1,-1))
            new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.max_pool(new_points, dim=2)[0]                                                    # [B, npoint, C']
        new_points = self.trans_block_2(new_points, new_xyz.transpose(1, -1))                               # [B, npoint, C'']
        new_points = new_points.transpose(1, -1)                                                            # [B, C', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        return new_xyz, new_points


class JointTransPointAbstraction_2(nn.Module):
    def __init__(self,
                npoint,
                radius, 
                nsample, 
                in_channel, 
                group_all, 
                transformer_1, 
                transformer_2, 
                pre_block=None,         # pre embedding
                back_block=None
                ):
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

        self.pre_block, self.back_block = pre_block, back_block
        self.trans_block_1 = transformer_1
        self.trans_block_2 = transformer_2

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
            group_xyz = xyz.unsqueeze(1)            # [B, 1, N, C]
        else:
            # new_points [B, npoint, nsample, C+D]   group_xyz [B, npoint, nsample, 3]
            new_xyz, new_points, group_xyz, _ = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, returnfps=True)     # [B, npoint, C] , [B, npoint, nsample, C+D]
        
        new_points = self.trans_block_1(new_points, group_xyz.transpose(1, -1))                             # [B, npoint, nsample, C+D]
        new_points = self.max_pool(new_points, dim=2)[0]                                                    # [B, npoint, C+D]
        if self.pre_block:
            new_points = self.pre_block(new_points.transpose(1, -1))
            new_points = new_points.permute(0, 2, 1)                                                        # [B, npoint, C']
        new_points = self.trans_block_2(new_points, new_xyz.transpose(1, -1))                               # [B, npoint, C']
        if self.back_block:
            new_points = self.back_block(new_points.transpose(1,-1))
            new_points = new_points.permute(0, 2, 1)                                                        # [B, npoint, C']
        new_points = new_points.transpose(1, -1)                                                            # [B, C', npoint]
        new_xyz = new_xyz.permute(0, 2, 1)                                                                  # [B, C, npoint]
        return new_xyz, new_points


# TODO:transformer Encoder layers 中不同level Encoder的output可以在最后concat起来

if __name__ == "__main__":
    model = JointTransPointAbstraction(512, 0.2, 32, 6, False,\
        transformer_1=Learn_Pos_TransEncoder(enc_num=2, dim_model=64, dim_hid=128, heads_count=8, in_size=4),\
            transformer_2=Learn_Pos_TransEncoder(enc_num=2, dim_model=64, dim_hid=128, heads_count=8, in_size=3),\
                pre_block=K_MLP_Layer(4, 9, [32, 64], [True, True], [True, True], [False, False])) 
    pts = torch.rand(2, 6, 2048, dtype=torch.float)
    new_y, new_p  = model(pts[:,:3,:], pts)
    print(new_y.shape, new_p.shape)