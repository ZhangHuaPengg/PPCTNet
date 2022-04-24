'''
Author       : ZHP
Date         : 2021-12-08 15:37:26
LastEditors  : ZHP
LastEditTime : 2022-03-21 21:29:55
FilePath     : /models/pointnet/PointNetPlusPlusModel.py
Description  : 
Copyright 2021 ZHP, All Rights Reserved. 
2021-12-08 15:37:26
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .pointNet2_Ops import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation
from utils.summary import summary


def get_perceptron_layer(in_channel, out_channel, bn=True, activate=True, dropout=0.4):
    """
    感知机，用于3D输入 [B, C, N]
    """
    model = nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=1)
    )
    if bn:
        model.add_module("BN", nn.BatchNorm1d(out_channel))
    if activate:
        model.add_module("activation", nn.ReLU())
    if dropout is not None:
        model.add_module("dropout", nn.Dropout(dropout))
    return model



class PointNetPlusPlusCls_SSG(nn.Module):

    def __init__(self, class_num, normal=True):
        '''
        Author: ZHP
        description: PointNet++ 分类网络,base Single Scale Grouping 
        param {int} class_num : 类别数
        param {bool} normal : 输入点云数据是否经过normal
        '''   
        super().__init__()
        in_channel = 3 if normal else 0
        self.normal = normal
        now_channel = in_channel
        self.SA_1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=now_channel,\
            mlp=[64, 64, 128], group_all=False)

        now_channel = 128              # mlp_list_1[-1]
        self.SA_2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=now_channel,\
            mlp=[128, 128, 256], group_all=False)                                   
        
        now_channel = 256               # mlp_list_2[-1]
        self.SA_3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=now_channel,\
            mlp=[256, 512, 1024], group_all=True)
        
        self.classifier = nn.Sequential(
            get_perceptron_layer(1024, 512, dropout=0.4),
            get_perceptron_layer(512, 256, dropout=0.4),
            nn.Conv1d(256, class_num, 1)
        )
        # self.fc1 = get_perceptron_layer(1024, 512, dropout=0.4)
        # self.fc2 = get_perceptron_layer(512, 256, dropout=0.4)
        # self.fc3 = nn.Linear(256, class_num)


    def forward(self, point_cloud):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} point_cloud : 输入点云数据 [B, C, N], self.normal==True则C=6
        
        return {torch.tensor} output : 预测点云标签 [B, class_num]
        return {torch.tensor} points_sa_3 : 最后一个SA输出的特征 [B, 1024, 1]
        '''    
        B, _, _ = point_cloud.shape
        if self.normal:
            norm_feature = point_cloud[:, 3:, :]                        # [B, C-3, N] 点云特征
            point_cloud = point_cloud[:, :3, :]                         # [B, 3, N] 点云坐标
        else:
            norm_feature = None
        
        # xyz_sa_i  [B, C, npoint_i/1]
        xyz_sa_1, points_sa_1 = self.SA_1(point_cloud, norm_feature)    # [B, 128, 512]   即[B, mlp_list_1[-1], npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_2(xyz_sa_1, points_sa_1)        # [B, 256, 128]   即[B, mlp_list_2[-1], npoint_2]
        xyz_sa_2, points_sa_3 = self.SA_3(xyz_sa_2, points_sa_2)        # [B, 1024, 1]

        output = points_sa_3.view(B, 1024).unsqueeze(2)                 # [B, 1024, 1]
        output = self.classifier(output)                                # [B, class_num, 1]

        output = F.log_softmax(output.squeeze(2), -1)                              # [B, class_num]

        return output, points_sa_3


class PointNetPlusPlusCls_MSG(nn.Module):

    def __init__(self, class_num, normal=True):
        '''
        Author: ZHP
        description: PointNet++ 分类网络,base Multi-Scale Grouping
        param {int} class_num : 类别数
        param {bool} normal : 输入点云数据是否经过normal
        '''    
        super().__init__()
        in_channel = 3 if normal else 0
        self.normal = normal
        now_channel = in_channel
        self.SA_1 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[16, 32, 128],\
            in_channel=now_channel, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])

        now_channel = 64 + 128 + 128        # sum(mlp_list_1[i][-1])
        self.SA_2 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.2, 0.4, 0.8], nsample_list=[32, 64, 128],\
            in_channel=now_channel, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        now_channel = 128 + 256 + 256       # sum(mlp_list_2[i][-1])
        self.SA_3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel= now_channel,\
            mlp=[256, 512, 1024], group_all=True)                           # 最后一个SA是group all

        self.classifier = nn.Sequential(
            get_perceptron_layer(1024, 512, dropout=0.4),
            get_perceptron_layer(512, 256, dropout=0.4),
            nn.Conv1d(256, class_num, 1)
        )

 
    def forward(self, point_cloud):
        '''
        Author: ZHP
        description: 每次经过一次set abstraction,点云基数降至相应的npoint,得到相应的npoint个点云的特征，最后经过分类器分类
        param {torch.tensor} point_cloud : 输入初始点云,[B, C, N] N为点云基数 self.normal==True 则 C=6
        
        return {torch.tnesor} output : 预测标签 [B, class_num]
        return {torch.tensor} point_sa_3 : 最后一个SA输出的点云特征 [B, 1024, 1]
        '''    
        if self.normal:
            norm_feature = point_cloud[:, 3:, :]                        # [B, C-3, N]
            point_cloud = point_cloud[:, :3, :]                         # [B, 3, N]
        else:
            norm_feature = None

        # xyz_sa_i [B, C, npoint_i/1]
        xyz_sa_1, points_sa_1 = self.SA_1(point_cloud, norm_feature)    # [B, 320, 512]  <->  [B, sum(mlp_list_1[i][-1]), npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_2(xyz_sa_1, points_sa_1)        # [B, 640, 128]  <->  [B, sum(mlp_list_2[i][-1]), npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_3(xyz_sa_2, points_sa_2)        # [B, 1024, 1]   <->  [B, mlp_list_3[-1], npoint_3(1)]

        points_feature = points_sa_3.view(-1, 1024).unsqueeze(2)        # [B, 1024, 1]
        output = self.classifier(points_feature)                   + 6
        return output, points_sa_3


class PointNetPlusPlusPartSeg_SSG(nn.Module):
    def __init__(self, part_num, normal=True):
        '''
        Author: ZHP
        description: PointNet++ part segmentaion with Single Scale Grouping
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        additional_channel = 6 if normal else 0
        self.normal = normal
        now_channel = additional_channel
        self.SA_1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=additional_channel,\
            mlp=[64, 64, 128], group_all=False)

        now_channel = 128                       # mlp_1[-1]
        self.SA_2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=now_channel,\
            mlp=[128, 128, 256], group_all=False)

        now_channel = 256                          # mlp_2[-1]
        self.SA_3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=now_channel,\
            mlp=[256, 512, 1024], group_all=True)

        now_channel = 1024 + 256                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + 128                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + additional_channel                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            get_perceptron_layer(128, 128, bn=True, activate=True, dropout=0.5),
            nn.Conv1d(128, part_num, 1)
        )

  
    def forward(self, point_cloud, cls_label):
        '''
        Author: ZHP
        description: 三层SA提取特征，分割上采样阶段通过 KNN插值 和 skip connection 拼接特征
        param {torch.tensor} point_cloud : 输入点云数据 [B, C, N] self.normal==True则C=6
        param {torch.tensor} cls_label : 点云object label [B, cls_num] cls_num = 16
        
        return {torch.tensor} pred:预测点云part label [B, N, part_num]
        return {torch.tensor} points_sa_3 : 编码器最后一个SA输出的特征 [B, 1024, 1] <->[B, mlp_3[-1], npoint_3(1)]
        '''    
        B, _, N = point_cloud.shape
        original = point_cloud
        if self.normal:
            point_cloud = point_cloud[:, :3, :]  # 提取点云坐标, [B, 3, N]

        # xyz_sa_i [B, 3, npoint_i][B, mlp_3[-1], npoint_3(1)]
        xyz_sa_1, points_sa_1 = self.SA_1(point_cloud, original)  # [B, 128, 512] <-> [B, mlp_1[-1], npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_2(xyz_sa_1, points_sa_1)    # [B, 256, 128] <-> [B, mlp_2[-1], npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_3(xyz_sa_2, points_sa_2)  # [B, 1024, 1] <-> [B, mlp_3[-1], npoint_3(1)]

        # upsample
        upsample_points_1 = self.FP_1(xyz_sa_2, xyz_sa_3, points_sa_2, points_sa_3)             # 第一层插值会与编码层SA_2的点云及特征concat  [B, 256, 128],点云基数恢复到npoint_2

        # 第二层插值会与编码层SA_1的点云及特征concat,这里当前点云基数恢复到SA_2，所以点云坐标用SA_2的点云坐标，特征用上一步插值得到的特征
        upsample_points_2 = self.FP_2(xyz_sa_1, xyz_sa_2, points_sa_1, upsample_points_1)       # [B, 128, 512] <-> [b, 128, npoint_1]

        # 以下实现必须与now_channel相对应
        # 1.第三层插值会从输入点云中做KNN插值，故输入原始点云坐标，高维点云特征，这里可以为None now_channel = 0 + 128(upsample_points_2)
        # upsample_points_3 = self.FP_3(point_cloud, xyz_sa_1, None, upsample_points_2)           # [B, 128, N]

        # 2.也有原始点云特征拼接object one-hot编码，坐标 now_channel= 16(one-hot) + 3(point_cloud) + 128(upsamle_points_2) + 6(original)
        label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)                                # [B, 16, N]
        co_feature = torch.cat([label_one_hot, point_cloud, original], dim=1)                   # [B, 16+3+C, N]
        upsample_points_3 = self.FP_3(point_cloud, xyz_sa_1, co_feature, upsample_points_2)     # [B, 128, N]

        pred = self.classifier(upsample_points_3)                                                   # [B, part_num, N]
        pred = F.log_softmax(pred, dim=1).transpose(2, 1)                                                      # [B, N, part_num]
        return pred, points_sa_3


class PointNetPlusPlusPartSeg_Msg(nn.Module):
    def __init__(self, part_num, normal=True):
        '''
        Author: ZHP
        description: PointNet++ part segmentaion with Multi Scale Grouping
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        self.normal = normal
        add_channel = 3 if normal else 0

        now_channel = add_channel + 3                       # dim(addition norm) + dim(coordinate) = dim(point_feature)
        self.SA_1 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.1, 0.2, 0.4], nsample_list=[32, 64, 128],\
            in_channel= now_channel, mlp_list=[[32, 32, 64], [64, 64, 128],  [64, 96, 128]])
        
        now_channel = 128 + 128 + 64                        # sum(mlp_list_1[i][-1])
        self.SA_2 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.4, 0.8], nsample_list=[64, 128],\
            in_channel= now_channel, mlp_list=[[128, 128, 256], [128, 196, 256]])

        now_channel = 256 + 256                       # sum(mlp_list_2[i][-1])
        self.SA_3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,\
            in_channel=now_channel, mlp=[256, 512, 1024], group_all=True)

        now_channel = 256 + 256 + 1024                      # sum(mlp_list_2[i][-1]) + mlp_list_3[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])

        now_channel = 256 + (64 + 128 + 128)                # fp_1_mlp[-1] + sum(mlp_list_1[i][-1])
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])

        now_channel = 128 + 16 + 6 + add_channel            # fp_2_mlp[-1] + count(cls) + dim(norm_feature) + dim(coordinate)
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128])

        self.classifier = nn.Sequential(
            get_perceptron_layer(128, 128, bn=True, activate=True, dropout=0.5),
            nn.Conv1d(128, part_num, kernel_size=1)
        )
 
    def forward(self, point_cloud, cls_label):
        '''
        Author: ZHP
        description: 三层SA提取特征，分割上采样阶段通过 KNN插值 和 skip connection 拼接特征
        param {torch.tensor} point_cloud : 输入点云数据，[B, C, N] self.normal==True 则 C=6
        param {torch.tensor} cls_label : 点云的object label, [B, cls_num]

        return {torch.tensor} pred : 预测点云part label [B, N, part_num]
        return {torch.tensor} points_sa_3 : 最后一层SA输出的特征 [B, 1024, 1] <-> [B, mlp_list_3[-1], npoint_3(1)]
        '''
        B, _, N = point_cloud.shape
        norm_feature = point_cloud
        if self.normal:
            point_cloud = point_cloud[:,:3, :]
        
        # xyz_sa_i [B, C, npoint_i]
        xyz_sa_1, points_sa_1 = self.SA_1(point_cloud, norm_feature)            # [B, 128, 512] <-> [B, sum(mlp_list_1[i][-1]), npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_2(xyz_sa_1, points_sa_1)                # [B, 256, 128] <-> [B, sum(mlp_list_2[i][-1]), npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_3(xyz_sa_2, points_sa_2)                # [B, 1024, 1] <-> [B, mlp_list_3[-1], npoint_3/1]

        upsample_points_1 = self.FP_1(xyz_sa=xyz_sa_2, xyz_now=xyz_sa_3, points_sa=points_sa_2, points_now=points_sa_3)             # [B, 256, 128]  <-> [B, fp_mlp_1[-1], npoint_2]
        upsample_points_2 = self.FP_2(xyz_sa=xyz_sa_1, xyz_now=xyz_sa_2, points_sa=points_sa_1, points_now=upsample_points_1)       # [B, 256, 512]  <-> [B, fp_mlp_2[-1], npoint_1]

        # 2.也有原始点云特征拼接object one-hot编码，坐标 now_channel= 16(one-hot) + 3(point_cloud) + 128(upsamle_points_2) + 6(original)
        label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)                                    # [B, 16, N]
        co_feature = torch.cat([label_one_hot, point_cloud, norm_feature], dim=1)                   # [B, 16+3+C, N]
        upsample_points_3 = self.FP_3(xyz_sa=point_cloud, xyz_now=xyz_sa_1, points_sa=co_feature, points_now=upsample_points_2)     # [B, 128, N]  <-> [B, fp_mlp_3[-1], N]

        pred = self.classifier(upsample_points_3)                                                   # [B, part_num, N]
        pred = F.log_softmax(pred, dim=1).transpose(1, 2)                                           # [B, N, part_num]
        return pred, points_sa_3


class PointNetPlusPlusSemanticSeg_SSG(nn.Module):
    def __init__(self, class_num, input_dim=9):
        '''
        Author: ZHP
        description: PointNet++ Semantic segmentaion
        param {int} class_num:类别数
        param {int} input_dim：输入点云数据的维度大小
        '''    
        super().__init__()
        self.input_dim = input_dim
        now_channel = input_dim
        self.SA_1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=32, in_channel=now_channel,\
            mlp=[32, 32, 64], group_all=False)

        now_channel = 64                   # mlp_1[-1]
        self.SA_2 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=now_channel,\
            mlp=[64, 64, 128], group_all=False)
        
        now_channel = 128                  # mlp_2[-1]
        self.SA_3 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=32, in_channel=now_channel,\
            mlp=[128, 128, 256], group_all=False)
        
        now_channel = 256               # mlp_3[-1]
        self.SA_4 = PointNetSetAbstraction(npoint=16, radius=0.8, nsample=32, in_channel=now_channel,\
            mlp=[256, 256, 512], group_all=False)

        now_channel = 512 + 256                 # sa_mlp_4[-1] + sa_mlp_3[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])

        now_channel = 256 + 128                 # fp_1_mlp[-1] + sa_mlp_2[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])

        now_channel = 256 + 64                  # fp_2_mlp[-1] + sa_mlp_1[-1]
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])

        now_channel = 128 + 0                   # fp_3_mlp[-1] + 0
        self.FP_4 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            get_perceptron_layer(in_channel=128, out_channel=128, bn=True, activate=True, dropout=0.5),
            nn.Conv1d(128, class_num, kernel_size=1)
        )

    def forward(self, point_cloud):
        ''' 
        Author: ZHP
        description: 
        param {torch.tensor} point_cloud ： 输入点云[B, C, N] C==9
        return {torch.tensor}
        '''
        original_feature = point_cloud
        coordinate = point_cloud[:, :3, :]  # [B, 3, N]

        # xyz_sa_i [B, 3, npoint_i]
        xyz_sa_1, points_sa_1 = self.SA_1(coordinate, original_feature)     # [B, 64, 1024] <-> [B, mlp_1[-1], npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_2(xyz_sa_1, points_sa_1)            # [B, 128, 256] <-> [B, mlp_2[-1], npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_3(xyz_sa_2, points_sa_2)            # [B, 256, 64] <-> [B, mlp_3[-1], npoint_3]
        xyz_sa_4, points_sa_4 = self.SA_4(xyz_sa_3, points_sa_3)            # [B, 512, 16] <-> [B, mlp_4[-1], npoint_4]

        upsample_points_1 = self.FP_1(xyz_sa=xyz_sa_3, xyz_now=xyz_sa_4,\
            points_sa=points_sa_3, points_now=points_sa_4)                  # [B, 256, 64]   <-> [B, fp_1_mlp[-1], npoint_3]
        upsample_points_2 = self.FP_2(xyz_sa=xyz_sa_2, xyz_now=xyz_sa_3,\
            points_sa=points_sa_2, points_now=upsample_points_1)            # [B, 256, 256]   <-> [B, fp_2_mlp[-1], npoint_2]
        upsample_points_3 = self.FP_3(xyz_sa=xyz_sa_1, xyz_now=xyz_sa_2,\
            points_sa=points_sa_1, points_now=upsample_points_2)            # [B, 128, 1024]   <-> [B, fp_3_mlp[-1], npoint_1]
        upsample_points_4 = self.FP_4(xyz_sa=coordinate, xyz_now=xyz_sa_1,\
            points_sa=None, points_now=upsample_points_3)                   # [B, 128, N]   <-> [B, fp_4_mlp[-1], N]
        
        pred = self.classifier(upsample_points_4)                           # [B, class_num, N]
        pred = F.log_softmax(pred, dim=1).transpose(1, 2)                   # [B, N, class_num]
        return pred, points_sa_4


class PointNetPlusPlusSemanticSeg_MSG(nn.Module):
    
    def __init__(self, class_num, in_dim=9):
        '''
        Author: ZHP
        description: PointNet++ Semantic segmentaion with Multi Scale Grouping
        param {int} class_num:类别数
        param {int} in_dim：输入点云数据的维度大小
        '''    
        super().__init__()
        now_channel = in_dim
        self.SA_1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.05, 0.1], nsample_list=[16, 32],\
            in_channel=now_channel, mlp_list=[[16, 16, 32], [32, 32, 64]])
        
        now_channel = 32 + 64                       # sum(mlp_list_1[i][-1])
        self.SA_2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.1, 0.2], nsample_list=[16, 32],\
            in_channel=now_channel, mlp_list=[[64, 64, 128], [64, 96, 128]])

        now_channel = 128 + 128                     # sum(mlp_list_2[i][-1])
        self.SA_3 = PointNetSetAbstractionMsg(npoint=64, radius_list=[0.2, 0.4], nsample_list=[16, 32],\
            in_channel=now_channel, mlp_list=[[128, 196, 256], [128, 196, 256]])

        now_channel = 256 + 256                     # sum(mlp_list_3[i][-1])
        self.SA_4 = PointNetSetAbstractionMsg(npoint=16, radius_list=[0.4, 0.8], nsample_list=[16, 32],\
            in_channel=now_channel, mlp_list=[[256, 256, 512], [256, 384, 512]])

        now_channel = 512 + 512 + 256 + 256         # sum(mlp_list_4[i][-1]) + sum(mlp_list_3[i][-1])    
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])

        now_channel = 128 + 128 + 256               # sum(mlp_list_2[i][-1]) + fp_1_mlp[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])

        now_channel = 32 + 64 + 256                 # sum(mlp_list_1[i][-1]) + fp_2_mlp[-1]
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])

        now_channel = 128                           # fp_3_mlp[-1] + 0
        self.FP_4 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            get_perceptron_layer(in_channel=128, out_channel=128, bn=True, activate=True, dropout=0.5),
            nn.Conv1d(128, class_num, kernel_size=1)
        )

    def forward(self, point_cloud):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} point_cloud : [B, C, N]   C = in_dim
        
        return {torch.tensor} pred : 预测的语义标签 [B, N, class_num]
        return {torch.tensor} 最后一个SA提取的特征 [B, 1024, 16]
        '''    
        norm_feature = point_cloud          # [B, C, N]
        coordinate = point_cloud[:, :3, :]                                       # [B, 3, N]
        
        # xyz_sa_i [B, C, npoint_i]
        xyz_sa_1, points_sa_1 = self.SA_1(coordinate, norm_feature)         # [B, 96, 1024] <-> [B, sum(mlp_1[i][-1]), npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_2(xyz_sa_1, points_sa_1)            # [B, 256, 256] <-> [B, sum(mlp_2[i][-1]), npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_3(xyz_sa_2, points_sa_2)            # [B, 512 ,64]  <-> [B, sum(mlp_3[i][-1]), npoint_3]
        xyz_sa_4, points_sa_4 = self.SA_4(xyz_sa_3, points_sa_3)            # [B, 1024, 16] <-> [B, sum(mlp_4[i][-1]), npoint_4]

        upsample_points_1 = self.FP_1(xyz_sa=xyz_sa_3, xyz_now=xyz_sa_4,\
            points_sa=points_sa_3, points_now=points_sa_4)                  # [B, 256, 64]   <-> [B, fp_1_mlp[-1], npoint_3]                               
        upsample_points_2 = self.FP_2(xyz_sa=xyz_sa_2, xyz_now=xyz_sa_3,\
            points_sa=points_sa_2, points_now=upsample_points_1)            # [B, 256, 256]  <-> [B, fp_2_mlp[-1], npoint_2]
        upsample_points_3 = self.FP_3(xyz_sa=xyz_sa_1, xyz_now=xyz_sa_2,\
            points_sa=points_sa_1, points_now=upsample_points_2)            # [B, 128, 1024] <-> [B, fp_3_mlp[-1], npoint_1]
        upsample_points_4 = self.FP_4(xyz_sa=coordinate, xyz_now=xyz_sa_1,\
            points_sa=None, points_now=upsample_points_3)                   # [B, 128, 2048] <-> [B, fp_4_mlp[-1], N]

        pred = self.classifier(upsample_points_4)                           # [B, class_num, N]
        pred = F.log_softmax(pred, dim=1).transpose(1, 2)                   # [B, N, class_num]
        return pred, points_sa_4


def get_loss(pred, target, weight=None):
    if weight is None:
        loss = F.nll_loss(pred, target)
    else:
        # semantic segmentation loss
        loss = F.nll_loss(pred, target, weight=weight)
    return loss


if __name__ == "__main__":
    pts = torch.rand((8, 6, 2048), dtype=torch.float)

    # model = PointNetPlusPlusPartSeg_SSG(50)
    # cls_label = torch.rand((8, 16), dtype=torch.float)
    # y1, y2 = model(pts, cls_label)
    # print(y1.shape, "  ", y2.shape)

    # model_cls_1 = PointNetPlusPlusCls_SSG(16)
    # cls_1, cls_2 = model_cls_1(pts)
    # print(cls_1.shape, '||', cls_2.shape)

    # model_msg_cls = PointNetPlusPlusCls_MSG(16)
    # msg_cls_1, msg_2 = model_msg_cls(pts)
    # print(msg_cls_1.shape, '||', msg_2.shape)

    # model = PointNetPlusPlusPartSeg_Msg(50)
    model = PointNetPlusPlusPartSeg_SSG(50)
    cls_label = torch.rand((8, 16), dtype=torch.float)
    y1, y2 = model(pts, cls_label)
    print(y1.shape, "  ", y2.shape)
    # print(model)
    # summary(model.cuda(), [(6, 2048), (16,)], device="cuda")
    with open("ppp.txt", 'a') as f:
        print(model, file=f)
    # model = PointNetPlusPlusSemanticSeg_MSG(50)
    # model = PointNetPlusPlusSemanticSeg_SSG(50)
    # model(pts)
