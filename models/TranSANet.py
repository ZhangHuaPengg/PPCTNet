'''
Author       : ZHP
Date         : 2022-01-04 13:54:09
LastEditors  : ZHP
LastEditTime : 2022-01-14 20:26:46
FilePath     : /models/TranSANet.py
Description  : 
Copyright 2022 ZHP, All Rights Reserved. 
2022-01-04 13:54:09
'''
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.summary import summary
from models.PointFormer.pointformer_utils import *

class TransPointPartSeg_SSG(nn.Module):
    def __init__(self, enc_num, dec_num, target_list=[128, 256, 1024], heads_list=[1, 8, 8], part_num=50, normal=True):
        '''
        Author: ZHP
        description: PointNet++ part segmentaion with Single Scale Grouping
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        self.normal = normal
        now_channel = 6        
        self.SA_Trans_1 = TransformerSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=now_channel, \
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=64, target_num=target_list[0], heads_count=heads_list[0]), group_all=False)

        now_channel = target_list[0]
        self.SA_Trans_2 = TransformerSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=now_channel,\
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=256, target_num=target_list[1], heads_count=heads_list[1]), group_all=False)
        
        now_channel = target_list[1]
        self.SA_Trans_3 = TransformerSetAbstraction(npoint=None, radius=0.2, nsample=None, in_channel=now_channel,\
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=512, target_num=target_list[2], heads_count=heads_list[2]), group_all=True)
        
        now_channel = target_list[2] + target_list[1]                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + 128                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + 6                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[False],\
                activate_list=[True], dropout_list=[0.5]),
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


        xyz_sa_1, points_sa_1 = self.SA_Trans_1(point_cloud, original)      # [B, 3, 512] / [B, 128, 512]  <-> [B, target_num_1, npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_Trans_2(xyz_sa_1, points_sa_1)      # [B, 3, 128] / [B, 256, 128]  <-> [B, target_num_2, npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_Trans_3(xyz_sa_2, points_sa_2)      # [B, 3, 1] / [B, 1024, 1]  <-> [B, target_num_2, npoint_2]

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


class Share_TransPointPartSeg_SSG(nn.Module):
    def __init__(self, enc_num, dec_num, target_list=[128, 256, 1024], heads_list=[1, 8, 8], part_num=50, normal=True):
        '''
        Author: ZHP
        description: PointNet++ part segmentaion with Single Scale Grouping
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        self.normal = normal
        now_channel = 6        
        self.SA_Trans_1 = TransformerSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=now_channel, \
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=64,\
            target_num=target_list[0], heads_count=heads_list[0], share=True), group_all=False)

        now_channel = target_list[0]
        self.SA_Trans_2 = TransformerSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=now_channel,\
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=256, \
            target_num=target_list[1], heads_count=heads_list[1], share=True), group_all=False)
        
        now_channel = target_list[1]
        self.SA_Trans_3 = TransformerSetAbstraction(npoint=None, radius=0.2, nsample=None, in_channel=now_channel,\
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=512, \
            target_num=target_list[2], heads_count=heads_list[2], share=True), group_all=True)
        
        now_channel = target_list[2] + target_list[1]                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + 128                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + 6                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[False],\
                activate_list=[True], dropout_list=[0.5]),
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


        xyz_sa_1, points_sa_1 = self.SA_Trans_1(point_cloud, original)      # [B, 3, 512] / [B, 128, 512]  <-> [B, target_num_1, npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_Trans_2(xyz_sa_1, points_sa_1)      # [B, 3, 128] / [B, 256, 128]  <-> [B, target_num_2, npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_Trans_3(xyz_sa_2, points_sa_2)      # [B, 3, 1] / [B, 1024, 1]  <-> [B, target_num_2, npoint_2]

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


class TransPointPartSeg_SSG_global(nn.Module):
    def __init__(self, enc_num, dec_num, target_list=[128, 256, 1024], heads_list=[1, 8, 8], part_num=50, normal=True):
        '''
        Author: ZHP
        description: PointNet++ part segmentaion with Single Scale Grouping
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        self.normal = normal
        now_channel = 6        
        self.SA_Trans_1 = TransformerSetAbstraction_global(npoint=256, radius=0.2, nsample=32, in_channel=now_channel, \
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=64,\
            target_num=target_list[0], heads_count=heads_list[0]), group_all=False)

        now_channel = target_list[0]
        self.SA_Trans_2 = TransformerSetAbstraction_global(npoint=128, radius=0.2, nsample=32, in_channel=now_channel,\
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=256,\
            target_num=target_list[1], heads_count=heads_list[1]), group_all=False)
        
        now_channel = target_list[1]
        self.SA_Trans_3 = TransformerSetAbstraction_global(npoint=None, radius=0.2, nsample=None, in_channel=now_channel,\
            transformer=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=512, \
            target_num=target_list[2], heads_count=heads_list[2]), group_all=True)
        
        now_channel = target_list[2] + target_list[1]                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + 128                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + 6                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[False],\
                activate_list=[True], dropout_list=[0.5]),
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


        xyz_sa_1, points_sa_1 = self.SA_Trans_1(point_cloud, original)      # [B, 3, 512] / [B, 128, 512]  <-> [B, target_num_1, npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_Trans_2(xyz_sa_1, points_sa_1)      # [B, 3, 128] / [B, 256, 128]  <-> [B, target_num_2, npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_Trans_3(xyz_sa_2, points_sa_2)      # [B, 3, 1] / [B, 1024, 1]  <-> [B, target_num_2, npoint_2]

        # upsample
        upsample_points_1 = self.FP_1(xyz_sa_2, xyz_sa_3, points_sa_2, points_sa_3)             # 第一层插值会与编码层SA_2的点云及特征concat  [B, 256, 128],点云基数恢复到npoint_2

        # 第二层插值会与编码层SA_1的点云及特征concat,这里当前点云基数恢复到SA_2，所以点云坐标用SA_2的点云坐标，特征用上一步插值得到的特征
        upsample_points_2 = self.FP_2(xyz_sa_1, xyz_sa_2, points_sa_1, upsample_points_1)       # [B, 128, 512] <-> [b, 128, npoint_1]

        # 以下实现必须与now_channel相对应"
        # 1.第三层插值会从输入点云中做KNN插值，故输入原始点云坐标，高维点云特征，这里可以为None now_channel = 0 + 128(upsample_points_2)
        # upsample_points_3 = self.FP_3(point_cloud, xyz_sa_1, None, upsample_points_2)           # [B, 128, N]

        # 2.也有原始点云特征拼接object one-hot编码，坐标 now_channel= 16(one-hot) + 3(point_cloud) + 128(upsamle_points_2) + 6(original)
        label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)                                # [B, 16, N]
        co_feature = torch.cat([label_one_hot, point_cloud, original], dim=1)                   # [B, 16+3+C, N]
        upsample_points_3 = self.FP_3(point_cloud, xyz_sa_1, co_feature, upsample_points_2)     # [B, 128, N]

        pred = self.classifier(upsample_points_3)                                                   # [B, part_num, N]
        pred = F.log_softmax(pred, dim=1).transpose(2, 1)                                                      # [B, N, part_num]
        return pred, points_sa_3

class Two_Dim_TransPointPartSeg_SSG(nn.Module):
    def __init__(self, enc_num, dec_num, target_list=[(32, 64), (128, 256), (256, 512)], heads_list=[1, 8, 8], part_num=50, normal=True):
        '''
        Author: ZHP
        description: 先在nsample维度做trans,再在npoint维度做
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        self.normal = normal
        now_channel = 6        
        self.SA_Trans_1 = Split_TransformerSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=now_channel, \
            transformer_1=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=16, target_num=target_list[0][0], heads_count=heads_list[0]),\
            transformer_2=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=target_list[0][0], dim_hid=64, target_num=target_list[0][1], heads_count=heads_list[1]), group_all=False)

        now_channel = target_list[0][1]
        self.SA_Trans_2 = Split_TransformerSetAbstraction(npoint=128, radius=0.2, nsample=32, in_channel=now_channel,\
            transformer_1=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=128, target_num=target_list[1][0], heads_count=heads_list[1]), \
            transformer_2=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=target_list[1][0], dim_hid=256, target_num=target_list[1][1], heads_count=heads_list[1]),group_all=False)
        
        now_channel = target_list[1][1]
        self.SA_Trans_3 = Split_TransformerSetAbstraction(npoint=None, radius=0.2, nsample=None, in_channel=now_channel,\
            transformer_1=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=now_channel, dim_hid=256, target_num=target_list[2][0], heads_count=heads_list[2]), \
            transformer_2=Mlp_Learn_Trans(enc_num=enc_num, dec_num=dec_num, dim_model=target_list[2][0], dim_hid=512, target_num=target_list[2][1], heads_count=heads_list[2]),group_all=True)
        
        now_channel = target_list[2][1] + target_list[1][1]                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + target_list[0][1]                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + 6                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[False],\
                activate_list=[True], dropout_list=[0.5]),
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


        xyz_sa_1, points_sa_1 = self.SA_Trans_1(point_cloud, original)      # [B, 3, 512] / [B, 128, 512]  <-> [B, target_num_1, npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_Trans_2(xyz_sa_1, points_sa_1)      # [B, 3, 128] / [B, 256, 128]  <-> [B, target_num_2, npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_Trans_3(xyz_sa_2, points_sa_2)      # [B, 3, 1] / [B, 1024, 1]  <-> [B, target_num_2, npoint_2]

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


class Local_TransPointPartSeg_SSG(nn.Module):
    def __init__(self, enc_num, channel_list=[[64, 64, 128], [128, 128, 256], [256, 512, 1024]], heads_list=[1, 8, 8], part_num=50, normal=True):
        '''
        Author: ZHP
        description: PointNet++ part segmentaion with Single Scale Grouping
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        self.normal = normal
        now_channel = 6        
        self.SA_Trans_1 = TransEncoderSA(npoint=512, radius=0.2, nsample=32, in_channel=now_channel, \
            linear_project=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=channel_list[0], bn_list=[True, True], \
                activate_list=[True, True], dropout_list=[False, False]),\
            transformer=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=128, dim_hid=256, heads_count=heads_list[0]), group_all=False)

        now_channel = 128
        self.SA_Trans_2 = TransEncoderSA(npoint=128, radius=0.2, nsample=32, in_channel=now_channel,\
            linear_project=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=channel_list[0], bn_list=[True, True], \
                activate_list=[True, True], dropout_list=[False, False]),\
            transformer=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=256, dim_hid=256, heads_count=heads_list[1]), group_all=False)
        
        now_channel = 256
        self.SA_Trans_3 = TransEncoderSA(npoint=None, radius=0.2, nsample=None, in_channel=now_channel,\
            linear_project=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=channel_list[0], bn_list=[True, True], \
                activate_list=[True, True], dropout_list=[False, False]),\
            transformer=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=1024, dim_hid=1024, heads_count=heads_list[2]), group_all=True)
        
        now_channel = 1024 + 256                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + 128                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + 6                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[False],\
                activate_list=[True], dropout_list=[0.5]),
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


        xyz_sa_1, points_sa_1 = self.SA_Trans_1(point_cloud, original)      # [B, 3, 512] / [B, 128, 512]  <-> [B, target_num_1, npoint_1]
        xyz_sa_2, points_sa_2 = self.SA_Trans_2(xyz_sa_1, points_sa_1)      # [B, 3, 128] / [B, 256, 128]  <-> [B, target_num_2, npoint_2]
        xyz_sa_3, points_sa_3 = self.SA_Trans_3(xyz_sa_2, points_sa_2)      # [B, 3, 1] / [B, 1024, 1]  <-> [B, target_num_2, npoint_2]

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

if __name__ == "__main__":
    model = TransPointPartSeg_SSG(50, True)
    pts = torch.rand((8, 6, 2048), dtype=torch.float)
    cls_label = torch.rand((8, 16), dtype=torch.float)
    y1, y2 = model(pts, cls_label)
    print(y1.shape, "  ", y2.shape)
    summary(model.cuda(), [(6, 2048), (16,)], device="cuda")
    with open("model.txt", 'a') as f:
        print(model, file=f)
    # print(model)