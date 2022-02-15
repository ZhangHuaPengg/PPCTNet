'''
Author       : ZHP
Date         : 2022-01-08 10:38:13
LastEditors  : ZHP
LastEditTime : 2022-02-15 13:52:31
FilePath     : /models/PointFormer/point_former.py
Description  : only transformer for point cloud
Copyright 2022 ZHP, All Rights Reserved. 
2022-01-08 10:38:13
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../..")

from models.PointFormer.trans_block import *
from models.PointFormer.pointformer_utils import *
from models.pointnet.pointNet2_Ops import *

class Tokenizer(nn.Module):
    def __init__(self, input_dim, embed_dim) -> None:
        super().__init__()

        # two linear
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim*4, embed_dim)
        )

        # single linear
        # self.layer = nn.Linear(input_dim, embed_dim)
    
    def forward(self, x):
        return self.layer(x)


class ProgressivePointCloudTransformer_Pre(nn.Module):
    def __init__(self, enc_num, part_num, share=False, normal=True):
        '''
        Author: ZHP
        description: 采样分组通过transformer(pre)-maxpool-transformer,PPCT
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        additional_channel = 6 if normal else 0
        self.normal = normal
        now_channel = additional_channel
        self.PPCT_1 = JointTransPointAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=additional_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=128, dim_hid=256, heads_count=8, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=128, dim_hid=256, heads_count=8, in_size=3, share=share),\
                    pre_block=K_MLP_Layer(in_size=4, in_channel=additional_channel+3, channel_list=[64, 128], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 128                       # mlp_1[-1]
        self.PPCT_2 = JointTransPointAbstraction(npoint=128, radius=0.4, nsample=32, in_channel=now_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=256, dim_hid=512, heads_count=8, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=256, dim_hid=512, heads_count=8, in_size=3, share=share),\
                    pre_block=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=[128, 256], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 256                          # mlp_2[-1]
        self.PPCT_3 = JointTransPointAbstraction(npoint=None, radius=None, nsample=None, in_channel=now_channel, group_all=True,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=1024, dim_hid=1024, heads_count=8, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=1024, dim_hid=1024, heads_count=8, in_size=3, share=share),\
                    pre_block=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=[512, 1024], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 1024 + 256                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + 128                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + additional_channel                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[True], activate_list=[True], dropout_list=[0.5]),
            nn.Conv1d(128, part_num, 1)
        )

  
    def forward(self, point_cloud, cls_label):
        '''
        Author: ZHP
        description: 三层SA提取特征，分割上采样阶段通过 KNN插值 和 skip connection 拼接特征
        param {torch.tensor} point_cloud : 输入点云数据 [B, C, N] self.normal==True则C=6
        param {torch.tensor} cls_label : 点云object label [B, cls_num] cls_num = 16
        
        return {torch.tensor} pred:预测点云part JointTransPointAbstraction_2label [B, N, part_num]
        return {torch.tensor} points_sa_3 : 编码器最后一个SA输出的特征 [B, 1024, 1] <->[B, mlp_3[-1], npoint_3(1)]
        '''    
        B, _, N = point_cloud.shape
        original = point_cloud
        if self.normal:
            point_cloud = point_cloud[:, :3, :]  # 提取点云坐标, [B, 3, N]

        # xyz_sa_i [B, 3, npoint_i][B, mlp_3[-1], npoint_3(1)]
        xyz_sa_1, points_sa_1 = self.PPCT_1(point_cloud, original)  # [B, 128, 512] <-> [B, mlp_1[-1], npoint_1]
        xyz_sa_2, points_sa_2 = self.PPCT_2(xyz_sa_1, points_sa_1)    # [B, 256, 128] <-> [B, mlp_2[-1], npoint_2]
        xyz_sa_3, points_sa_3 = self.PPCT_3(xyz_sa_2, points_sa_2)  # [B, 1024, 1] <-> [B, mlp_3[-1], npoint_3(1)]

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

        pred = self.classifier(upsample_points_3)                                                   # [B, part_num, N, 1]
        pred = F.log_softmax(pred, dim=1).transpose(2, 1)                                                      # [B, N, part_num]
        return pred, points_sa_3


class ProgressivePointCloudTransformer_Back(nn.Module):
    def __init__(self, enc_num, part_num, share=False, normal=True):
        '''
        Author: ZHP
        description: 采样分组通过transformer(back)-maxpool-transformer
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        additional_channel = 6 if normal else 0
        self.normal = normal
        now_channel = additional_channel
        self.PPCT_1 = JointTransPointAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=additional_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=additional_channel+3, dim_hid=32, heads_count=1, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=128, dim_hid=256, heads_count=8, in_size=3, share=share),\
                    back_block=K_MLP_Layer(in_size=4, in_channel=additional_channel+3, channel_list=[64, 128], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 128                       # mlp_1[-1]
        self.PPCT_2 = JointTransPointAbstraction(npoint=128, radius=0.4, nsample=32, in_channel=now_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=256, heads_count=1, in_size=4),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=256, dim_hid=512, heads_count=8, in_size=3),\
                    back_block=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=[128, 256], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 256                          # mlp_2[-1]
        self.PPCT_3 = JointTransPointAbstraction(npoint=None, radius=None, nsample=None, in_channel=now_channel, group_all=True,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=512, heads_count=1, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=1024, dim_hid=1024, heads_count=8, in_size=3, share=share),\
                    back_block=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=[512, 1024], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 1024 + 256                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + 128                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + additional_channel                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[True], activate_list=[True], dropout_list=[0.5]),
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
        xyz_sa_1, points_sa_1 = self.PPCT_1(point_cloud, original)  # [B, 128, 512] <-> [B, mlp_1[-1], npoint_1]
        xyz_sa_2, points_sa_2 = self.PPCT_2(xyz_sa_1, points_sa_1)    # [B, 256, 128] <-> [B, mlp_2[-1], npoint_2]
        xyz_sa_3, points_sa_3 = self.PPCT_3(xyz_sa_2, points_sa_2)  # [B, 1024, 1] <-> [B, mlp_3[-1], npoint_3(1)]

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

        pred = self.classifier(upsample_points_3)                                                   # [B, part_num, N, 1]
        pred = F.log_softmax(pred, dim=1).transpose(2, 1)                                                      # [B, N, part_num]
        return pred, points_sa_3


class ProgressivePointCloudTransformer_None(nn.Module):
    def __init__(self, enc_num, part_num, share=False, normal=True):
        '''
        Author: ZHP
        description: 采样分组通过transformer-maxpool-transformer,PPCT,no embedding,only transformer
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        additional_channel = 6 if normal else 0
        self.normal = normal
        now_channel = additional_channel + 3
        self.PPCT_1 = JointTransPointAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=additional_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel, dim_hid=128, heads_count=1, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel, dim_hid=128, heads_count=1, in_size=3, share=share))

        now_channel = 9                       # mlp_1[-1]
        self.PPCT_2 = JointTransPointAbstraction(npoint=128, radius=0.4, nsample=32, in_channel=now_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=256, heads_count=1, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=256, heads_count=1, in_size=3, share=share))

        now_channel = 12                          # mlp_2[-1]
        self.PPCT_3 = JointTransPointAbstraction(npoint=None, radius=None, nsample=None, in_channel=now_channel, group_all=True,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=512, heads_count=1, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=512, heads_count=1, in_size=3, share=share))

        now_channel = 15 + 12                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + 9                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + additional_channel                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[True], activate_list=[True], dropout_list=[0.5]),
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
        xyz_sa_1, points_sa_1 = self.PPCT_1(point_cloud, original)  # [B, 128, 512] <-> [B, mlp_1[-1], npoint_1]
        xyz_sa_2, points_sa_2 = self.PPCT_2(xyz_sa_1, points_sa_1)    # [B, 256, 128] <-> [B, mlp_2[-1], npoint_2]
        xyz_sa_3, points_sa_3 = self.PPCT_3(xyz_sa_2, points_sa_2)  # [B, 1024, 1] <-> [B, mlp_3[-1], npoint_3(1)]

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

        pred = self.classifier(upsample_points_3)                                                   # [B, part_num, N, 1]
        pred = F.log_softmax(pred, dim=1).transpose(2, 1)                                                      # [B, N, part_num]
        return pred, points_sa_3


class ProgressivePointCloudTransformer_Embed(nn.Module):
    def __init__(self, enc_num, part_num, share=False, normal=True):
        '''
        Author: ZHP
        description: 先embedding, 再采样分组通过transformer-maxpool-transformer,PPCTonly transformer
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        additional_channel = 6 if normal else 0
        self.normal = normal
        self.embedding = K_MLP_Layer(in_size=3, in_channel=additional_channel, channel_list=[64, 128, 256], bn_list=[True, True, True],
                    activate_list=[True, True, True], dropout_list=[False, False, False])
        
        now_channel = 256
        self.PPCT_1 = JointTransPointAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=now_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=256, heads_count=1, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=256, heads_count=1, in_size=3, share=share))

        now_channel = 256 + 3                      # mlp_1[-1]
        self.PPCT_2 = JointTransPointAbstraction(npoint=128, radius=0.4, nsample=32, in_channel=now_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=512, heads_count=1, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=512, heads_count=1, in_size=3, share=share))

        now_channel = 256 + 3 + 3                          # mlp_2[-1]
        self.PPCT_3 = JointTransPointAbstraction(npoint=None, radius=None, nsample=None, in_channel=now_channel, group_all=True,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=1024, heads_count=1, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel+3, dim_hid=1024, heads_count=1, in_size=3, share=share))

        now_channel = 256 + 3 + 3 +3 + (256 + 3 + 3)                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 256])  # 1280 = 1024 + 256

        now_channel = 256 + (256 + 3)                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[256, 128])   # 384 = 256 + 128

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        now_channel = 128 + 16 + 3 + additional_channel                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = PointNetFeaturePropagation(in_channel=now_channel, mlp=[128, 128, 128])

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=128, channel_list=[128], bn_list=[True], activate_list=[True], dropout_list=[0.5]),
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

        embed_feature = self.embedding(original) 
        
        # xyz_sa_i [B, 3, npoint_i][B, mlp_3[-1], npoint_3(1)]
        xyz_sa_1, points_sa_1 = self.PPCT_1(point_cloud, embed_feature)  # [B, 128, 512] <-> [B, mlp_1[-1], npoint_1]
        xyz_sa_2, points_sa_2 = self.PPCT_2(xyz_sa_1, points_sa_1)    # [B, 256, 128] <-> [B, mlp_2[-1], npoint_2]
        xyz_sa_3, points_sa_3 = self.PPCT_3(xyz_sa_2, points_sa_2)  # [B, 1024, 1] <-> [B, mlp_3[-1], npoint_3(1)]

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

        pred = self.classifier(upsample_points_3)                                                   # [B, part_num, N, 1]
        pred = F.log_softmax(pred, dim=1).transpose(2, 1)                                                      # [B, N, part_num]
        return pred, points_sa_3


class ProgressivePointCloudTransformer_Pre_Unsample(nn.Module):
    def __init__(self, enc_num, part_num, share=False, normal=True):
        '''
        Author: ZHP
        description: 采样分组通过transformer-maxpool-transformer,PPCT+trans_unsample
        param {int} part_num: part类别数
        param {bool} normal：输入点云数据是否经过normal,经过normal维度应为6，否则为3
        '''    
        super().__init__()
        additional_channel = 6 if normal else 0
        self.normal = normal
        now_channel = additional_channel
        self.PPCT_1 = JointTransPointAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=additional_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=128, dim_hid=256, heads_count=8, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=128, dim_hid=256, heads_count=8, in_size=3, share=share),\
                    pre_block=K_MLP_Layer(in_size=4, in_channel=additional_channel+3, channel_list=[64, 128], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 128                       # mlp_1[-1]
        self.PPCT_2 = JointTransPointAbstraction(npoint=128, radius=0.4, nsample=32, in_channel=now_channel, group_all=False,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=256, dim_hid=512, heads_count=8, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=256, dim_hid=512, heads_count=8, in_size=3, share=share),\
                    pre_block=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=[128, 256], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 256                          # mlp_2[-1]
        self.PPCT_3 = JointTransPointAbstraction(npoint=None, radius=None, nsample=None, in_channel=now_channel, group_all=True,\
            transformer_1=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=1024, dim_hid=1024, heads_count=8, in_size=4, share=share),\
                transformer_2=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=1024, dim_hid=1024, heads_count=8, in_size=3, share=share),\
                    pre_block=K_MLP_Layer(in_size=4, in_channel=now_channel+3, channel_list=[512, 1024], bn_list=[True, True],
                    activate_list=[True, True], dropout_list=[False, False]))

        now_channel = 1024 + 256                        # mlp_3[-1] + mlp_2[-1]
        self.FP_1 = TransFeaturePropagation(transformer=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel, dim_hid=now_channel, heads_count=8, in_size=3, share=share))

        now_channel = 1024 + 256 + 128                        # fp_1_mlp[-1] + mlp_1[-1]
        self.FP_2 = TransFeaturePropagation(transformer=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel, dim_hid=now_channel, heads_count=8, in_size=3, share=share))

        # fp_2_mlp[-1] + object_cls + coordinate + norm_feature
        # now_channel = 1024 + 256 + 128 + 16 + 3 + additional_channel                           # 16是点云共有16个object，每个样本只有一个object label
        now_channel = 1024 + 256 + 128 + 16                           # 16是点云共有16个object，每个样本只有一个object label
        self.FP_3 = TransFeaturePropagation(transformer=Learn_Pos_TransEncoder(enc_num=enc_num, dim_model=now_channel, dim_hid=now_channel, heads_count=8, in_size=3, share=share))

        self.classifier = nn.Sequential(
            K_MLP_Layer(in_size=3, in_channel=now_channel, channel_list=[128], bn_list=[True], activate_list=[True], dropout_list=[0.5]),
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
        xyz_sa_1, points_sa_1 = self.PPCT_1(point_cloud, original)  # [B, 128, 512] <-> [B, mlp_1[-1], npoint_1]
        xyz_sa_2, points_sa_2 = self.PPCT_2(xyz_sa_1, points_sa_1)    # [B, 256, 128] <-> [B, mlp_2[-1], npoint_2]
        xyz_sa_3, points_sa_3 = self.PPCT_3(xyz_sa_2, points_sa_2)  # [B, 1024, 1] <-> [B, mlp_3[-1], npoint_3(1)]

        # upsample
        upsample_points_1 = self.FP_1(xyz_sa_2, xyz_sa_3, points_sa_2, points_sa_3)             # 第一层插值会与编码层SA_2的点云及特征concat  [B, 256, 128],点云基数恢复到npoint_2

        # 第二层插值会与编码层SA_1的点云及特征concat,这里当前点云基数恢复到SA_2，所以点云坐标用SA_2的点云坐标，特征用上一步插值得到的特征
        upsample_points_2 = self.FP_2(xyz_sa_1, xyz_sa_2, points_sa_1, upsample_points_1)       # [B, 128, 512] <-> [b, 128, npoint_1]

        # 以下实现必须与now_channel相对应
        # 1.第三层插值会从输入点云中做KNN插值，故输入原始点云坐标，高维点云特征，这里可以为None now_channel = 0 + 128(upsample_points_2)
        # upsample_points_3 = self.FP_3(point_cloud, xyz_sa_1, None, upsample_points_2)           # [B, 128, N]

        # 2.也有原始点云特征拼接object one-hot编码，坐标 now_channel= 16(one-hot) + 3(point_cloud) + 128(upsamle_points_2) + 6(original)
        label_one_hot = cls_label.view(B, 16, 1).repeat(1, 1, N)                                # [B, 16, N]
        # co_feature = torch.cat([label_one_hot, point_cloud, original], dim=1)                   # [B, 16+3+C, N]
        co_feature = label_one_hot                   # [B, 16+3+C, N]
        upsample_points_3 = self.FP_3(point_cloud, xyz_sa_1, co_feature, upsample_points_2)     # [B, 128, N]

        pred = self.classifier(upsample_points_3)                                                   # [B, part_num, N, 1]
        pred = F.log_softmax(pred, dim=1).transpose(2, 1)                                                      # [B, N, part_num]
        return pred, points_sa_3


if __name__ == "__main__":
    model = ProgressivePointCloudTransformer_Pre_Unsample(2, 50)
    pts = torch.rand(2, 6, 2048, dtype=torch.float)
    cls_label = torch.rand(2, 16, dtype=torch.float)
    pred, _ = model(pts, cls_label)
    print(pred.shape)
