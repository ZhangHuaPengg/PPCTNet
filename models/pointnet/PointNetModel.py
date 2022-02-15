'''
Author       : ZHP
Date         : 2021-12-07 16:29:47
LastEditors  : ZHP
LastEditTime : 2022-01-14 20:11:55
FilePath     : /models/pointnet/PointNetModel.py
Description  : 
Copyright 2021 ZHP, All Rights Reserved. 
2021-12-07 16:29:47
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointNetUtils import PointNetEncoder, TNetkd, feature_transform_reguliarzer
from torchsummary import summary
class PointNetCls(nn.Module):
    def __init__(self, classes=40, normal_channel=True):
        super().__init__()
        in_channel = 6 if normal_channel else 3    # 只有坐标为3，有norm后的作为特征则为6

        self.encoder = PointNetEncoder(global_feature=True, feature_transform=True, in_channel=in_channel)
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.4),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        x, trans_matrix, trans_matrix_2 = self.encoder(x)   # x [B, 1024]
        x = self.fc2(self.fc1(x))                           # [B, 256]
        x = self.fc3(x)                                     # [B, classes]

        # 这里用了logSoftmax后，loss就应该用NllLoss
        x = F.log_softmax(x, dim=-1)                        # [B, classed]
        return x, trans_matrix_2

class PointNetPartSeg(nn.Module):
    def __init__(self, part_count=50, normal_channel=True):
        super().__init__()
        in_channel = 6 if normal_channel else 3                 # 只有坐标为3，有norm后的作为特征则为6
        self.part_count = part_count
        self.tnet1 = TNetkd(in_channel, 3)
        self.conv1 = TNetkd.get_single_conv(in_channel, 64, 1)
        self.conv2 = TNetkd.get_single_conv(64, 128, 1)
        self.conv3 = TNetkd.get_single_conv(128, 128, 1)
        self.conv4 = TNetkd.get_single_conv(128, 512, 1)
        self.conv5 = TNetkd.get_single_conv(512, 2048, 1, activate=False)

        self.tnet2 = TNetkd(in_channel=128, output_k=128)

        self.classifier = nn.Sequential(
            TNetkd.get_single_conv(3024, 256, 1),
            TNetkd.get_single_conv(256, 256, 1),
            TNetkd.get_single_conv(256, 128, 1),
            nn.Conv1d(128, part_count, 1)
        )
        self.max_pool = torch.max
    

    def forward(self, x, label):
        '''
        Author: ZHP
        description: PointNet part分割网络，详细结构在原文补充材料里，concat了多个局部feature
        param {torch.tensor} x : 输入点云 [B, C, N]
        param {torch.tensor} label ：one-hot编码，[B, 16], shapenet part有16个object，50个part，这里是点云的category label(object label)
        
        return {torch.tensor} output: 输出点云的part类别概率(经过LogSoftmax后), [B, N, 50]
        return {torch.tensor} trans_matrix_2 : 第二个T-Net学习到的transform matrix
        '''    
        B, C, N = x.shape
        trans_matrix = self.tnet1(x)                                                    # [B, C, C]
        x = x.transpose(2, 1)                                                           # [B, N, C]
        if C > 3:
            feature = x[:, :, 3:]                                                       # feature [B, N, C-3]
            x = x[:, :, :3]                                                             # coordinates [B, N, 3]
        x = torch.matmul(x, trans_matrix)                                               # 与学习到的矩阵相乘从而对齐, [B, N, 3]
        if C > 3:
            x = torch.cat([x, feature], dim=2)                                          # 再拼接feature, [B, N, C]
        
        x = x.transpose(2, 1)                                                           # [B, C, N]
        local_feature_1 = self.conv1(x)                                                 # [B, 64, N]
        local_feature_2 = self.conv2(local_feature_1)                                   # [B, 128, N]
        local_feature_3 = self.conv3(local_feature_2)                                   # [B, 128, N]

        trans_matrix_2 = self.tnet2(local_feature_3)                                    # [B, 128, 128]
        x = local_feature_3.transpose(2, 1)
        x = torch.matmul(x, trans_matrix_2)                                             # [B, N, 128]
        local_feature_4 = x.transpose(2, 1)                                             # [B, 128, N]
        local_feature_5 = self.conv4(local_feature_4)                                   # [B, 512, N]
        x = self.conv5(local_feature_5)                                                 # [B, 2048, N]
        x = self.max_pool(x, dim=2)[0]                                                  # [B, 2048]

        global_feature = x.unsqueeze(2).repeat(1, 1, N)                                 # [B, 2048, N]
        one_hot_label = label.unsqueeze(2).repeat(1, 1, N)
        concat = torch.cat([local_feature_1, local_feature_2, local_feature_3,\
            local_feature_4, local_feature_5, global_feature, one_hot_label], dim=1)    # [B, 3024, N]

        output = self.classifier(concat)                                                # [B, 50, N]
        output = output.transpose(2, 1).contiguous()                                    # [B, N, 50]
        output = F.log_softmax(output, dim=-1)                                          # [B, N, 50]
        return output, trans_matrix_2
        
class PointNetSemanticSeg(nn.Module):
    def __init__(self, class_num, in_channel=9):
        super().__init__()
        self.class_num = class_num
        self.encoder = PointNetEncoder(global_feature=False, feature_transform=True, in_channel=in_channel)
        
        self.segment_net = nn.Sequential(
            TNetkd.get_single_conv(1088, 512, 1),
            TNetkd.get_single_conv(512, 256, 1),
            TNetkd.get_single_conv(256, 128, 1),
            nn.Conv1d(128, class_num, 1)
        )
    
    def forward(self, point_cloud):
        B, _, N = point_cloud.shape
        x, trans_matrix_1, trans_matrix_2 = self.encoder(point_cloud)
        x = self.segment_net(x)                                         # [B, class_num, N]
        x = x.transpose(2, 1).contiguous()                              # [B, N, class_num]
        output = F.log_softmax(x, dim=-1)                               # [B, N, class_num]
        return output, trans_matrix_2


class get_loss(nn.Module):
    """
    PointNet Loss ,mat_diff_loss_scale是对特征转移矩阵的Loss施加的权重
    """
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight=None):
        if weight is None:
            loss = F.nll_loss(pred, target)
        else:
            loss = F.nll_loss(pred, target, weight=weight)          # semantic segmentation时需要weight
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)   # 转移矩阵的reguliarzer loss L_reg
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == "__main__":
    # model = PointNetPartSeg()
    # x = torch.rand((8, 6, 1000), dtype=torch.float)
    # label = torch.rand((8, 16), dtype=torch.float) 
    # result, feat = model(x, label)
    # print(result.shape, feat.shape)


    # semantic seg
    model = PointNetSemanticSeg(20)
    x = torch.rand((8, 9, 1000), dtype=torch.float)
    # result, feat = model(x)
    # print(result.shape, feat.shape)
    summary(model, (9, 1000), device='cpu')