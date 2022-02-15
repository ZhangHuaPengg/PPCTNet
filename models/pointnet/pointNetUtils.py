'''
Author       : ZHP
Date         : 2021-11-11 10:50:03
LastEditors  : ZHP
LastEditTime : 2021-12-07 20:30:47
FilePath     : /models/pointNetUtils.py
Description  : PointNet中的Ops
Copyright    : ZHP
'''
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d

class TNetkd(nn.Module):
    """
    PointNet中T-Net实现， 对于k维输入实现特征对齐，保证对特定空间转换的不变性
    T-Net是为了学习转移矩阵M，学习到M后与特征相乘
    """
    def __init__(self, in_channel, output_k=64, out_channels=[64, 128, 1024]):
        """
        in_channel : 输入channel大小
        output_k: 输出的旋转(转移)矩阵维度大小, PointNet 分类时有两个T-Net,分别是3和64
                    64时，in_channel也为64，用于中间feature的transform
                    3时，in_channel不一定为3，可能还有norm的feature,但是固定转移矩阵为3x3
        """
        super().__init__()
        self.output_k = output_k
        self.conv1 = TNetkd.get_single_conv(in_channel, out_channels[0], 1)
        self.conv2 = TNetkd.get_single_conv(out_channels[0], out_channels[1], 1)
        self.conv3 = TNetkd.get_single_conv(out_channels[1], out_channels[2], 1)
        self.max_pool = torch.max

        self.fc1 = TNetkd.get_single_linear(out_channels[2], 512)
        self.fc2 = TNetkd.get_single_linear(512, 256)
        self.fc3 = nn.Linear(256, output_k * output_k)


    def forward(self, x):
        '''
        Author: ZHP
        description: T-Net 转换，注意有单位矩阵
        param {torch.tensor} x : [B, in_channel, N]
        return {torch.tensor} output : 学习到的旋转(转移)矩阵M， [B, output_k, output_k]
        '''    
        batch_size = x.shape[0] 
        x = self.conv3(self.conv2(self.conv1(x)))                               # [B, 1024, N]
        x = self.max_pool(x, dim=-1)[0]                                         # [B, 1024]

        x = self.fc3(self.fc2(self.fc1(x)))                                     # [B, output_k]
        iden = torch.eye(self.output_k).view(1,-1).repeat(batch_size, 1)      # [B, output_k]
        if x.is_cuda:
            iden.to(x.device)
        x = x + iden
        output = x.view(-1, self.output_k, self.output_k)                   # [B, output_k, output_k]
        return output

    @staticmethod
    def get_single_conv(in_channel, out_channel, kernel_size, activate=True):
        model = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size),
            nn.BatchNorm1d(out_channel)
        )
        if activate:
            model.add_module("activateion_layer", nn.ReLU())
        return model
    
    @staticmethod
    def get_single_linear(in_channel, out_channel):
        model = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        return model

class PointNetEncoder(nn.Module):

    def __init__(self, global_feature=True, feature_transform=False, in_channel=3):
        '''
        Author: ZHP
        description: PointNet Encoder  
        param {bool} global_feature : 是否返回全局特征，分类时为True,返回全局特征。 分割任务时为False,返回全局+局部。
        param {bool} feature_transform : 是否有中间层feature与转移矩阵相乘对齐
        param {int} in_channel : 输入channel大小
        '''    
        super().__init__()
        self.tnet = TNetkd(in_channel, 3)
        self.conv1 = TNetkd.get_single_conv(in_channel, 64, 1)
        self.conv2 = TNetkd.get_single_conv(64, 128, 1)
        self.conv3 = TNetkd.get_single_conv(128, 1024, 1, False)    # 该层不需要激活函数
        self.max_pool = torch.max

        self.global_feature, self.feature_transform = global_feature, feature_transform
        if self.feature_transform:
            self.tnet64 = TNetkd(in_channel=64)             # 分割时不需要这层tnet

    def forward(self, x):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} x : [B, C, N]
        
        return {torch.tensor} x : global feature时[B, 1024]    point feature时 [B, 1088, N]
        return {torch.tensor} trans_matrix : 第一个T-Net学习到的转移矩阵 [B, C, C]
        return {torch.tensor/NoneType} trans_matrix_2 : 第一个T-Net学习到的转移矩阵 [B, 64, 64]/None
        '''    
        B, C, N = x.shape
        trans_matrix = self.tnet(x)                                 # [B, C, C]
        x = x.transpose(2, 1)                                       # [B, N, C]
        if C > 3:
            feature = x[:, :, 3:]                                   # feature [B, N, C-3]  
            x = x[:, :, :3]                                         # coordinates [B, N, 3]
        x = torch.matmul(x, trans_matrix)                           # 与学习到的矩阵相乘从而对齐, [B, N, 3]
        if C > 3:
            x = torch.cat([x, feature], dim=-1)                     # 再拼接feature, [B, N, C]
        x = x.transpose(2, 1)                                       # [B, C, N]
        x = self.conv1(x)                                           # [B, 64, N]

        if self.feature_transform:
            trans_matrix_2 = self.tnet64(x)                         # [B, 64, 64]
            x = torch.matmul(x.transpose(2, 1), trans_matrix_2)     # [B, N, 64]
            x = x.transpose(2, 1)                                   # [B, 64, N]
        else:
            trans_matrix_2 = None
        
        local_feature = x                                           # [B, 64, N], 这里是local feature
        x = self.conv3(self.conv2(x))                               # [B, 1024, N]
        x = self.max_pool(x, dim=-1)[0]                             # [B, 1024], 这里是global_feature

        if self.global_feature:
            return x, trans_matrix, trans_matrix_2
        x = x.view(-1, 1024, 1).repeat(1, 1, N)                     # [B, 1024, N]
        x = torch.cat([x, local_feature], 1)                        # [B, 1088, N]拼接前面的特征作为 point feature
        return x, trans_matrix, trans_matrix_2


def feature_transform_reguliarzer(trans):
    '''
    Author: ZHP
    func: 转移矩阵loss   L_reg = ||I-A*A^T||^2_F  Frobenius范数(矩阵元素和的开方) 
    description: 原文:constrain the feature transformation matrix to be close to orthogonal matrix 这里loss为了让转移矩阵接近正交矩阵
    param {torch.tensor} trans : 学习到的转移矩阵[B, C, C]
    return {*}
    '''

    C = trans.size()[1]
    I = torch.eye(C)[None, :, :]                            # [1, C, C]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.bmm(trans, trans.transpose(2, 1))  - I     # A*A^T - I  [B, C, C]
    loss = torch.norm(loss, dim=(1,2))                      # 求范数   [B]
    loss = torch.mean(loss)                                 # scalar
    return loss


if __name__ == "__main__":
    torch.manual_seed(100)
    x = torch.rand((8, 3, 3), dtype=torch.float)
    print(feature_transform_reguliarzer(x))  # tensor(2.5605)
