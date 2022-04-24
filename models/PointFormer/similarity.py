'''
Author       : ZHP
Date         : 2022-04-12 16:00:40
LastEditors  : ZHP
LastEditTime : 2022-04-12 17:01:01
FilePath     : /models/PointFormer/similarity.py
Description  : 
Copyright 2022 ZHP, All Rights Reserved. 
2022-04-12 16:00:40
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from models.pointnet.pointNet2_Ops import *
from models.PointFormer.basic_block import K_MLP_Layer

class Affinity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def forward(self, src, dst):
        pass

    def extra_repr(self) -> str:
        print_paras = ["sigma", "k", "mu", "epsilon"]
        s = ""
        for para in print_paras:
            if para in self.__dict__:
                s += f'{para}={self.__dict__[para]},'
        s = s[:-1]
        return s.format(**self.__dict__)

    
class pointnet2(Affinity):
    def __init__(self, k=3) -> None:
        super().__init__()
        self.k = k
        
    def forward(self, src, dst):
        '''
        Author: ZHP
        description: pointnet++ 中插值函数
        param {tensor} src：大基数点云 [B, N, 3]
        param {tensor} dst: 小基数点云 [B, S, 3]
        return {tensor} score: 相似度矩阵[B, N, S]
        '''    
        B, N, _ = src.shape
        # KNN 插值
        dists = square_distance(src, dst)                       # [B, N, S]，距离的平方
        dists, idx = dists.sort(dim=-1)                         # [B, N, S]
        dists, idx = dists[:, :, :self.k], idx[:, :, :self.k]             # [B, N, k] 
        # 以src的点云(N个)为中心，计算当前点云距离其最近的K(3)个点,记录距离和索引

        dist_recip = 1.0 / (dists + 1e-8)                       # 反向加权 w_i 防止为0 [B, N, k]
        norm = torch.sum(dist_recip, dim=2, keepdim=True)       # 分母，[B, N, 1]
        weight = dist_recip / norm                              # weight_i = w_i / sum(w_i)

        score = torch.zeros(B, N, dst.shape[1]).to(src.device)         # [B, N, S]
        score = score.scatter_(-1, idx, weight)         # [B, N, S]
        # 在当前点云中取出[B, N, k]个最近点数据[B, N, k, C]，score除了该k个点外，其他位置为0
        return score

class euclidean(Affinity):
    def __init__(self, mu=2, epsilon=1e-8) -> None:
        super().__init__()
        self.mu = mu
        self.epsilon = epsilon

    def forward(self, src, dst):
        '''
        Author: ZHP
        description: 基于欧氏距离反比的权重 1 / (||xi - yj||2)^mu + epsilon
        param {tensor} src：大基数点云 [B, N, 3]
        param {tensor} dst: 小基数点云 [B, S, 3]
        return {tensor} score 相似度矩阵 [B, N, S]
        ''' 
        dists = square_distance(src, dst)               # [B, N, S]
        dists = torch.pow(dists, exponent=self.mu)
        score = 1 / (dists + self.epsilon)                   # [B, N, S]
        score = F.softmax(score, dim=-1)
        return score

class cosine_similarity(Affinity):
    def __init__(self, epsilon=1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, src, dst):
        '''
        Author: ZHP
        description: 计算点之间余弦相似度   notion:F.cosine_similarity是向量对应相似度
        param {tensor} src：大基数点云 [B, N, 3]
        param {tensor} dst: 小基数点云 [B, S, 3]
        param {int} epsilon: 防止分母为0的极小值
        return {tensor} score 相似度矩阵 [B, N, S]
        ''' 
        B, N, _ = src.shape
        _, S, _ = dst.shape
        cdot = torch.matmul(src, dst.transpose(1,-1))               # [B, N, S]
        norm_src = torch.norm(src, dim=-1, keepdim=True)            # [B, N, 1]   ||src||2
        norm_dst = torch.norm(dst, dim=-1, keepdim=True)            # [B, S, 1]    ||dst||2
        norm_ = torch.matmul(norm_src, norm_dst.transpose(1,-1))    # [B, N, S]
        norm_ = torch.max(norm_, torch.ones_like(norm_) * self.epsilon)
        score = cdot / norm_                                        # [B, N, S]
        score = F.softmax(score, dim=-1)
        return score

class gaussian_kernel(Affinity):
    def __init__(self, sigma=1) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, src, dst):
        '''
        Author: ZHP
        description: 高斯核函数 k(x1,x2) = exp(- ||x1 - x2||^2 / (2*sigma^2))
        param {tensor} src：大基数点云 [B, N, 3]
        param {tensor} dst: 小基数点云 [B, S, 3]
        return {tensor} score 相似度矩阵 [B, N, S]
        ''' 
        gap = src[:,:,None] - dst[:,None]       # [B, N, S, 3]
        gap = torch.norm(gap, dim=-1)           # [B, N, S]
        gap = - (gap / (self.sigma ** 2)) * 0.5
        score = torch.exp(gap)                  # [B, N, S]
        score = F.softmax(score, dim=-1)
        return score

class chebyshev_distance(Affinity):
    def __init__(self, epsilon=1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, src, dst):
        '''
        Author: ZHP
        description: 切比雪夫距离  max|xi-yi|
        param {tensor} src：大基数点云 [B, N, 3]
        param {tensor} dst: 小基数点云 [B, S, 3]
        param {int} epsilon: 防止分母为0的极小值
        return {tensor} score 相似度矩阵 [B, N, S]
        ''' 
        dist = src[:,:,None] - dst[:,None]      # [B, N, S, 3]
        dist = torch.max(dist, dim=-1)[0]        # [B, N, S]
        dist = 1.0 / (dist + self.epsilon)
        score = F.softmax(dist, dim=-1)         # [B, N, S]
        return score

class minkowski_distance(Affinity):
    def __init__(self, p=1, epsilon=1e-8) -> None:
        super().__init__()
        self.p = p
        self.epsilon = epsilon

    def forward(self, src, dst):
        '''
        Author: ZHP
        description: 闵氏距离  [sum(|xi-yi|^p)]^(1/p)
        param {tensor} src：大基数点云 [B, N, 3]
        param {tensor} dst: 小基数点云 [B, S, 3]
        param {int} p: p=1表示曼哈顿距离，p=2表示欧氏距离，p=无穷大表示切比雪夫距离
        param {int} epsilon: 防止分母为0的极小值
        return {tensor} score 相似度矩阵 [B, N, S]
        ''' 
        # 
        dist = src[:,:,None] - dst[:,None]      # [B, N, S, 3]
        dist = torch.pow(dist, self.p)
        dist = torch.sum(dist, dim=-1)
        dist = torch.pow(dist, 1/self.p)
        dist = 1 / (dist + self.epsilon)
        score = F.softmax(dist, dim=-1)
        return score


class PointUpsampleAttn(nn.Module):
    def __init__(self, dim_in, relation=pointnet2(), dim_out=None, dropout=0.):
        super().__init__()
        if dim_out is None:
            self.embed = lambda x : x
        else:
            self.embed = K_MLP_Layer(3, dim_in, dim_out, True, True, dropout)
        self.relation = relation                     # 计算相似度方法


    def forward(self, q, k, v):
        '''
        Author: ZHP
        description: relation(qi,kj)*vj   1 / ||qi-kj||
        param {tensor} q : 原始点云坐标 [B, N, 3]
        param {tensor} k : 采样后的点云坐标 [B, S, 3]
        param {tensor} v : 采样后的点云特征 [B, S, C]
        return {tensor} extract: 上采样后的点云特征 [B, D, N]
        '''    
        score = self.relation(q, k)                     # [B, N, S]
        extract = torch.matmul(score, v)                # [B, N, C]
        extract = extract.transpose(1,-1)
        extract = self.embed(extract)   # [B, D, N]
        return extract


if __name__ == "__main__":
    p2 = euclidean()
    # src = torch.randn(1, 10, 3, dtype=torch.float)
    # dst = torch.randn(1, 10, 3, dtype=torch.float)
    # a = p2(src, dst)
    # print(a.shape)
    print(p2)