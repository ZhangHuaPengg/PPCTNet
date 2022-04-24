'''
Author       : ZHP
Date         : 2022-04-02 18:46:23
LastEditors  : ZHP
LastEditTime : 2022-04-24 12:34:39
FilePath     : /models/PointFormer/pplt.py
Description  : 
Copyright 2022 ZHP, All Rights Reserved. 
2022-04-02 18:46:23
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("../..")

from models.PointFormer.basic_block import *
from models.pointnet.pointNet2_Ops import *
from models.PointFormer.similarity import *


class SingleLayer(nn.Module):
    def __init__(self, dim_in, heads, share=False, dropout=0.5, dim_qkv=None, dim_hid=None):
        super().__init__()
        self.mha = MHAttention(dim_in=dim_in, heads_count=heads, share=share, dropout=dropout, dim_qkv=dim_qkv)
        hid_dim = dim_in * 2 if dim_hid is None else dim_hid
        self.ffn = FeedForwardNet(dim_in=dim_in, dim_hid=hid_dim, dropout=dropout)
    
    def forward(self, x, mask=None):
        context = self.mha(x, x, x, mask=mask)
        context = self.ffn(context)
        return context


class Token_Transformer(nn.Module):
    def __init__(self, 
                enc_num, # 编码层个数
                dim_in, # 输入维度
                heads=1, 
                share=False, 
                dropout=0.4, 
                dim_qkv=None, # q,k,v中间维度
                dim_hid=None):
        super().__init__()
        self.encoders = clones(SingleLayer(dim_in, heads, share, dropout, dim_qkv, dim_hid), enc_num)


    def forward(self, inputs, idx=0, mask=None):
        '''
        Author: ZHP
        description: 带token的transformer，当idx取到所有索引(如 range(S)),则退化为普通transformer
        param {tensor} inputs:特征序列 [B, N, C] / [B, N, S, C]
        param {int/list/tensor} idx:索引或索引矩阵
        return {tensor} : 取出token的特征
        '''    
        outputs = inputs
        for layer in self.encoders:
            outputs = layer(outputs, mask)
        
        if len(inputs.shape) == 3:
            return outputs[:,idx,:]
        elif len(inputs.shape) == 4:
            return outputs[:,:,idx,:]
        else:
            print("shape - ({})out of bounds.".format(inputs.shape))
            sys.exit(0)


class DualTransformer(nn.Module):
    def __init__(self, dim_in, pos_local, pos_global, enc_nums=[1,1], dropout=0.4, pos_mode="absolute", res=False):
        super().__init__()
        self.pos_local, self.pos_mode = pos_local, pos_mode
        self.local_former = Token_Transformer(enc_nums[0], dim_in, 8, dropout=dropout)
        self.pos_global = pos_global
        self.global_former = Token_Transformer(enc_nums[1], dim_in, 8, dropout=dropout)
        self.res = res

    def forward(self, xg, fg, xs, fs):
        '''
        Author: ZHP
        description: 双重点云Transformer，local用于提取每组内点云特征,球心为token，global提取组间点云特征。
        param {tensor} xg: 点云组集坐标，[B, S, K, 3]
        param {tensor} fg: 点云组集特征，[B, S, K, C]
        param {tensor} xs: 点云组中心点集坐标，[B, S, 3]
        param {tensor} fs: 点云组中心点集特征，[B, S, C]
        return {tensor} output: [B, S, C]
        '''    
        B, S, C = fs.shape
        if self.pos_mode == "absolute":
            pos_enc = self.pos_local(torch.cat([xg, xs[:,:,None]], dim=2), True)        # [B, S, K+1, C]
        elif self.pos_mode == "relative":
            centroid = xs[:,:,None]             # [B, S, 1, 3]
            re_xg = xg - centroid           # [B, S, K, 3]
            pos_enc = self.pos_local(torch.cat([re_xg, torch.zeros_like(centroid)], dim=2), True)        # [B, S, K+1, C]
        else:
            print("pos mode[{}] not support".format(self.pos_mode))
        group_feature = torch.cat([fg, fs[:,:,None]], dim=2)                        # [B, S, K+1, C]
        local_embed = self.local_former(group_feature + pos_enc, idx=-1)            # [B, S, C]

        global_pos = self.pos_global(xs, True)                                      # [B, S, C]
        output = self.global_former(local_embed + global_pos, idx=range(S))         # [B, S, C]
        if self.res:
            output = fs + output
        return output


class DualTransformer_Block(nn.Module):
    def __init__(self, dim_in, pos_local, pos_global, enc_nums=[1,1], dropout=0.4, pos_mode="absolute", res=False):
        super().__init__()
        self.pos_local, self.pos_mode = pos_local, pos_mode
        self.local_former = Token_Transformer(enc_nums[0], dim_in, 8, dropout=dropout)
        self.pos_global = pos_global
        self.global_former = Token_Transformer(enc_nums[1], dim_in, 8, dropout=dropout)
        self.res = res

    def forward(self, xg, fg, xs, fs):
        '''
        Author: ZHP
        description: 双重点云Transformer，local用于提取每组内点云特征,球心为token，global提取组中心点云特征，short-cut。
        param {tensor} xg: 点云组集坐标，[B, S, K, 3]
        param {tensor} fg: 点云组集特征，[B, S, K, C]
        param {tensor} xs: 点云组中心点集坐标，[B, S, 3]
        param {tensor} fs: 点云组中心点集特征，[B, S, C]
        return {tensor} output: [B, S, C]
        '''    
        B, S, C = fs.shape
        if self.pos_mode == "absolute":
            pos_enc = self.pos_local(torch.cat([xg, xs[:,:,None]], dim=2), True)        # [B, S, K+1, C]
        elif self.pos_mode == "relative":
            centroid = xs[:,:,None]             # [B, S, 1, 3]
            re_xg = xg - centroid           # [B, S, K, 3]
            pos_enc = self.pos_local(torch.cat([re_xg, torch.zeros_like(centroid)], dim=2), True)        # [B, S, K+1, C]
        else:
            print("pos mode[{}] not support".format(self.pos_mode))
        group_feature = torch.cat([fg, fs[:,:,None]], dim=2)                        # [B, S, K+1, C]
        local_embed = self.local_former(group_feature + pos_enc, idx=-1)            # [B, S, C]

        global_pos = self.pos_global(xs, True)                                      # [B, S, C]
        output = self.global_former(fs + global_pos, idx=range(S))                  # [B, S, C]
        return output + local_embed

class PointEmbeddingandGroup(nn.Module):
    def __init__(self, npoint, radius, nsample, dim_in, channel_list, group_all=False, dropout=0.4):
        super().__init__()
        self.npoint, self.nsample, self.radius, self.group_all = npoint, nsample, radius, group_all
        self.embedding = K_MLP_Layer(3, dim_in, channel_list, True, True, dropout_list=dropout)

    def forward(self, x, f):
        '''
        Author: ZHP
        description: 点云嵌入到高维，采样分组
        param {tensor} x : [B, N, 3]
        param {tensor} f : [B, N, C]
        
        return {tensor} xg: [B, S, K, 3]
        return {tensor} fg: [B, S, K, D]
        return {tensor} xs: [B, S, 3]
        return {tensor} xg: [B, S, D]
        '''    
        embed_ = self.embedding(f, transpose=True)   # [B, N, D]
        if self.group_all:
            # [B, S, 3] [B, S, K, D]
            xs, fg = sample_and_group_all(x, embed_, cat_xyz=False)
            xg = x[:,None]          # [B, 1, N, 3]
            fs = torch.mean(embed_, dim=1, keepdim=True) # [B, 1, D]
        else:
            # [B, S, 3] [B, S, K, D] [B, S, K, 3]
            xs, fg, xg, fps_idx = sample_and_group(self.npoint, self.radius, \
                self.nsample, x, embed_, returnfps=True, cat_xyz=False)
            fs = index_points(embed_, fps_idx)      # [B, S, D]
        
        return xg, fg, xs, fs


class PPLT_encoder_layer(nn.Module):
    def __init__(self, npoint, radius, nsample, dim_in, channel_list, group_all, pos_mode, dropout=0., res=False):
        super().__init__()
        self.PEG = PointEmbeddingandGroup(npoint, radius, nsample, dim_in, channel_list, group_all, dropout)
        dim_num = channel_list[-1]
        # self.transformer = DualTransformer(dim_num, K_MLP_Layer(4, 3, [3, dim_num], False, False, dropout),\
        #     pos_global=K_MLP_Layer(3, 3, [3, dim_num], False, False, 0.2), dropout=dropout,  pos_mode=pos_mode, res=res)

        self.transformer = DualTransformer_Block(dim_num, K_MLP_Layer(4, 3, [3, dim_num], False, False, dropout),\
            pos_global=K_MLP_Layer(3, 3, [3, dim_num], False, False, 0.2), dropout=dropout,  pos_mode=pos_mode, res=res)

    def forward(self, x, f):
        """
        x : [B, N, 3]
        f : [B, N, C]
        """
        xg, fg, xs, fs = self.PEG(x, f)         
        context = self.transformer(xg, fg, xs, fs)      # [B, S, D]
        return xs, context


class PPLT_UpsampleLayer(nn.Module):
    def __init__(self):
        super().__init__()


class PPLT_Model_Cls(nn.Module):
    def __init__(self, target_num, embed_list, pos_mode="absolute", res=False, init_dim=6):
        super().__init__()
        now_list = embed_list[0]
        pre_dim = init_dim
        self.encoder_1 = PPLT_encoder_layer(512, 0.2, 32, pre_dim, now_list, group_all=False, pos_mode=pos_mode, dropout=0., res=res)

        pre_dim = now_list[-1]
        now_list = embed_list[1]
        self.encoder_2 = PPLT_encoder_layer(128, 0.4, 32, pre_dim, now_list, group_all=False, pos_mode=pos_mode, dropout=0., res=res)

        pre_dim = now_list[-1]
        now_list = embed_list[2]
        self.encoder_3 = PPLT_encoder_layer(None, None, None, pre_dim, now_list, group_all=True, pos_mode=pos_mode, dropout=0., res=res)
        
        pre_dim=now_list[-1]
        self.classifier = nn.Sequential(
            K_MLP_Layer(3, pre_dim, [512, 256], True, True, 0.5),
            nn.Conv1d(256, target_num, 1)
        )


    def forward(self, 
                points      # 原始特征[B, init_dim, N]
                ):
        xyz = points[:,:3,:].transpose(1,-1)                # [B, N, 3]
        points = points.transpose(1,-1)                     # [B, N, C]
        xs_1, context = self.encoder_1(xyz, points)
        xs_2, context = self.encoder_2(xs_1, context)
        xs_3, context = self.encoder_3(xs_2, context)       # [B, 1, C]

        pred = self.classifier(context.transpose(1,-1))     # [B, target_num, 1]
        pred = F.log_softmax(pred, dim=1).squeeze(-1)       # [B, target_num]
        return pred



class PPLT_Decoder_Layer(nn.Module):
    def __init__(self, pre_dim, cat_dim, upsample, dim_out, channel_list, dropout=0.):
        super().__init__()
        self.upsample = PointUpsampleAttn(pre_dim, upsample, dim_out, dropout=dropout)
        self.layers = K_MLP_Layer(3, pre_dim+cat_dim, channel_list, True, True, dropout)
    
    def forward(self, xs_s, fs_s, xs_l, fs_l):
        extract = self.upsample(xs_l, xs_s, fs_s)       # [B, D, N]
        cat_fea = torch.cat([fs_l.transpose(1,-1), extract], dim=1)  # [B, C+D, N]
        outs = self.layers(cat_fea).transpose(1,-1)             # [B, N, D']   
        return outs

class PPLT_Model_Seg(nn.Module):
    def __init__(self, target_num, embed_list, pos_mode="absolute", res=False, init_dim=6):
        super().__init__()
        now_list = embed_list[0]
        pre_dim = init_dim
        cat_dim = [init_dim]
        self.encoder_1 = PPLT_encoder_layer(512, 0.2, 32, pre_dim, now_list, group_all=False, pos_mode=pos_mode, dropout=0., res=res)

        pre_dim = now_list[-1]
        now_list = embed_list[1]
        cat_dim.append(pre_dim)
        self.encoder_2 = PPLT_encoder_layer(128, 0.4, 32, pre_dim, now_list, group_all=False, pos_mode=pos_mode, dropout=0., res=res)

        pre_dim = now_list[-1]
        now_list = embed_list[2]
        cat_dim.append(pre_dim)
        self.encoder_3 = PPLT_encoder_layer(None, None, None, pre_dim, now_list, group_all=True, pos_mode=pos_mode, dropout=0., res=res)
        
        pre_dim=now_list[-1]
        self.decoder_1 = PPLT_Decoder_Layer(pre_dim, cat_dim.pop(), pointnet2(), None, [512, 256])

        pre_dim = 256
        self.decoder_2 = PPLT_Decoder_Layer(pre_dim, cat_dim.pop(), pointnet2(), None, [256, 128])

        pre_dim = 128
        self.decoder_3 = PPLT_Decoder_Layer(pre_dim, cat_dim.pop() + 16 + 3, pointnet2(), None, [128, 128], 0.4)

        pre_dim = 128
        self.classifier = nn.Sequential(
            K_MLP_Layer(3, pre_dim, [128], True, True, 0.5),
            nn.Conv1d(128, target_num, 1)
        )

    
    def forward(self, 
                points,      # 原始特征 [B, init_dim, N]
                cls_label
                ):
        B, _, N = points.shape

        xyz = points[:,:3,:].transpose(1, -1)   # [B, N, 3]
        points = points.transpose(1,-1)     # [B, N, C]
        xs_1, fs_1 = self.encoder_1(xyz, points)
        xs_2, fs_2 = self.encoder_2(xs_1, fs_1)
        xs_3, fs_3 = self.encoder_3(xs_2, fs_2)

        feature = self.decoder_1(xs_3, fs_3, xs_2, fs_2)
        feature = self.decoder_2(xs_2, feature, xs_1, fs_1)
        label_one_hot = cls_label.view(B, 1, 16).repeat(1, N, 1)                                # [B, N, 16]
        co_feature = torch.cat([label_one_hot, xyz, points], dim=-1)                             # [B, 16+3+C, N]
        feature = self.decoder_3(xs_1, feature, xyz, co_feature)

        pred = self.classifier(feature.transpose(1,-1))                     # [B, target_num, N]
        pred = F.log_softmax(pred, dim=1).transpose(1,-1)   # [B, N, target_num]
        return pred


    
if __name__ == "__main__":
    model = PPLT_Model_Seg(50, [[32, 64, 128], [256, 256, 512], [512, 512, 1024]], res=True, init_dim=6)
    pts = torch.rand(1, 6, 2048, dtype=torch.float)
    cls_label = torch.rand(1, 16, dtype=torch.float)
    pred = model(pts, cls_label)
    print(pred.shape)
