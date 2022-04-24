'''
Author       : ZHP
Date         : 2022-04-02 18:52:04
LastEditors  : ZHP
LastEditTime : 2022-04-03 09:17:23
FilePath     : /models/PointFormer/basic_block.py
Description  : 
Copyright 2022 ZHP, All Rights Reserved. 
2022-04-02 18:52:04
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

import sys
sys.path.append("../..")
from models.transformer.transformer_model import DotAttention, AddAndNorm, FeedForwardNet, clones


class MHSA(nn.Module):
    def __init__(self, dim_num, heads_count, share=False, dropout=0.1):
        '''
        Author: ZHP
        description: MultiHeadAttention 
        param {torch.tensor} dim_Num : dimension
        param {torch.tensor} heads_count : heads
        param {torch.tensor} share : 是否权重共享
        param {torch.tensor} dropout: 
        '''
        super().__init__()
        assert dim_num % heads_count == 0, "In Multihead-Attention，heads count({0}) is not divisible by the vector\'s dimension({1})".format(heads_count, dim_num)
        self.dim_num = dim_num
        self.d_k = dim_num // heads_count
        self.heads, self.share = heads_count, share
        if share:
            self.w = nn.Linear(dim_num, dim_num, bias=False)
        else:
            self.w_q = nn.Linear(dim_num, dim_num, bias=False)
            self.w_k = nn.Linear(dim_num, dim_num, bias=False)
            self.w_v = nn.Linear(dim_num, dim_num, bias=False)
        self.attention = DotAttention()
        self.proj = nn.Linear(dim_num, dim_num, bias=False)
        # add & norm
        self.add_norm = AddAndNorm(dim_num, dropout)

    def forward(self, q, k, v, mask=None):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} q : query矩阵，[B, *, N, dim_k]   dim_k == self.dim_num
        param {torch.tensor} k : key矩阵，[B, *, N, dim_k]
        param {torch.tensor} v : value矩阵, [B, *, N, dim_k]
        param {torch.tensor} mask : mask矩阵

        return {torch.tensor} feature : 经过Multihead-attention得到的特征输出 [B, *, N, dim_k]
        return {torch.tensor} attn : attention map
        '''    
        assert k.shape == q.shape and v.shape == q.shape and q.shape[-1] == self.dim_num, \
            "In Multihead-Attention, query.shape={0}, key.shape={1}, value.shape={2}, dim_num={3}, heads={4}".\
                format(q.shape, k.shape, v.shape, self.dim_num, self.heads)
        

        assert q.dtype == torch.float and v.dtype == torch.float and k.dtype == torch.float, \
            "In Multihead-Attention, query/key/value dtype should be torch.float"

        residual, _ = q, q.shape[0] 
        original_shape = list(q.shape)
        view_shape = original_shape[:-1] + [self.heads, self.d_k]   # [B, *, N, heads, d_k]
        if self.share:
            k = self.w(k).view(view_shape).transpose(-3, -2)        # [B, *, heads, N, d_k]
            q = self.w(q).view(view_shape).transpose(-3, -2)        
            v = self.w(v).view(view_shape).transpose(-3, -2)        
        else:
            k = self.w_k(k).view(view_shape).transpose(-3, -2)          # [B, *, heads, N, d_k] 
            q = self.w_q(q).view(view_shape).transpose(-3, -2)
            v = self.w_v(v).view(view_shape).transpose(-3, -2)

        feature, _ = self.attention(q, k, v, mask=mask)                          # [B, *, heads, N, d_k] [B, *, heads, N, N]
        feature = feature.transpose(-3, -2).contiguous().view(original_shape)       # [B, *, N, dim_k]
        feature = self.proj(feature)                                                  # [B, *, N, dim_k]
        feature = self.add_norm(residual, feature)                                  # [B, *, N, dim_k]
        return feature


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_hid, heads_count, share=False, dropout=0.1):
        '''
        Author: ZHP
        description: transformer single encoder
        param {int} dim_model : channels,last dimension
        param {int} dim_hid : hidden dimension
        param {int} heads_count : multi-head attention
        param {float} dropout
        '''    
        super().__init__()
        self.self_attention = MHSA(dim_model, heads_count, share, dropout)
        self.feed_forward = FeedForwardNet(dim_model, dim_hid, dropout)

    def forward(self, encoder_input, attn_mask=None):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} encoder_input : 输入的embedding [B, *, N, dim_model]
        param {torch.tensor} attn_mask : mask

        return {torch.tensor} output : encoder编码的特征，shape [B, *, N, dim_model]
        return {torch.tensor} attn_map : multihead 的 attention map [B, heads_count, N, N]
        '''    
        context = self.self_attention(encoder_input, encoder_input, encoder_input, attn_mask)
        output = self.feed_forward(context)
        return output 


class Encoder(nn.Module):
    """
    多层EncoderLayer
    """
    def __init__(self, layer, N, save_fea=False):
        super().__init__()
        self.save_fea = save_fea
        self.layers = clones(layer, N)
        
    def forward(self, inputs, mask=None):
        """
        这里没有实现Embedding
        """
        outputs = inputs
        feature_list = []
        for layer in self.layers:
            if self.save_fea:
                feature_list.append(outputs)
            outputs = layer(outputs, mask)
        
        if self.save_fea:
            return outputs, feature_list
        return outputs


class Basic_Transformer_Encoder(nn.Module):
    def __init__(self, enc_num, dim_model, dim_hid, heads_count, in_size=4, share=False, dropout=0.5):
        '''
        Author: ZHP
        description: transformer结构
        param {scalar} enc_num : encoder包含的encoderlayer数量
        param {scalar} dim_model : 序列维度，输入最后一维大小
        param {scalar} dim_hid : hidden dimension
        param {scalar} heads_count : multihead-attention中head数量
        param {bool} share : multihead-attention中中是否权重共享
        param {scalar} dropout : dropout概率值
        '''    
        super().__init__()
        self.input_dim = dim_model
        self.in_size= in_size
        self.encoder = Encoder(EncoderLayer(dim_model, dim_hid, heads_count, share, dropout), enc_num)
        self.config_pos()
        self.add_layer()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_input, postion_input):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} enc_input : 输入数据 [B, N, C]
        param {torch.tensor} postion_input : 位置数据 [B, C', N]
        '''    
        # 缺省生成mask
        enc_input = self.enc_pos_embed(postion_input, enc_input)
        if self.pre_layer is not None:
            enc_input = self.pre_layer(enc_input)
        
        enc_output = self.encoder(enc_input)
        if self.back_layer is not None:
            enc_output = self.back_layer(enc_output)
        return enc_output

    def config_pos(self):
        self.enc_pos_embed = None
        self.dec_pos_embed = None
    
    def add_layer(self, pre_layer=None, back_layer=None):
        self.pre_layer, self.back_layer = pre_layer, back_layer


class PositionEmbedding(nn.Module):
    def __init__(self, pos_embed, aggregate_func):
        super().__init__()
        self.layer = pos_embed
        self.aggregate_func = aggregate_func

    def forward(self, x, feature):
        '''
        Author: ZHP
        description: 位置编码与feature聚合成向量
        param {torch.tensor} x : 待编码的位置向量, [B, C, N]/[B, C, N, S]
        param {torch.tensor} feature : 与位置编码聚合的特征 [B, N, C]/[B, S, N, C]

        return {torch.tensor} output : 聚合后的特征 [B, N, C]/[B, S, N, C]
        '''    
        if self.layer is not None:
            x = self.layer(x)
        x = x.transpose(1, -1)      # [B, *, N, C]
        output = self.aggregate_func(x, feature)
        return output


class Learn_Pos_TransEncoder(Basic_Transformer_Encoder):
    def config_pos(self):
        self.enc_pos_embed = PositionEmbedding(K_MLP_Layer(in_size=self.in_size, in_channel=3, channel_list=[3, self.input_dim],\
            bn_list=[False, False], activate_list=[False, False], dropout_list=[False, False]), Pos_Aggregate.add_)


class Pos_Aggregate():
    def __init__(self) -> None:
        pass

    @staticmethod
    def add_(arg1, arg2):
        return arg1 + arg2
    
    @staticmethod
    def concat_(arg1, arg2):
        return torch.cat([arg1, arg2], dim=-1)


class K_MLP_Layer(nn.Module):
    def __init__(self, in_size, in_channel, channel_list, bn_list, activate_list, dropout_list):
        super().__init__()
        self.model = nn.ModuleList()
        self.in_size = in_size
        assert in_size == 3 or in_size == 4, "not implement for the input size as {}".format(in_size)
        if in_size == 4:
            conv, bn = "Conv2d", "BatchNorm2d"
        else:
            conv, bn = "Conv1d", "BatchNorm1d"

        self.length = len(channel_list)
        if self.length == 0:
            return
        self.config_list(bn_list, activate_list, dropout_list)

        for i in range(0, len(channel_list)):
            self.model.append(getattr(nn, conv)(in_channel, channel_list[i], 1))
            in_channel = channel_list[i]
            if self.bn_list[i]:
                self.model.append(getattr(nn, bn)(in_channel))
            if self.activate_list[i]:
                self.model.append(nn.ReLU())
            if self.dropout_list[i]:
                self.model.append(nn.Dropout(self.dropout_list[i]))
    
    def forward(self, x, transpose=False):
        # x [B, C, N] or [B, C, H, W]  ==> [B, D, N] [B, D, H, W](transpose=False)
        # x [B, N, C] or [B, H, W, C]  ==> [B, N, C] [B, H, W, C](transpose=True)

        if self.length == 0:
            return x
        shape_len = len(list(x.shape))
        assert shape_len == self.in_size, "expect the input size length as {}, but got {}".format(self.in_size, shape_len)
        if transpose:
            pts = x.transpose(1, -1)
        else:
            pts = x
        for layer in self.model:
            pts = layer(pts)
        pts = pts.transpose(1,-1).contiguous() if transpose else pts
        return pts

    def config_list(self, bn_list, activate_list, dropout_list):
        if isinstance(bn_list, bool):
            bool_v = bn_list
            self.bn_list = [bool_v for i in range(self.length)]
        elif isinstance(bn_list, list):
            self.bn_list = bn_list
        else:
            self.bn_list = [True for i in range(self.length)]

        if isinstance(activate_list, bool):
            bool_v = activate_list
            self.activate_list = [bool_v for i in range(self.length)]
        elif isinstance(bn_list, list):
            self.activate_list = bn_list
        else:
            self.activate_list = [True for i in range(self.length)]

        if isinstance(dropout_list, float):
            bool_v = dropout_list
            self.dropout_list = [bool_v for i in range(self.length)]
        elif isinstance(dropout_list, list):
            self.dropout_list = dropout_list
        else:
            self.dropout_list = [False for i in range(self.length)]


class MHAttention(nn.Module):
    def __init__(self, dim_in, heads_count, share=False, dropout=0.5, dim_qkv=None):
        '''
        Author: ZHP
        description: q,k dim应一样， k和v 序列长度应一样
        param {int} dim_in : input dimension
        param {int} heads_count : heads
        param {bool} share : 是否权重共享
        param {float} dropout: 
        param {list/int/dict} dim_qkv: q,k,v的中间维度 
        '''
        super().__init__()
        if dim_qkv is None:
            dim_q, dim_k, dim_v = dim_in, dim_in, dim_in
        elif isinstance(dim_qkv, int):
            dim_q, dim_k, dim_v = dim_qkv, dim_qkv, dim_qkv
        elif isinstance(dim_qkv, list):
            if len(dim_qkv) == 3:
                dim_q, dim_k, dim_v = dim_qkv[0], dim_qkv[1], dim_qkv[2]
            if len(dim_qkv) == 2:
                dim_q, dim_k = dim_qkv[0], dim_qkv[0]
                dim_v = dim_qkv[-1]
        elif isinstance(dim_qkv, dict):
            dim_q = dim_qkv.get('q') or dim_qkv.get('k')
            dim_k = dim_q
            dim_v = dim_qkv["v"]
        else:
            print('error for dim_qkv {}.'.format(dim_qkv))
        assert dim_q == dim_k and dim_q > 0 and dim_v > 0, "error: dim_q={},dim-k={},dim_v={}.".format(dim_q, dim_k, dim_v)

        assert dim_q % heads_count == 0, "In Multihead-Attention，heads count({0}) is not divisible by the query and key vector\'s dimension({1})".format(heads_count, dim_q)
        assert dim_v % heads_count == 0, "In Multihead-Attention，heads count({0}) is not divisible by the value vector\'s dimension({1})".format(heads_count, dim_q)
        self.heads, self.share = heads_count, share
        self.dim_in = dim_in
        if share:
            self.w_qk = nn.Linear(dim_in, dim_q)
        else:
            self.w_q = nn.Linear(dim_in, dim_q, bias=False)
            self.w_k = nn.Linear(dim_in, dim_q, bias=False)
        self.w_v = nn.Linear(dim_in, dim_v)
        self.attention = DotAttention()

        self.proj = nn.Linear(dim_v, dim_in)
        # add & norm
        self.add_norm = AddAndNorm(dim_in, dropout)

    def forward(self, q, k, v, mask=None):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} q : query矩阵，[B, *, Nq, dim_in]  
        param {torch.tensor} k : key矩阵，[B, *, Nv, dim_in]
        param {torch.tensor} v : value矩阵, [B, *, Nv, dim_in]
        param {torch.tensor} mask : mask矩阵

        return {torch.tensor} feature : 经过Multihead-attention得到的特征输出 [B, *, Nq, dim_in]
        return {torch.tensor} attn : attention map
        '''    
        assert k.shape[-1] == q.shape[-1] and v.shape[-2] == k.shape[-2] and q.shape[-1] == self.dim_in, \
            "In Multihead-Attention, query.shape={0}, key.shape={1}, value.shape={2}, dim_num={3}, heads={4}".\
                format(q.shape, k.shape, v.shape, self.dim_in, self.heads)
        

        assert q.dtype == torch.float and v.dtype == torch.float and k.dtype == torch.float, \
            "In Multihead-Attention, query/key/value dtype should be torch.float"

        residual = q
        if self.share:
            q = self.w_qk(q)
            k = self.w_qk(k)
        else:
            q, k = self.w_q(q), self.w_k(k)
        v = self.w_v(v)
        pre_dim, Nq, Nv, d_q, d_v = list(q.shape)[:-2], q.shape[-2], v.shape[-2], q.shape[-1], v.shape[-1]
        q = q.view(pre_dim + [Nq, self.heads, d_q // self.heads]).transpose(-3, -2)   # [B, *, heads, Nq, d_q']
        k = k.view(pre_dim + [Nv, self.heads, d_q // self.heads]).transpose(-3, -2)   # [B, *, heads, Nv, d_q']
        v = v.view(pre_dim + [Nv, self.heads, d_v // self.heads]).transpose(-3, -2)   # [B, *, heads, Nv, d_v']
        feature, attn_map = self.attention(q, k, v, mask=mask)    # [B, *, heads, Nq, d_v']   [B, *, heads, Nq, Nv]
        feature = feature.transpose(-3, -2).contiguous().view(pre_dim + [Nq, d_v])   # [B, *, Nq, d_v]
        feature = self.proj(feature)                                                  # [B, *, Nq, dim_in]
        feature = self.add_norm(residual, feature)                                  # [B, *, Nq, dim_in]
        return feature


def get_loss(pred, target, weight=None):
    if weight is None:
        loss = F.nll_loss(pred, target)
    else:
        # semantic segmentation loss
        loss = F.nll_loss(pred, target, weight=weight)
    return loss

if __name__ == "__main__":
    pass