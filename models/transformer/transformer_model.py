'''
Author       : ZHP
Date         : 2021-12-03 16:06:21
LastEditors  : ZHP
LastEditTime : 2022-01-14 14:56:31
FilePath     : /models/transformer_model.py
Description  : transformer model, refer http://nlp.seas.harvard.edu/2018/04/03/attention.html
Copyright 2021 ZHP, All Rights Reserved. 
2021-12-03 16:06:21
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
import math, copy, traceback

from torch.nn.modules.activation import ReLU


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DotAttention(nn.Module):
    '''
    Author: ZHP
    description: Scale Dot-Product Attention   Attention(Q,K,V) = softmax(QK^T/sqrt(dk))*V
    param {scalr} dropout_attn : dropout概率，默认没有dropout，用于softmax后，与value乘积前
    '''    
    def __init__(self, dropout_attn=None):
        super().__init__()
        if dropout_attn is not None:
            self.dropout = nn.Dropout(dropout_attn)
        self.softmax = nn.Softmax(dim=-1)
   
    def forward(self, q, k, v, scale=True, mask=None):
        '''
        Author: ZHP
        description: self-attention implement
        param {torch.tensor} q : query矩阵，[B, *, N, dim_k]   
        param {torch.tensor} k : key矩阵，[B, *, N, dim_k]
        param {torch.tensor} v : value矩阵, [B, *, N, dim_k]
        param {torch.tensor} scale :scalar,温度因子(temperature)，默认为sqrt(dim_k)
        param {torch.tensor} mask : mask,用于消去对padding位置的attention
        
        return {torch.tensor} feature : 经过self-attention后的特征， [B, *, N, dim_k]
        rerutn {torch.tensor} scores : attention map, [B, *, N, N]
        ''' 
        # 对于视觉常用序列，输入肯定是对齐的(等长),故不用mask
        # scale是防止点积过大，用于缩放
        dim_num = q.shape[-1]
        assert k.shape == q.shape and v.shape == q.shape, "In Self-Attention, query.shape={0}, key.shape={1}, value.shape={3}".format(q.shape, k.shape, v.shape)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        if scale:
            scores =  scores / math.sqrt(dim_num)   # 通过温度因子(temperature) sqrt(dk) 进行缩放(scale)
        
        if mask is not None:
            # mask 可见 https://www.cnblogs.com/zhouxiaosong/p/11032431.html
            # padding mask主要是nlp中词向量长度不一致，为了对齐需要对padding 0 的地方进行处理，加上负无穷，经过softmax，这些位置概率会接近0
            # sequence mask 为了让decoder无法看到未来信息，通过上三角矩阵进行
            scores = scores.masked_fill(mask == 0, -np.inf)   # mask for the variety dimension of input tensor

        # softmax
        scores = self.softmax(scores)

        if self.dropout is not None:
            scores = self.dropout(scores)
        
        # 与value(V)点积
        feature = torch.matmul(scores, v.to(scores.dtype))
        return feature, scores


class AddAndNorm(nn.Module):
    '''
    Author: ZHP
    description: 实现transformer中Add&Norm(残差连接和LayerNormalization)
    '''    
    def __init__(self, normalized_shape, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, residual, prelayer_output):
        '''
        Author: ZHP
        description: 每个子层输出为LayerNorm(x + prelayer(x))
        param {*} residual : 残差连接项
        param {*} pre_layer_output : 子网络Layer的输出， 残差连接Layer(args) + residual, prelayer_output + residual
        return {*} output : 经过add和norm后的输出
        '''
        # 通常BN层后面接dropout,但是结合残差连接后，可以先dropout再进行layernorm
        # > CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        output = self.dropout(prelayer_output) + residual
        output = self.norm(output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_num, heads_count, dropout=0.1):
        super().__init__()
        assert dim_num % heads_count == 0, "In Multihead-Attention，heads count({0}) is not divisible by the vector\'s dimension({1})".format(heads_count, dim_num)
        self.dim_num = dim_num
        self.d_k = dim_num // heads_count
        self.heads = heads_count
        self.w_q = nn.Linear(dim_num, dim_num, bias=False)
        self.w_k = nn.Linear(dim_num, dim_num, bias=False)
        self.w_v = nn.Linear(dim_num, dim_num, bias=False)
        self.fc = nn.Linear(dim_num, dim_num, bias=False)

        # self.linears = nn.ModuleList([nn.Linear(dim_num, dim_num) for _ in range(4)])  # 4个转移矩阵(线性层)
        self.attention = DotAttention(dropout)

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

        return {torch.tensor} feature : 经过Multihead-attention得到的特征输出就
        return {torch.tensor} attn : attention map
        '''    
        assert k.shape == q.shape and v.shape == q.shape and q.shape[-1] == self.dim_num, \
            "In Multihead-Attention, query.shape={0}, key.shape={1}, value.shape={3}, dim_num={4}, heads={5}".\
                format(q.shape, k.shape, v.shape, self.dim_num, self.heads)
        

        assert q.dtype == torch.float and v.dtype == torch.float and k.dtype == torch.float, \
            "In Multihead-Attention, query/key/value dtype should be torch.float"

        residual, batch_size = q, q.shape[0]  
        k = self.w_k(k).view(batch_size, -1, self.heads, self.d_k).transpose(1,2)   # [B, heads, N, d_k]   
        q = self.w_q(q).view(batch_size, -1, self.heads, self.d_k).transpose(1,2)
        v = self.w_v(v).view(batch_size, -1, self.heads, self.d_k).transpose(1,2)

        feature, attn = self.attention(k, q, v, mask=mask)                          # [B, N, ]
        feature = feature.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)
        feature = self.fc(feature)
        feature = self.add_norm(residual, feature)
        return feature, attn


class FeedForwardNet(nn.Module):
    '''
    Feed-Forward network,FFN
    '''
    def __init__(self, dim_in, dim_hid, dropout=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hid, bias=False),
            nn.ReLU(),
            nn.Linear(dim_hid, dim_in, bias=False),
            nn.Dropout(dropout)
        )

        # add & norm
        self.add_norm = AddAndNorm(dim_in, dropout)    
        '''
        ViT
        self.net_v2 = nn.Sequential(
            nn.Linear(dim_in, dim_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hid, dim_in),
            nn.Dropout(dropout)
        )
        '''

    def forward(self, x):
        '''
        Author: ZHP
        description: Position-wise Feed-Forward network
        param {torch.tensor} x : input [B, *, N, dim_in]
        return {torch.tensor} output : [B, * , N, dim_in]
        '''    
        output = self.net(x)
        output = self.add_norm(x, output)
        return output


class PositionalEncoding(nn.Module):
    """位置编码,nlp的transfomer实现"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_hid, heads_count, dropout=0.1):
        '''
        Author: ZHP
        description: transformer single encoder
        param {int} dim_model : channels,last dimension
        param {int} dim_hid : hidden dimension
        param {int} heads_count : multi-head attention
        param {float} dropout
        '''    
        super().__init__()
        self.self_attention = MultiHeadAttention(dim_model, heads_count, dropout)
        self.feed_forward = FeedForwardNet(dim_model, dim_hid, dropout)

    def forward(self, encoder_input, attn_mask=None):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} encoder_input : 输入的embedding [B, N, dim_model]
        param {torch.tensor} attn_mask : mask

        return {torch.tensor} output : encoder编码的特征，shape [B, N, dim_model]
        return {torch.tensor} attn_map : multihead 的 attention map [B, heads_count, N, N]
        '''    
        context, attn_map = self.self_attention(encoder_input, encoder_input, encoder_input, attn_mask)
        output = self.feed_forward(context)
        return output, attn_map


class Encoder(nn.Module):
    """
    多层EncoderLayer
    """
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        
    def forward(self, inputs, mask=None):
        """
        这里没有实现Embedding
        """
        outputs = inputs
        attn_maps = []
        for layer in self.layers:
            outputs, attn_map = layer(outputs, mask)
            attn_maps.append(copy.deepcopy(attn_map))
        return outputs, attn_maps


class DecoderLayer(nn.Module):
    """
    单个Decoder，包含两个multi-attention和一个FFN
    """
    def __init__(self, dim_model, dim_hid, heads_count, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(dim_num=dim_model, heads_count=heads_count, dropout=dropout)
        self.decoder_encoder_attention = MultiHeadAttention(dim_num=dim_model, heads_count=heads_count, dropout=dropout)
        self.ffn = FeedForwardNet(dim_in=dim_model, dim_hid=dim_hid, dropout=dropout)

    def forward(self, decoder_input, encoder_output, self_attn_mask, dec_enc_attn_mask):
        # 第一个attention是self-attention，q,k,v都是decoder的输入
        dec_output, dec_self_attn = self.self_attention(decoder_input, decoder_input, decoder_input, self_attn_mask)
        
        # Decoder第二个mutlihead-attention的query来自第一个attention的输出，key和value来自encoder的输出
        dec_output, dec_enc_attn = self.decoder_encoder_attention(dec_output, encoder_output, encoder_output, dec_enc_attn_mask)
        dec_output = self.ffn(dec_output)
        return dec_output, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    """
    多层Decoder
    """
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)

    def forward(self, dec_input, enc_output, self_attn_mask, dec_enc_attn_mask):
        dec_self_attn_maps, dec_enc_attn_maps = [], []
        dec_output = dec_input
        for layer in self.layers:
            dec_output, dec_self_attn_map, dec_enc_attn_map = layer(dec_output, enc_output, self_attn_mask, dec_enc_attn_mask)
            dec_self_attn_maps.append(dec_self_attn_map)
            dec_enc_attn_maps.append(dec_enc_attn_map)
        return dec_output, dec_self_attn_maps, dec_enc_attn_maps


class PostionalEmbedding(nn.Module):
    def __init__(self, embed_pos_layer):
        super().__init__()
        self.layer = embed_pos_layer

    def forward(self, x, feature, aggregate_func):
        '''
        Author: ZHP
        description: 位置编码与feature聚合成向量
        param {*} x : 待编码的位置向量
        param {*} feature : 特征
        param {*} aggregate_func ： 聚合函数，接收两个参数
        return {*} output : 聚合后的特征
        '''    
        output = self.layer(x)
        output = aggregate_func(feature, output)
        return output


class Transformer(nn.Module):
    def __init__(self, enc_pos_embed, dec_pos_embed, enc_num, dec_num, dim_model, dim_hid, target_num, heads_count, dropout=0.1):
        '''
        Author: ZHP
        description: transformer结构
        param {nn.Module} enc_pos_embed ：encoder位置编码 block
        param {nn.Module} dec_pos_embed : decoder位置编码 block
        param {scalar} enc_num : encoder包含的encoderlayer数量
        param {scalar} dec_num : decoder包含的decoderlayer数量
        param {scalar} dim_model : 序列维度，输入最后一维大小
        param {scalar} dim_hid : hidden dimension
        param {scalar} target_num : 经过linear+softmax输出的维度
        param {scalar} heads_count : multihead-attention中head数量
        param {scalar} dropout : dropout概率值
        '''    
        super().__init__()
        self.enc_pos_embed, self.dec_pos_embed = enc_pos_embed, dec_pos_embed
        self.encoder = Encoder(EncoderLayer(dim_model, dim_hid, heads_count, dropout), enc_num)
        self.decoder = Decoder(DecoderLayer(dim_model, dim_hid, heads_count, dropout), dec_num)
        self.classifier = nn.Sequential(
            nn.Linearx(dim_model, target_num),
            nn.Softmax(dim=-1)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, enc_input, dec_input):
        '''
        Author: ZHP
        description: 
        param {*} enc_input : 输入数据 [B, N, C]
        param {*} dec_input : decoder输入 [B, N, C] (通常等价enc_input)
        return {*}
        '''    
        # 缺省生成mask
        enc_input = self.enc_pos_embed(enc_input)
        enc_output, _ = self.encoder(enc_input)

        dec_input = self.dec_pos_embed(dec_input)
        dec_output, _, _ = self.decoder(dec_input, enc_output, None, None)
        return self.classifier(dec_output)


if __name__ == "__main__":
    net = EncoderLayer(80, 100, 8)
    # net = Transformer(None, None, 4, 4, 64, 128, 4, 8, 0.1)
    print(net)
    
