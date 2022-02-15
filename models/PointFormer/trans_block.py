'''
Author       : ZHP
Date         : 2021-12-03 16:06:21
LastEditors  : ZHP
LastEditTime : 2022-02-15 13:55:28
FilePath     : /models/PointFormer/trans_block.py
Description  : transformer model, refer http://nlp.seas.harvard.edu/2018/04/03/attention.html
Copyright 2021 ZHP, All Rights Reserved. 
2021-12-03 16:06:21
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

import sys
sys.path.append("../..")
from models.transformer.transformer_model import DotAttention, AddAndNorm, FeedForwardNet, clones
from torchsummary import summary

# from utils.summary import summary

class MultiHeadAttention_Share(nn.Module):
    def __init__(self, dim_num, heads_count, share=False, dropout=0.1):
        '''
        Author: ZHP
        description: 
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
        self.fc = nn.Linear(dim_num, dim_num, bias=False)

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

        return {torch.tensor} feature : 经过Multihead-attention得到的特征输出 [B, *, N, dim_k]
        return {torch.tensor} attn : attention map
        '''    
        assert k.shape == q.shape and v.shape == q.shape and q.shape[-1] == self.dim_num, \
            "In Multihead-Attention, query.shape={0}, key.shape={1}, value.shape={3}, dim_num={4}, heads={5}".\
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

        feature, _ = self.attention(k, q, v, mask=mask)                          # [B, *, heads, N, d_k] [B, *, heads, N, N]
        feature = feature.transpose(-3, -2).contiguous().view(original_shape)       # [B, *, N, dim_k]
        feature = self.fc(feature)                                                  # [B, *, N, dim_k]
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
        self.self_attention = MultiHeadAttention_Share(dim_model, heads_count, share, dropout)
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
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        
    def forward(self, inputs, mask=None):
        """
        这里没有实现Embedding
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, mask)
        return outputs


class DecoderLayer(nn.Module):
    """
    单个Decoder，包含两个multi-attention和一个FFN
    """
    def __init__(self, dim_model, dim_hid, heads_count, share=False, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention_Share(dim_num=dim_model, heads_count=heads_count, share=share, dropout=dropout)
        self.decoder_encoder_attention = MultiHeadAttention_Share(dim_num=dim_model, heads_count=heads_count, share=share, dropout=dropout)
        self.ffn = FeedForwardNet(dim_in=dim_model, dim_hid=dim_hid, dropout=dropout)

    def forward(self, decoder_input, encoder_output, self_attn_mask, dec_enc_attn_mask):
        # 第一个attention是self-attention，q,k,v都是decoder的输入
        dec_output = self.self_attention(decoder_input, decoder_input, decoder_input, self_attn_mask)
        
        # Decoder第二个mutlihead-attention的query来自第一个attention的输出，key和value来自encoder的输出
        dec_output = self.decoder_encoder_attention(dec_output, encoder_output, encoder_output, dec_enc_attn_mask)
        dec_output = self.ffn(dec_output)
        return dec_output


class Decoder(nn.Module):
    """
    多层Decoder
    """
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_output = dec_input
        for layer in self.layers:
            dec_output = layer(dec_output, enc_output, self_attn_mask, dec_enc_attn_mask)
        return dec_output


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
            nn.Linear(dim_model, target_num),
            nn.Softmax(dim=-1)
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, enc_input, dec_input):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} enc_input : 输入数据 [B, N, C]
        param {torch.tensor} dec_input : decoder输入 [B, N, C] (通常等价enc_input)
        return {torch.tensor} : [B, N, target_num]
        '''    
        # 缺省生成mask
        enc_input = self.enc_pos_embed(enc_input)
        enc_output = self.encoder(enc_input)

        dec_input = self.dec_pos_embed(dec_input)
        dec_output = self.decoder(dec_input, enc_output, None, None)
        return self.classifier(dec_output)


class Basic_Transformer(nn.Module):
    def __init__(self, enc_num, dec_num, dim_model, dim_hid, target_num, heads_count, share=False, dropout=0.1):
        '''
        Author: ZHP
        description: transformer结构
        param {scalar} enc_num : encoder包含的encoderlayer数量
        param {scalar} dec_num : decoder包含的decoderlayer数量
        param {scalar} dim_model : 序列维度，输入最后一维大小
        param {scalar} dim_hid : hidden dimension
        param {scalar} target_num : 经过linear+softmax输出的维度
        param {scalar} heads_count : multihead-attention中head数量
        param {bool} share : multihead-attention中中是否权重共享
        param {scalar} dropout : dropout概率值
        '''    
        super().__init__()
        self.input_dim = dim_model
        
        self.encoder = Encoder(EncoderLayer(dim_model, dim_hid, heads_count, share, dropout), enc_num)
        self.decoder = Decoder(DecoderLayer(dim_model, dim_hid, heads_count, share, dropout), dec_num)
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, target_num),
            nn.Softmax(dim=-1)
        )
        self.config_pos()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, enc_input, postion_input, dec_input):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} enc_input : 输入数据 [B, N, C]
        param {torch.tensor} postion_input : 位置数据 [B, N, C']
        param {torch.tensor} dec_input : decoder输入 [B, N, C] (通常等价enc_input)
        return {torch.tensor} : [B, N, target_num]
        '''    
        # 缺省生成mask
        enc_input = self.enc_pos_embed(postion_input, enc_input)
        enc_output = self.encoder(enc_input)

        dec_input = self.dec_pos_embed(postion_input, dec_input)
        dec_output = self.decoder(dec_input, enc_output)
        return self.classifier(dec_output)

    def config_pos(self):
        self.enc_pos_embed = None
        self.dec_pos_embed = None


class Basic_Transformer_Encoder(nn.Module):
    def __init__(self, enc_num, dim_model, dim_hid, heads_count, in_size=4, share=False, dropout=0.1):
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

class Learn_Pos_TransEncoder(Basic_Transformer_Encoder):
    def config_pos(self):
        self.enc_pos_embed = PositionEmbedding(K_MLP_Layer(in_size=self.in_size, in_channel=3, channel_list=[3, self.input_dim],\
            bn_list=[False, False], activate_list=[False, False], dropout_list=[False, False]), Pos_Aggregate.add_)


class Mlp_Learn_Trans(Basic_Transformer):
    def config_pos(self):
        self.enc_pos_embed = PositionEmbedding(K_MLP_Layer(in_size=self.in_size, in_channel=3, channel_list=[3, self.input_dim],\
            bn_list=[False, False], activate_list=[True, False], dropout_list=[False, False]), Pos_Aggregate.add_)

        self.dec_pos_embed = PositionEmbedding(K_MLP_Layer(in_size=self.in_size, in_channel=3, channel_list=[3, self.input_dim],\
            bn_list=[False, False], activate_list=[True, False], dropout_list=[False, False]), Pos_Aggregate.add_)


class PositionEmbedding(nn.Module):
    def __init__(self, pos_embed, aggregate_func):
        super().__init__()
        self.layer = pos_embed
        self.aggregate_func = aggregate_func

    def forward(self, x, feature):
        '''
        Author: ZHP
        description: 位置编码与feature聚合成向量
        param {torch.tensor} x : 待编码的位置向量, [B, C, N, *]
        param {torch.tensor} feature : 与位置编码聚合的特征 [B, *, N, C]

        return {torch.tensor} output : 聚合后的特征 [B, *, N, C]
        '''    
        if self.layer is not None:
            x = self.layer(x)
            x = x.transpose(1, -1)      # [B, *, N, C]
        output = self.aggregate_func(x, feature)
        return output


class Pos_Aggregate():
    def __init__(self) -> None:
        pass

    @staticmethod
    def add_(arg1, arg2):
        return arg1 + arg2
    
    @staticmethod
    def concat_(arg1, arg2):
        return torch.cat([arg1, arg2], dim=-1)

class PointPyramidTransformer(nn.Module):
    def __init__(self, enc_num, dim_model, dim_hid, heads_count, in_size=4, share=False, dropout=0.1):
        '''
        Author: ZHP
        description: Point Pyramid Transformer,每个encoder输出最终concat
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
        self.pre_layer, self.back_layer = None, None
        self.encoder_s = clones(EncoderLayer(dim_model, dim_hid, heads_count, share, dropout), enc_num)
        self.config_pos()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, enc_input, postion_input):
        '''
        Author: ZHP
        description: 
        param {torch.tensor} enc_input : 输入数据 [B, N, C]
        param {torch.tensor} postion_input : 位置数据 [B, N, C']
        '''    
        # 缺省生成mask
        enc_input = self.enc_pos_embed(postion_input, enc_input)
        if self.pre_layer is not None:
            enc_input = self.pre_layer(enc_input)
        feature_list = []
        enc_out = enc_input
        for encoder in self.encoder_s:
            enc_out = encoder(enc_out)
            feature_list.append(enc_out)
        if self.back_layer is not None:
            enc_out = self.back_layer(enc_out)
        return enc_out, feature_list

    def config_pos(self):
        self.enc_pos_embed = None
        self.dec_pos_embed = None
    
    def add_layer(self):
        pass

class Learn_Pos_PPT(PointPyramidTransformer):
    def config_pos(self):
        self.enc_pos_embed = PositionEmbedding(K_MLP_Layer(in_size=self.in_size, in_channel=3, channel_list=[3, self.input_dim],\
            bn_list=[False, False], activate_list=[False, False], dropout_list=[False, False]), Pos_Aggregate.add_)

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

        for i in range(0, len(channel_list)):
            self.model.append(getattr(nn, conv)(in_channel, channel_list[i], 1))
            in_channel = channel_list[i]
            if bn_list[i]:
                self.model.append(getattr(nn, bn)(in_channel))
            if activate_list[i]:
                self.model.append(nn.ReLU())
            if dropout_list[i]:
                self.model.append(nn.Dropout(dropout_list[i]))
    
    def forward(self, x):
        # x [B, C, N] or [B, C, H, W]
        shape_len = len(list(x.shape))
        assert shape_len == self.in_size, "expect the input size length as {}, but got {}".format(self.in_size, shape_len)
        pts = x
        for layer in self.model:
            pts = layer(pts)
        return pts



if __name__ == "__main__":
    # model = Mlp_Learn_Trans(enc_num=2, dec_num=2, dim_model=128, dim_hid=64, target_num=256, heads_count=8).cuda()
    # summary(model, [(1024, 6, 128), (1024, 6, 3), (1024, 6, 128)], device="cuda")
    # with open("m1.txt", 'a') as f:
    #     print(model, file=f)

    model = Learn_Pos_PPT(3, 3, 64, 1, 3, False)
    pts = torch.rand(1, 2, 3, dtype=torch.float)
    y, out_list = model(pts, pts.transpose(1,2))
    for i, feas in enumerate(out_list):
        print("the {} feature:".format(i), feas)
    print(y)

    
