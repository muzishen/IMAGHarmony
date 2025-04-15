import torch
from torch import nn
import math
import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import torch.nn as nn
from diffusers.models.attention_processor import Attention
import torch
import torch.nn.functional as F
import numpy as np
class Cross_Attention(nn.Module):
    def __init__(self, 
                 query_dim,         # Q 投影输入维度
                 context_dim,       # K/V 投影输入维度
                 heads=8, 
                 head_dim=64, 
                 value_dim=None,    # V 降维后维度（默认同 head_dim）
                 out_dim=None):     # 输出维度
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = math.sqrt(head_dim)
        self.value_dim = value_dim if value_dim is not None else head_dim
        self.out_dim = out_dim if out_dim is not None else heads * self.value_dim

        # 线性投影层
        self.to_q = nn.Linear(query_dim, heads * head_dim)
        self.to_k = nn.Linear(context_dim, heads * head_dim)
        self.to_v = nn.Linear(context_dim, heads * self.value_dim)

        # 可选输出投影
        self.out_proj = nn.Linear(heads * self.value_dim, self.out_dim)

    def forward(self, query_input, context_input):
        """
        query_input: [B, Q_len, query_dim]
        context_input: [B, K_len, context_dim]
        """
        B = query_input.size(0)

        # 投影 Q, K, V
        q = self.to_q(query_input).view(B, -1, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, Q_len, head_dim]
        k = self.to_k(context_input).view(B, -1, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, K_len, head_dim]
        v = self.to_v(context_input).view(B, -1, self.heads, self.value_dim).transpose(1, 2)  # [B, heads, K_len, v_dim]

        # Attention 权重
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, heads, Q_len, K_len]
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 加权求和
        attn_output = torch.matmul(attn_probs, v)  # [B, heads, Q_len, v_dim]

        # 拼接 heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.heads * self.value_dim)  # [B, Q_len, heads * v_dim]

        # 输出投影
        output = self.out_proj(attn_output)  # [B, Q_len, out_dim]
        return output
class ImageProjModel(torch.nn.Module):
    """投影模型 - 将CLIP图像特征转换为适合UNet交叉注意力的格式"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim  # UNet交叉注意力的维度
        self.clip_extra_context_tokens = clip_extra_context_tokens  # 额外上下文token数量
        # 线性投影层，将CLIP嵌入转换为多个额外的上下文token
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)  # 标准化层

    def forward(self, image_embeds):
        # 投影CLIP图像嵌入到多个上下文token
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens



class Composed_Attention(torch.nn.Module):#Number_Class_crossAttention
    def __init__(self, hidden_size=1280, cross_attention_dim=64, scale=1.0):
        super().__init__()
        
        # Cross Attention 层
        self.cross_attention = Cross_Attention(query_dim=640, context_dim=2048, heads=10, value_dim=32)
        self.scale=scale
        # self.cross_attention=Attention(query_dim=640, cross_attention_dim=2048, heads=10, dim_head=64)
        
        #图像从1280->2560
        self.fc1=nn.Linear(hidden_size, hidden_size*2)
        
        
        # Layer Normalization 层
        self.ln = nn.LayerNorm(hidden_size)
        
        # FC 层 1280->1280
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        

    def forward(self, text_embeds,image_embeds):

        image_embeds=self.fc1(image_embeds) #[1, 2560]
      
        image_embeds=image_embeds.reshape(1,4,640)
        output = self.cross_attention(image_embeds, text_embeds) #[1,4,320]
        output=output.reshape(1,1280)
        
        # 对输出进行 Layer Normalization
        output=self.ln(output)
        
        #  FC 层[1,4,2048]->[1,4,2048]
        output=self.fc2(output)
        
        return output*self.scale
    
    def load_from_checkpoint(self, ckpt_path: str):
        from safetensors.torch import load_file
        from collections import OrderedDict
        
        # 加载权重文件
        weights = load_file(ckpt_path)
        
        # 初始化两个字典分别存储不同模块的权重
        image_proj_weights = OrderedDict()
        attn_weights = OrderedDict()
        
        # 分离权重到不同模块
        for k, v in weights.items():
            # 处理image_proj_model权重 (匹配两种可能的键名前缀)
            if k.startswith("image_proj_model.") or k.startswith("image_proj."):
                new_key = k.replace("image_proj_model.", "").replace("image_proj.", "")
                if hasattr(self, "image_proj_model") and hasattr(self.image_proj_model, new_key.split('.')[0]):
                    image_proj_weights[new_key] = v
            
            # 处理目标注意力层权重 (匹配两种可能的键名格式)
            elif "down_blocks.2.attentions.1" in k:
                # 转换键名格式: composed_modules.down_blocks.2.attentions.1 -> down_blocks.2.attentions.1
                new_key = k.replace("composed_modules.", "").replace("ip_adapter.", "")
                if hasattr(self, new_key.split('.')[0]):
                    attn_weights[new_key] = v
        
        # 加载image_proj_model权重(严格模式)
        if image_proj_weights:
            self.image_proj_model.load_state_dict(image_proj_weights, strict=True)
            print(f"Loaded image_proj_model weights: {len(image_proj_weights)} params")
        
        # 加载注意力层权重(非严格模式)
        if attn_weights:
            # 创建临时ModuleDict来加载权重
            temp_dict = {k: v for k, v in self.named_modules() 
                        if "down_blocks.2.attentions.1" in k}
            temp_model = torch.nn.ModuleDict(temp_dict)
            
            missing, unexpected = temp_model.load_state_dict(attn_weights, strict=False)
            if missing:
                print(f"Missing keys in attention blocks: {missing}")
            if unexpected:
                print(f"Unexpected keys in attention blocks: {unexpected}")
            
            print(f"Loaded attention weights: {len(attn_weights)} params")
        
        print(f"Successfully loaded target modules from {ckpt_path}")
        return self