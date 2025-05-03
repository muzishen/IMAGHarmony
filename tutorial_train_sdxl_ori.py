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
from collections import OrderedDict
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from safetensors import safe_open
from shared_models import ImageProjModel

from safetensors.torch import save_file, load_file
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor

else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
from ip_adapter.attention_processor import Cross_Attention


def count_model_params(model):
    return sum([p.numel() for p in model.parameters()]) / 1e6

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
    
        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        text_extra = item['extra_text']
        image_file = item["image_file"]
        
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        
        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])
        
        image_tensor = self.transform(raw_image.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        #extra text
        text_extra_input_ids=self.tokenizer(
            text_extra,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_extra_input_ids_2 = self.tokenizer_2(
            text_extra,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "text_extra_input_ids":text_extra_input_ids,
            "text_extra_input_ids_2":text_extra_input_ids_2,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),

        }
        
    
    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    original_size = torch.stack([example["original_size"] for example in data])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data])
    target_size = torch.stack([example["target_size"] for example in data])

    #extra text
    text_extra_input_ids=torch.stack([example["text_extra_input_ids"] for example in data])
    text_extra_input_ids_2=torch.stack([example["text_extra_input_ids_2"] for example in data])
    
    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "text_extra_input_ids":text_extra_input_ids,
        "text_extra_input_ids_2":text_extra_input_ids_2,
    }



class HarmonyAttention(nn.Module):
    def __init__(self,
                 image_hidden_size=1280,     # 输入图像特征维度
                 text_context_dim=2048,      # 输入文本上下文特征维度
                 inter_dim=2560,             # 中间投影维度
                 cross_heads=10,             # cross-attention 多头数量
                 reshape_blocks=8,           # 图像特征分块数量
                 cross_value_dim=64,         # 降维后每个 head 的 value 维度
                 scale=1.0,                  # 输出缩放因子
                 fusion_method="qformer"): # 融合方法选择mlp,cross_attention,qformer
        super().__init__()
        
        self.scale = scale
        self.reshape_blocks = reshape_blocks
        self.cross_query_dim = inter_dim // reshape_blocks
        self.fusion_method = fusion_method
        self.image_hidden_size = image_hidden_size
        self.text_context_dim = text_context_dim
        
        # 所有方法都需要的图像投影
        self.fc1 = nn.Linear(image_hidden_size, inter_dim)
        
        # 1. 交叉注意力
        self.cross_attention = Cross_Attention(
            query_dim=self.cross_query_dim,
            context_dim=text_context_dim,
            heads=cross_heads,
            value_dim=cross_value_dim
        )
        
        # 后处理组件
        flattened_dim = cross_value_dim * cross_heads * reshape_blocks
        self.ln = nn.LayerNorm(flattened_dim)
        self.fc2 = nn.Linear(flattened_dim, image_hidden_size)
        
        # 2. Q-Former
        self.query_tokens = None
        self.qformer_attention = None
        self.qformer_self_attention = None
        
        # 3. MLP
        self.mlp_fusion = None
        self.text_proj = None
        
    def _init_qformer(self, cross_heads, cross_value_dim):
       
        if self.qformer_attention is None:
            
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            
           
            self.query_tokens = nn.Parameter(torch.zeros(1, self.reshape_blocks, self.cross_query_dim, device=device, dtype=dtype))
            nn.init.normal_(self.query_tokens, std=0.02)
            
            # 计算第一次注意力后的维度
            attention_output_dim = cross_heads * cross_value_dim
            
            # 创建注意力层并转移到正确设备和数据类型
            self.qformer_attention = Cross_Attention(
                query_dim=self.cross_query_dim,
                context_dim=self.text_context_dim,
                heads=cross_heads,
                value_dim=cross_value_dim
            ).to(device).to(dtype)
            
            # 修改自注意力层的query_dim和context_dim，匹配第一个注意力层的输出维度
            self.qformer_self_attention = Cross_Attention(
                query_dim=attention_output_dim,  # cross_heads * cross_value_dim
                context_dim=attention_output_dim,  # 对应512维
                heads=cross_heads,
                value_dim=cross_value_dim
            ).to(device).to(dtype)
            
         
            self.img_proj = nn.Linear(
                self.cross_query_dim * self.reshape_blocks, 
                attention_output_dim
            ).to(device).to(dtype)
    
    def _init_mlp_fusion(self):
   
        if self.mlp_fusion is None:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            
            self.mlp_fusion = nn.Sequential(
                nn.Linear(self.image_hidden_size + self.text_context_dim, self.cross_query_dim * self.reshape_blocks),
                nn.GELU(),
                nn.Linear(self.cross_query_dim * self.reshape_blocks, self.image_hidden_size)
            ).to(device).to(dtype)
            
            self.text_proj = nn.Linear(self.text_context_dim, self.text_context_dim).to(device).to(dtype)

    def forward(self, text_embeds, image_embeds):

        B = image_embeds.size(0)
        
        # 根据融合方法选择实现
        if self.fusion_method == "qformer":
            # 延迟初始化Q-Former组件
            self._init_qformer(cross_heads=self.cross_attention.heads, 
                            cross_value_dim=self.cross_attention.value_dim)
            
            # 获取相关维度
            query_dim = self.cross_query_dim  # 通常是320
            attention_output_dim = self.cross_attention.heads * self.cross_attention.value_dim  # 8*64=512
            
            # 扩展查询token以匹配批次大小
            query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, 8, 320]
            
            # 先与文本交互
            attended = self.qformer_attention(query_tokens, text_embeds)  # [B, 8, 512]
            
            # 图像特征处理 - 调整到与attended相同的尺寸
            x = self.fc1(image_embeds)  # [B, 2560]
            x = x.view(B, self.reshape_blocks, self.cross_query_dim)  # [B, 8, 320]
            x = self.cross_attention(x, text_embeds)  # [B, 8, 512] 
            
            # 融合特征 (相加)
            attended = attended + x  # 现在两者维度匹配: [B, 8, 512]
            
            # 再进行自注意力，聚合查询token之间的信息
           
            attended = self.qformer_self_attention(attended, attended)  # [B, 8, 512]
            
        elif self.fusion_method == "mlp":
      
            self._init_mlp_fusion()
            
            # 处理文本特征（如果是序列，则取平均）
            if len(text_embeds.shape) == 3:  # [B, T, D]
                text_feat = self.text_proj(text_embeds).mean(dim=1)  # [B, D]
            else:
                text_feat = self.text_proj(text_embeds)  # [B, D]
                
            # 拼接图像和文本特征
            concat_feat = torch.cat([image_embeds, text_feat], dim=1)
            
            # 通过MLP处理拼接特征
            return self.mlp_fusion(concat_feat) * self.scale
            
        else:  # 默认使用cross_attention
      
            x = self.fc1(image_embeds)  # [B, inter_dim]
            x = x.view(B, self.reshape_blocks, self.cross_query_dim)  # [B, N_blocks, query_dim]
    
            # 交叉注意力：图像块与文本交互
            attended = self.cross_attention(x, text_embeds)  # [B, N_blocks, value_dim * heads]
        
  
        attended = attended.view(B, -1)     # [B, flattened_dim]
        out = self.ln(attended)
        out = self.fc2(out) * self.scale
            
        return out

           
           
class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None, 
                 inter_dim=None,
                 cross_heads=None,
                 reshape_blocks=None,
                 cross_value_dim=None,
                 fusion_method="cross_attention"):  # 融合方法参数
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        
        # 直接使用 fusion_method 参数，不需要转换为布尔标志
        self.composed_modules = HarmonyAttention(
            image_hidden_size=1280,     # 图像特征维度固定
            text_context_dim=2048,      # 文本上下文维度固定
            inter_dim=inter_dim,
            cross_heads=cross_heads,
            reshape_blocks=reshape_blocks,
            cross_value_dim=cross_value_dim,
            scale=1.0,                  # 缩放因子固定
            fusion_method=fusion_method  # 直接传递融合方法参数
        )
        
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)
    
    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds, text_extra_embeds):
        composed_embeds = self.composed_modules(text_extra_embeds, image_embeds)
        image_embeds = image_embeds + composed_embeds
        
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
        
        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--composed_inter_dim",
        type=int,
        default=None,
        help="HarmonyAttention 的中间投影维度 [例如: 1280, 2560]。"
    )
    parser.add_argument(
        "--composed_cross_heads",
        type=int,
        default=None,
        help="HarmonyAttention 中的 cross-attention 多头数量 [例如: 8, 10]。"
    )
    parser.add_argument(
        "--composed_reshape_blocks",
        type=int,
        default=None,
        help="HarmonyAttention 中图像特征的分块数量 [例如: 4, 8]。"
    )
    parser.add_argument(
        "--composed_cross_value_dim",
        type=int,
        default=None,
        help="HarmonyAttention 中降维后每个 head 的 value 维度 [例如: 32, 64]。"
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    # 解析命令行参数
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    # 配置accelerate，用于混合精度训练和分布式训练
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # 创建输出目录
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 加载各种预训练模型组件
    # 噪声添加控制、分词器、文本编码器、VAE和UNet等
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    
    # 冻结基础模型参数，只训练适配器部分
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    # 初始化IP-Adapter模型组件
    # 图像投影模型将CLIP图像嵌入映射到UNet的交叉注意力维度
    num_tokens = 4  # 额外上下文token数量
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=num_tokens,
    )
    
    # 初始化适配器注意力处理器
    # 为UNet中的每个注意力块创建适当的注意力处理器
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    #初始化unet的attention+IPA的cross_attention参数
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            #需要添加IP的层加入额外的attention参数
            if 'down_blocks.2.attentions.1' in name:
        
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }  
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens, skip=False)
                
                attn_procs[name].load_state_dict(weights, strict=False)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens, skip=True)

    #把attention处理都加载到unet中
    unet.set_attn_processor(attn_procs)
    #把所有attention层提出来变成一个list
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())



  
    # 创建完整的IP-Adapter模型
    ip_adapter = IPAdapter(
        unet,
        image_proj_model,
        adapter_modules,
        args.pretrained_ip_adapter_path,
        inter_dim=args.composed_inter_dim,           # 使用命令行参数
        cross_heads=args.composed_cross_heads,         # 使用命令行参数
        reshape_blocks=args.composed_reshape_blocks,   # 使用命令行参数
        cross_value_dim=args.composed_cross_value_dim, # 使用命令行参数
    )

    # ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules)
    # 设置混合精度训练的数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # 将模型移至适当的设备并转换为适当的数据类型
    vae.to(accelerator.device)  # VAE使用fp32以提高稳定性
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # 设置优化器 - 只优化IP-Adapter组件
    params_to_opt = itertools.chain(ip_adapter.adapter_modules.parameters(), ip_adapter.composed_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    accelerator.print("Trainable parameters:  adapter_modules:{:.2f}M, composed_modules:{:.2f}M".format(
    count_model_params(ip_adapter.adapter_modules),
    count_model_params(ip_adapter.composed_modules)))
    # 创建数据集和数据加载器
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # 使用accelerator准备模型、优化器和数据加载器
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    # 开始训练循环
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # 使用VAE将图像转换到潜在空间
                with torch.no_grad():
                    # SDXL的VAE使用fp32以提高数值稳定性
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # 生成随机噪声添加到潜在表示
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # 使用噪声偏移技术提高训练稳定性
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # 为每个图像采样随机时间步
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # 根据时间步添加噪声（前向扩散过程）
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                # 获取CLIP图像嵌入
                with torch.no_grad():
                    image_embeds = image_encoder(batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                
                # 应用图像嵌入丢弃策略
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
            
                # 获取文本嵌入（SDXL使用两个文本编码器）
                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)  # 连接两个文本编码器的输出
                    
                    #extra text
                    encoder_extra_output = text_encoder(batch['text_extra_input_ids'].to(accelerator.device),output_hidden_states=True)
                    text_extra_embeds = encoder_extra_output.hidden_states[-2]
                    encoder_extra_output_2 = text_encoder_2(batch['text_extra_input_ids_2'].to(accelerator.device),output_hidden_states=True)
                    text_extra_embeds_2 = encoder_extra_output_2.hidden_states[-2]
                    text_extra_embeds=torch.concat([text_extra_embeds,text_extra_embeds_2],dim=-1)
                                        
                # 添加SDXL所需的额外条件（图像尺寸和裁剪信息）
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                # 使用IP-Adapter预测噪声
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds, text_extra_embeds)
                
                # 计算MSE损失（预测噪声与实际添加的噪声之间）
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # 收集分布式训练中的损失
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # 反向传播和优化器步骤
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                # 打印训练信息
                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            # 定期保存检查点
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path, safe_serialization=False)
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
