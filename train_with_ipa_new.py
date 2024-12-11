import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
from safetensors import safe_open
from diffusers.models.attention_processor import Attention
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05,
                 i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = '/home/yj/sketch/InstantStyle-main/image/2'

        # self.data = json.load(open(json_file))  # list of dict: [{"image_file": "1.png", "text": "A dog"}]
        self.image_list = ['1.jpg']
        
        # self.data = []
        # self.text_list = ["a photo of three cats", 
        #                   "a photo of three cats",
        #                   "a photo of three sheep", 
        #                   "a photo of three birds",
        #                   "a photo of three dogs"]
        with open('/home/yj/sketch/InstantStyle-main/data_5/caption_empty.json','r',encoding='utf-8') as file:
            data=json.load(file)
        
        self.text_list = []
        self.text_list_extra=['three dogs']
        # for img in self.image_list:
        #     #print(img)
        #     for i in range(0,len(data)):
        #         if data[i]['image_file']==img:
        #             self.text_list.append(data[i]['caption'])
        for i in range(0,len(self.image_list)):
            self.text_list.append('')
        
        for i,text in enumerate(self.text_list):
            print(os.path.join(self.image_root_path,self.image_list[i]))
            print(text)
            print(self.text_list_extra[i])
        # quit()
        

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        # item = self.data[idx]
        #text = ""
        text = self.text_list[idx]
        # image_file = item["image_file"]
        image_file = self.image_list[idx]

        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        # raw_image_style = Image.open(os.path.join(self.image_root_path, '2', image_file))

        # original size
        original_width, original_height = raw_image.size
        original_size = torch.tensor([original_height, original_width])

        # original size
        # original_width_style, original_height_style = raw_image_style.size
        # original_size_out = torch.tensor([original_height_style, original_width_style])

        image_tensor = self.transform(raw_image.convert("RGB"))
        # image_tensor_style = self.transform(raw_image_style.convert("RGB"))
        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size

        # delta_h_style = image_tensor_style.shape[1] - self.size
        # delta_w_style = image_tensor_style.shape[2] - self.size
        assert not all([delta_h, delta_w])

        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
            # top_style = delta_h_style // 2
            # left_style = delta_w_style // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
            # top_style = np.random.randint(0, delta_h_style + 1)
            # left_style = np.random.randint(0, delta_w_style + 1)

        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left])
        #
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        # image = transforms.functional.crop(
        #     raw_image_style, top=top_style, left=left_style, height=self.size, width=self.size
        # )
        # crop_coords_top_left = torch.tensor([top_style, left_style])

        # clip_image = self.clip_image_processor(images=raw_image_style, return_tensors="pt").pixel_values

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
        text_extra=self.text_list_extra[idx]
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
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
            "text_extra_input_ids":text_extra_input_ids,
            "text_extra_input_ids_2":text_extra_input_ids_2,
        }

    def __len__(self):
        # return len(self.data)
        return len(self.image_list)


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


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds,extra=None):
        
        ip_tokens = self.image_proj_model(image_embeds)
        if extra is not None:
            extra_ip_tokens=ip_tokens+extra
            encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens,extra_ip_tokens], dim=1)
        else:
            encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states,
                               added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        # orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

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
        # self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        # new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        # assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
        
class AttentionProcess(torch.nn.Module):
    def __init__(self,hidden_size,cross_attention_dim=None):
        super().__init__()
        
        self.hidden_size=hidden_size
        self.cross_atteiton=cross_attention_dim
    def __call__(
        self,
        attn,
        text_embeds,
        image_embeds
    ):
        #图像做q
        batch_size,sequence_length,_=text_embeds.shape
        query=attn.to_q(image_embeds)
        #文本做k,v
        key=attn.to_k(text_embeds)
        value=attn.to_v(text_embeds)
        
        
        #多头注意力机制
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query, key, value,  dropout_p=0.0, is_causal=False
        )

        output = output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        output = output.to(query.dtype)
        
        return output
        
        
class Number_Class_crossAttention(torch.nn.Module):#Number_Class_crossAttention
    def __init__(self, hidden_size, cross_attention_dim,scale=1.0):
        super().__init__()
        
        # Cross Attention 层
        self.cross_attention = AttentionProcess(hidden_size==hidden_size,cross_attention_dim=cross_attention_dim)
        self.scale=scale
        self.attention=Attention(query_dim=cross_attention_dim,out_dim=cross_attention_dim)
        
        #图像从1280->2048*4
        self.fc1=nn.Linear(hidden_size,cross_attention_dim*4)
        
        
        
        # Layer Normalization 层
        self.ln = nn.LayerNorm(cross_attention_dim)
        
        # FC 层 2048->2048
        self.fc2 = nn.Linear(cross_attention_dim,cross_attention_dim)
        

    def forward(self, text_embeds,image_embeds):
        # image_features 和 text_features 维度应为 [seq_len, batch_size, input_dim]
        #[1,1,1280] Linear ->[1,1,2048]->q
        #[1,77,2048]->decoder1,decoder2->[1,77,2048]->k,v
        #图像q，文本k,v,cross_attention，LN,FC(2048->1280)->[1,1,1280]->[1,1280]*scale+image_embeds
        
        # 使用 Cross Attention 层
        image_embeds = image_embeds.unsqueeze(1)#[1,1,1280]
       
        #image [1,1,1280]->[1,1,2048*4]->[1,4,2048]
        image_embeds=self.fc1(image_embeds)
        image_embeds=image_embeds.reshape(1,4,2048)
        output=self.cross_attention(self.attention,text_embeds,image_embeds)#[1,4,2048]
        # 对输出进行 Layer Normalization
        output=self.ln(output)
        
        #  FC 层[1,4,2048]->[1,4,2048]
        output=self.fc2(output)
        
        return output*self.scale
    def load_from_checkpoint(self, ckpt_path: str):

        weights = load_file(ckpt_path)

        # 加载权重到模型
        self.load_state_dict(weights)
        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default='/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/ip-adapter_sdxl.safetensors',
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default='/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/image_encoder',
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
        default=1000,
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

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    print("************")
    print("output_dir:{}".format(args.output_dir))
    
    #路径拼接
    logging_dir = Path(args.output_dir, args.logging_dir)
    #accelerator输出、日志文件路径
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    #定义Acceleartor加速器，实现混合精度、多卡训练
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path,
                                                                 subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    # ip-adapter
    #配置投影层的维度
    num_tokens = 4
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=num_tokens,
    )
    
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
                    "to_k_s.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_s.weight": unet_sd[layer_name + ".to_v.weight"],
                    
                }
                
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens, skip=False, inference=True)
                
                attn_procs[name].load_state_dict(weights, strict=False)
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens, skip=True)
    #把attention处理都加载到unet中
    unet.set_attn_processor(attn_procs)
    #把所有attention层提出来变成一个list
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    # load ipa
    if os.path.splitext(args.pretrained_ip_adapter_path)[-1] == ".safetensors":
        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(args.pretrained_ip_adapter_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    else:
        state_dict = torch.load(args.pretrained_ip_adapter_path, map_location="cpu")
    
    
    print('load IPA!')
    adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)
    
    #IPA初始化
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    #prompt生成图片的风格、场景，extra_text控生成图片数量和种类 
    #原始IPA控风格，训好的，直接用，
    #只更新down_blocks.2.attentions.1层的IPA的参数
    for name, param in ip_adapter.unet.named_parameters():
        if 'to_k_ip' in name or 'to_v_ip' in name:
                param.requires_grad_(False)
    #k_s v_s k_ip v_ip都更新了
    #参数没更新
    
    for name, param in ip_adapter.unet.named_parameters():
        if param.requires_grad:
            print(name)
    
    #设置精度
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)  # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    
    #额外文本和参考图的cross_attention
    number_class_crossattention= Number_Class_crossAttention(hidden_size=1280, cross_attention_dim=2048)
    number_class_crossattention.requires_grad_(True)
    
    
    
    # optimizer
    # params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters())
    # 自己的模块更新
    
    
    #使用Adaw优化更新梯度
    params_to_opt = itertools.chain(ip_adapter.adapter_modules.parameters(),number_class_crossattention.parameters())
    
    for name, param in number_class_crossattention.named_parameters():
        if param.requires_grad:
            print(name)
    
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = MyDataset(tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader,number_class_crossattention = accelerator.prepare(ip_adapter, optimizer, train_dataloader,number_class_crossattention)

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(
                        batch["images"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(
                        accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    image_embeds = image_encoder(
                        batch["clip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)
                

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device),
                                                  output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device),
                                                      output_hidden_states=True)
                    
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                   
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)  # concat
                    
                    #extra text
                    encoder_extra_output = text_encoder(batch['text_extra_input_ids'].to(accelerator.device),
                                                  output_hidden_states=True)
                    text_extra_embeds = encoder_extra_output.hidden_states[-2]
                    
                    encoder_extra_output_2 = text_encoder_2(batch['text_extra_input_ids_2'].to(accelerator.device),
                                                      output_hidden_states=True)
                    
                    text_extra_embeds_2 = encoder_extra_output_2.hidden_states[-2]
                    
                    text_extra_embeds=torch.concat([text_extra_embeds,text_extra_embeds_2],dim=-1)
                    
                
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
               
                
                #extra text
                output=number_class_crossattention(text_extra_embeds,image_embeds)
                # print(image_embeds.shape)
                # print(f"out:{output.shape}")
                #image_embeds=image_embeds+output
                # print(f"image:{image_embeds.shape}")
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds,extra=output)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))

            global_step += 1

            # if global_step % args.save_steps == 0:
                # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                # accelerator.save_state(save_path)

            if global_step==1000 or global_step==5000 or global_step==10000:
                print(f"global_step={global_step}  save checkpoint!!!!")
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
            if global_step>=10000:
                print(f"global_step={global_step},all done!!!")
                quit()
            begin = time.perf_counter()


if __name__ == "__main__":
    
    main()
