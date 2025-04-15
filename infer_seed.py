import torch
import os
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterXL
from tutorial_train_sdxl_ori import ComposedAttention

# 与训练参数保持一致
ckpt_inter_dim = 2560
ckpt_cross_heads = 8
ckpt_reshape_blocks = 8
ckpt_cross_value_dim = 64

def generate_images_with_seeds(
    ip_model,
    input_path,
    prompt,
    extra_text,
    output_dir,
    start_seed=1,
    end_seed=100,
    number_class_crossattention=None
):
    """为多个种子值生成图像"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载输入图像
    input_image = Image.open(input_path).resize((512, 512))
    
    # 循环生成每个种子的图像
    for seed in range(start_seed, end_seed + 1):
        # 设置输出文件名为种子值
        output_path = os.path.join(output_dir, f"{seed}.png")
        
        # 检查是否已存在该图像（便于中断后继续）
        if os.path.exists(output_path):
            print(f"图像 seed={seed} 已存在，跳过...")
            continue
            
        print(f"\n正在生成 seed={seed} 的图像: {prompt} + {extra_text}")
        
        # 生成图像
        images = ip_model.generate(
            pil_image=input_image,
            prompt=prompt,
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            scale=1.0,
            guidance_scale=5.0,
            num_samples=1,
            num_inference_steps=30,
            seed=seed,  # 使用当前种子
            extra_text=extra_text,
            number_class_crossattention=number_class_crossattention
        )
        
        # 保存结果
        images[0].save(output_path)
        print(f"已保存 seed={seed} 的图像到: {output_path}")


if __name__ == "__main__":
    input_image = "/home/sf/code/IMAGHarmony-2/sdxl-fine-tuning/data/images/five cats 3.png"
    device = "cuda:1"  

    # 模型路径配置
    base_model_path = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0"
    image_encoder_path = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/image_encoder"
    fine_tuned_ckpt = "/aigc_data_hdd/all_logs/IMAGHarmony_fivecats/checkpoint-200/ip_adapter.bin"

    # 创建专门存储多种子结果的目录
    save_root = os.path.join(
        '/home/sf/code/IMAGHarmony_new/results/',
        fine_tuned_ckpt.split('/')[3],  
        fine_tuned_ckpt.split('/')[4],
        "seeds_1_to_100"  # 创建专门存放种子结果的子目录
    )
    
    # 确保目录存在
    os.makedirs(save_root, exist_ok=True)
    print(f"保存目录创建于: {save_root}")
    
    # 加载SDXL基础模型
    print("加载SDXL基础模型...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    pipe.to(device)

    # 初始化ComposedAttention模块
    number_class_crossattention = ComposedAttention(
        image_hidden_size=1280,
        text_context_dim=2048,
        inter_dim=ckpt_inter_dim,
        cross_heads=ckpt_cross_heads,
        reshape_blocks=ckpt_reshape_blocks,
        cross_value_dim=ckpt_cross_value_dim,
        scale=1.0
    ).to(device).half() 

    # 初始化IP-Adapter
    print("初始化IP-Adapter...")
    ip_model = IPAdapterXL(
        pipe, 
        image_encoder_path, 
        fine_tuned_ckpt, 
        device,
        target_blocks=["down_blocks.2.attentions.1"],  
        num_tokens=4,
        inference=True,
        number_class_crossattention=number_class_crossattention  
    )

    # 执行批量生成
    generate_images_with_seeds(
        ip_model=ip_model,
        input_path=input_image,
        prompt="dogs,roses in the background, high quality",
        extra_text="Five cats",
        output_dir=save_root,
        start_seed=1,
        end_seed=100,
        number_class_crossattention=number_class_crossattention
    )