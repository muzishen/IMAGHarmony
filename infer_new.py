import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterXL
from tutorial_train_sdxl_ori import ComposedAttention
import os 


'''
须手动调整以下参数与训练的参数相符合
'''
ckpt_inter_dim = 2560
ckpt_cross_heads = 8
ckpt_reshape_blocks = 8
ckpt_cross_value_dim = 64




# 图像生成函数
def generate_image(input_path, prompt, extra_text, output_path="output.png"):
    print(f"\nGenerating image for: {prompt} + {extra_text}")
    
    # 准备输入图像
    input_image = Image.open(input_path).resize((512, 512))
    
    # 生成图像
    images = ip_model.generate(
        pil_image=input_image,
        prompt=prompt,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        scale=1.0,
        guidance_scale=5.0,
        num_samples=1,
        num_inference_steps=30,
        seed=4,
        extra_text=extra_text,
        number_class_crossattention=number_class_crossattention
    )
    
    # 保存结果
    images[0].save(output_path)
    print(f"Saved generated image to: {output_path}")
    return images[0]


if __name__ == "__main__":
    input_image = "/home/sf/code/IMAGHarmony-2/sdxl-fine-tuning/data/images/five cats 3.png"
    device = "cuda:0"  

    # 模型路径配置
    base_model_path = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0"
    image_encoder_path = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/image_encoder"
    # ip_ckpt = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/ip-adapter_sdxl.safetensors"
    fine_tuned_ckpt = "/aigc_data_hdd/all_logs/IMAGHarmony_fivecats/checkpoint-200/ip_adapter.bin"  # 微调后的权重

    save_root = os.path.join(
        '/home/sf/code/IMAGHarmony_new/results/',
        fine_tuned_ckpt.split('/')[3],  # IMAGHarmonyv2
        fine_tuned_ckpt.split('/')[4],  # checkpoint-600
    )
    # 创建路径（如果不存在）
    os.makedirs(save_root, exist_ok=True)
    print(f"Save directory created at: {save_root}")
    
    # 加载SDXL基础模型
    print("Loading base SDXL model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    pipe.to(device)


    number_class_crossattention = ComposedAttention(
        image_hidden_size=1280, # 保持不变
        text_context_dim=2048,  # 保持不变
        inter_dim=ckpt_inter_dim,
        cross_heads=ckpt_cross_heads,
        reshape_blocks=ckpt_reshape_blocks,
        cross_value_dim=ckpt_cross_value_dim,
        scale=1.0 # 保持不变或根据需要调整
    ).to(device).half() 


    # 初始化IP-Adapter
    print("Initializing IP-Adapter with target blocks...")
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

    # 加载微调后的ComposedAttention模块
    print("Loading fine-tuned ComposedAttention...")
 

    generate_image(
        input_path=input_image,
        prompt="pigs,roses in the background, high quality",
        #prompt="dogs,high quaility",
        extra_text="Five cats",#用训练的caption
        output_path=save_root + '/' + input_image.split('/')[-1]
    )