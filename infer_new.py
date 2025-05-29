import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterXL
from tutorial_train_sdxl_ori import HarmonyAttention
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
        seed=42,
        extra_text=extra_text,
        number_class_crossattention=number_class_crossattention
    )
    
    # 保存结果
    images[0].save(output_path)
    print(f"Saved generated image to: {output_path}")
    return images[0]


if __name__ == "__main__":
    input_image = "your path to inputimage"
    device = "cuda:2"  

    # 模型路径配置
    base_model_path = "your path"
    image_encoder_path = "your path"

    fine_tuned_ckpt = "fine_tuned model path"  # 微调后的权重

    save_root = os.path.join(
        'your path',
        fine_tuned_ckpt.split('/')[3], 
        fine_tuned_ckpt.split('/')[4], 
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


        # 定义使用的融合方法
    fusion_method = "cross_attention"  # 可选: "cross_attention", "qformer", "mlp"

    number_class_crossattention = HarmonyAttention(
        image_hidden_size=1280,     # 保持不变
        text_context_dim=2048,      # 保持不变
        inter_dim=ckpt_inter_dim,
        cross_heads=ckpt_cross_heads,
        reshape_blocks=ckpt_reshape_blocks,
        cross_value_dim=ckpt_cross_value_dim,
        scale=1.0,                  # 保持不变或根据需要调整
        fusion_method=fusion_method  # 添加融合方法选择
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

    # 加载微调后的HarmonyAttention模块
    print("Loading fine-tuned HarmonyAttention...")
 

    generate_image(
        input_path=input_image,
        prompt="lions",
        extra_text="eight sheep",#用训练的caption
        output_path=save_root + '/' + input_image.split('/')[-1]
    )
