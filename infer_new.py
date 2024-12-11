import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os
from ip_adapter import IPAdapterXL
from train_with_ipa_new import Number_Class_crossAttention
base_model_path = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0"
image_encoder_path = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/image_encoder"
ip_ckpt = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/ip-adapter_sdxl.safetensors"

style_ckpt = "/home/yj/sketch/InstantStyle-main/checkpoint-3/checkpoint-5000/model.safetensors"
net_ckpt='/home/yj/sketch/InstantStyle-main/checkpoint-3/checkpoint-5000/model_1.safetensors'
device = "cuda:2"

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    
)
pipe.enable_vae_tiling()



ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, style_ckpt=style_ckpt,
                target_blocks=[ "down_blocks.2.attentions.1"], inference=True)

number_class_crossattention=Number_Class_crossAttention(hidden_size=1280,cross_attention_dim=2048,scale=0)
number_class_crossattention.load_from_checkpoint(net_ckpt)


#single-image
image = "/home/yj/sketch/InstantStyle-main/image/2/1.jpg"
image = Image.open(image)
image.resize((512, 512))


images = ip_model.generate(pil_image=image,
                           prompt=" on the beach",#场景/风格
                           negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30,
                           seed=42,
                           extra_text="three pigs",#想要生成的
                           number_class_crossattention=number_class_crossattention,
                          )
images[0].save("test5.png")
