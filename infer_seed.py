import torch
import os
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterXL
from tutorial_train_sdxl_ori import ComposedAttention # Assuming tutorial_train_sdxl_ori.py contains ComposedAttention

# Consistent with training parameters
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
    """Generate images for multiple seed values"""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the input image
    input_image = Image.open(input_path).resize((512, 512))
    
    # Loop to generate images for each seed
    for seed in range(start_seed, end_seed + 1):
        # Set the output filename as the seed value
        output_path = os.path.join(output_dir, f"{seed}.png")
        
        # Check if the image already exists (to resume after interruption)
        if os.path.exists(output_path):
            print(f"Image for seed={seed} already exists, skipping...")
            continue
            
        print(f"\nGenerating image for seed={seed}: {prompt} + {extra_text}")
        
        # Generate image
        images = ip_model.generate(
            pil_image=input_image,
            prompt=prompt,
            negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
            scale=1.0,
            guidance_scale=5.0,
            num_samples=1,
            num_inference_steps=30,
            seed=seed,  # Use the current seed
            extra_text=extra_text,
            number_class_crossattention=number_class_crossattention
        )
        
        # Save the result
        images[0].save(output_path)
        print(f"Saved image for seed={seed} to: {output_path}")


if __name__ == "__main__":
    input_image = "/home/sf/code/IMAGHarmony-2/sdxl-fine-tuning/data/images/five cats 3.png"
    device = "cuda:1"  

    # Model path configuration
    base_model_path = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0"
    image_encoder_path = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/image_encoder"
    fine_tuned_ckpt = "/aigc_data_hdd/all_logs/IMAGHarmony_fivecats/checkpoint-200/ip_adapter.bin"

    # Create a dedicated directory to store multi-seed results
    save_root = os.path.join(
        '/home/sf/code/IMAGHarmony_new/results/',
        fine_tuned_ckpt.split('/')[3],  
        fine_tuned_ckpt.split('/')[4],
        "seeds_1_to_100"  # Create a subdirectory specifically for seed results
    )
    
    # Ensure the directory exists
    os.makedirs(save_root, exist_ok=True)
    print(f"Save directory created at: {save_root}")
    
    # Load the SDXL base model
    print("Loading SDXL base model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    pipe.to(device)

    # Initialize the ComposedAttention module
    # Assuming ComposedAttention is defined in tutorial_train_sdxl_ori.py and takes these arguments
    number_class_crossattention = ComposedAttention(
        image_hidden_size=1280, # This argument might be named differently in your ComposedAttention class
        text_context_dim=2048,  # This argument might be named differently in your ComposedAttention class
        inter_dim=ckpt_inter_dim,
        cross_heads=ckpt_cross_heads,
        reshape_blocks=ckpt_reshape_blocks,
        cross_value_dim=ckpt_cross_value_dim,
        scale=1.0
    ).to(device).half() 

    # Initialize IP-Adapter
    print("Initializing IP-Adapter...")
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

    # Perform batch generation
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
