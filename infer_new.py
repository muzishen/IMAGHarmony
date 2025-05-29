import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterXL # Assuming IPAdapterXL is correctly defined in ip_adapter
from tutorial_train_sdxl_ori import HarmonyAttention # Assuming HarmonyAttention is correctly defined
import os


'''
The following parameters must be manually adjusted to match the training parameters.
'''
ckpt_inter_dim = 2560
ckpt_cross_heads = 8
ckpt_reshape_blocks = 8
ckpt_cross_value_dim = 64




# Image generation function
def generate_image(input_path, prompt, extra_text, output_path="output.png"):
    print(f"\nGenerating image for: {prompt} + {extra_text}")
    
    # Prepare input image
    input_image = Image.open(input_path).resize((512, 512))
    
    # Generate image
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
        number_class_crossattention=number_class_crossattention # This should be the HarmonyAttention instance
    )
    
    # Save result
    images[0].save(output_path)
    print(f"Saved generated image to: {output_path}")
    return images[0]


if __name__ == "__main__":
    input_image = "your path to inputimage" # Replace with your actual input image path
    device = "cuda:2"  

    # Model path configuration
    base_model_path = "your path" # Replace with your actual base model path
    image_encoder_path = "your path" # Replace with your actual image encoder path

    fine_tuned_ckpt = "fine_tuned model path"  # Path to the fine-tuned weights (ip_adapter.bin or similar)

    save_root = os.path.join(
        'your path', # Replace with your desired save directory
        fine_tuned_ckpt.split('/')[3] if len(fine_tuned_ckpt.split('/')) > 3 else "default_folder1", 
        fine_tuned_ckpt.split('/')[4] if len(fine_tuned_ckpt.split('/')) > 4 else "default_folder2", 
    )
    # Create path (if it doesn't exist)
    os.makedirs(save_root, exist_ok=True)
    print(f"Save directory created at: {save_root}")
    
    # Load SDXL base model
    print("Loading base SDXL model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )
    pipe.enable_vae_tiling()
    pipe.to(device)


    # Define the fusion method to be used
    fusion_method = "cross_attention"  # Options: "cross_attention", "qformer", "mlp"

    # Initialize HarmonyAttention (this is the `number_class_crossattention` module)
    print("Initializing HarmonyAttention module...")
    number_class_crossattention = HarmonyAttention( # Renamed for clarity, this is your custom attention module
        image_hidden_size=1280,     # Keep unchanged or match your training
        text_context_dim=2048,      # Keep unchanged or match your training
        inter_dim=ckpt_inter_dim,
        cross_heads=ckpt_cross_heads,
        reshape_blocks=ckpt_reshape_blocks,
        cross_value_dim=ckpt_cross_value_dim,
        scale=1.0,                  # Keep unchanged or adjust as needed
        fusion_method=fusion_method  # Add fusion method selection
    ).to(device).half()

    # Initialize IP-Adapter
    print("Initializing IP-Adapter with target blocks...")
    ip_model = IPAdapterXL(
        pipe, 
        image_encoder_path, 
        fine_tuned_ckpt, 
        device,
        target_blocks=["down_blocks.2.attentions.1"],  # Or your specific target blocks
        num_tokens=4, # Or your specific number of tokens
        inference=True,
        number_class_crossattention=number_class_crossattention  # Pass the initialized HarmonyAttention module here
    )


    print("HarmonyAttention weights are expected to be loaded as part of the IP-Adapter checkpoint.")
 

    generate_image(
        input_path=input_image,
        prompt="lions",
        extra_text="eight sheep", # Use the caption from training
        output_path=os.path.join(save_root, os.path.basename(input_image)) # Safer way to construct output path
    )
