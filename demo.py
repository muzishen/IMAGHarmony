import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterXL
from huggingface_hub import hf_hub_download
import os
import time

try:
    from tutorial_train_sdxl_ori import ComposedAttention
except ImportError:
    print("Error: Could not import ComposedAttention.")
    print("Please ensure 'tutorial_train_sdxl_ori.py' is in the same directory as this script.")
    exit()

print("Loading models, please wait...")

CKPT_INTER_DIM = 2560
CKPT_CROSS_HEADS = 8
CKPT_RESHAPE_BLOCKS = 8
CKPT_CROSS_VALUE_DIM = 64

BASE_MODEL_PATH = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0"
IMAGE_ENCODER_PATH = os.path.join(BASE_MODEL_PATH, "image_encoder")

if not os.path.exists(BASE_MODEL_PATH) or not os.path.exists(IMAGE_ENCODER_PATH):
    print(f"Error: Model or image encoder path not found: {BASE_MODEL_PATH}")
    exit()

IP_ADAPTER_REPO_ID = "kkkkggg/IMAGHarmony"
IP_ADAPTER_FILENAME = "IMAGHarmony_variant1.bin"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Downloading weights file {IP_ADAPTER_FILENAME} from repository {IP_ADAPTER_REPO_ID}...")
try:
    fine_tuned_ckpt_path = hf_hub_download(
        repo_id=IP_ADAPTER_REPO_ID,
        filename=IP_ADAPTER_FILENAME,
    )
    print(f"Weights file downloaded to: {fine_tuned_ckpt_path}")
except Exception as e:
    print(f"Failed to download weights: {e}")
    exit()

print(f"Loading Stable Diffusion XL pipeline from local path: {BASE_MODEL_PATH}")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(DEVICE)
pipe.enable_vae_tiling()

print("Instantiating custom ComposedAttention module...")
number_class_crossattention = ComposedAttention(
    image_hidden_size=1280,
    text_context_dim=2048,
    inter_dim=CKPT_INTER_DIM,
    cross_heads=CKPT_CROSS_HEADS,
    reshape_blocks=CKPT_RESHAPE_BLOCKS,
    cross_value_dim=CKPT_CROSS_VALUE_DIM,
    scale=1.0
).to(DEVICE).half()

print("Extracting and loading ComposedAttention weights from the main checkpoint...")
try:
    state_dict = torch.load(fine_tuned_ckpt_path, map_location="cpu")
    composed_attention_weights = state_dict["composed_adapter"]
    number_class_crossattention.load_state_dict(composed_attention_weights)
    print("Successfully loaded fine-tuned ComposedAttention weights.")
except KeyError:
    print(f"Error: Key 'composed_adapter' not found in weights file {IP_ADAPTER_FILENAME}.")
    exit()
except Exception as e:
    print(f"An unknown error occurred while loading ComposedAttention weights: {e}")
    exit()

print("Initializing IP-Adapter...")
ip_model = IPAdapterXL(
    pipe,
    IMAGE_ENCODER_PATH,
    fine_tuned_ckpt_path,
    DEVICE,
    target_blocks=["down_blocks.1.attentions.1"],
    num_tokens=4,
    inference=True,
    number_class_crossattention=number_class_crossattention
)

print("Models loaded. Gradio application is ready!")


def generate_image(uploaded_image: Image.Image, local_path: str, save_path: str,
                   prompt: str, extra_text: str, negative_prompt: str,
                   guidance_scale: float, num_inference_steps: int, seed: int, progress=gr.Progress()):
    
    pil_image = None
    if uploaded_image is not None:
        pil_image = uploaded_image
    elif local_path and local_path.strip():
        try:
            pil_image = Image.open(local_path.strip())
        except FileNotFoundError:
            raise gr.Error(f"File not found. Please check the path: {local_path.strip()}")
        except Exception as e:
            raise gr.Error(f"Cannot open image file. Error: {e}")
    else:
        raise gr.Error("Please upload a reference image or provide a valid local file path!")

    input_image = pil_image.resize((512, 512))
    progress(0, desc="Generating image...")

    images = ip_model.generate(
        pil_image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        scale=1.0,
        guidance_scale=guidance_scale,
        num_samples=1,
        num_inference_steps=int(num_inference_steps),
        seed=int(seed),
        extra_text=extra_text,
        number_class_crossattention=number_class_crossattention
    )
    generated_image = images[0]
    progress(1, desc="Generation complete!")

    if save_path and save_path.strip():
        try:
            save_dir = save_path.strip()
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"output_{timestamp}_seed{seed}.png"
            full_path = os.path.join(save_dir, filename)
            
            generated_image.save(full_path)
            gr.Info(f"Image successfully saved to: {full_path}")
        except Exception as e:
            gr.Warning(f"Could not save the image! Error: {e}")
            print(f"Error saving image: {e}")

    return generated_image

with gr.Blocks() as demo:
    gr.Markdown("# IMAGHarmony: Image Generation Demo")
    gr.Markdown(
        "**Upload a reference image from your computer, or enter the full local path in the text box below.**\n"
        "Then, enter a **Target Prompt** and a **Reference Content** description to generate a new image."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Your Reference Image")
            local_path_input = gr.Textbox(
                label="Or Enter Local Image Path",
                placeholder="/home/user/images/photo.jpg",
                info="If an image is uploaded, it will be prioritized over the path."
            )
            prompt = gr.Textbox(label="Target Prompt", value="four cats")
            extra_text = gr.Textbox(
                label="Reference Content",
                info="Enter text that describes the reference image, typically the caption used during training.",
                value="four dogs"
            )
            neg_prompt = gr.Textbox(
                label="Negative Prompt",
                value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
            )
            save_path_input = gr.Textbox(
                label="Save to Local Directory (Optional)",
                placeholder="/your/path",
                info="If left empty, the image will not be saved."
            )
            run_button = gr.Button("Generate Image", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Generated Image")

    with gr.Accordion("Advanced Settings", open=False):
        guidance_scale = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=10.0, label="Guidance Scale")
        num_inference_steps = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Inference Steps")
        seed = gr.Slider(minimum=0, maximum=999999, step=1, value=8, label="Seed", randomize=True)

    run_button.click(
        fn=generate_image,
        inputs=[input_image, local_path_input, save_path_input, prompt, extra_text, neg_prompt, guidance_scale, num_inference_steps, seed],
        outputs=output_image
    )

demo.launch(share=True)