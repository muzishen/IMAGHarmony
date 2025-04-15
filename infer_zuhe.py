import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterXL 
from tutorial_train_sdxl_ori import ComposedAttention 
import os
import re 
import gc 


BASE_LOG_DIR = "/aigc_data_hdd/all_logs/"             
BASE_RESULTS_DIR = "/home/sf/code/IMAGHarmony_zuhe/results_dog/" 
BASE_MODEL_PATH = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0" 
IMAGE_ENCODER_PATH = "/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/image_encoder" 
INPUT_IMAGE_PATH = "/home/sf/code/IMAGHarmony_zuhe/assets/8/eight sheep.jpg"
DEVICE = "cuda:2"                                      
PROMPT = "dogs, roses in background"                   # 固定的图像生成提示词
EXTRA_TEXT = "eight sheep"                             # 用于ComposedAttention的文本（应与训练输入图像时使用的caption一致）

# 从目录名解析超参数

def parse_hyperparameters_from_dirname(dirname):

    params = {
        'inter_dim': None,
        'cross_heads': None,
        'reshape_blocks': None,
        'cross_value_dim': None,
    }
    try:
        match = re.search(r"interdim(\d+)", dirname)
        if match: params['inter_dim'] = int(match.group(1))

        match = re.search(r"crossheads(\d+)", dirname)
        if match: params['cross_heads'] = int(match.group(1))

        match = re.search(r"reshapeblocks(\d+)", dirname)
        if match: params['reshape_blocks'] = int(match.group(1))

        match = re.search(r"crossvaluedim(\d+)", dirname)
        if match: params['cross_value_dim'] = int(match.group(1))

        if not all(params.values()):
            print(f"  [警告] 未能从 '{dirname}' 中解析所有超参数。找到: {params}")
            return None
        return params
    except Exception as e:
        print(f"  [错误] 从 '{dirname}' 解析超参数失败: {e}")
        return None


def generate_image(input_path, prompt, extra_text, output_path):

    print(f"\n正在生成图像: {prompt} + {extra_text}")
    print(f"输出路径: {output_path}")
    try:
        input_image = Image.open(input_path).resize((512, 512))
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        images[0].save(output_path)
        print(f"已保存生成的图像到: {output_path}")
        return True
    except Exception as e:
        print(f"[错误] 为 {output_path} 生成图像失败: {e}")
        import traceback
        traceback.print_exc() 
        return False



if __name__ == "__main__":
    
    print("加载基础 SDXL 模型...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(DEVICE)
    pipe.enable_vae_tiling()
    print("基础模型加载完成。")

    total_processed = 0 
    total_errors = 0    

    for training_run_dir_name in sorted(os.listdir(BASE_LOG_DIR)):
        training_run_dir_path = os.path.join(BASE_LOG_DIR, training_run_dir_name)

        if not os.path.isdir(training_run_dir_path):
            continue 

        print(f"\n=============================================")
        print(f"处理训练运行: {training_run_dir_name}")
        print(f"=============================================")

        hyperparams = parse_hyperparameters_from_dirname(training_run_dir_name)
        if hyperparams is None:
            print("  由于解析错误，跳过此训练运行。")
            continue

        for checkpoint_dir_name in sorted(os.listdir(training_run_dir_path)):
            if checkpoint_dir_name.startswith("checkpoint-") and \
               os.path.isdir(os.path.join(training_run_dir_path, checkpoint_dir_name)):

                checkpoint_dir_path = os.path.join(training_run_dir_path, checkpoint_dir_name)
                checkpoint_step = checkpoint_dir_name.split('-')[-1] 
                print(f"\n--- 处理检查点: {checkpoint_step} ---")

                ip_adapter_ckpt_path = os.path.join(checkpoint_dir_path, "ip_adapter.bin")
                

                # 保持目录结构不变
                output_image_dir = os.path.join(
                    BASE_RESULTS_DIR, 
                    training_run_dir_name, # 包含训练运行参数
                    checkpoint_dir_name    # 包含检查点步骤
                )

                base, ext = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))

                output_image_filename = f"{training_run_dir_name}_step{checkpoint_step}_{base}{ext}" 

                output_image_path = os.path.join(output_image_dir, output_image_filename)



                if not os.path.exists(ip_adapter_ckpt_path):
                    print(f"  [跳过] 未找到 ip_adapter.bin 文件: {ip_adapter_ckpt_path}")
                    continue
                

                # if os.path.exists(output_image_path):
                #      print(f"  [跳过] 输出图像已存在: {output_image_path}")
                #      continue

                try:
                    print("  初始化 ComposedAttention...")
                    number_class_crossattention = ComposedAttention(
                        image_hidden_size=1280,
                        text_context_dim=2048,
                        inter_dim=hyperparams['inter_dim'],
                        cross_heads=hyperparams['cross_heads'],
                        reshape_blocks=hyperparams['reshape_blocks'],
                        cross_value_dim=hyperparams['cross_value_dim'],
                        scale=1.0 
                    ).to(DEVICE).half()

                    print(f"  使用检查点初始化 IP-Adapter: {ip_adapter_ckpt_path}")
                    ip_model = IPAdapterXL(
                        pipe, 
                        IMAGE_ENCODER_PATH,
                        ip_adapter_ckpt_path, 
                        DEVICE,
                        target_blocks=None, 
                        num_tokens=4,
                        inference=True,
                        number_class_crossattention=number_class_crossattention 
                    )
                    print("  IP-Adapter 初始化完成并加载权重。")

                    success = generate_image(
                        input_path=INPUT_IMAGE_PATH,
                        prompt=PROMPT,
                        extra_text=EXTRA_TEXT,
                        output_path=output_image_path # 使用新的路径
                    )
                    if success:
                        total_processed += 1
                    else:
                        total_errors += 1

                    print("  清理 GPU 显存...")
                    del ip_model 
                    del number_class_crossattention
                    gc.collect()            
                    torch.cuda.empty_cache() 
                    print("  GPU 显存清理完毕。")

                except Exception as e:
                    print(f"[错误] 处理 {training_run_dir_name} 中的检查点 {checkpoint_step} 失败: {e}")
                    import traceback
                    traceback.print_exc() 
                    total_errors += 1
                    if 'ip_model' in locals(): del ip_model
                    if 'number_class_crossattention' in locals(): del number_class_crossattention
                    gc.collect()
                    torch.cuda.empty_cache()

    print("\n--- 自动化处理总结 ---")
    print(f"成功处理的检查点总数: {total_processed}")
    print(f"遇到的错误总数: {total_errors}")