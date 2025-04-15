accelerate launch --gpu_ids 0 --num_processes 1 --mixed_precision "fp16" \
  tutorial_train_sdxl_ori.py \
  --pretrained_model_name_or_path="/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0" \
  --pretrained_ip_adapter_path="/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/ip-adapter_sdxl.safetensors" \
  --image_encoder_path="/aigc_data_hdd/checkpoints/stable-diffusion-xl-base-1.0/image_encoder" \
  --data_root_path='/home/sf/code/IMAGHarmony-2/sdxl-fine-tuning/data/images' \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --learning_rate=2.5e-04 \
  --data_json_file="IMAGHarmony-2/sdxl-fine-tuning/data/train.json" \
  --weight_decay=0.01 \
  --output_dir="/aigc_data_hdd/all_logs/IMAGHarmony_interdim2560_crossheads10_reshapeblocks8_crossvaluedim64" \
  --save_steps=100 \
  --num_train_epochs 2100 \
  --composed_inter_dim=2560 \
  --composed_cross_heads=10 \
  --composed_reshape_blocks=8 \
  --composed_cross_value_dim=64