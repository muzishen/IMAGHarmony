accelerate launch --gpu_ids 0 --num_processes 1 --mixed_precision "fp16" \
  train.py \
  --pretrained_model_name_or_path="your path" \
  --pretrained_ip_adapter_path="your path" \
  --image_encoder_path="your path" \
  --data_root_path='your path' \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --learning_rate=2.5e-04 \
  --data_json_file="your path" \
  --weight_decay=0.01 \
  --output_dir="your path" \
  --save_steps=100 \
  --num_train_epochs 2100 \
  --composed_inter_dim=2560 \
  --composed_cross_heads=8 \
  --composed_reshape_blocks=8 \
  --composed_cross_value_dim=64
