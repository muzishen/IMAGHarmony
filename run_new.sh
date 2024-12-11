accelerate launch --gpu_ids 2 --num_processes 1 --mixed_precision "fp16" \
  train_with_ipa_new.py \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --learning_rate=5e-04 \
  --weight_decay=0.01 \
  --output_dir="/home/yj/sketch/InstantStyle-main/checkpoint-3" \
  --save_steps=10000