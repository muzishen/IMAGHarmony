import os
import torch
from collections import OrderedDict

def convert_checkpoint_to_ip_adapter(pytorch_model_path, output_ip_adapter_path):
    
    if not os.path.exists(pytorch_model_path):
        print(f"  [Warning] Source file not found, skipping: {pytorch_model_path}")
        return False

    print(f"  Converting: {pytorch_model_path}")
    try:

        sd = torch.load(pytorch_model_path, map_location="cpu")

        image_proj_sd = OrderedDict()
        ip_sd = OrderedDict()
        composed_sd = OrderedDict()
        

        for k in sd:
            if k.startswith("image_proj_model."):
                image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
            elif k.startswith("adapter_modules."):

                ip_sd[k.replace("adapter_modules.", "")] = sd[k] 
            elif k.startswith("composed_modules."):
                composed_sd[k.replace("composed_modules.", "")] = sd[k]


        if not image_proj_sd and not ip_sd and not composed_sd:
             print(f"  [Warning] No expected keys (image_proj_model, adapter_modules, composed_modules) found in {pytorch_model_path}. Skipping save.")
             return False


        final_sd = {
            "image_proj": image_proj_sd, 
            "ip_adapter": ip_sd, 
            'composed_adapter': composed_sd
        }
        

        torch.save(final_sd, output_ip_adapter_path)
        print(f"  Successfully saved: {output_ip_adapter_path}")
        return True

    except Exception as e:
        print(f"  [Error] Failed to convert {pytorch_model_path}: {e}")
        return False




'''
在下列main中修改目录

'''




if __name__ == "__main__":
    base_log_dir = "/aigc_data_hdd/all_logs/"
    total_converted = 0
    total_skipped = 0
    total_errors = 0

    print(f"Starting conversion process in base directory: {base_log_dir}")


    for training_run_dir_name in os.listdir(base_log_dir):
        training_run_dir_path = os.path.join(base_log_dir, training_run_dir_name)
        
        # Check if it's actually a directory
        if os.path.isdir(training_run_dir_path):
            print(f"\nProcessing training run: {training_run_dir_name}")
            
            # Iterate through items inside the training run directory
            for checkpoint_dir_name in os.listdir(training_run_dir_path):
     
                if checkpoint_dir_name.startswith("checkpoint-") and \
                   os.path.isdir(os.path.join(training_run_dir_path, checkpoint_dir_name)):
                    
                    checkpoint_dir_path = os.path.join(training_run_dir_path, checkpoint_dir_name)
                    print(f"- Found checkpoint directory: {checkpoint_dir_name}")
                    

                    pytorch_model_path = os.path.join(checkpoint_dir_path, "pytorch_model.bin")
                    output_ip_adapter_path = os.path.join(checkpoint_dir_path, "ip_adapter.bin")
                    

                    if os.path.exists(output_ip_adapter_path):
                         print(f"  Output file already exists, skipping: {output_ip_adapter_path}")
                         total_skipped += 1
                         continue 


                    success = convert_checkpoint_to_ip_adapter(pytorch_model_path, output_ip_adapter_path)
                    if success:
                        total_converted += 1
                    else:
                       
                         if not os.path.exists(pytorch_model_path):
                            total_skipped += 1 
                         else:
                            total_errors +=1 

    print("\n--- Conversion Summary ---")
    print(f"Total checkpoints converted: {total_converted}")
    print(f"Total checkpoints skipped (e.g., source missing): {total_skipped}")
    print(f"Total errors during conversion: {total_errors}")