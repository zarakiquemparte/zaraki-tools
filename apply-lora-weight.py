from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import shutil

import os
import argparse
import json

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_name_or_path", type=str)
  parser.add_argument("--peft_model_path", type=str)
  parser.add_argument("--peft_model_weight", type=float, default=1.0)
  parser.add_argument("--output_dir", type=str)
  parser.add_argument("--cpu_mode", action="store_true")
  return parser.parse_args()

def merge(base_model, lora_model, scaling, merge_weight=1.0):
    weights_list = []

    # Loop over all parameters
    for name, param in lora_model.named_parameters():
        # If the parameter name ends with '.weight', it's an original weight
        if name.endswith('.weight'):
            # Make sure it's not a lora_A or lora_B weight
            if not any(substring in name for substring in ['lora_A', 'lora_B']):
                # Construct the names of the corresponding lora_A and lora_B weights
                layers = name.split('.')
                try:
                    layer = lora_model
                    for item in layers[:-1]:  # We go until the penultimate item (excluding the 'weight' part)
                        if 'lora' in item:  # Split further if lora_A or lora_B
                            item, lora_item = item.split('_')
                            layer = getattr(layer, item)
                            layer = getattr(layer, lora_item)
                        else:
                            layer = getattr(layer, item)
                        
                    # Try to get lora_A and lora_B weights
                    lora_A = getattr(layer, 'lora_A').default.weight
                    lora_B = getattr(layer, 'lora_B').default.weight

                    # Add a tuple to the list with the parameter name as the first item
                    weights_list.append((name, param.data, lora_A, lora_B))

                except AttributeError:
                    pass
                    #print(f"Unable to find lora_A or lora_B weights for {name}")

    for (name,weight,a,b) in weights_list:
        ab = b @ a
        weight += ab * scaling * merge_weight
        print(f"Did thing for layer named {name}")
    
    #clean lora loading trash
    for name, module in base_model.named_modules():
        if 'lora_A' in dir(module):
            delattr(module, 'lora_A')
        if 'lora_B' in dir(module):
            delattr(module, 'lora_B')

def get_lora_scaling(lora_model):
    r = lora_model.peft_config["default"].r
    alpha = lora_model.peft_config["default"].lora_alpha

    scaling = alpha/r
    return scaling

def load_model(model_path, lora_path, cpu_mode=False):
    offload_model_path = "./offload"
    offload_peft_path = "./offload_peft"
    shutil.rmtree(offload_model_path, ignore_errors=True)
    shutil.rmtree(offload_peft_path, ignore_errors=True)
    os.makedirs(offload_model_path, exist_ok=True)
    os.makedirs(offload_peft_path, exist_ok=True)

    device_map = "auto"
    float_type = torch.float16
    if cpu_mode:
        float_type = torch.float32
        device_map = torch.device("cpu")

    base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    return_dict=True,
    torch_dtype=float_type,
    device_map = device_map,
    offload_folder=offload_model_path,
    )

    print(f"Loading PEFT: {lora_path}")
    lora_model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=float_type, device_map = device_map, offload_folder=offload_peft_path)
    
    return base_model, lora_model

def initiate_model_lora_merge(model_path, lora_path, output_dir, merge_weight, cpu_mode=False):
    print(model_path)
    print(lora_path)

    if cpu_mode:
        print("Using CPU mode")

    base_model, lora_model = load_model(model_path, lora_path, cpu_mode)
    scaling = get_lora_scaling(lora_model)
    
    print(f"Lora Scaling: {scaling}")
    
    merge(base_model, lora_model, scaling, merge_weight=merge_weight)
    
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if cpu_mode:
        base_model.half()
    final_model = base_model.save_pretrained(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)
    
    print("Done merging.")
    return final_model

def main():

  args = get_args()
  initiate_model_lora_merge(args.base_model_name_or_path, args.peft_model_path, args.output_dir, args.peft_model_weight, args.cpu_mode)
  
if __name__ == "__main__" :
  main()
