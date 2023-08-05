import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig
import torch
import argparse
from huggingface_hub import HfApi, login
import json


#mixer output settings

fp16 = True                 #perform operations in fp16. Saves memory, but CPU inference will not be possible.
always_output_fp16 = True   #if true, will output fp16 even if operating in fp32
max_shard_size = "8000MiB"  #set output shard size
verbose_info = True        #will show model information when loading
force_cpu = False            #only use cpu


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_model_path", type=str)
    parser.add_argument("--second_model_path", type=str)
    parser.add_argument("--merged_model_path", type=str)
    parser.add_argument("--merge_ratios", type=str)
    return parser.parse_args()

args = get_args()
first_model_path  = args.first_model_path
second_model_path = args.second_model_path
merged_model_path = args.merged_model_path
merge_ratios_list = args.merge_ratios

with torch.no_grad(): 
    if fp16:
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)

    device = torch.device("cuda") if (torch.cuda.is_available() and not force_cpu) else torch.device("cpu")
    print(device)
    
    # Load the first and second models
    print("Loading Model 1...")
    first_model = AutoModelForCausalLM.from_pretrained(first_model_path)
    first_model = first_model.to(device)
    first_model.eval()
    print("Model 1 Loaded. Dtype: " + str(first_model.dtype))
    
    print("Loading Model 2...")
    second_model = AutoModelForCausalLM.from_pretrained(second_model_path)
    second_model = second_model.to(device)
    second_model.eval()
    print("Model 2 Loaded. Dtype: " + str(second_model.dtype))
    
    # Determine the number of layers in the first model
    num_layers = first_model.config.num_hidden_layers
    #num_layers = len(first_model.transformer.h)
    #model.transformer.h
    #num_layers = len(first_model.encoder.layer)

# Create a "commit and merge" button

def merge_models():
    global first_model, second_model, num_layers, merge_ratios_list, verbose_info, device, merged_model_path, first_model_path, always_output_fp16, max_shard_size, args

    with torch.no_grad():
        # Read the merge ratios from the sliders
        merge_ratios = [float(i) for i in merge_ratios_list.split(',')]
        # Merge the models using the merge ratios
        for i in range(num_layers):
            # Determine how much of each layer to use from each model
            first_ratio = merge_ratios[i]
            second_ratio = 1 - first_ratio
# gpt-j
            # Merge the layer from the two models
            if hasattr(first_model, "transformer"):# and hasattr(first_model.transformer, "h"):
                merged_layer = (first_model.transformer.h[i].state_dict(), second_model.transformer.h[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.transformer.h[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
# maybe BERT
            elif hasattr(first_model, "encoder"):#and hasattr(first_model.encoder, "layer"):
                merged_layer = (first_model.encoder.layer[i].state_dict(), second_model.encoder.layer[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.encoder.layer[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
# opt
            elif hasattr(first_model.model, "decoder"):#and hasattr(first_model.decoder, "layers"):
                merged_layer = (first_model.model.decoder.layers[i].state_dict(), second_model.model.decoder.layers[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.model.decoder.layers[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
# neox/pythia
            elif hasattr(first_model, "gpt_neox"):#and hasattr(first_model.decoder, "layers"):
                tokenizer = AutoTokenizer.from_pretrained(first_model_path, use_fast=True)
                merged_layer = (first_model.gpt_neox.layers[i].state_dict(), second_model.gpt_neox.layers[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.gpt_neox.layers[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
# llama
            elif hasattr(first_model, "model"):#and hasattr(first_model.decoder, "layers"):
                merged_layer = (first_model.model.layers[i].state_dict(), second_model.model.layers[i].state_dict())
                for key in merged_layer[0].keys():
                    merged_layer[0][key] = first_ratio * merged_layer[0][key] + second_ratio * merged_layer[1][key]

                if verbose_info:
                    print("Merging tensor " + str(i))

                # Create the merged model by replacing the layers in the second model with the merged layers
                second_model.model.layers[i].load_state_dict(merged_layer[0])
                if verbose_info:
                    print("Migrating tensor " + str(i))
 
            else:
# model isn't supported
                raise ValueError("Unsupported model architecture")

#anchor got rid of the script generating a converted_model folder, simply adds / to the path now.
        if merged_model_path:
            print("Saving new model...")
            newsavedpath = merged_model_path + "/"
            #copies necessary files from the first selected model folder into the merged model folder
# Define a list of the files to copy
            files_to_copy = ["special_tokens_map.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
# Copy each file to the new folder
            for filename in files_to_copy:
                src_path = f"{first_model_path}/{filename}"
                dst_path = f"{merged_model_path}/{filename}"
                try:
                    shutil.copy2(src_path, dst_path)
                except FileNotFoundError:
                    print("\nFile " + filename + " not found in" + first_model_path + ". Skipping.")
            if always_output_fp16 and not fp16:
                second_model.half()

            second_model.save_pretrained(newsavedpath, max_shard_size=max_shard_size)
            print("\nSaved to: " + newsavedpath)

        else:
            print("\nOutput model was not saved as no output path was selected.")


print(f"Loaded {first_model_path} and {second_model_path}")
print(f"Ratios:\n{merge_ratios_list}")
print(f"Output path: {merged_model_path}")
input("\n Press Enter to continue...")
merge_models()
