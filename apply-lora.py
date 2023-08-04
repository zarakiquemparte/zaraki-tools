from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import argparse

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_model_name_or_path", type=str)
  parser.add_argument("--peft_model_path", type=str)
  parser.add_argument("--peft_model_path2", type=str)
  parser.add_argument("--peft_model_path3", type=str)
  parser.add_argument("--output_dir", type=str)
  parser.add_argument("--device", type=str, default="auto")
  parser.add_argument("--push_to_hub", action="store_true")
  parser.add_argument("--hub_repo", type=str, default="")
  parser.add_argument("--cpu_mode", action="store_true")
  return parser.parse_args()

def main():

  args = get_args()
  
  if args.device == 'auto':
    device_arg = { 'device_map': 'auto' }
  else:
    device_arg = { 'device_map': { "": args.device} }

  if args.cpu_mode:
    device_arg = { 'device_map': torch.device("cpu") }

  torch_dtype = torch.float16
  if args.cpu_mode:
    torch_dtype = torch.float32

  print(f"Loading base model: {args.base_model_name_or_path}")
  base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch_dtype,
    **device_arg
  )

  print(f"Loading PEFT: {args.peft_model_path}")
  model = PeftModel.from_pretrained(base_model, args.peft_model_path, torch_dtype=torch_dtype, **device_arg)

  print(f"Running merge_and_unload")
  model = model.merge_and_unload()

  if args.peft_model_path2:
    print(f"Loading PEFT2: {args.peft_model_path2}")
    model = PeftModel.from_pretrained(model, args.peft_model_path2, torch_dtype=torch_dtype, **device_arg)
    print(f"Running merge_and_unload2")
    model = model.merge_and_unload()
  if args.peft_model_path3:
    print(f"Loading PEFT3: {args.peft_model_path3}")
    model = PeftModel.from_pretrained(model, args.peft_model_path3, torch_dtype=torch_dtype, **device_arg)
    print(f"Running merge_and_unload3")
    model = model.merge_and_unload()

  tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)

  if args.cpu_mode:
    model.half()

  if args.push_to_hub:
    print(f"Saving to hub ...")
    model.push_to_hub(args.hub_repo, use_temp_dir=True, overwrite=True)
    tokenizer.push_to_hub(args.hub_repo, use_temp_dir=True, overwrite=True)
  else:
    model.save_pretrained(f"{args.output_dir}")
    tokenizer.save_pretrained(f"{args.output_dir}")
    print(f"Model saved to {args.output_dir}")
  
if __name__ == "__main__" :
  main()
