# A simple tools for LLama

## Examples

### Apply Lora

```
python3 apply-lora.py --base_model_name_or_path "./base_model/" \
--peft_model_path "./peft_model/" \
--output_dir "/output_model/" \
--cpu_mode
```

### Apply Lora with Weight

```
python3 apply-lora-weight.py --base_model_name_or_path "./base_model/" \
--peft_model_path "./peft_model/" \
--peft_model_weight 0.54 \
--output_dir "./gpt/hermes-limarp-half-7b" \
--cpu_mode
```

### Multiple apply lora with weight

```
./apply-lora-multiple.sh --base_model "./models/base_model" \
--pefts_paths "./loras/lora_1,./loras/lora2" \
--pefts_weights "0.65,0.35" \
--output_dir "./models/output_model" \
--cpu_mode
```

### Merge Model

```
python3 merge-cli.py \
--first_model_path "./models/first_model" \
--second_model_path "./models/second_model" \
--merged_model_path "./models/merged_model" \
--merge_ratios "0.96,0.88,0.8,0.72,0.64,0.56,0.48,0.4,0.32,0.24,0.16,0.08,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
```

