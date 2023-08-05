#!/bin/bash

declare -a pefts
declare -a pefts_weights
output_dir=""
cpu_mode=0
base_model=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --pefts_paths)
      pefts_paths=($(echo "$2" | tr ',' ' '))
      shift 2
      ;;
    --pefts_weights)
      pefts_weights=($(echo "$2" | tr ',' ' '))
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      shift 2
      ;;
    --cpu_mode)
      cpu_mode=1
      shift 1
      ;;
    --base_model)
      base_model="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $key"
      usage
      ;;
  esac
done

if [[ ${#pefts_paths[@]} -eq 0 || ${#pefts_weights[@]} -eq 0 || -z $output_dir ]]; then
    echo "Missing required parameters."
    usage
fi

echo "pefts_paths: ${pefts_paths[@]}"
echo "pefts_weights: ${pefts_weights[@]}"
echo "output_dir: $output_dir"
echo "cpu_mode: $cpu_mode"

tmpdir_mother="$HOME/.tmp_model"
rm -rf $tmpdir_mother
mkdir $tmpdir_mother

for ((i=0; i < ${#pefts_paths[@]}; i++)) do
  peft_path=${pefts_paths[$i]}
  peft_weight=${pefts_weights[$i]}
  echo "peft_path: $peft_path"
  echo "peft_weight: $peft_weight"

  current_output_dir="$output_dir"
  
  if [[ $i -lt ${#pefts_paths[@]}-1 ]]; then
    current_output_dir="$tmpdir_mother/model-$i"
    mkdir $current_output_dir
  fi

  if [[ $i -gt 0 ]]; then
    base_model="$tmpdir_mother/model-$((i-1))"
  fi

  if [[ $cpu_mode -eq 1 ]]; then
    echo "Running on CPU"
    python3 apply-lora-weight.py --base_model_name_or_path $base_model --peft_model_path $peft_path --peft_model_weight $peft_weight --output_dir $current_output_dir --cpu_mode
  else
    echo "Running on GPU"
    python3 apply-lora-weight.py --base_model_name_or_path $base_model --peft_model_path $peft_path --peft_model_weight $peft_weight --output_dir $current_output_dir
  fi 
done

rm -rf $tmpdir_mother